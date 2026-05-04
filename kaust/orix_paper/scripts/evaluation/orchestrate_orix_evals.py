#!/usr/bin/env python
"""Orchestrate orix nanotron-checkpoint evaluations across 8 GPUs.

For each (run, step) work item:
  1. rsync the nanotron checkpoint's model/ + small files from the remote SSH host
     (skipping optimizer/ — saves ~7x bandwidth).
  2. Convert nanotron → HF format (using nanotron conda env).
  3. Run lighteval CF/MC/GEN tasks (using lighteval conda env).
  4. Delete local copies of the nanotron + HF checkpoints to free disk.

Thread-safety: each work item is claimed by exactly one worker via an atomic
mkdir on a per-item lock dir. A worker is bound to a single GPU id and pulls
items off a multiprocessing.Queue serially. There is no inter-worker contention
on disk paths beyond the lock dir.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Hard-coded layout (per plan). Override via CLI flags if needed.
# ---------------------------------------------------------------------------

REMOTE_HOST = "alrashsm@89.169.123.118"
REMOTE_BASE = "/mnt/data/u/alrashsm/nanotron_checkpoints"
# Cross-machine "this checkpoint has been evaluated" flag, written into each
# step dir on the shared checkpoint mount. Both the SSH-based and the
# local-mount orchestrators check this flag before claiming work and write it
# after a successful eval, so two machines never duplicate work on the same
# (run, step). The flag is a 0-byte file: <CHECKPOINT_BASE>/<run>/<step>/.eval_done
EVAL_DONE_FLAG = ".eval_done"

# Shared results bundle on the SSH machine. Every successful (run, step) eval
# pushes its lighteval JSONs here so all machines see one merged result tree.
# Layout: <SHARED_RESULTS_BASE>/<run>/step_<N>/{results_*.json, .completed_*}
# In SSH mode this is a remote path written via rsync. In local mode this is
# a local path on the shared mount written via cp -r.
SHARED_RESULTS_BASE = "/mnt/data/u/alrashsm/orix-results-multiseed-nanotron-evals"
# SSH connection multiplexing: all rsyncs share one TCP connection so the
# remote sshd doesn't rate-limit / reset on burst. Master is started lazily
# by the first rsync that needs it.
SSH_CONTROL_PATH = str(Path.home() / ".ssh/cm/orix-%r@%h:%p")
SSH_OPTS = (
    f"-o ControlMaster=auto -o ControlPath={SSH_CONTROL_PATH} "
    f"-o ControlPersist=600 -o ServerAliveInterval=30 "
    f"-o ConnectTimeout=20"
)
RSYNC_CONCURRENCY = 3  # max simultaneous rsyncs against remote
LOCAL_CKPT_BASE = "/home/alrashsm/orix_local_ckpts"  # transient nanotron copies
LOCAL_HF_BASE = "/home/alrashsm/nanotron-orix-neurips-results-260430/hf_models"
RESULTS_DIR = "/home/alrashsm/nanotron-orix-neurips-results-260430/results"
LOGS_DIR = "/home/alrashsm/nanotron-orix-neurips-results-260430/logs"

# Filled in from CLI args before workers start.
MODE = "ssh"            # "ssh" or "local"
LOCAL_MOUNT_BASE = ""   # only used in local mode
LOCKS_DIR = "/home/alrashsm/nanotron-orix-neurips-results-260430/locks"

NANOTRON_ROOT = "/home/alrashsm/github/nanotron"
NANOTRON_PYTHON = "/home/alrashsm/miniconda3/envs/nanotron/bin/python"
LIGHTEVAL_PYTHON = "/home/alrashsm/miniconda3/envs/lighteval/bin/python"
TOKENIZER = "google/gemma-2b"
# bs=32 OOMs on A100-80GB during multi-choice loglikelihood (4 choices × seq=2048
# blows up activations). bs=16 is the sustainable setting.
BATCH_SIZE = 16

CONFIG_DIR = Path("/home/alrashsm/github/nanotron/kaust/orix_paper/config/evaluation")

# Run → language map (drives which task config to use)
RUN_TO_LANG: dict[str, str] = {
    # Arabic
    "consensus_seed1": "arabic",
    "consensus_seed2": "arabic",
    "arabicweb24_seed1": "arabic",
    "arabicweb24_seed2": "arabic",
    "consensus_dup3": "arabic",
    "consensus_dup4": "arabic",
    # Hindi
    "hinmix_seed1": "hindi",
    "hinmix_seed2": "hindi",
    "culturax_hindi_seed1": "hindi",
    "culturax_hindi_seed2": "hindi",
    "hinmix_dup3": "hindi",
    # Turkish
    "turmix_seed1": "turkish",
    "turmix_seed2": "turkish",
    "fineweb2_turkish_seed1": "turkish",
    "turmix_dup3": "turkish",
    "turmix_dup4": "turkish",
    # Indonesian
    "indmix_seed42": "indonesian",
    "hplt2_indonesian_seed42": "indonesian",
}

LANG_TO_CONFIG: dict[str, str] = {
    "arabic": "arabic_finetasks.txt",
    "hindi": "hindi_finetasks.txt",
    "turkish": "turkish_finetasks.txt",
    "indonesian": "indonesian_finetasks.txt",
    "italian": "italian_finetasks.txt",
    "japanese": "japanese_finetasks.txt",
}

# Steps to evaluate — closest 500-multiples to the prior aramix_paper sampling
# pattern (every 1200 steps). Orix runs only save at 500-step intervals, so we
# can't hit 1200/2400/... exactly; max offset is 200 steps (~0.4 BT on a 30 BT
# horizon, ≈1.4% of x-axis range — visually indistinguishable when overlaid).
#
#   prior (1200-step): 1200 2400 3600 4800 6000 7200 8400 9600 10800 12000 13200 14000
#   orix  (500-step):  1000 2500 3500 5000 6000 7000 8500 9500 11000 12000 13000 14000
DEFAULT_STEPS = [1000, 2500, 3500, 5000, 6000, 7000, 8500, 9500, 11000, 12000, 13000, 14000]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def log(worker_id: int, msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}][gpu{worker_id}] {msg}", flush=True)


def parse_task_config(path: Path) -> tuple[str, str, str]:
    """Read a task config file and return (CF, MC, GEN) comma-joined task lists."""
    cf, mc, gen = "", "", ""
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("CF:"):
            cf = line[3:]
        elif line.startswith("MC:"):
            mc = line[3:]
        elif line.startswith("GEN:"):
            gen = line[4:]
    return cf, mc, gen


def shared_done_check(run: str, step: int) -> bool:
    """Check the cross-machine .eval_done flag on the shared checkpoint mount.
    Returns True if another machine has already evaluated this (run, step)."""
    if MODE == "local":
        return (Path(LOCAL_MOUNT_BASE) / run / str(step) / EVAL_DONE_FLAG).exists()
    # SSH mode: probe the remote filesystem. test -f returns 0 if exists.
    rc = subprocess.run(
        ["ssh"] + SSH_OPTS.split() + [REMOTE_HOST,
         f"test -f {REMOTE_BASE}/{run}/{step}/{EVAL_DONE_FLAG}"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30,
    ).returncode
    return rc == 0


def shared_done_mark(run: str, step: int) -> None:
    """Write the cross-machine .eval_done flag so other machines skip this item."""
    if MODE == "local":
        flag = Path(LOCAL_MOUNT_BASE) / run / str(step) / EVAL_DONE_FLAG
        flag.parent.mkdir(parents=True, exist_ok=True)
        flag.touch()
        return
    # SSH mode
    subprocess.run(
        ["ssh"] + SSH_OPTS.split() + [REMOTE_HOST,
         f"touch {REMOTE_BASE}/{run}/{step}/{EVAL_DONE_FLAG}"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30,
    )


def push_results_to_shared(run: str, step: int, worker_id: int) -> None:
    """Mirror the local results dir for this (run, step) into SHARED_RESULTS_BASE.

    The lighteval JSON files are tiny (a few hundred KB total per step), so we
    sync the whole step_<N>/ tree including .completed_* markers and the
    details/ parquets. Idempotent: rsync only sends what changed."""
    src = Path(RESULTS_DIR) / run / f"step_{step}"
    if not src.exists():
        return
    log_file = Path(LOGS_DIR) / f"push_{run}_step_{step}.log"
    if MODE == "local":
        # Direct copy to local-mount path
        dst = Path(SHARED_RESULTS_BASE) / run / f"step_{step}"
        dst.parent.mkdir(parents=True, exist_ok=True)
        cmd = ["rsync", "-a", f"{src}/", f"{dst}/"]
        run_cmd(cmd, log_file)
        return
    # SSH mode: ensure remote parent dir then rsync up
    subprocess.run(
        ["ssh"] + SSH_OPTS.split() + [REMOTE_HOST,
         f"mkdir -p {SHARED_RESULTS_BASE}/{run}/step_{step}"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30,
    )
    cmd = [
        "rsync", "-az", "--partial",
        "-e", f"ssh {SSH_OPTS}",
        f"{src}/",
        f"{REMOTE_HOST}:{SHARED_RESULTS_BASE}/{run}/step_{step}/",
    ]
    run_cmd(cmd, log_file)


def claim(run: str, step: int) -> bool:
    """Atomically claim a work item via mkdir. Returns True on success, False if
    already claimed/done — including by another machine (via the shared flag)."""
    item_lock = Path(LOCKS_DIR) / f"{run}__step_{step}"
    done_marker = Path(LOCKS_DIR) / f"{run}__step_{step}.done"
    if done_marker.exists():
        return False
    if shared_done_check(run, step):
        # Another machine already finished this; record locally so we don't
        # repeatedly probe over SSH for the same item.
        done_marker.write_text("done by other machine\n")
        return False
    try:
        item_lock.mkdir(parents=True, exist_ok=False)
        return True
    except FileExistsError:
        return False


def mark_done(run: str, step: int, success: bool) -> None:
    item_lock = Path(LOCKS_DIR) / f"{run}__step_{step}"
    done_marker = Path(LOCKS_DIR) / f"{run}__step_{step}.done"
    fail_marker = Path(LOCKS_DIR) / f"{run}__step_{step}.failed"
    target = done_marker if success else fail_marker
    target.write_text(time.strftime("%Y-%m-%dT%H:%M:%S\n"))
    if item_lock.exists() and item_lock.is_dir():
        try:
            item_lock.rmdir()
        except OSError:
            pass


def already_completed(run: str, step: int) -> bool:
    """Skip work that has the lighteval marker files in the results dir."""
    out = Path(RESULTS_DIR) / run / f"step_{step}"
    if not out.exists():
        return False
    cfg_path = CONFIG_DIR / LANG_TO_CONFIG[RUN_TO_LANG[run]]
    cf, mc, gen = parse_task_config(cfg_path)
    needed = []
    if cf:
        needed.append(out / ".completed_cf")
    if mc:
        needed.append(out / ".completed_mc")
    if gen:
        needed.append(out / ".completed_gen")
    return all(p.exists() for p in needed) if needed else False


def run_cmd(cmd: list[str], log_file: Path, env: Optional[dict] = None,
            cwd: Optional[str] = None) -> int:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    full_env = os.environ.copy()
    if env:
        full_env.update(env)
    with open(log_file, "ab", buffering=0) as f:
        f.write(f"\n=== {time.strftime('%Y-%m-%dT%H:%M:%S')} CMD: {' '.join(cmd)}\n".encode())
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=full_env, cwd=cwd)
    return proc.returncode


_RSYNC_SEMAPHORE: Optional["mp.Semaphore"] = None  # populated in main()


def fetch_step(run: str, step: int, worker_id: int) -> Path:
    """Make the nanotron checkpoint accessible locally and return its path.

    In SSH mode: rsync model/ + metadata from remote into LOCAL_CKPT_BASE,
    skipping optimizer/lr_scheduler/random (saves ~7x bandwidth). Uses SSH
    connection multiplexing (ControlMaster) so every rsync shares one TCP+SSH
    connection — avoids remote sshd rate-limit / connection-reset on burst.
    A multiprocessing.Semaphore caps simultaneous rsyncs at RSYNC_CONCURRENCY.

    In local mode: the checkpoints are already present on a local mount, so
    just return the mount path directly (zero copy, zero deletion later)."""
    if MODE == "local":
        local_path = Path(LOCAL_MOUNT_BASE) / run / str(step)
        if not local_path.exists():
            raise RuntimeError(
                f"local-mount path missing: {local_path} "
                f"(check --local-mount and that the checkpoint exists)"
            )
        return local_path

    # SSH mode
    remote_path = f"{REMOTE_HOST}:{REMOTE_BASE}/{run}/{step}/"
    local_path = Path(LOCAL_CKPT_BASE) / run / str(step)
    local_path.mkdir(parents=True, exist_ok=True)
    log_file = Path(LOGS_DIR) / f"rsync_{run}_step_{step}.log"
    # Pull only the model/ dir + small metadata files. Skip optimizer/, lr_scheduler/, random/.
    cmd = [
        "rsync", "-az", "--partial",
        "-e", f"ssh {SSH_OPTS}",
        "--include=/model/***",
        "--include=*.json",
        "--include=*.yaml",
        "--exclude=/optimizer",
        "--exclude=/optimizer/***",
        "--exclude=/lr_scheduler",
        "--exclude=/lr_scheduler/***",
        "--exclude=/random",
        "--exclude=/random/***",
        "--exclude=*",
        remote_path, f"{local_path}/",
    ]
    sem = _RSYNC_SEMAPHORE
    last_err = None
    for attempt in range(4):
        if sem is not None:
            sem.acquire()
        try:
            rc = run_cmd(cmd, log_file)
        finally:
            if sem is not None:
                sem.release()
        if rc == 0:
            return local_path
        last_err = rc
        # Backoff before retrying — longer per attempt
        time.sleep(5 + attempt * 10)
    raise RuntimeError(
        f"rsync failed for {run}/step_{step} (rc={last_err}, see {log_file})"
    )


def convert_to_hf(run: str, step: int, worker_id: int, nanotron_dir: Path,
                  suffix: str = "") -> Path:
    hf_dir = Path(LOCAL_HF_BASE) / f"{run}_step_{step}{suffix}"
    if (hf_dir / "config.json").exists():
        return hf_dir
    log_file = Path(LOGS_DIR) / f"convert_{run}_step_{step}{suffix}.log"
    port = 29500 + worker_id * 100 + (os.getpid() % 50)
    cmd = [
        NANOTRON_PYTHON, "-m", "torch.distributed.run",
        "--nproc_per_node=1", f"--master_port={port}",
        "-m", "examples.llama.convert_nanotron_to_hf",
        "--checkpoint_path", str(nanotron_dir),
        "--save_path", str(hf_dir),
        "--tokenizer_name", TOKENIZER,
    ]
    env = {"CUDA_VISIBLE_DEVICES": str(worker_id)}
    rc = run_cmd(cmd, log_file, env=env, cwd=NANOTRON_ROOT)
    if rc != 0 or not (hf_dir / "config.json").exists():
        raise RuntimeError(f"nanotron->HF convert failed for {run}/step_{step} (rc={rc})")
    return hf_dir


def lighteval(run: str, step: int, hf_path: Path, tasks_csv: str,
              task_type: str, worker_id: int) -> None:
    out_dir = Path(RESULTS_DIR) / run / f"step_{step}"
    out_dir.mkdir(parents=True, exist_ok=True)
    marker = out_dir / f".completed_{task_type}"
    if marker.exists():
        return
    log_file = Path(LOGS_DIR) / f"eval_{run}_step_{step}_{task_type}.log"
    cmd = [
        LIGHTEVAL_PYTHON, "-m", "lighteval", "accelerate",
        f"model_name={hf_path},dtype=bfloat16,batch_size={BATCH_SIZE}",
        tasks_csv,
        "--load-tasks-multilingual",
        "--output-dir", str(out_dir),
        "--save-details",
    ]
    env = {
        "CUDA_VISIBLE_DEVICES": str(worker_id),
        "HF_DATASETS_TRUST_REMOTE_CODE": "1",
        "TOKENIZERS_PARALLELISM": "false",
    }
    rc = run_cmd(cmd, log_file, env=env)
    if rc != 0:
        raise RuntimeError(f"lighteval {task_type} failed for {run}/step_{step} (rc={rc})")
    marker.write_text(time.strftime("%Y-%m-%dT%H:%M:%S\n"))


def process_item(run: str, step: int, worker_id: int) -> bool:
    """Returns True on success, False on failure."""
    if already_completed(run, step):
        log(worker_id, f"{run}/step_{step}: already completed — skipping")
        return True

    if not claim(run, step):
        log(worker_id, f"{run}/step_{step}: claimed by another worker — skipping")
        return True  # not our work, treat as success in dispatch terms

    try:
        log(worker_id, f"{run}/step_{step}: claimed; starting")
        cfg_path = CONFIG_DIR / LANG_TO_CONFIG[RUN_TO_LANG[run]]
        cf, mc, gen = parse_task_config(cfg_path)

        nanotron_dir = fetch_step(run, step, worker_id)

        if cf or mc:
            hf = convert_to_hf(run, step, worker_id, nanotron_dir, suffix="")
            if cf:
                log(worker_id, f"{run}/step_{step}: lighteval CF ({cf.count(',')+1} tasks)")
                lighteval(run, step, hf, cf, "cf", worker_id)
            if mc:
                log(worker_id, f"{run}/step_{step}: lighteval MC")
                lighteval(run, step, hf, mc, "mc", worker_id)
            shutil.rmtree(hf, ignore_errors=True)

        if gen:
            hf_gen = convert_to_hf(run, step, worker_id, nanotron_dir, suffix="_gen")
            log(worker_id, f"{run}/step_{step}: lighteval GEN")
            lighteval(run, step, hf_gen, gen, "gen", worker_id)
            shutil.rmtree(hf_gen, ignore_errors=True)

        # In SSH mode, drop the locally-rsynced nanotron copy. In local mode,
        # nanotron_dir IS the shared mount — never delete from it.
        if MODE == "ssh":
            shutil.rmtree(Path(LOCAL_CKPT_BASE) / run / str(step), ignore_errors=True)

        # Push results JSONs to the shared bundle so the plotter on either
        # machine picks them up.
        try:
            push_results_to_shared(run, step, worker_id)
        except Exception as e:
            log(worker_id, f"{run}/step_{step}: WARN — failed to push shared results: {e}")

        # Tell the other machines this (run, step) is done so they skip it.
        try:
            shared_done_mark(run, step)
        except Exception as e:
            log(worker_id, f"{run}/step_{step}: WARN — failed to write shared done flag: {e}")

        mark_done(run, step, success=True)
        log(worker_id, f"{run}/step_{step}: done")
        return True
    except Exception as e:
        log(worker_id, f"{run}/step_{step}: FAILED — {type(e).__name__}: {e}")
        mark_done(run, step, success=False)
        # Best-effort cleanup of partial state (only for SSH mode — local mode
        # keeps the shared mount untouched)
        if MODE == "ssh":
            shutil.rmtree(Path(LOCAL_CKPT_BASE) / run / str(step), ignore_errors=True)
        for suf in ("", "_gen"):
            shutil.rmtree(Path(LOCAL_HF_BASE) / f"{run}_step_{step}{suf}", ignore_errors=True)
        return False


def worker(worker_id: int, queue: "mp.Queue", rsync_sem: "mp.Semaphore",
           cfg: dict) -> None:
    # Each worker is bound to one GPU; SIGINT kills cleanly.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(1))
    global _RSYNC_SEMAPHORE, MODE, LOCAL_MOUNT_BASE, RESULTS_DIR, LOGS_DIR, LOCKS_DIR
    _RSYNC_SEMAPHORE = rsync_sem
    MODE = cfg["mode"]
    LOCAL_MOUNT_BASE = cfg["local_mount_base"]
    RESULTS_DIR = cfg["results_dir"]
    LOGS_DIR = cfg["logs_dir"]
    LOCKS_DIR = cfg["locks_dir"]
    while True:
        try:
            item = queue.get(timeout=1.0)
        except Exception:
            return
        if item is None:
            return
        run, step = item
        process_item(run, step, worker_id)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="*", default=None,
                        help="Subset of run names; default: all in RUN_TO_LANG with completed remote ckpts")
    parser.add_argument("--steps", nargs="*", type=int, default=DEFAULT_STEPS,
                        help="Step numbers to evaluate; default: 1000..14000 step 1000")
    parser.add_argument("--num-gpus", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--mode", choices=["ssh", "local"], default="ssh",
                        help="ssh: rsync from REMOTE_HOST; local: read directly from --local-mount")
    parser.add_argument("--local-mount", type=str, default=None,
                        help="Local path where nanotron_checkpoints/<run>/<step>/ is mounted. "
                             "Required when --mode local.")
    parser.add_argument("--results-dir", type=str, default=None,
                        help="Override RESULTS_DIR (where lighteval JSONs land)")
    parser.add_argument("--logs-dir", type=str, default=None,
                        help="Override LOGS_DIR")
    parser.add_argument("--locks-dir", type=str, default=None,
                        help="Override LOCKS_DIR (per-machine atomic lock dir)")
    args = parser.parse_args()

    global MODE, LOCAL_MOUNT_BASE, RESULTS_DIR, LOGS_DIR, LOCKS_DIR
    MODE = args.mode
    if MODE == "local":
        if not args.local_mount:
            print("--mode local requires --local-mount /path/to/nanotron_checkpoints",
                  file=sys.stderr)
            return 2
        LOCAL_MOUNT_BASE = args.local_mount
    if args.results_dir:
        RESULTS_DIR = args.results_dir
    if args.logs_dir:
        LOGS_DIR = args.logs_dir
    if args.locks_dir:
        LOCKS_DIR = args.locks_dir

    for d in (LOCAL_CKPT_BASE, LOCAL_HF_BASE, RESULTS_DIR, LOGS_DIR, LOCKS_DIR):
        Path(d).mkdir(parents=True, exist_ok=True)

    runs = args.runs or list(RUN_TO_LANG.keys())
    work_items: list[tuple[str, int]] = []
    for run in runs:
        if run not in RUN_TO_LANG:
            print(f"Unknown run: {run}", file=sys.stderr)
            return 2
        for step in args.steps:
            if already_completed(run, step):
                continue
            work_items.append((run, step))

    print(f"Total work items: {len(work_items)} ({len(runs)} runs × up to {len(args.steps)} steps)")
    if args.dry_run:
        for run, step in work_items[:30]:
            print(f"  {run}/step_{step}")
        if len(work_items) > 30:
            print(f"  ... and {len(work_items)-30} more")
        return 0

    queue: mp.Queue = mp.Queue()
    for item in work_items:
        queue.put(item)
    for _ in range(args.num_gpus):
        queue.put(None)  # sentinel

    rsync_sem = mp.Semaphore(RSYNC_CONCURRENCY)

    if MODE == "ssh":
        # Start the SSH control master eagerly so the first batch of rsyncs
        # reuses it (avoids a thundering-herd of new TCP+SSH handshakes against
        # the remote sshd, which can otherwise rate-limit / reset the burst).
        Path(SSH_CONTROL_PATH).parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["ssh"] + SSH_OPTS.split() + ["-N", "-f", REMOTE_HOST],
            check=False,
        )

    cfg = {
        "mode": MODE,
        "local_mount_base": LOCAL_MOUNT_BASE,
        "results_dir": RESULTS_DIR,
        "logs_dir": LOGS_DIR,
        "locks_dir": LOCKS_DIR,
    }
    workers = []
    for gpu_id in range(args.num_gpus):
        p = mp.Process(target=worker, args=(gpu_id, queue, rsync_sem, cfg), daemon=False)
        p.start()
        workers.append(p)

    try:
        for p in workers:
            p.join()
    except KeyboardInterrupt:
        print("\nInterrupted; terminating workers...", flush=True)
        for p in workers:
            p.terminate()
        for p in workers:
            p.join(timeout=10)
        return 130

    failed = list(Path(LOCKS_DIR).glob("*.failed"))
    print(f"\nDone. {len(failed)} failed item(s).")
    if failed:
        for f in failed[:20]:
            print(f"  FAIL: {f.stem}")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())

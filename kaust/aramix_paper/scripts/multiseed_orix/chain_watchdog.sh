#!/bin/bash
# Chain watchdog: periodically ensures every training run has an active
# queue entry as long as training isn't complete. If a chain breaks
# (no queue entry + latest.txt < 14000), resubmit its run script.
#
# Safe to run alongside the in-job trap resubmit — the name-match check
# avoids double-submitting (a pending+running job with the matching name
# is enough to satisfy "active").
set -u

SCRIPTS_DIR="/home/alrashsm/github/nanotron/kaust/aramix_paper/scripts/multiseed_orix"
CKPT_BASE="/mnt/data/u/alrashsm/nanotron_checkpoints"
TRAIN_STEPS=14000
POLL_INTERVAL=600  # 10 minutes

# (job_name, ckpt_dir, run_script) triples
RUNS=(
    "fwtr-s1  fineweb2_turkish_seed1    run_fineweb2_turkish_seed1.sh"
    "fwtr-s2  fineweb2_turkish_seed2    run_fineweb2_turkish_seed2.sh"
    "hinm-d3  hinmix_dup3               run_hinmix_dup3.sh"
    "hinm-d4  hinmix_dup4               run_hinmix_dup4.sh"
    "turm-d4  turmix_dup4               run_turmix_dup4.sh"
    "itam-s42 itamix_seed42             run_itamix_seed42.sh"
    "indm-s42 indmix_seed42             run_indmix_seed42.sh"
    "fwit-s42 fineweb2_italian_seed42   run_fineweb2_italian_seed42.sh"
    "hpid-s42 hplt2_indonesian_seed42   run_hplt2_indonesian_seed42.sh"
    "itam-s1  itamix_seed1              run_itamix_seed1.sh"
    "itam-s2  itamix_seed2              run_itamix_seed2.sh"
    "fwit-s1  fineweb2_italian_seed1    run_fineweb2_italian_seed1.sh"
    "fwit-s2  fineweb2_italian_seed2    run_fineweb2_italian_seed2.sh"
    "itam-nfw2 itamix_no_fw2            run_itamix_no_fw2.sh"
    "indm-nhpl indmix_no_hplt2          run_indmix_no_hplt2.sh"
    "jpnm-s42  jpnmix_seed42            run_jpnmix_seed42.sh"
    "fwjp-s42  fineweb2_japanese_seed42 run_fineweb2_japanese_seed42.sh"
    "jpnm-nfw2 jpnmix_no_fw2            run_jpnmix_no_fw2.sh"
)

echo "[$(date)] Chain watchdog starting. Poll every ${POLL_INTERVAL}s."
while true; do
    for entry in "${RUNS[@]}"; do
        read -r name ckpt script <<< "$entry"
        # Skip if training complete
        latest=$(cat "${CKPT_BASE}/${ckpt}/latest.txt" 2>/dev/null || echo 0)
        if [ "$latest" -ge "$TRAIN_STEPS" ]; then
            continue
        fi
        # Count active queue entries with this name
        active=$(squeue -u alrashsm -h -o "%j" | grep -cxF "$name" || true)
        if [ "$active" -gt 0 ]; then
            continue  # Already in queue
        fi
        # Try to resubmit
        echo "[$(date +%H:%M)] $name: broken chain (latest=$latest), resubmitting $script..."
        out=$(sbatch "${SCRIPTS_DIR}/${script}" 2>&1)
        if echo "$out" | grep -q "Submitted batch job"; then
            echo "  OK: $out"
        else
            echo "  FAIL: $out"
        fi
    done
    sleep "$POLL_INTERVAL"
done

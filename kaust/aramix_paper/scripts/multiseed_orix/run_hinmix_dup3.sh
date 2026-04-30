#!/bin/bash
#SBATCH --job-name=hinm-d3
#SBATCH --partition=freecycle
#SBATCH --qos=freecycle
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --signal=B:TERM@90
#SBATCH --output=/mnt/data/u/alrashsm/nanotron_checkpoints/logs/hinmix_dup3_%j.out
#SBATCH --error=/mnt/data/u/alrashsm/nanotron_checkpoints/logs/hinmix_dup3_%j.err

set -e

source /home/alrashsm/miniconda3/etc/profile.d/conda.sh
conda activate nanotron

# CHAIN_MODE_UPDATED v2 — do not remove this marker
export HF_HOME="/mnt/data/u/alrashsm/hf_home"
export HF_HUB_CACHE="/mnt/data/u/alrashsm/hf_home/hub"
export HF_DATASETS_CACHE="/mnt/data/u/alrashsm/hf_home/datasets"
export TRANSFORMERS_CACHE="/mnt/data/u/alrashsm/hf_home/transformers"

_CKPT_DIR="/mnt/data/u/alrashsm/nanotron_checkpoints/hinmix_dup3"
_TRAIN_STEPS=14000
_SELF_SCRIPT="/home/alrashsm/github/nanotron/kaust/aramix_paper/scripts/multiseed_orix/run_hinmix_dup3.sh"

_LATEST_STEP=$(cat "${_CKPT_DIR}/latest.txt" 2>/dev/null || echo 0)
if [ "$_LATEST_STEP" -ge "$_TRAIN_STEPS" ]; then
    echo "Training complete at step $_LATEST_STEP. Chain terminates."
    exit 0
fi
echo "Chain link start — resuming from step $_LATEST_STEP, target $_TRAIN_STEPS (job ${SLURM_JOB_ID:-})"

_submit_successor() {
    local ec=$?
    local latest=$(cat "${_CKPT_DIR}/latest.txt" 2>/dev/null || echo 0)
    if [ "$latest" -ge "$_TRAIN_STEPS" ]; then
        echo "Training done at step $latest. Chain terminates."
        return 0
    fi
    echo "Chain link exiting (code=$ec, latest=$latest); submitting successor."
    # Retry sbatch up to 3 times (25s apart) — survives transient
    # AssocMaxSubmitJobLimit hits when slot budget is briefly full.
    for _attempt in 1 2 3; do
        local _out
        _out=$(sbatch "${_SELF_SCRIPT}" 2>&1)
        if echo "$_out" | grep -q "Submitted batch job"; then
            echo "  successor submitted (attempt $_attempt): ${_out##*Submitted batch job }"
            return 0
        fi
        echo "  attempt $_attempt failed: $_out"
        [ "$_attempt" -lt 3 ] && sleep 25
    done
    echo "WARN: chain broke — successor not submitted after 3 attempts."
}
trap _submit_successor EXIT

NANOTRON_DIR="/home/alrashsm/github/nanotron"
SCRIPTS_DIR="${NANOTRON_DIR}/kaust/aramix_paper/scripts/multiseed_orix"
CONFIG_PATH="${NANOTRON_DIR}/kaust/aramix_paper/configs/multiseed_hindi/config_llama_1.46B_hinmix_dup3.yaml"

DATA_BASE="/mnt/data/u/alrashsm"
RAW_SRC="${DATA_BASE}/datasets/hinmix-matched/consensus"
FILTERED_DIR="${DATA_BASE}/datasets/hinmix-matched-3/consensus"
TOKENIZED_DIR="${DATA_BASE}/nanotron_checkpoints/tokenized_data/hinmix-matched-dup3"

mkdir -p "${DATA_BASE}/nanotron_checkpoints/hinmix_dup3"
mkdir -p "${DATA_BASE}/nanotron_checkpoints/logs"

export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=4

echo "=== Training: hinmix dup>=3 ==="
echo "    Config: ${CONFIG_PATH}"
echo "    Node: $(hostname)"
echo "    Job ID: ${SLURM_JOB_ID}"
echo "    Conda env: ${CONDA_DEFAULT_ENV}"

# --- Filter (idempotent: per-file skip if already filtered) ---
echo ""
echo "=== Filter step ==="
python "${SCRIPTS_DIR}/filter_source_count.py" "${RAW_SRC}" "${FILTERED_DIR}" 3

# --- Tokenize (thread-safe via tokenize_helper.sh) ---
echo ""
echo "=== Tokenize step ==="
source "${SCRIPTS_DIR}/tokenize_helper.sh"
tokenize_dataset \
    "${FILTERED_DIR}" \
    "${TOKENIZED_DIR}" \
    "hinmix-matched-dup3" "hf" "text"

# --- Train ---
echo ""
echo "=== Training step ==="
cd "${NANOTRON_DIR}"

srun /home/alrashsm/miniconda3/envs/nanotron/bin/python -m torch.distributed.run \
    --nproc_per_node=8 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=29530 \
    run_train.py --config-file "${CONFIG_PATH}"

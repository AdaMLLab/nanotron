#!/bin/bash
#SBATCH --job-name=cons-dup3
#SBATCH --partition=pi-orabonf
#SBATCH --qos=pi-orabonf
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --time=3-00:00:00
#SBATCH --requeue
#SBATCH --output=/mnt/data/u/alrashsm/nanotron_checkpoints/logs/consensus_dup3_%j.out
#SBATCH --error=/mnt/data/u/alrashsm/nanotron_checkpoints/logs/consensus_dup3_%j.err

set -e

source /home/alrashsm/miniconda3/etc/profile.d/conda.sh
conda activate nanotron

NANOTRON_DIR="/home/alrashsm/github/nanotron"
SCRIPTS_DIR="${NANOTRON_DIR}/kaust/aramix_paper/scripts/multiseed_orix"
CONFIG_PATH="${NANOTRON_DIR}/kaust/aramix_paper/configs/multiseed_aramix/config_llama_1.46B_consensus_dup3.yaml"

DATA_BASE="/mnt/data/u/alrashsm"
RAW_SRC="${DATA_BASE}/datasets/aramix-matched/data/consensus"
FILTERED_DIR="${DATA_BASE}/datasets/aramix-matched-3/data/consensus"
TOKENIZED_DIR="${DATA_BASE}/nanotron_checkpoints/tokenized_data/aramix-consensus-dup3"

mkdir -p "${DATA_BASE}/nanotron_checkpoints/consensus_dup3"
mkdir -p "${DATA_BASE}/nanotron_checkpoints/logs"

export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=4

echo "=== Training: consensus duplicated>=3 ==="
echo "    Config: ${CONFIG_PATH}"
echo "    Node: $(hostname)"
echo "    Job ID: ${SLURM_JOB_ID}"
echo "    Conda env: ${CONDA_DEFAULT_ENV}"

# --- Step 1: Filter parquet files (duplicated >= 3) ---
echo ""
echo "=== Filter step ==="
if ls "${FILTERED_DIR}"/train-*.parquet 1>/dev/null 2>&1; then
    echo "[filter] Already filtered at ${FILTERED_DIR}, skipping."
else
    python "${SCRIPTS_DIR}/filter_parquet.py" "${RAW_SRC}" "${FILTERED_DIR}" 3
fi

# --- Step 2: Tokenize ---
echo ""
echo "=== Tokenize step ==="
if ls "${TOKENIZED_DIR}"/*.ds 1>/dev/null 2>&1; then
    echo "[tokenize] Already tokenized at ${TOKENIZED_DIR}, skipping."
else
    mkdir -p "${TOKENIZED_DIR}"
    cd "${NANOTRON_DIR}"
    python3 tools/preprocess_data.py \
        --tokenizer-name-or-path "google/gemma-2b" \
        --output-folder "${TOKENIZED_DIR}" \
        --n-tasks 32 \
        hf \
        --dataset "${DATA_BASE}/datasets/aramix-matched-3/data/consensus" \
        --column text \
        --split train
fi

# --- Step 3: Train ---
echo ""
echo "=== Training step ==="
cd "${NANOTRON_DIR}"

srun /home/alrashsm/miniconda3/envs/nanotron/bin/python -m torch.distributed.run \
    --nproc_per_node=8 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=29510 \
    run_train.py --config-file "${CONFIG_PATH}"

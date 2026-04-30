#!/bin/bash
#SBATCH --job-name=turm-d3
#SBATCH --partition=freecycle
#SBATCH --qos=freecycle
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --time=3-00:00:00
#SBATCH --requeue
#SBATCH --output=/mnt/data/u/alrashsm/nanotron_checkpoints/logs/turmix_dup3_%j.out
#SBATCH --error=/mnt/data/u/alrashsm/nanotron_checkpoints/logs/turmix_dup3_%j.err

set -e

source /home/alrashsm/miniconda3/etc/profile.d/conda.sh
conda activate nanotron

NANOTRON_DIR="/home/alrashsm/github/nanotron"
SCRIPTS_DIR="${NANOTRON_DIR}/kaust/aramix_paper/scripts/multiseed_orix"
CONFIG_PATH="${NANOTRON_DIR}/kaust/aramix_paper/configs/multiseed_turkish/config_llama_1.46B_turmix_dup3.yaml"

DATA_BASE="/mnt/data/u/alrashsm"
RAW_SRC="${DATA_BASE}/datasets/turmix-matched/consensus"
FILTERED_DIR="${DATA_BASE}/datasets/turmix-matched-3/consensus"
TOKENIZED_DIR="${DATA_BASE}/nanotron_checkpoints/tokenized_data/turmix-matched-dup3"

mkdir -p "${DATA_BASE}/nanotron_checkpoints/turmix_dup3"
mkdir -p "${DATA_BASE}/nanotron_checkpoints/logs"

export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=4

echo "=== Training: turmix dup>=3 ==="
echo "    Config: ${CONFIG_PATH}"
echo "    Node: $(hostname)"
echo "    Job ID: ${SLURM_JOB_ID}"
echo "    Conda env: ${CONDA_DEFAULT_ENV}"

echo ""
echo "=== Filter step ==="
python "${SCRIPTS_DIR}/filter_source_count.py" "${RAW_SRC}" "${FILTERED_DIR}" 3

echo ""
echo "=== Tokenize step ==="
source "${SCRIPTS_DIR}/tokenize_helper.sh"
tokenize_dataset \
    "${FILTERED_DIR}" \
    "${TOKENIZED_DIR}" \
    "turmix-matched-dup3" "hf" "text"

echo ""
echo "=== Training step ==="
cd "${NANOTRON_DIR}"

srun /home/alrashsm/miniconda3/envs/nanotron/bin/python -m torch.distributed.run \
    --nproc_per_node=8 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=29532 \
    run_train.py --config-file "${CONFIG_PATH}"

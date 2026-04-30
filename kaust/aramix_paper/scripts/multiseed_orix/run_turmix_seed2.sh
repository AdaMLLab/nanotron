#!/bin/bash
#SBATCH --job-name=turm-s2
#SBATCH --partition=freecycle
#SBATCH --qos=freecycle
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --time=3-00:00:00
#SBATCH --requeue
#SBATCH --output=/mnt/data/u/alrashsm/nanotron_checkpoints/logs/turmix_seed2_%j.out
#SBATCH --error=/mnt/data/u/alrashsm/nanotron_checkpoints/logs/turmix_seed2_%j.err

set -e

source /home/alrashsm/miniconda3/etc/profile.d/conda.sh
conda activate nanotron

NANOTRON_DIR="/home/alrashsm/github/nanotron"
SCRIPTS_DIR="${NANOTRON_DIR}/kaust/aramix_paper/scripts/multiseed_orix"
CONFIG_PATH="${NANOTRON_DIR}/kaust/aramix_paper/configs/multiseed_turkish/config_llama_1.46B_turmix_seed2.yaml"

mkdir -p /mnt/data/u/alrashsm/nanotron_checkpoints/turmix_seed2
mkdir -p /mnt/data/u/alrashsm/nanotron_checkpoints/logs

export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=4

echo "=== Training: turmix seed2 ==="
echo "    Config: ${CONFIG_PATH}"
echo "    Node: $(hostname)"
echo "    Job ID: ${SLURM_JOB_ID}"
echo "    Conda env: ${CONDA_DEFAULT_ENV}"

source "${SCRIPTS_DIR}/tokenize_helper.sh"
tokenize_dataset \
    "/mnt/data/u/alrashsm/datasets/turmix-matched/consensus" \
    "/mnt/data/u/alrashsm/nanotron_checkpoints/tokenized_data/turmix-matched" \
    "turmix-matched" "hf" "text"

cd "${NANOTRON_DIR}"

srun /home/alrashsm/miniconda3/envs/nanotron/bin/python -m torch.distributed.run \
    --nproc_per_node=8 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=29527 \
    run_train.py --config-file "${CONFIG_PATH}"

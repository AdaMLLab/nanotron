#!/bin/bash
#
# Full pretraining run for LLaMA 1.46B on FinePDFs Arabic dataset
#
# Training configuration:
#   - Batch size: 1024 samples = 2,097,152 tokens/step
#   - Achieved via: 8 GPUs x micro_batch=4 x accumulation=32
#   - Total steps: 14,000
#   - Checkpoint every: 1,200 steps (~2.5B tokens)
#   - Expected checkpoints: ~12 checkpoints
#
# LR schedule:
#   - Warmup: 500 steps (linear)
#   - Decay: 13,500 steps (cosine) starting at step 500
#   - Peak LR: 3e-4, Min LR: 3e-5
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NANOTRON_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
CONFIG_FILE="${SCRIPT_DIR}/config_llama_1.46B_finepdfs_arabic.yaml"

NUM_GPUS=8

export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=4

mkdir -p "/home/alrashsm/github/nanotron/kaust/checkpoints/checkpoints_finepdfs_arabic"

echo "=========================================="
echo "LLaMA 1.46B Pretraining - FinePDFs Arabic"
echo "=========================================="
echo "Config: ${CONFIG_FILE}"
echo "GPUs: ${NUM_GPUS}"
echo ""
echo "Training details:"
echo "  - Batch size: 1024 samples (2,097,152 tokens/step)"
echo "  - Total steps: 14,000"
echo "  - Checkpoint interval: 1,200 steps (~2.5B tokens)"
echo "  - LR: 3e-4 -> 3e-5 (cosine decay)"
echo "=========================================="
echo ""

cd "${NANOTRON_DIR}"

torchrun --nproc_per_node=${NUM_GPUS} \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=29500 \
    run_train.py --config-file "${CONFIG_FILE}"

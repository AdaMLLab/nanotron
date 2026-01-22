#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARAMIX_PAPER_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
NANOTRON_DIR="$(dirname "$(dirname "$ARAMIX_PAPER_DIR")")"
CONFIGS_DIR="${ARAMIX_PAPER_DIR}/configs/hindi"
CHECKPOINTS_DIR="$HOME/nanotron_checkpoints/aramix_paper"

NUM_GPUS=8
export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=4

CONFIG="config_llama_1.46B_hinmix_consensus.yaml"
CKPT_DIR="hinmix_consensus_30bt"

echo "============================================================"
echo "Training: Llama 1.46B on HinMix-Consensus (30BT)"
echo "Config: ${CONFIGS_DIR}/${CONFIG}"
echo "Checkpoints: ${CHECKPOINTS_DIR}/${CKPT_DIR}"
echo "============================================================"

mkdir -p "${CHECKPOINTS_DIR}/${CKPT_DIR}"
cd "${NANOTRON_DIR}"

START_TIME=$(date +%s)

torchrun --nproc_per_node=${NUM_GPUS} \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=29500 \
    run_train.py --config-file "${CONFIGS_DIR}/${CONFIG}"

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
echo "============================================================"
echo "Complete! Duration: $((DURATION / 3600))h $((DURATION % 3600 / 60))m $((DURATION % 60))s"
echo "============================================================"

#!/bin/bash
# Usage: ./run_pretrain.sh <config_path>
#
# Examples:
#   ./run_pretrain.sh /path/to/config_llama_1.46B_aramix.yaml
#   ./run_pretrain.sh /path/to/config_llama_1.46B_consensus.yaml
set -e

CONFIG_PATH="$1"

if [ $# -ne 1 ]; then
    echo "Usage: $0 <config_path>"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NANOTRON_DIR="$(dirname "$SCRIPT_DIR")"

NUM_GPUS=8
export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=4

if [ ! -f "${CONFIG_PATH}" ]; then
    echo "ERROR: Config not found: ${CONFIG_PATH}"
    exit 1
fi

echo "=== Training ==="
echo "    Config: ${CONFIG_PATH}"

cd "${NANOTRON_DIR}"

START_TIME=$(date +%s)

torchrun --nproc_per_node=${NUM_GPUS} \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=29500 \
    run_train.py --config-file "${CONFIG_PATH}"

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
echo "Complete! Duration: $((DURATION / 3600))h $((DURATION % 3600 / 60))m $((DURATION % 60))s"

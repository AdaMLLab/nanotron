#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARAMIX_PAPER_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
NANOTRON_DIR="$(dirname "$(dirname "$ARAMIX_PAPER_DIR")")"
CONFIGS_DIR="${ARAMIX_PAPER_DIR}/configs/arabic"
CHECKPOINTS_DIR="$HOME/nanotron_checkpoints/aramix_paper"

NUM_GPUS=8
export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=4

declare -A MODELS=(
    ["arabicweb24"]="config_llama_1.46B_arabicweb24.yaml|arabicweb24"
    ["consensus"]="config_llama_1.46B_consensus.yaml|consensus_30bt"
    ["finepdfs_arabic"]="config_llama_1.46B_finepdfs_arabic.yaml|finepdfs_arabic"
    ["fineweb_edu_ar"]="config_llama_1.46B_fineweb_edu_ar.yaml|fineweb_edu_ar_30bt"
    ["aramix"]="config_llama_1.46B_aramix.yaml|aramix_30bt"
)

train_model() {
    local name=$1
    local config=$2
    local ckpt_dir=$3

    echo "============================================================"
    echo "Training: $name"
    echo "Config: ${CONFIGS_DIR}/${config}"
    echo "Checkpoints: ${CHECKPOINTS_DIR}/${ckpt_dir}"
    echo "============================================================"

    mkdir -p "${CHECKPOINTS_DIR}/${ckpt_dir}"
    cd "${NANOTRON_DIR}"

    torchrun --nproc_per_node=${NUM_GPUS} \
        --nnodes=1 \
        --node_rank=0 \
        --master_addr=localhost \
        --master_port=29500 \
        run_train.py --config-file "${CONFIGS_DIR}/${config}"
}

if [ $# -eq 0 ]; then
    echo "Usage: $0 <model_name|all>"
    echo "Available models: ${!MODELS[*]}"
    exit 1
fi

START_TIME=$(date +%s)

if [ "$1" == "all" ]; then
    for name in arabicweb24 consensus finepdfs_arabic fineweb_edu_ar; do
        IFS='|' read -r config ckpt_dir <<< "${MODELS[$name]}"
        train_model "$name" "$config" "$ckpt_dir"
    done
else
    if [ -z "${MODELS[$1]}" ]; then
        echo "Unknown model: $1"
        echo "Available models: ${!MODELS[*]}"
        exit 1
    fi
    IFS='|' read -r config ckpt_dir <<< "${MODELS[$1]}"
    train_model "$1" "$config" "$ckpt_dir"
fi

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
echo "============================================================"
echo "Complete! Duration: $((DURATION / 3600))h $((DURATION % 3600 / 60))m $((DURATION % 60))s"
echo "============================================================"

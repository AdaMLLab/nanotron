#!/bin/bash
# Usage: ./run_pretrain.sh <language> <config_yaml> <checkpoint_dir>
#
# Arabic:
#   ./run_pretrain.sh arabic config_llama_1.46B_arabicweb24.yaml arabicweb24
#   ./run_pretrain.sh arabic config_llama_1.46B_aramix.yaml aramix_30bt
#   ./run_pretrain.sh arabic config_llama_1.46B_consensus.yaml consensus_30bt
#   ./run_pretrain.sh arabic config_llama_1.46B_finepdfs_arabic.yaml finepdfs_arabic
#   ./run_pretrain.sh arabic config_llama_1.46B_fineweb_edu_ar.yaml fineweb_edu_ar_30bt
#   ./run_pretrain.sh arabic config_llama_1.46B_aramix_hq.yaml aramix_hq_30bt
#   ./run_pretrain.sh arabic config_llama_1.46B_consensus_no_aw24.yaml consensus_no_aw24_30bt
#   ./run_pretrain.sh arabic config_llama_1.46B_minhash_no_aw24.yaml minhash_no_aw24_30bt
#
# Hindi:
#   ./run_pretrain.sh hindi config_llama_1.46B_culturax_hindi.yaml culturax_hindi_30bt
#   ./run_pretrain.sh hindi config_llama_1.46B_hinmix_hindi.yaml hinmix_minhash_deduped_31bt
#   ./run_pretrain.sh hindi config_llama_1.46B_hinmix_consensus.yaml hinmix_consensus_30bt
#   ./run_pretrain.sh hindi config_llama_1.46B_hinmix_consensus_no_culturax.yaml hinmix_consensus_no_culturax_30bt
#
# Turkish:
#   ./run_pretrain.sh turkish config_llama_1.46B_fineweb2_turkish.yaml fineweb2_turkish_30bt
#   ./run_pretrain.sh turkish config_llama_1.46B_turmix_turkish.yaml turmix_minhash_deduped_31bt
#   ./run_pretrain.sh turkish config_llama_1.46B_turmix_consensus.yaml turmix_consensus_30bt
#   ./run_pretrain.sh turkish config_llama_1.46B_turmix_no_fineweb2_consensus_checked_turkish.yaml turmix_minhash_no_fineweb2_consensus_checked_31bt
set -e

LANGUAGE="$1"
CONFIG="$2"
CKPT_DIR="$3"

if [ $# -ne 3 ]; then
    echo "Usage: $0 <language> <config_yaml> <checkpoint_dir>"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARAMIX_PAPER_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
NANOTRON_DIR="$(dirname "$(dirname "$ARAMIX_PAPER_DIR")")"
CONFIGS_DIR="${ARAMIX_PAPER_DIR}/configs/${LANGUAGE}"
CHECKPOINTS_DIR="$HOME/nanotron_checkpoints/aramix_paper"

NUM_GPUS=8
export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=4

if [ ! -f "${CONFIGS_DIR}/${CONFIG}" ]; then
    echo "ERROR: Config not found: ${CONFIGS_DIR}/${CONFIG}"
    exit 1
fi

echo "=== Training: $CKPT_DIR ==="
echo "    Config: ${CONFIGS_DIR}/${CONFIG}"
echo "    Checkpoints: ${CHECKPOINTS_DIR}/${CKPT_DIR}"

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
echo "Complete! Duration: $((DURATION / 3600))h $((DURATION % 3600 / 60))m $((DURATION % 60))s"

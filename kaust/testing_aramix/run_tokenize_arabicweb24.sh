#!/bin/bash
#
# Tokenize the full ArabicWeb24 dataset for nanotron pretraining
# Uses tools/preprocess_data.py with HuggingFace dataset reader
#
# Dataset info:
#   - Source: /home/alrashsm/data/arabicweb24/ArabicWeb24
#   - Format: HuggingFace datasets (arrow format)
#   - Documents: ~39M
#   - Tokens: ~29.3B
#
# Output:
#   - Tokenized data in nanotron format
#   - Location: /home/alrashsm/tokenized_data/gemma_tokenizer/arabicweb24
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NANOTRON_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

INPUT_DIR="/home/alrashsm/data/arabicweb24/ArabicWeb24"
OUTPUT_DIR="/home/alrashsm/tokenized_data/gemma_tokenizer/arabicweb24"
TOKENIZER="google/gemma-2b"
N_TASKS=32

echo "=========================================="
echo "ArabicWeb24 Tokenization"
echo "=========================================="
echo "Input: ${INPUT_DIR}"
echo "Output: ${OUTPUT_DIR}"
echo "Tokenizer: ${TOKENIZER}"
echo "Parallel tasks: ${N_TASKS}"
echo "=========================================="
echo ""

mkdir -p "${OUTPUT_DIR}"

cd "${NANOTRON_DIR}"

python3 tools/preprocess_data.py \
    --tokenizer-name-or-path "${TOKENIZER}" \
    --output-folder "${OUTPUT_DIR}" \
    --n-tasks ${N_TASKS} \
    --logging-dir "${SCRIPT_DIR}/logs/tokenization_arabicweb24" \
    hf \
    --dataset "${INPUT_DIR}" \
    --column text \
    --split train

echo ""
echo "=========================================="
echo "Tokenization complete!"
echo "Output saved to: ${OUTPUT_DIR}"
echo "=========================================="

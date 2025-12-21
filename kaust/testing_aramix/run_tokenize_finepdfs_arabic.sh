#!/bin/bash
#
# Tokenize the FinePDFs Arabic dataset for nanotron pretraining
# Uses tools/preprocess_data.py with HuggingFace dataset reader
#
# Dataset info:
#   - Source: /home/alrashsm/data/finepdfs-arabic/data/arb_Arab
#   - 16 parquet files (~53GB)
#
# Output:
#   - Tokenized data in nanotron format
#   - Location: /home/alrashsm/tokenized_data/gemma_tokenizer/finepdfs-arabic
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NANOTRON_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

INPUT_DIR="/home/alrashsm/data/finepdfs-arabic/data/arb_Arab"
OUTPUT_DIR="/home/alrashsm/tokenized_data/gemma_tokenizer/finepdfs-arabic"
TOKENIZER="google/gemma-2b"
N_TASKS=16

echo "=========================================="
echo "FinePDFs Arabic Tokenization"
echo "=========================================="
echo "Input: ${INPUT_DIR}"
echo "Output: ${OUTPUT_DIR}"
echo "Tokenizer: ${TOKENIZER}"
echo "Parallel tasks: ${N_TASKS}"
echo "=========================================="
echo ""

# Check if data exists
if [ ! -d "${INPUT_DIR}" ]; then
    echo "ERROR: Data not found at ${INPUT_DIR}"
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"

cd "${NANOTRON_DIR}"

python3 tools/preprocess_data.py \
    --tokenizer-name-or-path "${TOKENIZER}" \
    --output-folder "${OUTPUT_DIR}" \
    --n-tasks ${N_TASKS} \
    --logging-dir "${SCRIPT_DIR}/logs" \
    hf \
    --dataset "${INPUT_DIR}" \
    --column text \
    --split train

echo ""
echo "=========================================="
echo "Tokenization complete!"
echo "Output saved to: ${OUTPUT_DIR}"
echo "=========================================="

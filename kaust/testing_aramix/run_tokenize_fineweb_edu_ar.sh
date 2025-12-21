#!/bin/bash
#
# Tokenize the sampled FineWeb-Edu Arabic dataset for nanotron pretraining
# Uses tools/preprocess_data.py with HuggingFace dataset reader
#
# Prerequisites:
#   - Run run_sample_fineweb_edu_ar.sh first to create the sampled dataset
#
# Dataset info:
#   - Source: /home/alrashsm/data/fineweb-edu-ar-30bt (sampled parquet files)
#   - Target: ~30B tokens
#
# Output:
#   - Tokenized data in nanotron format
#   - Location: /home/alrashsm/tokenized_data/gemma_tokenizer/fineweb-edu-ar-30bt
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NANOTRON_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

INPUT_DIR="/home/alrashsm/data/fineweb-edu-ar-30bt/data"
OUTPUT_DIR="/home/alrashsm/tokenized_data/gemma_tokenizer/fineweb-edu-ar-30bt"
TOKENIZER="google/gemma-2b"
N_TASKS=32

echo "=========================================="
echo "FineWeb-Edu Arabic Tokenization"
echo "=========================================="
echo "Input: ${INPUT_DIR}"
echo "Output: ${OUTPUT_DIR}"
echo "Tokenizer: ${TOKENIZER}"
echo "Parallel tasks: ${N_TASKS}"
echo "=========================================="
echo ""

# Check if sampled data exists
if [ ! -d "${INPUT_DIR}" ]; then
    echo "ERROR: Sampled data not found at ${INPUT_DIR}"
    echo "Please run run_sample_fineweb_edu_ar.sh first."
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

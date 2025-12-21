#!/bin/bash
#
# Sample ~30B tokens from FineWeb-Edu Arabic dataset
#
# Dataset info:
#   - Source: /home/alrashsm/data/fineweb-edu-ar-kaust-gen-ai/ar
#   - Format: HuggingFace dataset (zip files with jsonl)
#   - Target: ~30B tokens
#
# Output:
#   - Sampled parquet files
#   - Location: /home/alrashsm/data/fineweb-edu-ar-30bt/
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

INPUT_DIR="/home/alrashsm/data/fineweb-edu-ar-kaust-gen-ai/ar"
OUTPUT_DIR="/home/alrashsm/data/fineweb-edu-ar-30bt"
TOKENIZER="google/gemma-2b"
TARGET_TOKENS=30.5  # 30.5B to ensure we get ~30B

echo "=========================================="
echo "FineWeb-Edu Arabic Sampling (~30B tokens)"
echo "=========================================="
echo "Input: ${INPUT_DIR}"
echo "Output: ${OUTPUT_DIR}"
echo "Tokenizer: ${TOKENIZER}"
echo "Target: ${TARGET_TOKENS}B tokens"
echo "=========================================="
echo ""

python3 "${SCRIPT_DIR}/sample_fineweb_edu_ar_30bt.py" \
    --input-dir "${INPUT_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --tokenizer "${TOKENIZER}" \
    --target-tokens ${TARGET_TOKENS} \
    --seed 42

echo ""
echo "=========================================="
echo "Sampling complete!"
echo "Output saved to: ${OUTPUT_DIR}"
echo "=========================================="

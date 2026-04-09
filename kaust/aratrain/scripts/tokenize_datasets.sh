#!/bin/bash
# Tokenize all datasets for aratrain experiments.
#
# Datasets:
#   - AdaMLLab/AraMix-HQ (column: text) — from HF
#   - SultanR/finewiki-en-ar (column: ar_text) — from HF
#   - SultanR/ultradata-math-qa-ar — sanitized to local jsonl first
#   - SultanR/ultradata-math-textbook-exercise-ar — sanitized to local jsonl first
#   - SultanR/openresearcher-corpus-ar (column: ar_text) — from HF
#   - SultanR/nyu-aco-ocr (column: text) — from HF
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NANOTRON_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
OUTPUT_BASE=~/tokenized_data/gemma_tokenizer/aratrain

TOKENIZER="google/gemma-2b"
N_TASKS=16

cd "$NANOTRON_DIR"

tokenize_hf() {
    local DATASET="$1"
    local NAME="$2"
    local COLUMN="$3"
    local OUTPUT_DIR="$OUTPUT_BASE/$NAME"

    echo "=== Tokenizing $DATASET (HF) ==="
    echo "    Output: $OUTPUT_DIR"
    mkdir -p "$OUTPUT_DIR"

    python3 tools/preprocess_data.py \
        --tokenizer-name-or-path "$TOKENIZER" \
        --output-folder "$OUTPUT_DIR" \
        --n-tasks "$N_TASKS" \
        hf \
        --dataset "$DATASET" \
        --column "$COLUMN" \
        --split train

    echo "Done: $OUTPUT_DIR"
    echo ""
}

tokenize_jsonl() {
    local INPUT_DIR="$1"
    local NAME="$2"
    local OUTPUT_DIR="$OUTPUT_BASE/$NAME"

    echo "=== Tokenizing $INPUT_DIR (jsonl) ==="
    echo "    Output: $OUTPUT_DIR"
    mkdir -p "$OUTPUT_DIR"

    python3 tools/preprocess_data.py \
        --tokenizer-name-or-path "$TOKENIZER" \
        --output-folder "$OUTPUT_DIR" \
        --n-tasks "$N_TASKS" \
        jsonl \
        --dataset "$INPUT_DIR" \
        --column text

    echo "Done: $OUTPUT_DIR"
    echo ""
}

# Step 1: Sanitize datasets with inconsistent parquet schemas
# echo "=== Sanitizing ultradata-math datasets ==="
# python3 "$SCRIPT_DIR/sanitize_datasets.py"

# Step 2: Tokenize all datasets
# tokenize_hf "AdaMLLab/AraMix-HQ" "aramix-hq" "text"
tokenize_hf "SultanR/finewiki-ar-checked" "finewiki-ar-checked" "ar_text"
# tokenize_jsonl ~/data/aratrain/ultradata-math-qa-ar "ultradata-math-qa-ar"
# tokenize_jsonl ~/data/aratrain/ultradata-math-textbook-exercise-ar "ultradata-math-textbook-exercise-ar"
# tokenize_hf "SultanR/openresearcher-corpus-ar" "openresearcher-corpus-ar" "ar_text"
tokenize_hf "SultanR/nyu-aco-ocr" "nyu-aco-ocr" "text"

echo "=== All datasets tokenized ==="
echo "Verify: ls $OUTPUT_BASE/"

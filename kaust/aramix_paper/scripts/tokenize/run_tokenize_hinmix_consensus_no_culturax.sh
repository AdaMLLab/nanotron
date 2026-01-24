#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARAMIX_PAPER_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
NANOTRON_DIR="$(dirname "$(dirname "$ARAMIX_PAPER_DIR")")"

TOKENIZER="google/gemma-2b"
OUTPUT_BASE="$HOME/tokenized_data/gemma_tokenizer"

# HinMix consensus with CulturaX-only pairs filtered out
INPUT_DIR="$HOME/data/aramix_paper/hinmix-consensus-no_culturax-30bt/data"
OUTPUT_NAME="hinmix-consensus-no_culturax-30bt"
N_TASKS=32

OUTPUT_DIR="${OUTPUT_BASE}/${OUTPUT_NAME}"

echo "============================================================"
echo "Tokenizing: HinMix Consensus (No CulturaX-only pairs)"
echo "Input: $INPUT_DIR"
echo "Output: $OUTPUT_DIR"
echo "============================================================"

if [ ! -d "$INPUT_DIR" ]; then
    echo "ERROR: Data not found at $INPUT_DIR"
    echo "Please run sample_consensus_30bt_no_culturax.py first"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"
cd "$NANOTRON_DIR"

python3 tools/preprocess_data.py \
    --tokenizer-name-or-path "$TOKENIZER" \
    --output-folder "$OUTPUT_DIR" \
    --n-tasks "$N_TASKS" \
    hf \
    --dataset "$INPUT_DIR" \
    --column text \
    --split train

echo "============================================================"
echo "Done: $OUTPUT_DIR"
echo "============================================================"

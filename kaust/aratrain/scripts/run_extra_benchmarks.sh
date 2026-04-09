#!/bin/bash
# Extra benchmark continuation runs (all resume from AraMix-HQ 10B checkpoint).
#   1. 100% ultradata-math-qa
#   2. 100% ultradata-math-textbook
#   3. 50% aramix-hq + 25% ultradata-math-textbook + 25% ultradata-math-qa
#   4. 85% aramix-hq + 15% nyu-aco-ocr
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KAUST_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONFIGS_DIR="$SCRIPT_DIR/../configs"

echo "=== Running extra benchmark continuations ==="
echo "    Base checkpoint: ~/nanotron_checkpoints/aratrain/aramix_hq_10bt"
echo ""

echo "--- 1/4: ultradata-math-qa only ---"
bash "$KAUST_DIR/run_pretrain.sh" "$CONFIGS_DIR/config_llama_1.46B_ultradata_math_qa_only.yaml"

echo "--- 2/4: ultradata-math-textbook only ---"
bash "$KAUST_DIR/run_pretrain.sh" "$CONFIGS_DIR/config_llama_1.46B_ultradata_math_textbook_only.yaml"

echo "--- 3/4: aramix-hq + math blend ---"
bash "$KAUST_DIR/run_pretrain.sh" "$CONFIGS_DIR/config_llama_1.46B_aramix_math_blend.yaml"

echo "--- 4/4: aramix-hq + nyu-aco-ocr ---"
echo "Tokenizing SultanR/nyu-aco-ocr first..."
NANOTRON_DIR="$(cd "$KAUST_DIR/.." && pwd)"
OUTPUT_DIR=~/tokenized_data/gemma_tokenizer/aratrain/nyu-aco-ocr
mkdir -p "$OUTPUT_DIR"
$HOME/miniconda3/envs/nanotron/bin/python "$NANOTRON_DIR/tools/preprocess_data.py" \
    --tokenizer-name-or-path google/gemma-2b \
    --output-folder "$OUTPUT_DIR" \
    --n-tasks 16 \
    hf \
    --dataset SultanR/nyu-aco-ocr \
    --column text \
    --split train
bash "$KAUST_DIR/run_pretrain.sh" "$CONFIGS_DIR/config_llama_1.46B_aramix_nyu_aco_ocr.yaml"

echo "=== All extra benchmark runs complete ==="

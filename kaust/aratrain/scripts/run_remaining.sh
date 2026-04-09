#!/bin/bash
# Run remaining training + evaluate all aratrain checkpoints on Arabic finetasks.
#
# Remaining training:
#   1. Resume aramix_math_blend (interrupted at step 13200)
#   2. Tokenize nyu-aco-ocr + train aramix_nyu_aco_ocr
#
# Then benchmark all completed checkpoints.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KAUST_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
NANOTRON_DIR="$(cd "$KAUST_DIR/.." && pwd)"
CONFIGS_DIR="$SCRIPT_DIR/../configs"
EVAL_SCRIPT="$KAUST_DIR/evaluations/run_finetasks.sh"
TASK_CONFIG="$HOME/github/nanotron/kaust/aramix_paper/config/evaluation/arabic_finetasks.txt"
OUTPUT_DIR="$HOME/nanotron_checkpoint_results/aratrain"
NANOTRON_PYTHON="$HOME/miniconda3/envs/nanotron/bin/python"

echo "=== Step 1/4: Resume aramix_math_blend (from step 13200) ==="
# Temporarily use a config that resumes from its own checkpoint dir
BLEND_CONFIG="$CONFIGS_DIR/config_llama_1.46B_aramix_math_blend.yaml"
BLEND_RESUME_CONFIG="/tmp/config_llama_1.46B_aramix_math_blend_resume.yaml"
sed 's|resume_checkpoint_path: ~/nanotron_checkpoints/aratrain/aramix_hq_10bt|resume_checkpoint_path: ~/nanotron_checkpoints/aratrain/aramix_math_blend_25bt|' \
    "$BLEND_CONFIG" > "$BLEND_RESUME_CONFIG"
bash "$KAUST_DIR/run_pretrain.sh" "$BLEND_RESUME_CONFIG"

echo "=== Step 2/4: Tokenize nyu-aco-ocr ==="
NYU_OUTPUT=~/tokenized_data/gemma_tokenizer/aratrain/nyu-aco-ocr
mkdir -p "$NYU_OUTPUT"
cd "$NANOTRON_DIR"
$NANOTRON_PYTHON tools/preprocess_data.py \
    --tokenizer-name-or-path google/gemma-2b \
    --output-folder "$NYU_OUTPUT" \
    --n-tasks 16 \
    hf \
    --dataset SultanR/nyu-aco-ocr \
    --column text \
    --split train

echo "=== Step 3/4: Train aramix_nyu_aco_ocr ==="
bash "$KAUST_DIR/run_pretrain.sh" "$CONFIGS_DIR/config_llama_1.46B_aramix_nyu_aco_ocr.yaml"

echo "=== Step 4/4: Evaluate all checkpoints on Arabic finetasks ==="
CHECKPOINTS_BASE=~/nanotron_checkpoints/aratrain
MODELS=(
    finewiki_ar_checked_25bt
    ultradata_math_25bt
    openresearcher_25bt
    ultradata_math_qa_only_25bt
    ultradata_math_textbook_only_25bt
    aramix_math_blend_25bt
    aramix_nyu_aco_ocr_25bt
)

for model in "${MODELS[@]}"; do
    checkpoint_path="$CHECKPOINTS_BASE/$model"
    if [ ! -d "$checkpoint_path" ]; then
        echo "WARNING: $checkpoint_path does not exist, skipping"
        continue
    fi
    echo "--- Evaluating $model ---"
    "$EVAL_SCRIPT" \
        "$checkpoint_path" \
        "$OUTPUT_DIR" \
        9999 \
        "$TASK_CONFIG" \
        "google/gemma-2b" \
        8 \
        8
    echo ""
done

echo "=== All done. Results in $OUTPUT_DIR ==="

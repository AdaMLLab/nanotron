#!/bin/bash
# Train all 50/50 aramix-hq blends then evaluate on Arabic finetasks.
#
# Runs (all 50% aramix-hq + 50% X, resuming from base 10B checkpoint):
#   1. aramix50 + nyu-aco-ocr50
#   2. aramix50 + openresearcher50
#   3. aramix50 + finewiki50
#   4. aramix50 + math-qa50
#   5. aramix50 + math-textbook50
#
# Then evaluates all 5 on Arabic finetasks.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KAUST_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONFIGS_DIR="$SCRIPT_DIR/../configs"
EVAL_SCRIPT="$KAUST_DIR/evaluations/run_finetasks.sh"
TASK_CONFIG="$HOME/github/nanotron/kaust/aramix_paper/config/evaluation/arabic_finetasks.txt"
OUTPUT_DIR="$HOME/nanotron_checkpoint_results/aratrain"

MODELS=(
    aramix50_nyu_aco_ocr50
    aramix50_openresearcher50
    aramix50_finewiki50
    aramix50_math_qa50
    aramix50_math_textbook50
)

echo "=== Training 50/50 blends ==="

for i in "${!MODELS[@]}"; do
    model="${MODELS[$i]}"
    echo "--- $((i+1))/${#MODELS[@]}: $model ---"
    bash "$KAUST_DIR/run_pretrain.sh" "$CONFIGS_DIR/config_llama_1.46B_${model}.yaml"
    echo ""
done

echo "=== Evaluating all 50/50 blends ==="

for model in "${MODELS[@]}"; do
    checkpoint_path="$HOME/nanotron_checkpoints/aratrain/${model}_25bt"
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

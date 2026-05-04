#!/bin/bash
# Tokenize, train, and evaluate DCLM-Pro-IBEX and FineWeb-Edu-IBEX experiments.
#
# Tokenization (2 datasets):
#   1. SultanR/dclm-pro-ibex-merged-260315
#   2. SultanR/fineweb-edu-ibex-merged-260315
#
# Training (8 runs, all resume from AraMix-HQ 10B base checkpoint):
#   1. 100% DCLM-Pro-IBEX
#   2. 50% AraMix + 50% DCLM-Pro-IBEX
#   3. 25% AraMix + 75% DCLM-Pro-IBEX
#   4. 75% AraMix + 25% DCLM-Pro-IBEX
#   5. 100% FineWeb-Edu-IBEX
#   6. 50% AraMix + 50% FineWeb-Edu-IBEX
#   7. 25% AraMix + 75% FineWeb-Edu-IBEX
#   8. 75% AraMix + 25% FineWeb-Edu-IBEX
#
# Then evaluates all 8 on Arabic finetasks.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KAUST_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
NANOTRON_DIR="$(cd "$KAUST_DIR/.." && pwd)"
CONFIGS_DIR="$SCRIPT_DIR/../configs"
EVAL_SCRIPT="$KAUST_DIR/evaluations/run_finetasks.sh"
TASK_CONFIG="$HOME/github/nanotron/kaust/aramix_paper/config/evaluation/arabic_finetasks.txt"
OUTPUT_DIR="$HOME/nanotron_checkpoint_results/aratrain"
NANOTRON_PYTHON="$HOME/miniconda3/envs/nanotron/bin/python"
OUTPUT_BASE=~/tokenized_data/gemma_tokenizer/aratrain

echo "=== Step 1: Tokenize datasets ==="

cd "$NANOTRON_DIR"

# Tokenize DCLM-Pro-IBEX (from local download)
DCLM_DATA=~/data/aratrain/dclm-pro-ibex
DCLM_OUTPUT="$OUTPUT_BASE/dclm-pro-ibex"
if [ -d "$DCLM_OUTPUT" ] && [ "$(ls -A "$DCLM_OUTPUT"/*.ds 2>/dev/null | head -1)" ]; then
    echo "DCLM-Pro-IBEX already tokenized, skipping"
else
    echo "--- Tokenizing DCLM-Pro-IBEX from $DCLM_DATA ---"
    mkdir -p "$DCLM_OUTPUT"
    $NANOTRON_PYTHON tools/preprocess_data.py \
        --tokenizer-name-or-path google/gemma-2b \
        --output-folder "$DCLM_OUTPUT" \
        --n-tasks 16 \
        hf \
        --dataset "$DCLM_DATA" \
        --column text \
        --split train
fi

# Tokenize FineWeb-Edu-IBEX (from local download)
FWE_DATA=~/data/aratrain/fineweb-edu-ibex
FWE_OUTPUT="$OUTPUT_BASE/fineweb-edu-ibex"
if [ -d "$FWE_OUTPUT" ] && [ "$(ls -A "$FWE_OUTPUT"/*.ds 2>/dev/null | head -1)" ]; then
    echo "FineWeb-Edu-IBEX already tokenized, skipping"
else
    echo "--- Tokenizing FineWeb-Edu-IBEX from $FWE_DATA ---"
    mkdir -p "$FWE_OUTPUT"
    $NANOTRON_PYTHON tools/preprocess_data.py \
        --tokenizer-name-or-path google/gemma-2b \
        --output-folder "$FWE_OUTPUT" \
        --n-tasks 16 \
        hf \
        --dataset "$FWE_DATA" \
        --column text \
        --split train
fi

# Clean up empty unshuffled files that can cause training crashes
find "$DCLM_OUTPUT" "$FWE_OUTPUT" -name "*_unshuffled.ds" -size 0 -delete 2>/dev/null || true

echo ""
echo "=== Step 2: Train all 8 runs ==="

MODELS=(
    dclm_pro_ibex_100
    aramix50_dclm_pro_ibex50
    aramix25_dclm_pro_ibex75
    aramix75_dclm_pro_ibex25
    fineweb_edu_ibex_100
    aramix50_fineweb_edu_ibex50
    aramix25_fineweb_edu_ibex75
    aramix75_fineweb_edu_ibex25
)

for i in "${!MODELS[@]}"; do
    model="${MODELS[$i]}"
    echo "--- Training $((i+1))/${#MODELS[@]}: $model ---"
    bash "$KAUST_DIR/run_pretrain.sh" "$CONFIGS_DIR/config_llama_1.46B_${model}.yaml"
    echo ""
done

echo "=== Step 3: Evaluate all 8 runs ==="

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

echo "=== All done ==="

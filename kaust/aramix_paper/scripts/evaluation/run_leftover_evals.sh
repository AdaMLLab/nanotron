#!/bin/bash
# Usage: ./run_leftover_evals.sh <base_path> <output_dir> <task_config> <model1> [model2 ...]
#
# Example (all new H100 models):
#   ./run_leftover_evals.sh \
#     ~/new_h100_nanotron_checkpoints \
#     ~/nanotron_checkpoint_results_from_h100s \
#     arabic_finetasks.txt \
#     fineweb2_hq_arabic_30bt
#
#   ./run_leftover_evals.sh \
#     ~/new_h100_nanotron_checkpoints \
#     ~/nanotron_checkpoint_results_from_h100s \
#     turkish_finetasks.txt \
#     fineweb2_hq_turkish_30bt turmix_consensus_no_fineweb2_30bt
#
#   ./run_leftover_evals.sh \
#     ~/new_h100_nanotron_checkpoints \
#     ~/nanotron_checkpoint_results_from_h100s \
#     hindi_finetasks.txt \
#     hinmix_minhash_no_culturax_consensus_checked_30bt

cleanup() {
    pkill -P $$ 2>/dev/null
    pkill -f "convert_nanotron_to_hf|lighteval|run_finetasks" 2>/dev/null
    exit 1
}
trap cleanup SIGINT SIGTERM

BASE_PATH="$1"
OUTPUT_DIR="$2"
TASK_CONFIG="$3"
shift 3

SCRIPT_DIR="$(dirname "$0")"
CONFIG_DIR="$HOME/github/nanotron/kaust/aramix_paper/config/evaluation"

for model_dir in "$@"; do
    checkpoint_path="$BASE_PATH/$model_dir"
    echo "=== $model_dir ==="

    if [ ! -d "$checkpoint_path" ]; then
        echo "    WARNING: $checkpoint_path does not exist, skipping"
        continue
    fi

    "$SCRIPT_DIR/run_finetasks.sh" \
        "$checkpoint_path" \
        9999 \
        "$CONFIG_DIR/$TASK_CONFIG" \
        "$OUTPUT_DIR" \
        "google/gemma-2b" \
        12 \
        8

    echo ""
done

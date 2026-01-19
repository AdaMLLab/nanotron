#!/bin/bash
# Evaluate all checkpoints from each aramix_paper model using 8 GPUs

# Trap Ctrl+C and kill all child processes
cleanup() {
    echo ""
    echo "Caught interrupt signal. Killing all child processes..."
    pkill -P $$ 2>/dev/null
    pkill -f "convert_nanotron_to_hf|lighteval|run_finetasks" 2>/dev/null
    exit 1
}
trap cleanup SIGINT SIGTERM

SCRIPT_DIR="$(dirname "$0")"
CHECKPOINT_BASE="/scratch/nanotron_checkpoint"
CONFIG_DIR="$HOME/github/nanotron/kaust/aramix_paper/config/evaluation"
OUTPUT_DIR="/scratch/nanotron_results"

# Model configurations: checkpoint_dir|task_config
MODELS=(
    # "consensus_no_aw24_30bt|arabic_finetasks.txt"
    # "minhash_no_aw24_30bt|arabic_finetasks.txt"
    "culturax_hindi_30bt|hindi_finetasks.txt"
)

for model_cfg in "${MODELS[@]}"; do
    model_dir="${model_cfg%%|*}"
    task_file="${model_cfg##*|}"
    echo "=== Evaluating $model_dir ==="
    "$SCRIPT_DIR/run_finetasks.sh" \
        "$CHECKPOINT_BASE/$model_dir" \
        9999 \
        "$CONFIG_DIR/$task_file" \
        "$OUTPUT_DIR" \
        "google/gemma-2b" \
        12 \
        8
done

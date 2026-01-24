#!/bin/bash
# Evaluate leftover checkpoints from aramix_paper models using 8 GPUs
# Evaluates 5 evenly-spaced checkpoints from each model

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
ARAMIX_BASE="$HOME/nanotron_checkpoints/aramix_paper"
H100S_BASE="$HOME/nanotron_checkpoint_from_h100s"
CONFIG_DIR="$HOME/github/nanotron/kaust/aramix_paper/config/evaluation"
OUTPUT_DIR="$HOME/nanotron_checkpoint_results"
OUTPUT_DIR_H100S="$HOME/nanotron_checkpoint_results_from_h100s"

NUM_CHECKPOINTS=5
NUM_GPUS=8
BATCH_SIZE=12
TOKENIZER="google/gemma-2b"

# Model configurations: checkpoint_dir|task_config|base_path|output_dir
MODELS=(
    # Arabic models from aramix_paper (already evaluated in eval_results_fw2)
    # "arabicweb24|arabic_finetasks.txt|$ARAMIX_BASE|$OUTPUT_DIR"
    # "aramix_30bt|arabic_finetasks.txt|$ARAMIX_BASE|$OUTPUT_DIR"
    # "consensus_30bt|arabic_finetasks.txt|$ARAMIX_BASE|$OUTPUT_DIR"
    # "finepdfs_arabic|arabic_finetasks.txt|$ARAMIX_BASE|$OUTPUT_DIR"
    # "fineweb_edu_ar_30bt|arabic_finetasks.txt|$ARAMIX_BASE|$OUTPUT_DIR"

    # Hindi models from aramix_paper
    "hinmix_minhash_deduped_31bt|hindi_finetasks.txt|$ARAMIX_BASE|$OUTPUT_DIR"

    # Turkish models from aramix_paper
    "fineweb2_turkish_30bt|turkish_finetasks.txt|$ARAMIX_BASE|$OUTPUT_DIR"
    "turmix_minhash_deduped_31bt|turkish_finetasks.txt|$ARAMIX_BASE|$OUTPUT_DIR"

    # Arabic models from h100s
    "consensus_no_aw24_30bt|arabic_finetasks.txt|$H100S_BASE|$OUTPUT_DIR_H100S"
    "minhash_no_aw24_30bt|arabic_finetasks.txt|$H100S_BASE|$OUTPUT_DIR_H100S"

    # Hindi models from h100s
    "culturax_hindi_30bt|hindi_finetasks.txt|$H100S_BASE|$OUTPUT_DIR_H100S"
    "hinmix_consensus_30bt|hindi_finetasks.txt|$H100S_BASE|$OUTPUT_DIR_H100S"

    # Turkish models from h100s
    "turmix_consensus_30bt|turkish_finetasks.txt|$H100S_BASE|$OUTPUT_DIR_H100S"
)

echo "========================================="
echo "Leftover Checkpoints Evaluation Script"
echo "========================================="
echo "Evaluating $NUM_CHECKPOINTS evenly-spaced checkpoints per model"
echo "Using $NUM_GPUS GPUs with batch size $BATCH_SIZE"
echo "Output directories:"
echo "  - aramix_paper models: $OUTPUT_DIR"
echo "  - h100s models: $OUTPUT_DIR_H100S"
echo ""

for model_cfg in "${MODELS[@]}"; do
    IFS='|' read -r model_dir task_file base_path model_output_dir <<< "$model_cfg"
    checkpoint_path="$base_path/$model_dir"

    echo "=== Evaluating $model_dir ==="
    echo "    Checkpoint path: $checkpoint_path"
    echo "    Task config: $task_file"
    echo "    Output dir: $model_output_dir"

    # Check if checkpoint directory exists and has checkpoints
    if [ ! -d "$checkpoint_path" ]; then
        echo "    WARNING: Checkpoint directory does not exist, skipping"
        echo ""
        continue
    fi

    checkpoint_count=$(ls -d "$checkpoint_path"/*/ 2>/dev/null | xargs -n1 basename 2>/dev/null | grep -E '^[0-9]+$' | wc -l)
    if [ "$checkpoint_count" -eq 0 ]; then
        echo "    WARNING: No checkpoints found, skipping"
        echo ""
        continue
    fi

    echo "    Found $checkpoint_count checkpoints"

    "$SCRIPT_DIR/run_finetasks.sh" \
        "$checkpoint_path" \
        "$NUM_CHECKPOINTS" \
        "$CONFIG_DIR/$task_file" \
        "$model_output_dir" \
        "$TOKENIZER" \
        "$BATCH_SIZE" \
        "$NUM_GPUS"

    echo ""
done

echo "========================================="
echo "All evaluations complete!"
echo "Results saved to:"
echo "  - $OUTPUT_DIR"
echo "  - $OUTPUT_DIR_H100S"
echo "========================================="

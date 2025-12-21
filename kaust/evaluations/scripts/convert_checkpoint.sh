#!/bin/bash
#
# Convert a single nanotron checkpoint to HuggingFace format
#
# Usage:
#   ./convert_checkpoint.sh <nanotron_checkpoint_path> <hf_save_path> [tokenizer_name]
#
# Example:
#   ./convert_checkpoint.sh /path/to/checkpoints/1200 /path/to/hf_models/step_1200 meta-llama/Llama-3.2-1B
#

set -e

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <nanotron_checkpoint_path> <hf_save_path> [tokenizer_name]"
    echo ""
    echo "Arguments:"
    echo "  nanotron_checkpoint_path  Path to the nanotron checkpoint directory"
    echo "  hf_save_path              Path to save the converted HF model"
    echo "  tokenizer_name            (Optional) HuggingFace tokenizer name"
    exit 1
fi

CHECKPOINT_PATH="$1"
SAVE_PATH="$2"
TOKENIZER_NAME="${3:-meta-llama/Llama-3.2-1B}"

# Get the nanotron repo root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NANOTRON_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

echo "=========================================="
echo "Converting Nanotron to HuggingFace"
echo "=========================================="
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Save path: $SAVE_PATH"
echo "Tokenizer: $TOKENIZER_NAME"
echo "=========================================="

# Set distributed training environment for single GPU
export MASTER_ADDR=${MASTER_ADDR:-localhost}
export MASTER_PORT=${MASTER_PORT:-29501}
export RANK=${RANK:-0}
export WORLD_SIZE=${WORLD_SIZE:-1}
export LOCAL_RANK=${LOCAL_RANK:-0}

# Run conversion - use module import style
cd "$NANOTRON_ROOT"
PYTHONPATH="$NANOTRON_ROOT:$PYTHONPATH" torchrun --nproc_per_node=1 -m examples.llama.convert_nanotron_to_hf \
    --checkpoint_path "$CHECKPOINT_PATH" \
    --save_path "$SAVE_PATH" \
    --tokenizer_name "$TOKENIZER_NAME"

echo ""
echo "=========================================="
echo "Conversion complete!"
echo "HF model saved to: $SAVE_PATH"
echo "=========================================="

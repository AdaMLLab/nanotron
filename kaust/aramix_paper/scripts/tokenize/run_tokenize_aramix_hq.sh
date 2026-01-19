#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARAMIX_PAPER_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
NANOTRON_DIR="$(dirname "$(dirname "$ARAMIX_PAPER_DIR")")"

TOKENIZER="google/gemma-2b"
OUTPUT_BASE="/scratch/tokenized_data"

declare -A DATASETS=(
    ["aramix_hq"]="/scratch/data/aramix-hq|aramix-hq|32"
)

tokenize_dataset() {
    local name=$1
    local input_dir=$2
    local output_name=$3
    local n_tasks=$4
    local output_dir="${OUTPUT_BASE}/${output_name}"

    echo "============================================================"
    echo "Tokenizing: $name"
    echo "Input: $input_dir"
    echo "Output: $output_dir"
    echo "============================================================"

    [ ! -d "$input_dir" ] && echo "ERROR: Data not found at $input_dir" && exit 1

    mkdir -p "$output_dir"
    cd "$NANOTRON_DIR"

    python3 tools/preprocess_data.py \
        --tokenizer-name-or-path "$TOKENIZER" \
        --output-folder "$output_dir" \
        --n-tasks "$n_tasks" \
        hf \
        --dataset "$input_dir" \
        --column text \
        --split train

    echo "Done: $output_dir"
}

if [ $# -eq 0 ]; then
    echo "Usage: $0 <dataset_name|all>"
    echo "Available datasets: ${!DATASETS[*]}"
    exit 1
fi

if [ "$1" == "all" ]; then
    for name in "${!DATASETS[@]}"; do
        IFS='|' read -r input_dir output_name n_tasks <<< "${DATASETS[$name]}"
        tokenize_dataset "$name" "$input_dir" "$output_name" "$n_tasks"
    done
else
    if [ -z "${DATASETS[$1]}" ]; then
        echo "Unknown dataset: $1"
        echo "Available datasets: ${!DATASETS[*]}"
        exit 1
    fi
    IFS='|' read -r input_dir output_name n_tasks <<< "${DATASETS[$1]}"
    tokenize_dataset "$1" "$input_dir" "$output_name" "$n_tasks"
fi

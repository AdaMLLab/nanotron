#!/bin/bash
# Usage: ./run_tokenize.sh <input_dir> <output_dir> [n_tasks]
#
# Arabic:
#   ./run_tokenize.sh ~/data/arabicweb24/ArabicWeb24 ~/tokenized_data/gemma_tokenizer/arabicweb24 32
#   ./run_tokenize.sh ~/data/aramix-consensus-30bt/data/sampled ~/tokenized_data/gemma_tokenizer/aramix-consensus-30bt 16
#   ./run_tokenize.sh ~/data/finepdfs-arabic/data/arb_Arab ~/tokenized_data/gemma_tokenizer/finepdfs-arabic 16
#   ./run_tokenize.sh ~/data/fineweb-edu-ar-30bt/data ~/tokenized_data/gemma_tokenizer/fineweb-edu-ar-30bt 32
#   ./run_tokenize.sh ~/data/aramix_paper/aramix-minhash-no_aw24-30bt/data/sampled ~/tokenized_data/gemma_tokenizer/aramix-minhash-no_aw24-30bt 32
#   ./run_tokenize.sh ~/data/aramix_paper/aramix-consensus-no_aw24-30bt/data ~/tokenized_data/gemma_tokenizer/aramix-consensus-no_aw24-30bt 16
#   ./run_tokenize.sh ~/data/aramix_paper/aramix-minhash-no_aw24_consensus_checked-30bt/data/sampled ~/tokenized_data/gemma_tokenizer/aramix-minhash-no_aw24_consensus_checked-30bt 32
#   ./run_tokenize.sh /scratch/data/aramix-hq /scratch/tokenized_data/aramix-hq 32
#
# Hindi:
#   ./run_tokenize.sh ~/data/hinmix-minhash-deduped-31bt/data/sampled ~/tokenized_data/gemma_tokenizer/hinmix-minhash-deduped-31bt 32
#   ./run_tokenize.sh ~/data/hinmix-minhash-no_culturax_consensus_checked-31bt/data/sampled ~/tokenized_data/gemma_tokenizer/hinmix-minhash-no_culturax_consensus_checked-31bt 32
#   ./run_tokenize.sh /scratch/data/hinmix-consensus /scratch/tokenized_data/hinmix-consensus 16
#
# Turkish:
#   ./run_tokenize.sh ~/data/turmix-minhash-deduped-31bt/data/sampled ~/tokenized_data/gemma_tokenizer/turmix-minhash-deduped-31bt 32
#   ./run_tokenize.sh ~/data/turmix-no_fineweb2_consensus_checked-31bt/data/sampled ~/tokenized_data/gemma_tokenizer/turmix-no_fineweb2_consensus_checked-31bt 32
#   ./run_tokenize.sh /scratch/data/turmix-consensus /scratch/tokenized_data/turmix-consensus 16
set -e

INPUT_DIR="$1"
OUTPUT_DIR="$2"
N_TASKS="${3:-16}"

if [ $# -lt 2 ]; then
    echo "Usage: $0 <input_dir> <output_dir> [n_tasks]"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NANOTRON_DIR="$(dirname "$(dirname "$(dirname "$(dirname "$SCRIPT_DIR")")")")"

if [ ! -d "$INPUT_DIR" ]; then
    echo "ERROR: Data not found at $INPUT_DIR"
    exit 1
fi

echo "=== Tokenizing ==="
echo "    Input: $INPUT_DIR"
echo "    Output: $OUTPUT_DIR"
echo "    Tasks: $N_TASKS"

mkdir -p "$OUTPUT_DIR"
cd "$NANOTRON_DIR"

python3 tools/preprocess_data.py \
    --tokenizer-name-or-path "google/gemma-2b" \
    --output-folder "$OUTPUT_DIR" \
    --n-tasks "$N_TASKS" \
    hf \
    --dataset "$INPUT_DIR" \
    --column text \
    --split train

echo "Done: $OUTPUT_DIR"

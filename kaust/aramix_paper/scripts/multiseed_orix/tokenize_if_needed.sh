#!/bin/bash
# Tokenize datasets if not already tokenized.
# Usage: source tokenize_if_needed.sh
# Requires NANOTRON_DIR to be set.
#
# This script tokenizes the two raw datasets (consensus and arabicweb24)
# into the format nanotron expects (.ds + .metadata files).
# It uses a lock file to prevent multiple jobs from tokenizing simultaneously,
# and checks for existing .ds files to skip if already done.
set -e

TOKENIZER="google/gemma-2b"
N_TASKS=16
DATA_BASE="/mnt/data/u/alrashsm"
TOKENIZED_BASE="${DATA_BASE}/nanotron_checkpoints/tokenized_data"

# --- Consensus dataset ---
CONSENSUS_RAW="${DATA_BASE}/datasets/aramix-matched/data/consensus"
CONSENSUS_OUT="${TOKENIZED_BASE}/aramix-consensus-30bt"

# --- ArabicWeb24 dataset ---
ARABICWEB24_RAW="${DATA_BASE}/datasets/arabicweb24/ArabicWeb24-no-sentence-dedup"
ARABICWEB24_OUT="${TOKENIZED_BASE}/arabicweb24"

tokenize_dataset() {
    local RAW_PATH="$1"
    local OUT_PATH="$2"
    local DATASET_NAME="$3"
    local READER_TYPE="$4"  # "hf" or "jsonl"
    local COLUMN="$5"
    local LOCKFILE="${OUT_PATH}.lock"

    # Check if already tokenized (.ds files exist)
    if ls "${OUT_PATH}"/*.ds 1>/dev/null 2>&1; then
        echo "[tokenize] ${DATASET_NAME}: already tokenized at ${OUT_PATH}, skipping."
        return 0
    fi

    # Try to acquire lock (avoid parallel tokenization from multiple jobs)
    if mkdir "${LOCKFILE}" 2>/dev/null; then
        echo "[tokenize] ${DATASET_NAME}: lock acquired, starting tokenization..."
        mkdir -p "${OUT_PATH}"

        cd "${NANOTRON_DIR}"
        python3 tools/preprocess_data.py \
            --tokenizer-name-or-path "${TOKENIZER}" \
            --output-folder "${OUT_PATH}" \
            --n-tasks "${N_TASKS}" \
            "${READER_TYPE}" \
            --dataset "${RAW_PATH}" \
            --column "${COLUMN}" \
            --split train

        echo "[tokenize] ${DATASET_NAME}: tokenization complete."
        rmdir "${LOCKFILE}"
    else
        # Another job is tokenizing — wait for it
        echo "[tokenize] ${DATASET_NAME}: another job is tokenizing, waiting..."
        while [ -d "${LOCKFILE}" ]; do
            sleep 30
        done
        # Verify it succeeded
        if ls "${OUT_PATH}"/*.ds 1>/dev/null 2>&1; then
            echo "[tokenize] ${DATASET_NAME}: tokenization completed by another job."
        else
            echo "[tokenize] ERROR: ${DATASET_NAME} lock released but no .ds files found!"
            exit 1
        fi
    fi
}

# Ensure base directories exist
mkdir -p "${TOKENIZED_BASE}"

echo "=== Checking/running tokenization ==="
tokenize_dataset "${CONSENSUS_RAW}" "${CONSENSUS_OUT}" "consensus" "hf" "text"
tokenize_dataset "${ARABICWEB24_RAW}" "${ARABICWEB24_OUT}" "arabicweb24" "hf" "text"
echo "=== Tokenization check complete ==="

#!/bin/bash
# Generic tokenize helper with lock-based coordination.
# Usage: source tokenize_helper.sh, then call tokenize_dataset with args.
# Requires NANOTRON_DIR to be set.
#
# tokenize_dataset <raw_path> <out_path> <dataset_name> <reader_type> <column>
#
# Lock mechanism:
#   - Uses a directory as a lockfile (atomic mkdir).
#   - Writes the holder's SLURM_JOB_ID inside the lock.
#   - Waiters poll squeue to detect stale locks (e.g. after preemption).
#   - trap on EXIT/TERM/INT releases the lock if the holder dies.
# Completion marker: *.metadata files (written last by datatrove).

TOKENIZER="google/gemma-2b"
N_TASKS=16

# Force HF datasets library cache to /mnt instead of ~/.cache/huggingface/datasets
# (load_dataset() creates massive Arrow copies that otherwise fill /home).
export HF_DATASETS_CACHE="/mnt/data/u/alrashsm/hf_home/datasets"
export HF_HOME="/mnt/data/u/alrashsm/hf_home"
mkdir -p "${HF_DATASETS_CACHE}"

tokenize_dataset() {
    local RAW_PATH="$1"
    local OUT_PATH="$2"
    local DATASET_NAME="$3"
    local READER_TYPE="$4"
    local COLUMN="$5"
    local LOCKFILE="${OUT_PATH}.lock"
    local HOLDER

    mkdir -p "$(dirname "${OUT_PATH}")"

    while true; do
        # Done if the final metadata files exist.
        if ls "${OUT_PATH}"/*.metadata 1>/dev/null 2>&1; then
            echo "[tokenize] ${DATASET_NAME}: already tokenized at ${OUT_PATH}, skipping."
            return 0
        fi

        # Try to acquire the lock atomically.
        if mkdir "${LOCKFILE}" 2>/dev/null; then
            echo "${SLURM_JOB_ID:-unknown}" > "${LOCKFILE}/slurm_job_id"
            trap "rm -rf '${LOCKFILE}' 2>/dev/null; echo '[tokenize] ${DATASET_NAME}: lock released on exit'" EXIT TERM INT
            echo "[tokenize] ${DATASET_NAME}: lock acquired by job ${SLURM_JOB_ID:-unknown}, tokenizing..."
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
            rm -rf "${LOCKFILE}"
            trap - EXIT TERM INT
            return 0
        fi

        # Lock held; check if the holder is still alive.
        HOLDER=$(cat "${LOCKFILE}/slurm_job_id" 2>/dev/null)
        if [ -n "$HOLDER" ] && [ "$HOLDER" != "unknown" ]; then
            if ! squeue -j "$HOLDER" -h -o "%T" 2>/dev/null | grep -qE "RUNNING|PENDING"; then
                echo "[tokenize] ${DATASET_NAME}: stale lock from job $HOLDER (not active), stealing..."
                rm -rf "${LOCKFILE}"
                continue
            fi
        fi
        echo "[tokenize] ${DATASET_NAME}: waiting on lock (held by job ${HOLDER:-unknown})..."
        sleep 30
    done
}

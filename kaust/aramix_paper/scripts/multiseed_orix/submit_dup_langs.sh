#!/bin/bash
# Submit 4 filtered (dup>=3, dup>=4) jobs for Hindi and Turkish.
# 4 x 8 GPUs = 32 GPUs total (within freecycle QoS limit).
# Jobs independently filter/tokenize/train; each uses a distinct dataset so no
# cross-job races. Requeue-safe via trap-based lock cleanup.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Submitting 4 filtered-language jobs to freecycle..."

sbatch "${SCRIPT_DIR}/run_hinmix_dup3.sh"
sbatch "${SCRIPT_DIR}/run_hinmix_dup4.sh"
sbatch "${SCRIPT_DIR}/run_turmix_dup3.sh"
sbatch "${SCRIPT_DIR}/run_turmix_dup4.sh"

echo "4 jobs submitted. Check with: squeue -u \$USER"

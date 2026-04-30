#!/bin/bash
# Submit 4 baseline jobs: itamix + fineweb2-italian + indmix + hplt2-indonesian.
# Seed 42 for all. 4 x 8 GPUs = 32 GPUs (within freecycle QoS limit).
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Submitting 4 Italian/Indonesian jobs to freecycle..."

sbatch "${SCRIPT_DIR}/run_itamix_seed42.sh"
sbatch "${SCRIPT_DIR}/run_fineweb2_italian_seed42.sh"
sbatch "${SCRIPT_DIR}/run_indmix_seed42.sh"
sbatch "${SCRIPT_DIR}/run_hplt2_indonesian_seed42.sh"

echo "4 jobs submitted. Check with: squeue -u \$USER"

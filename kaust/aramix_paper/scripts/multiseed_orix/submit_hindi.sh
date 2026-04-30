#!/bin/bash
# Submit 4 Hindi training jobs (hinmix + culturax-hindi, seeds 1 & 2).
# 4 x 8 GPUs = 32 GPUs total (within freecycle QoS limit).
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Submitting 4 Hindi jobs to freecycle (32 GPUs total)..."

sbatch "${SCRIPT_DIR}/run_hinmix_seed1.sh"
sbatch "${SCRIPT_DIR}/run_hinmix_seed2.sh"
sbatch "${SCRIPT_DIR}/run_culturax_hindi_seed1.sh"
sbatch "${SCRIPT_DIR}/run_culturax_hindi_seed2.sh"

echo "All 4 Hindi jobs submitted. Check status with: squeue -u \$USER"

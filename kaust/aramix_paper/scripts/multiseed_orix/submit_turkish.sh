#!/bin/bash
# Submit 4 Turkish training jobs (fineweb2-turkish + turmix, seeds 1 & 2).
# 4 x 8 GPUs = 32 GPUs total (within freecycle QoS limit).
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Submitting 4 Turkish jobs to freecycle (32 GPUs total)..."

sbatch "${SCRIPT_DIR}/run_fineweb2_turkish_seed1.sh"
sbatch "${SCRIPT_DIR}/run_fineweb2_turkish_seed2.sh"
sbatch "${SCRIPT_DIR}/run_turmix_seed1.sh"
sbatch "${SCRIPT_DIR}/run_turmix_seed2.sh"

echo "All 4 Turkish jobs submitted. Check status with: squeue -u \$USER"

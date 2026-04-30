#!/bin/bash
# Submit 4 multi-seed training jobs (seeds 1 & 2 for both datasets)
# Uses 4 × 8 = 32 H100 GPUs total (freecycle QoS limit)
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Submitting 4 multi-seed jobs to ORIX freecycle (32 GPUs total)..."

sbatch "${SCRIPT_DIR}/run_consensus_seed1.sh"
sbatch "${SCRIPT_DIR}/run_consensus_seed2.sh"
sbatch "${SCRIPT_DIR}/run_arabicweb24_seed1.sh"
sbatch "${SCRIPT_DIR}/run_arabicweb24_seed2.sh"

echo "All 4 jobs submitted. Check status with: squeue -u \$USER"

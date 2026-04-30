#!/bin/bash
# Submit seed 3 jobs (run after seed 1/2 jobs complete to stay within QoS limits)
# Uses 2 × 8 = 16 H100 GPUs
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Submitting seed 3 jobs to ORIX freecycle (16 GPUs)..."

sbatch "${SCRIPT_DIR}/run_consensus_seed3.sh"
sbatch "${SCRIPT_DIR}/run_arabicweb24_seed3.sh"

echo "Seed 3 jobs submitted. Check status with: squeue -u \$USER"

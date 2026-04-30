#!/bin/bash
# Submit dup3 and dup4 training jobs to PI allocation back-to-back.
# dup4 only starts after dup3 finishes successfully (--dependency=afterok).
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Submitting dup3 to pi-orabonf..."
JOB_DUP3=$(sbatch --parsable "${SCRIPT_DIR}/run_consensus_dup3.sh")
echo "  dup3 job id: ${JOB_DUP3}"

echo "Submitting dup4 (dependency: after dup3)..."
JOB_DUP4=$(sbatch --parsable --dependency=afterok:${JOB_DUP3} "${SCRIPT_DIR}/run_consensus_dup4.sh")
echo "  dup4 job id: ${JOB_DUP4}"

echo ""
echo "Sequence submitted:"
echo "  ${JOB_DUP3} (cons-dup3) -> ${JOB_DUP4} (cons-dup4)"
echo "Check status with: squeue -u \$USER"

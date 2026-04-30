#!/bin/bash
# Submit a chain of N dependent jobs. Each job does a done-check and exits
# early if training is complete — excess chain slots are safe no-ops.
#
# Usage: ./submit_chain.sh <run_script> [N=20]
#
# Uses --dependency=afterany so even if a job fails or is preempted, the next
# link can still run (and nanotron auto-resumes from latest.txt).
set -e

RUN_SCRIPT="$1"
N="${2:-20}"

if [ -z "$RUN_SCRIPT" ] || [ ! -f "$RUN_SCRIPT" ]; then
    echo "Usage: $0 <run_script> [N=20]" >&2
    exit 1
fi

PREV=""
echo "Submitting ${N}-link chain for $(basename "$RUN_SCRIPT")..."
for i in $(seq 1 "$N"); do
    if [ -z "$PREV" ]; then
        JOB=$(sbatch --parsable "$RUN_SCRIPT")
    else
        JOB=$(sbatch --parsable --dependency=afterany:"$PREV" "$RUN_SCRIPT")
    fi
    printf "  link %2d: job %s\n" "$i" "$JOB"
    PREV="$JOB"
done
echo "Chain head: first job, tail: last job."

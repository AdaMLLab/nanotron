#!/bin/bash
# Run all orix evaluations (8 GPUs, every 1000 steps from 1000-14000, 18 runs).
# Wraps orchestrate_orix_evals.py and ensures clean shutdown on Ctrl-C.

set -euo pipefail

cleanup() {
    echo ""
    echo "Caught interrupt; killing workers..."
    pkill -P $$ 2>/dev/null || true
    pkill -f "orchestrate_orix_evals|convert_nanotron_to_hf|lighteval accelerate" 2>/dev/null || true
    exit 1
}
trap cleanup SIGINT SIGTERM

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="$HOME/miniconda3/envs/lighteval/bin/python"

exec "$PYTHON" "$SCRIPT_DIR/orchestrate_orix_evals.py" "$@"

#!/bin/bash
# Run base AraMix-HQ 10B token training.
# This must complete before running continuations.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KAUST_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

bash "$KAUST_DIR/run_pretrain.sh" "$SCRIPT_DIR/../configs/config_llama_1.46B_aramix_hq_10bt.yaml"

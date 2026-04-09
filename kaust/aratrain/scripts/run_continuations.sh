#!/bin/bash
# Run all 3 continuation trainings sequentially.
# Requires the base AraMix-HQ 10B run to be complete first.
# Each continues from the base checkpoint for 15B more tokens (25B total).
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KAUST_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONFIGS_DIR="$SCRIPT_DIR/../configs"

echo "=== Running continuation trainings ==="
echo "    Base checkpoint: ~/nanotron_checkpoints/aratrain/aramix_hq_10bt"
echo ""

# echo "--- 1/3: finewiki-en-ar ---"
# bash "$KAUST_DIR/run_pretrain.sh" "$CONFIGS_DIR/config_llama_1.46B_finewiki_en_ar.yaml"

echo "--- 1/2: ultradata-math ---"
bash "$KAUST_DIR/run_pretrain.sh" "$CONFIGS_DIR/config_llama_1.46B_ultradata_math.yaml"

echo "--- 2/2: openresearcher ---"
bash "$KAUST_DIR/run_pretrain.sh" "$CONFIGS_DIR/config_llama_1.46B_openresearcher.yaml"

echo "=== All continuation trainings complete ==="

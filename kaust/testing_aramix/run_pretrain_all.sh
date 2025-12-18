#!/bin/bash
#
# Train TWO SEPARATE LLaMA 1.46B models back-to-back:
#
#   Model 1: Trained on Aramix MinHash Deduped 30BT (~29.36B tokens)
#            Checkpoints: checkpoints/
#
#   Model 2: Trained on ArabicWeb24 (~29.3B tokens)
#            Checkpoints: checkpoints_arabicweb24/
#
# Each model is trained independently from scratch (not resumed from the other).
#
# Model configuration (same for both):
#   - Architecture: LLaMA 1.46B
#   - Batch size: 1024 samples (2,097,152 tokens/step)
#   - Steps: 14,000 each
#   - Checkpoint interval: 1,200 steps (~2.5B tokens)
#   - LR: 3e-4 -> 3e-5 (cosine decay)
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "############################################################"
echo "#                                                          #"
echo "#  LLaMA 1.46B - Train Two Separate Models                 #"
echo "#                                                          #"
echo "############################################################"
echo ""
echo "This script trains TWO INDEPENDENT models back-to-back:"
echo ""
echo "  Model 1: Aramix MinHash Deduped 30BT"
echo "           - Tokens: ~29.36B"
echo "           - Checkpoints: ${SCRIPT_DIR}/checkpoints_aramix_30bt/"
echo ""
echo "  Model 2: ArabicWeb24"
echo "           - Tokens: ~29.3B"
echo "           - Checkpoints: ${SCRIPT_DIR}/checkpoints_arabicweb24/"
echo ""
echo "Each model starts from scratch with random initialization."
echo "############################################################"
echo ""

# Record start time
START_TIME=$(date +%s)

echo ""
echo "============================================================"
echo "[Model 1/2] Training on Aramix MinHash Deduped 30BT"
echo "============================================================"
echo "Start time: $(date)"
echo "Checkpoints: ${SCRIPT_DIR}/checkpoints_aramix_30bt/"
echo ""

"${SCRIPT_DIR}/run_pretrain.sh"

MODEL1_END_TIME=$(date +%s)
MODEL1_DURATION=$((MODEL1_END_TIME - START_TIME))

echo ""
echo "============================================================"
echo "[Model 1/2] COMPLETE - Aramix model trained!"
echo "Duration: $((MODEL1_DURATION / 3600))h $((MODEL1_DURATION % 3600 / 60))m $((MODEL1_DURATION % 60))s"
echo "Checkpoints: ${SCRIPT_DIR}/checkpoints_aramix_30bt/"
echo "============================================================"
echo ""

echo ""
echo "============================================================"
echo "[Model 2/2] Training on ArabicWeb24 (starting fresh)"
echo "============================================================"
echo "Start time: $(date)"
echo "Checkpoints: ${SCRIPT_DIR}/checkpoints_arabicweb24/"
echo ""

"${SCRIPT_DIR}/run_pretrain_arabicweb24.sh"

END_TIME=$(date +%s)
MODEL2_DURATION=$((END_TIME - MODEL1_END_TIME))
TOTAL_DURATION=$((END_TIME - START_TIME))

echo ""
echo "############################################################"
echo "#                                                          #"
echo "#  Both Models Trained Successfully!                       #"
echo "#                                                          #"
echo "############################################################"
echo ""
echo "Summary:"
echo ""
echo "  Model 1 (Aramix):"
echo "    - Duration: $((MODEL1_DURATION / 3600))h $((MODEL1_DURATION % 3600 / 60))m $((MODEL1_DURATION % 60))s"
echo "    - Checkpoints: ${SCRIPT_DIR}/checkpoints_aramix_30bt/"
echo ""
echo "  Model 2 (ArabicWeb24):"
echo "    - Duration: $((MODEL2_DURATION / 3600))h $((MODEL2_DURATION % 3600 / 60))m $((MODEL2_DURATION % 60))s"
echo "    - Checkpoints: ${SCRIPT_DIR}/checkpoints_arabicweb24/"
echo ""
echo "  Total duration: $((TOTAL_DURATION / 3600))h $((TOTAL_DURATION % 3600 / 60))m $((TOTAL_DURATION % 60))s"
echo ""
echo "End time: $(date)"
echo "############################################################"

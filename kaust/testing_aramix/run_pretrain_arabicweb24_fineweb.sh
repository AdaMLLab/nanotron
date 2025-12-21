#!/bin/bash
#
# Train TWO SEPARATE LLaMA 1.46B models back-to-back:
#
#   Model 1: Trained on ArabicWeb24 (~29.3B tokens)
#            Checkpoints: kaust/checkpoints/checkpoints_arabicweb24/
#
#   Model 2: Trained on FineWeb-Edu Arabic (~30B tokens)
#            Checkpoints: kaust/checkpoints/checkpoints_fineweb_edu_ar_30bt/
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
CHECKPOINTS_DIR="/home/alrashsm/github/nanotron/kaust/checkpoints"

echo "############################################################"
echo "#                                                          #"
echo "#  LLaMA 1.46B - Train Two Separate Models                 #"
echo "#                                                          #"
echo "############################################################"
echo ""
echo "This script trains TWO INDEPENDENT models back-to-back:"
echo ""
echo "  Model 1: ArabicWeb24"
echo "           - Tokens: ~29.3B"
echo "           - Checkpoints: ${CHECKPOINTS_DIR}/checkpoints_arabicweb24/"
echo ""
echo "  Model 2: FineWeb-Edu Arabic"
echo "           - Tokens: ~30B"
echo "           - Checkpoints: ${CHECKPOINTS_DIR}/checkpoints_fineweb_edu_ar_30bt/"
echo ""
echo "Each model starts from scratch with random initialization."
echo "############################################################"
echo ""

# Record start time
START_TIME=$(date +%s)

echo ""
echo "============================================================"
echo "[Model 1/2] Training on ArabicWeb24"
echo "============================================================"
echo "Start time: $(date)"
echo "Checkpoints: ${CHECKPOINTS_DIR}/checkpoints_arabicweb24/"
echo ""

"${SCRIPT_DIR}/run_pretrain_arabicweb24.sh"

MODEL1_END_TIME=$(date +%s)
MODEL1_DURATION=$((MODEL1_END_TIME - START_TIME))

echo ""
echo "============================================================"
echo "[Model 1/2] COMPLETE - ArabicWeb24 model trained!"
echo "Duration: $((MODEL1_DURATION / 3600))h $((MODEL1_DURATION % 3600 / 60))m $((MODEL1_DURATION % 60))s"
echo "Checkpoints: ${CHECKPOINTS_DIR}/checkpoints_arabicweb24/"
echo "============================================================"
echo ""

echo ""
echo "============================================================"
echo "[Model 2/2] Training on FineWeb-Edu Arabic (starting fresh)"
echo "============================================================"
echo "Start time: $(date)"
echo "Checkpoints: ${CHECKPOINTS_DIR}/checkpoints_fineweb_edu_ar_30bt/"
echo ""

"${SCRIPT_DIR}/run_pretrain_fineweb_edu_ar.sh"

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
echo "  Model 1 (ArabicWeb24):"
echo "    - Duration: $((MODEL1_DURATION / 3600))h $((MODEL1_DURATION % 3600 / 60))m $((MODEL1_DURATION % 60))s"
echo "    - Checkpoints: ${CHECKPOINTS_DIR}/checkpoints_arabicweb24/"
echo ""
echo "  Model 2 (FineWeb-Edu Arabic):"
echo "    - Duration: $((MODEL2_DURATION / 3600))h $((MODEL2_DURATION % 3600 / 60))m $((MODEL2_DURATION % 60))s"
echo "    - Checkpoints: ${CHECKPOINTS_DIR}/checkpoints_fineweb_edu_ar_30bt/"
echo ""
echo "  Total duration: $((TOTAL_DURATION / 3600))h $((TOTAL_DURATION % 3600 / 60))m $((TOTAL_DURATION % 60))s"
echo ""
echo "End time: $(date)"
echo "############################################################"

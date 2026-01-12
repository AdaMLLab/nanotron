#!/bin/bash

CHECKPOINT_DIR="${1:-/home/alrashsm/github/nanotron/kaust/checkpoints/checkpoints_aramix_30bt}"
NUM_CHECKPOINTS="${2:-5}"
TASK_CONFIG="${3:-/home/alrashsm/github/nanotron/kaust/aramix_paper/config/evaluation/arabic_finetasks.txt}"
OUTPUT_DIR="${4:-/home/alrashsm/github/nanotron/kaust/aramix_paper/results}"
TOKENIZER="${5:-google/gemma-2b}"
BATCH_SIZE="${6:-8}"
NUM_GPUS="${7:-8}"

NANOTRON_ROOT="/home/alrashsm/github/nanotron"
NANOTRON_PYTHON="/home/alrashsm/miniconda3/envs/nanotron/bin/python"
LIGHTEVAL_PYTHON="/home/alrashsm/miniconda3/envs/lighteval/bin/python"

CF_TASKS=$(grep "^CF:" "$TASK_CONFIG" | cut -d: -f2)
MC_TASKS=$(grep "^MC:" "$TASK_CONFIG" | cut -d: -f2)
GEN_TASKS=$(grep "^GEN:" "$TASK_CONFIG" | cut -d: -f2)

STEPS=($(ls -d "$CHECKPOINT_DIR"/*/ 2>/dev/null | xargs -n1 basename | grep -E '^[0-9]+$' | sort -n))
TOTAL=${#STEPS[@]}
if [ "$TOTAL" -eq 0 ]; then
    echo "No checkpoints found in $CHECKPOINT_DIR"
    exit 1
fi

SELECTED=()
if [ "$NUM_CHECKPOINTS" -ge "$TOTAL" ]; then
    SELECTED=("${STEPS[@]}")
else
    for i in $(seq 0 $((NUM_CHECKPOINTS - 1))); do
        idx=$(( i * (TOTAL - 1) / (NUM_CHECKPOINTS - 1) ))
        SELECTED+=("${STEPS[$idx]}")
    done
fi

MODEL_NAME=$(basename "$CHECKPOINT_DIR" | sed 's/checkpoints_//' | sed 's/_30bt//')
HF_MODELS_DIR="$OUTPUT_DIR/hf_models"
RESULTS_DIR="$OUTPUT_DIR/results/$MODEL_NAME"
mkdir -p "$HF_MODELS_DIR" "$RESULTS_DIR"

echo "Model: $MODEL_NAME"
echo "Checkpoints: ${SELECTED[*]}"
echo "Tasks: CF=$CF_TASKS MC=$MC_TASKS GEN=$GEN_TASKS"

convert_checkpoint() {
    local step=$1
    local gpu=$2
    local suffix=$3
    local nanotron_path="$CHECKPOINT_DIR/$step"
    local hf_path="$HF_MODELS_DIR/${MODEL_NAME}_step_${step}${suffix}"

    if [ -d "$hf_path" ] && [ -f "$hf_path/config.json" ]; then
        return 0
    fi

    local port=$((29500 + gpu * 100 + RANDOM % 50))
    (
        cd "$NANOTRON_ROOT"
        CUDA_VISIBLE_DEVICES=$gpu MASTER_ADDR=localhost MASTER_PORT=$port RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 \
        $NANOTRON_PYTHON -m torch.distributed.run --nproc_per_node=1 --master_port=$port \
            examples/llama/convert_nanotron_to_hf.py \
            --checkpoint_path "$nanotron_path" \
            --save_path "$hf_path" \
            --tokenizer_name "$TOKENIZER"
    ) > /dev/null 2>&1
}

run_eval() {
    local step=$1
    local gpu=$2
    local tasks=$3
    local suffix=$4
    local hf_path="$HF_MODELS_DIR/${MODEL_NAME}_step_${step}${suffix}"
    local out_dir="$RESULTS_DIR/step_${step}"
    mkdir -p "$out_dir"

    CUDA_VISIBLE_DEVICES=$gpu $LIGHTEVAL_PYTHON -m lighteval accelerate \
        "model_name=${hf_path},dtype=bfloat16,batch_size=${BATCH_SIZE}" \
        "$tasks" \
        --load-tasks-multilingual \
        --output-dir "$out_dir" \
        --save-details \
        > /dev/null 2>&1
}

run_step() {
    local step=$1
    local gpu=$2

    echo "[GPU $gpu] Step $step: Converting..."
    convert_checkpoint "$step" "$gpu" ""

    echo "[GPU $gpu] Step $step: Running CF..."
    run_eval "$step" "$gpu" "$CF_TASKS" ""

    echo "[GPU $gpu] Step $step: Running MC..."
    run_eval "$step" "$gpu" "$MC_TASKS" ""

    if [ -n "$GEN_TASKS" ]; then
        convert_checkpoint "$step" "$gpu" "_gen"
        echo "[GPU $gpu] Step $step: Running GEN..."
        run_eval "$step" "$gpu" "$GEN_TASKS" "_gen"
        rm -rf "$HF_MODELS_DIR/${MODEL_NAME}_step_${step}_gen"
    fi

    rm -rf "$HF_MODELS_DIR/${MODEL_NAME}_step_${step}"
    echo "[GPU $gpu] Step $step: Done"
}

PIDS=()
GPU_IDX=0

for step in "${SELECTED[@]}"; do
    gpu=$((GPU_IDX % NUM_GPUS))
    run_step "$step" "$gpu" &
    PIDS+=($!)
    GPU_IDX=$((GPU_IDX + 1))

    if [ ${#PIDS[@]} -ge $NUM_GPUS ]; then
        wait "${PIDS[0]}"
        PIDS=("${PIDS[@]:1}")
    fi
    sleep 2
done

for pid in "${PIDS[@]}"; do
    wait $pid
done

echo "Evaluation complete. Results in $RESULTS_DIR"

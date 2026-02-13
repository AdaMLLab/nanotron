#!/bin/bash

# Trap Ctrl+C and kill all child processes
cleanup() {
    echo ""
    echo "Caught interrupt signal. Killing all child processes..."
    pkill -P $$ 2>/dev/null
    pkill -f "convert_nanotron_to_hf|lighteval" 2>/dev/null
    exit 1
}
trap cleanup SIGINT SIGTERM

CHECKPOINT_DIR="${1:-$HOME/github/nanotron/kaust/checkpoints/checkpoints_aramix_30bt}"
NUM_CHECKPOINTS="${2:-9999}"
TASK_CONFIG="${3:-$HOME/github/nanotron/kaust/aramix_paper/config/evaluation/arabic_finetasks.txt}"
OUTPUT_DIR="${4:-$HOME/github/nanotron/kaust/aramix_paper/results}"
TOKENIZER="${5:-google/gemma-2b}"
BATCH_SIZE="${6:-8}"
NUM_GPUS="${7:-8}"

NANOTRON_ROOT="$HOME/github/nanotron"
NANOTRON_PYTHON="$HOME/miniconda3/envs/nanotron/bin/python"
LIGHTEVAL_PYTHON="$HOME/miniconda3/envs/lighteval/bin/python"

CF_TASKS=$(grep "^CF:" "$TASK_CONFIG" | cut -d: -f2-)
MC_TASKS=$(grep "^MC:" "$TASK_CONFIG" | cut -d: -f2-)
GEN_TASKS=$(grep "^GEN:" "$TASK_CONFIG" | cut -d: -f2-)

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
    local log_file="$OUTPUT_DIR/logs/convert_${MODEL_NAME}_step_${step}${suffix}.log"
    mkdir -p "$OUTPUT_DIR/logs"
    (
        cd "$NANOTRON_ROOT"
        CUDA_VISIBLE_DEVICES=$gpu $NANOTRON_PYTHON -m torch.distributed.run --nproc_per_node=1 --master_port=$port \
            -m examples.llama.convert_nanotron_to_hf \
            --checkpoint_path "$nanotron_path" \
            --save_path "$hf_path" \
            --tokenizer_name "$TOKENIZER"
    ) 2>&1 | tee "$log_file"
}

run_eval() {
    local step=$1
    local gpu=$2
    local tasks=$3
    local suffix=$4
    local task_type=$5
    local hf_path="$HF_MODELS_DIR/${MODEL_NAME}_step_${step}${suffix}"
    local out_dir="$RESULTS_DIR/step_${step}"
    local log_file="$OUTPUT_DIR/logs/eval_${MODEL_NAME}_step_${step}_${task_type}.log"
    local marker="$out_dir/.completed_${task_type}"
    mkdir -p "$out_dir" "$OUTPUT_DIR/logs"

    # Skip if already completed
    if [ -f "$marker" ]; then
        echo "[GPU $gpu] Step $step: $task_type already completed, skipping"
        return 0
    fi

    HF_DATASETS_TRUST_REMOTE_CODE=1 CUDA_VISIBLE_DEVICES=$gpu $LIGHTEVAL_PYTHON -m lighteval accelerate \
        "model_name=${hf_path},dtype=bfloat16,batch_size=${BATCH_SIZE}" \
        "$tasks" \
        --load-tasks-multilingual \
        --output-dir "$out_dir" \
        --save-details \
        2>&1 | tee "$log_file"
    local eval_exit=${PIPESTATUS[0]}

    # Mark as completed if successful
    if [ $eval_exit -eq 0 ]; then
        echo "$(date -Iseconds)" > "$marker"
    fi
}

run_step() {
    local step=$1
    local gpu=$2
    local out_dir="$RESULTS_DIR/step_${step}"
    local nanotron_path="$CHECKPOINT_DIR/$step"

    # Skip incomplete checkpoints (missing model_config.json)
    if [ ! -f "$nanotron_path/model_config.json" ]; then
        echo "[GPU $gpu] Step $step: Skipping (incomplete checkpoint - missing model_config.json)"
        return 0
    fi

    # Check if all tasks are already completed (or empty)
    local cf_done=0 mc_done=0 gen_done=0
    [ -z "$CF_TASKS" ] || [ -f "$out_dir/.completed_cf" ] && cf_done=1
    [ -z "$MC_TASKS" ] || [ -f "$out_dir/.completed_mc" ] && mc_done=1
    [ -z "$GEN_TASKS" ] || [ -f "$out_dir/.completed_gen" ] && gen_done=1

    if [ $cf_done -eq 1 ] && [ $mc_done -eq 1 ] && [ $gen_done -eq 1 ]; then
        echo "[GPU $gpu] Step $step: All tasks already completed, skipping"
        return 0
    fi

    # Only convert if CF or MC needs to run
    if [ $cf_done -eq 0 ] || [ $mc_done -eq 0 ]; then
        echo "[GPU $gpu] Step $step: Converting..."
        convert_checkpoint "$step" "$gpu" ""

        if [ -n "$CF_TASKS" ] && [ $cf_done -eq 0 ]; then
            echo "[GPU $gpu] Step $step: Running CF..."
            run_eval "$step" "$gpu" "$CF_TASKS" "" "cf"
        fi

        if [ -n "$MC_TASKS" ] && [ $mc_done -eq 0 ]; then
            echo "[GPU $gpu] Step $step: Running MC..."
            run_eval "$step" "$gpu" "$MC_TASKS" "" "mc"
        fi

        rm -rf "$HF_MODELS_DIR/${MODEL_NAME}_step_${step}"
    fi

    if [ -n "$GEN_TASKS" ] && [ $gen_done -eq 0 ]; then
        convert_checkpoint "$step" "$gpu" "_gen"
        echo "[GPU $gpu] Step $step: Running GEN..."
        run_eval "$step" "$gpu" "$GEN_TASKS" "_gen" "gen"
        rm -rf "$HF_MODELS_DIR/${MODEL_NAME}_step_${step}_gen"
    fi

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

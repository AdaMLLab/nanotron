#!/bin/bash
#
# Parallel evaluation of checkpoints across multiple GPUs
#
# Usage:
#   ./parallel_eval.sh <checkpoint_base> <model_name> [tokenizer_name] [num_gpus]
#
# Example:
#   ./parallel_eval.sh /path/to/checkpoints_aramix_30bt aramix google/gemma-2b 8
#

set -e

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <checkpoint_base> <model_name> [tokenizer_name] [num_gpus]"
    exit 1
fi

CHECKPOINT_BASE="$1"
MODEL_NAME="$2"
TOKENIZER_NAME="${3:-google/gemma-2b}"
NUM_GPUS="${4:-8}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_DIR="$(dirname "$SCRIPT_DIR")"
HF_MODELS_DIR="${EVAL_DIR}/hf_models/${MODEL_NAME}"
RESULTS_DIR="${EVAL_DIR}/results/${MODEL_NAME}"
LIGHTEVAL_CONFIG="${EVAL_DIR}/configs/arabic_finetasks.yaml"
LOG_DIR="${EVAL_DIR}/logs/${MODEL_NAME}"
QUEUE_DIR="${EVAL_DIR}/.queue/${MODEL_NAME}"

echo "=========================================="
echo "Parallel FineWeb2 Arabic Evaluation"
echo "=========================================="
echo "Model: $MODEL_NAME"
echo "Checkpoint base: $CHECKPOINT_BASE"
echo "Number of GPUs: $NUM_GPUS"
echo "HF models: $HF_MODELS_DIR"
echo "Results: $RESULTS_DIR"
echo "=========================================="

# Create directories
mkdir -p "$HF_MODELS_DIR" "$RESULTS_DIR" "$LOG_DIR" "$QUEUE_DIR"

# Find all checkpoints
CHECKPOINTS=($(find "$CHECKPOINT_BASE" -maxdepth 1 -type d -name '[0-9]*' | sort -V))
NUM_CHECKPOINTS=${#CHECKPOINTS[@]}

if [ "$NUM_CHECKPOINTS" -eq 0 ]; then
    echo "No checkpoints found in $CHECKPOINT_BASE"
    exit 1
fi

echo "Found $NUM_CHECKPOINTS checkpoints"
echo ""

# ============================================
# PHASE 1: Convert all checkpoints to HF format
# ============================================
echo "=========================================="
echo "PHASE 1: Converting checkpoints to HF format"
echo "=========================================="

for ckpt in "${CHECKPOINTS[@]}"; do
    step=$(basename "$ckpt")
    hf_path="${HF_MODELS_DIR}/step_${step}"

    if [ -f "${hf_path}/config.json" ]; then
        echo "Step $step: Already converted"
    else
        echo "Step $step: Converting..."
        "${SCRIPT_DIR}/convert_checkpoint.sh" "$ckpt" "$hf_path" "$TOKENIZER_NAME" 2>&1 | tail -5
    fi
done

echo ""
echo "All checkpoints converted!"
echo ""

# ============================================
# PHASE 2: Parallel evaluation across GPUs
# ============================================
echo "=========================================="
echo "PHASE 2: Running parallel evaluations"
echo "=========================================="

# Read tasks from config
TASKS=$(grep "^  tasks:" "$LIGHTEVAL_CONFIG" | sed 's/.*tasks: "\(.*\)"/\1/' | tr -d '[:space:]')

# Clean up old queue files
rm -rf "${QUEUE_DIR:?}"/*

# Create job files for each step that needs evaluation
job_count=0
for ckpt in "${CHECKPOINTS[@]}"; do
    step=$(basename "$ckpt")
    output_dir="${RESULTS_DIR}/step_${step}"

    # Check if already evaluated
    if [ -f "${output_dir}/results.json" ]; then
        echo "Step $step: Already evaluated, skipping"
    elif ls "${output_dir}"/*/results*.json >/dev/null 2>&1; then
        echo "Step $step: Already evaluated, skipping"
    else
        # Create a job file
        echo "$step" > "${QUEUE_DIR}/pending_${step}"
        ((job_count++)) || true
    fi
done

echo "Jobs to run: $job_count"
echo ""

if [ "$job_count" -eq 0 ]; then
    echo "All checkpoints already evaluated!"
    exit 0
fi

# Worker function - runs in background for each GPU
worker_script() {
    local gpu_id=$1
    local queue_dir=$2
    local hf_models_dir=$3
    local results_dir=$4
    local log_dir=$5
    local tasks=$6

    while true; do
        # Try to claim a pending job by atomically renaming it
        local job_file=""
        for f in "${queue_dir}"/pending_*; do
            if [ -f "$f" ]; then
                local step=$(cat "$f")
                local running_file="${queue_dir}/running_${step}_gpu${gpu_id}"
                # Try to atomically claim this job
                if mv "$f" "$running_file" 2>/dev/null; then
                    job_file="$running_file"
                    break
                fi
            fi
        done

        # No more jobs
        if [ -z "$job_file" ]; then
            echo "[GPU $gpu_id] No more jobs available, exiting"
            break
        fi

        local step=$(cat "$job_file")
        local hf_path="${hf_models_dir}/step_${step}"
        local output_dir="${results_dir}/step_${step}"
        local log_file="${log_dir}/step_${step}.log"

        echo "[GPU $gpu_id] Starting evaluation for step $step"

        mkdir -p "$output_dir"

        # Run evaluation with vLLM backend
        if CUDA_VISIBLE_DEVICES=$gpu_id VLLM_WORKER_MULTIPROC_METHOD=spawn python3 -c "
from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.models.vllm.vllm_model import VLLMModelConfig
from lighteval.pipeline import ParallelismManager, Pipeline, PipelineParameters

pipeline_params = PipelineParameters(
    launcher_type=ParallelismManager.VLLM,
    load_tasks_multilingual=True,
)
model_config = VLLMModelConfig.from_args('model_name=${hf_path},dtype=bfloat16,gpu_memory_utilization=0.9,max_model_length=2048')
evaluation_tracker = EvaluationTracker(output_dir='${output_dir}', save_details=True)

pipeline = Pipeline(
    tasks='${tasks}',
    pipeline_parameters=pipeline_params,
    evaluation_tracker=evaluation_tracker,
    model_config=model_config,
)
pipeline.evaluate()
pipeline.show_results()
pipeline.save_and_push_results()
" > "$log_file" 2>&1; then
            echo "[GPU $gpu_id] Completed step $step successfully"
            mv "$job_file" "${queue_dir}/done_${step}"
        else
            echo "[GPU $gpu_id] FAILED step $step (see $log_file)"
            mv "$job_file" "${queue_dir}/failed_${step}"
        fi
    done
}

# Start workers
echo "Starting $NUM_GPUS parallel workers..."
echo ""

pids=()
for ((gpu=0; gpu<NUM_GPUS && gpu<job_count; gpu++)); do
    worker_script "$gpu" "$QUEUE_DIR" "$HF_MODELS_DIR" "$RESULTS_DIR" "$LOG_DIR" "$TASKS" &
    pids+=($!)
    echo "Started worker on GPU $gpu (PID: ${pids[-1]})"
done

echo ""
echo "All workers started. Waiting for completion..."
echo "Monitor progress:"
echo "  - Main log: tail -f logs/parallel_eval_${MODEL_NAME}.log"
echo "  - Per-step logs: tail -f ${LOG_DIR}/*.log"
echo "  - Queue status: ls ${QUEUE_DIR}/"
echo ""

# Wait for all workers to complete
for pid in "${pids[@]}"; do
    wait $pid
done

# Count results
done_count=$(ls "${QUEUE_DIR}"/done_* 2>/dev/null | wc -l)
failed_count=$(ls "${QUEUE_DIR}"/failed_* 2>/dev/null | wc -l)

echo ""
echo "=========================================="
echo "Evaluation complete!"
echo "  Successful: $done_count"
echo "  Failed: $failed_count"
echo "Results saved to: $RESULTS_DIR"
echo "=========================================="

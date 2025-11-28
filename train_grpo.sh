#!/bin/bash

set -e

# GPU Configuration
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:$PYTHONPATH"

# Model and Data Configuration
# TODO: Update these paths to your local setup
MODEL_NAME_OR_PATH="Qwen/Qwen2.5-Math-1.5B"  # HuggingFace model ID or local path
OUTPUT_DIR="./output/Qwen2.5-Math-1.5B-Level0"
DATASET_NAME="./data/train/your-training-data.json"  # Path to your training dataset

DATASET_TRAIN_SPLIT="train"
DATASET_TEST_SPLIT="test"

mkdir -p "$OUTPUT_DIR"
REWARD_FUNCS="box_accuracy_reward"

# WandB Configuration (Optional - comment out if not using)
# export WANDB_PROJECT="math_grpo"
# export WANDB_API_KEY="your-wandb-api-key"

# --- Run the Training Script ---
accelerate launch --config_file "${SCRIPT_DIR}/trl_scope/accelerate_configs/zero3.yaml" ${SCRIPT_DIR}/trl_scope/scripts/grpo.py \
    --model_name_or_path "$MODEL_NAME_OR_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --dataset_name "$DATASET_NAME" \
    --dataset_train_split "$DATASET_TRAIN_SPLIT" \
    --dataset_test_split "$DATASET_TEST_SPLIT" \
    --num_train_epochs 1 \
    --max_completion_length 1024 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --logging_steps 1 \
    --save_steps 100 \
    --eval_strategy "no" \
    --save_strategy "epoch" \
    --report_to "wandb" \
    --reward_funcs "$REWARD_FUNCS" \
    --beta 0.0 \
    --num_generations 4 \
    --use_vllm True \
    --learning_rate 1e-6 \
    --vllm_mode "colocate" \
    --temperature 0.6 \
    2>&1 | tee "$OUTPUT_DIR/training.log"

# Optional arguments (uncomment as needed):
#     --log_completions \
#     --report_to "tensorboard" \ 
#!/bin/bash

# Enhanced Monopoly RL Training Script
# Incorporates Curriculum Learning, Self-Play, and Population-Based Training

set -e  # Exit on any error

# Configuration
EXPERIMENT_NAME="monopoly_enhanced_$(date +%Y%m%d_%H%M%S)"
BASE_DIR="rl_agent/checkpoints/${EXPERIMENT_NAME}"
TENSORBOARD_DIR="runs/${EXPERIMENT_NAME}"
HYPERPARAMS_FILE="enhanced_hyperparameters.json"
BOARD_JSON="/home/srinivasan/PycharmProjects/monopoly-rl/monopoly_env/core/data.json"

# Create experiment directory
mkdir -p "${BASE_DIR}"
mkdir -p "${TENSORBOARD_DIR}"

# Log file
LOG_FILE="${BASE_DIR}/training.log"
exec > >(tee -a "${LOG_FILE}")
exec 2>&1

echo "=== Enhanced Monopoly RL Training ==="
echo "Experiment: ${EXPERIMENT_NAME}"
echo "Start time: $(date)"
echo "GPU available: $(python3 -c 'import torch; print(torch.cuda.is_available())')"
echo "CUDA devices: $(python3 -c 'import torch; print(torch.cuda.device_count())')"
echo "========================================="

# Function to check if training should continue
check_training_status() {
    local phase=$1
    local expected_steps=$2
    echo "Checking ${phase} training status..."
    
    # Check if checkpoints exist and get the latest step count
    if [ -d "${BASE_DIR}" ]; then
        latest_checkpoint=$(find "${BASE_DIR}" -name "policy_player_0_update_*.pt" | sort -V | tail -1)
        if [ -n "$latest_checkpoint" ]; then
            echo "Found checkpoint: $latest_checkpoint"
            return 0
        fi
    fi
    return 1
}

# Function to run tensorboard in background
start_tensorboard() {
    echo "Starting TensorBoard..."
    tensorboard --logdir=runs --port=6006 --bind_all &
    TENSORBOARD_PID=$!
    echo "TensorBoard started with PID: $TENSORBOARD_PID"
    echo "Access TensorBoard at: http://localhost:6006"
}

# Function to cleanup background processes
cleanup() {
    echo "Cleaning up background processes..."
    if [ -n "$TENSORBOARD_PID" ]; then
        kill $TENSORBOARD_PID 2>/dev/null || true
    fi
}

# Set up cleanup trap
trap cleanup EXIT

# Start TensorBoard
start_tensorboard

echo "Phase 1: Curriculum Learning (5M steps)"
echo "Training against scripted opponents with simplified mechanics..."

python3 train_ippo.py \
    --hyperparameters "${HYPERPARAMS_FILE}" \
    --board-json "${BOARD_JSON}" \
    --num-players 4 \
    --total-timesteps 5000000 \
    --curriculum-steps 5000000 \
    --max-steps 1000 \
    --num-envs 16 \
    --num-steps 512 \
    --cuda \
    --track \
    --save-path "${BASE_DIR}/curriculum" \
    --checkpoint-freq 100 \
    --seed 42 \
    --compile

echo "Phase 1 completed. Checkpoints saved to: ${BASE_DIR}/curriculum"

echo "Phase 2: Self-Play Training (15M steps)"
echo "Self-play with opponent sampling from checkpoint pool..."

# Find the latest curriculum checkpoint to continue from
CURRICULUM_CHECKPOINT=$(find "${BASE_DIR}/curriculum" -name "policy_player_0_update_*.pt" | sort -V | tail -1)

python3 train_ippo.py \
    --hyperparameters "${HYPERPARAMS_FILE}" \
    --board-json "${BOARD_JSON}" \
    --num-players 4 \
    --total-timesteps 15000000 \
    --curriculum-steps 0 \
    --max-steps 2000 \
    --num-envs 32 \
    --num-steps 512 \
    --cuda \
    --track \
    --save-path "${BASE_DIR}/self_play" \
    --checkpoint-freq 200 \
    --seed 43 \
    --compile \
    --load-checkpoint "${CURRICULUM_CHECKPOINT}"

echo "Phase 2 completed. Checkpoints saved to: ${BASE_DIR}/self_play"

echo "Phase 3: Population-Based Training (5M steps)"
echo "Population-based training with diverse strategies..."

# Find the latest self-play checkpoint
SELFPLAY_CHECKPOINT=$(find "${BASE_DIR}/self_play" -name "policy_player_0_update_*.pt" | sort -V | tail -1)

python3 train_ippo.py \
    --hyperparameters "${HYPERPARAMS_FILE}" \
    --board-json "${BOARD_JSON}" \
    --num-players 4 \
    --total-timesteps 5000000 \
    --curriculum-steps 0 \
    --max-steps 3000 \
    --num-envs 48 \
    --num-steps 512 \
    --cuda \
    --track \
    --save-path "${BASE_DIR}/population" \
    --checkpoint-freq 150 \
    --seed 44 \
    --compile \
    --load-checkpoint "${SELFPLAY_CHECKPOINT}"

echo "Phase 3 completed. Checkpoints saved to: ${BASE_DIR}/population"

echo "=== Training Complete ==="
echo "Total training time: 25M steps across 3 phases"
echo "Final checkpoints: ${BASE_DIR}/population"
echo "TensorBoard logs: ${TENSORBOARD_DIR}"
echo "End time: $(date)"

# Final evaluation
echo "Running final evaluation..."
python3 evaluate.py \
    --checkpoint-dir "${BASE_DIR}/population" \
    --num-games 100 \
    --max-steps 5000 \
    --board-json "${BOARD_JSON}" \
    --output-file "${BASE_DIR}/final_evaluation.json"

echo "Evaluation complete. Results saved to: ${BASE_DIR}/final_evaluation.json"

# Generate training summary
echo "Generating training summary..."
python3 -c "
import json
import os
import glob

base_dir = '${BASE_DIR}'
summary = {
    'experiment_name': '${EXPERIMENT_NAME}',
    'total_timesteps': 25000000,
    'phases': {
        'curriculum': {'steps': 5000000, 'description': 'Learning against scripted opponents'},
        'self_play': {'steps': 15000000, 'description': 'Self-play with opponent sampling'},
        'population': {'steps': 5000000, 'description': 'Population-based training'}
    },
    'checkpoints': {
        'curriculum': len(glob.glob(f'{base_dir}/curriculum/policy_*.pt')),
        'self_play': len(glob.glob(f'{base_dir}/self_play/policy_*.pt')),
        'population': len(glob.glob(f'{base_dir}/population/policy_*.pt'))
    }
}

with open(f'{base_dir}/training_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print('Training summary saved to: {}/training_summary.json'.format(base_dir))
"

echo "All training phases completed successfully!"
echo "Experiment directory: ${BASE_DIR}"
echo "TensorBoard: http://localhost:6006" 
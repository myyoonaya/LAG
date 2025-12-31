#!/bin/bash
# Training script for DQN on SingleCombat Selfplay Shoot task

# Set environment variables
export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0

# Training parameters
env="SingleCombat"
scenario="1v1/ShootMissile/Selfplay"
algo="dqn"
exp="dqn_selfplay_shoot"
seed=1

# System parameters
num_env_steps=10000000  # 10M steps
n_rollout_threads=8
n_eval_rollout_threads=1

# DQN hyperparameters
dqn_batch_size=32
dqn_update_interval=4
dqn_target_update_interval=10000
dqn_warmup_steps=20000
dqn_buffer_size=100000
epsilon_start=1.0
epsilon_end=0.05
epsilon_decay_steps=100000

# Network parameters
hidden_size="256 256"
lr=0.0001
gamma=0.99

# Self-play parameters
use_selfplay="--use-selfplay"
selfplay_algorithm="sp"
n_choose_opponents=4

# Logging and evaluation
use_eval="--use-eval"
eval_interval=10
eval_episodes=32
log_interval=1
save_interval=10

# Optional: use wandb for logging
# use_wandb="--use-wandb"
# wandb_name="your_wandb_username"

# Run training
python scripts/train/train_dqn_jsbsim.py \
    --env-name="${env}" \
    --algorithm-name="${algo}" \
    --experiment-name="${exp}" \
    --scenario-name="${scenario}" \
    --seed=${seed} \
    --n-rollout-threads=${n_rollout_threads} \
    --n-eval-rollout-threads=${n_eval_rollout_threads} \
    --num-env-steps=${num_env_steps} \
    --dqn-batch-size=${dqn_batch_size} \
    --dqn-update-interval=${dqn_update_interval} \
    --dqn-target-update-interval=${dqn_target_update_interval} \
    --dqn-warmup-steps=${dqn_warmup_steps} \
    --dqn-buffer-size=${dqn_buffer_size} \
    --epsilon-start=${epsilon_start} \
    --epsilon-end=${epsilon_end} \
    --epsilon-decay-steps=${epsilon_decay_steps} \
    --hidden-size="${hidden_size}" \
    --lr=${lr} \
    --gamma=${gamma} \
    ${use_selfplay} \
    --selfplay-algorithm="${selfplay_algorithm}" \
    --n-choose-opponents=${n_choose_opponents} \
    ${use_eval} \
    --eval-interval=${eval_interval} \
    --eval-episodes=${eval_episodes} \
    --log-interval=${log_interval} \
    --save-interval=${save_interval} \
    --cuda
    # --use-wandb \
    # --wandb-name="${wandb_name}"

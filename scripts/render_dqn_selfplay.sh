#!/bin/bash

# Render DQN selfplay model to ACMI file for Tacview visualization
# Usage: bash scripts/render_dqn_selfplay.sh [options]

env="SingleCombat"
scenario="1v1/ShootMissile/Selfplay"
algo="dqn"
exp="dqn_selfplay"
seed=1

# Model directory (REQUIRED - change this to your trained model path)
# Example: results/SingleCombat/1v1/ShootMissile/Selfplay/dqn/dqn_selfplay
model_dir="results/${env}/${scenario}/${algo}/${exp}"

# Which policy checkpoint to render
render_index=100000  # Episode/step number of ego policy
render_opponent_index=100000  # Episode/step number of opponent policy

# Rendering settings
n_rollout_threads=1  # Use 1 thread for rendering
render_mode="txt"  # txt for ACMI file output

echo "Rendering DQN model to ACMI file..."
echo "Model directory: ${model_dir}"
echo "Ego policy: ${render_index}, Opponent policy: ${render_opponent_index}"
echo ""

python scripts/render/render_dqn_jsbsim.py \
    --env-name ${env} \
    --algorithm-name ${algo} \
    --experiment-name ${exp} \
    --scenario-name ${scenario} \
    --seed ${seed} \
    --n-rollout-threads ${n_rollout_threads} \
    --model-dir ${model_dir} \
    --render-index ${render_index} \
    --render-opponent-index ${render_opponent_index} \
    --render-mode ${render_mode}

echo ""
echo "Rendering completed! Check the output ACMI file in ${model_dir}/results/"
echo "Open the .txt.acmi file in Tacview to visualize the combat."

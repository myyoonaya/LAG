@echo off
REM Training script for DQN on SingleCombat Selfplay Shoot task (Windows)

REM Set environment variables
set OMP_NUM_THREADS=1
set CUDA_VISIBLE_DEVICES=0

REM Training parameters
set env=SingleCombat
set scenario=1v1/ShootMissile/Selfplay
set algo=dqn
set exp=dqn_selfplay_shoot
set seed=1

REM System parameters
set num_env_steps=10000000
set n_rollout_threads=8
set n_eval_rollout_threads=1

REM DQN hyperparameters
set dqn_batch_size=32
set dqn_update_interval=4
set dqn_target_update_interval=10000
set dqn_warmup_steps=20000
set dqn_buffer_size=100000
set epsilon_start=1.0
set epsilon_end=0.05
set epsilon_decay_steps=100000

REM Network parameters
set hidden_size=256 256
set lr=0.0001
set gamma=0.99

REM Self-play parameters
set selfplay_algorithm=sp
set n_choose_opponents=4

REM Logging and evaluation
set eval_interval=10
set eval_episodes=32
set log_interval=1
set save_interval=10

REM Run training
python scripts/train/train_dqn_jsbsim.py ^
    --env-name=%env% ^
    --algorithm-name=%algo% ^
    --experiment-name=%exp% ^
    --scenario-name=%scenario% ^
    --seed=%seed% ^
    --n-rollout-threads=%n_rollout_threads% ^
    --n-eval-rollout-threads=%n_eval_rollout_threads% ^
    --num-env-steps=%num_env_steps% ^
    --dqn-batch-size=%dqn_batch_size% ^
    --dqn-update-interval=%dqn_update_interval% ^
    --dqn-target-update-interval=%dqn_target_update_interval% ^
    --dqn-warmup-steps=%dqn_warmup_steps% ^
    --dqn-buffer-size=%dqn_buffer_size% ^
    --epsilon-start=%epsilon_start% ^
    --epsilon-end=%epsilon_end% ^
    --epsilon-decay-steps=%epsilon_decay_steps% ^
    --hidden-size=%hidden_size% ^
    --lr=%lr% ^
    --gamma=%gamma% ^
    --use-selfplay ^
    --selfplay-algorithm=%selfplay_algorithm% ^
    --n-choose-opponents=%n_choose_opponents% ^
    --use-eval ^
    --eval-interval=%eval_interval% ^
    --eval-episodes=%eval_episodes% ^
    --log-interval=%log_interval% ^
    --save-interval=%save_interval% ^
    --cuda

pause

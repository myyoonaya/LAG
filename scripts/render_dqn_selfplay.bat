@echo off
REM Render DQN selfplay model to ACMI file for Tacview visualization
REM Usage: scripts\render_dqn_selfplay.bat

set env=SingleCombat
set scenario=1v1/ShootMissile/Selfplay
set algo=dqn
set exp=dqn_selfplay
set seed=1

REM Model directory (REQUIRED - change this to your trained model path)
REM Example: results/SingleCombat/1v1/ShootMissile/Selfplay/dqn/dqn_selfplay
set model_dir=results/%env%/%scenario%/%algo%/%exp%

REM Which policy checkpoint to render
set render_index=100000
set render_opponent_index=100000

REM Rendering settings
set n_rollout_threads=1
set render_mode=txt

echo Rendering DQN model to ACMI file...
echo Model directory: %model_dir%
echo Ego policy: %render_index%, Opponent policy: %render_opponent_index%
echo.

python scripts/render/render_dqn_jsbsim.py ^
    --env-name %env% ^
    --algorithm-name %algo% ^
    --experiment-name %exp% ^
    --scenario-name %scenario% ^
    --seed %seed% ^
    --n-rollout-threads %n_rollout_threads% ^
    --model-dir %model_dir% ^
    --render-index %render_index% ^
    --render-opponent-index %render_opponent_index% ^
    --render-mode %render_mode%

echo.
echo Rendering completed! Check the output ACMI file in %model_dir%/results/
echo Open the .txt.acmi file in Tacview to visualize the combat.
pause

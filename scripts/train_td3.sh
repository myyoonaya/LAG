#!/bin/sh
env="SingleControl"
scenario="1/heading"
algo="td3"
exp="v1"
seed=5

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, seed is ${seed}"
CUDA_VISIBLE_DEVICES=0 python train/train_jsbsim.py \
    --env-name ${env} --algorithm-name ${algo} --scenario-name ${scenario} --experiment-name ${exp} \
    --seed ${seed} --n-training-threads 1 --n-rollout-threads 32 --cuda \
    --log-interval 1 --save-interval 1 \
    --num-env-steps 1e8 --use-recurrent-policy \
    --gamma 0.99 --td3-buffer-size 200000 --batch-size 256 --start-timesteps 10000 \
    --actor-lr 1e-3 --critic-lr 1e-3 --tau 0.005 --policy-delay 2 \
    --target-noise 0.2 --noise-clip 0.5 --explore-noise 0.1 --updates-per-step 1 \
    --hidden-size "128 128" --act-hidden-size "128 128"

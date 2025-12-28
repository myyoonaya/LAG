# DQN训练指南

本文档介绍如何使用Dueling-DQN在SingleCombat_selfplay_shoot环境中进行训练。

## 算法概述

### Dueling-DQN特性
- **多头分解动作空间**: 针对复杂动作空间`Tuple([MultiDiscrete([41,41,41,30]), Discrete(2)])`，将其分解为5个独立的决策头
- **Dueling架构**: 每个动作头都有独立的Value流和Advantage流，提高Q值估计的稳定性
- **经验回放**: 使用经验回放缓冲区打破数据相关性
- **目标网络**: 使用目标网络稳定训练过程

### 训练配置
- **Batch size**: 32
- **更新间隔**: 每4个env step更新1次
- **每次更新**: 1个gradient step
- **经验回放warmup**: 20,000 steps
- **Target network更新**: 每10,000 env steps同步一次
- **Epsilon衰减**: 从1.0衰减到0.05，共100,000 steps

## 文件结构

```
algorithms/dqn/
├── __init__.py              # 模块初始化
├── dqn_network.py           # Dueling-DQN网络实现
├── dqn_policy.py            # DQN策略类
├── dqn_trainer.py           # DQN训练器
└── replay_buffer.py         # 经验回放缓冲区

runner/
└── dqn_selfplay_jsbsim_runner.py  # DQN自博弈训练流程

scripts/
├── train/
│   └── train_dqn_jsbsim.py        # 训练脚本
├── train_dqn_selfplay.sh          # Linux/Mac启动脚本
└── train_dqn_selfplay.bat         # Windows启动脚本
```

## 快速开始

### Windows系统
```bash
cd e:\code\flyctrl\LAG
scripts\train_dqn_selfplay.bat
```

### Linux/Mac系统
```bash
cd /path/to/LAG
chmod +x scripts/train_dqn_selfplay.sh
./scripts/train_dqn_selfplay.sh
```

### Python直接运行
```bash
python scripts/train/train_dqn_jsbsim.py \
    --env-name=SingleCombat \
    --scenario-name=1v1/ShootMissile/Selfplay \
    --algorithm-name=dqn \
    --experiment-name=dqn_test \
    --seed=1 \
    --n-rollout-threads=8 \
    --num-env-steps=10000000 \
    --dqn-batch-size=32 \
    --dqn-update-interval=4 \
    --dqn-target-update-interval=10000 \
    --dqn-warmup-steps=20000 \
    --dqn-buffer-size=100000 \
    --epsilon-start=1.0 \
    --epsilon-end=0.05 \
    --epsilon-decay-steps=100000 \
    --hidden-size="256 256" \
    --lr=0.0001 \
    --gamma=0.99 \
    --use-selfplay \
    --selfplay-algorithm=sp \
    --n-choose-opponents=4 \
    --use-eval \
    --eval-interval=10 \
    --eval-episodes=32 \
    --cuda
```

## 主要参数说明

### DQN核心参数
- `--dqn-batch-size`: 训练批次大小，默认32
- `--dqn-update-interval`: 每多少env step更新一次网络，默认4
- `--dqn-target-update-interval`: 每多少env step同步目标网络，默认10000
- `--dqn-warmup-steps`: 开始训练前收集的步数，默认20000
- `--dqn-buffer-size`: 经验回放缓冲区大小，默认100000

### 探索参数
- `--epsilon-start`: 初始探索率，默认1.0
- `--epsilon-end`: 最终探索率，默认0.05
- `--epsilon-decay-steps`: 探索率衰减步数，默认100000

### 网络参数
- `--hidden-size`: 隐藏层维度，默认"256 256"
- `--lr`: 学习率，默认0.0001
- `--gamma`: 折扣因子，默认0.99
- `--activation-id`: 激活函数(0:Tanh, 1:ReLU, 2:LeakyReLU)

### 自博弈参数
- `--use-selfplay`: 启用自博弈训练
- `--selfplay-algorithm`: 自博弈算法，可选sp/fsp/pfsp
- `--n-choose-opponents`: 对手数量

### 系统参数
- `--n-rollout-threads`: 并行环境数量
- `--num-env-steps`: 总训练步数
- `--seed`: 随机种子
- `--cuda`: 使用GPU训练

## 训练监控

### 日志输出
训练过程会输出以下信息：
- **Loss**: DQN损失值
- **Q_mean**: 平均Q值
- **Q_max**: 最大Q值
- **Epsilon**: 当前探索率
- **Buffer_size**: 缓冲区当前大小
- **Average_episode_rewards**: 平均回合奖励

### WandB集成
如需使用WandB记录训练过程：
1. 在脚本中取消注释wandb相关行
2. 设置`--use-wandb`和`--wandb-name`参数

```bash
python scripts/train/train_dqn_jsbsim.py \
    ... 其他参数 ... \
    --use-wandb \
    --wandb-name=your_username
```

## 模型保存与加载

### 保存
模型会自动保存在 `results/SingleCombat/1v1/ShootMissile/Selfplay/dqn/experiment_name/` 目录下

保存间隔由`--save-interval`参数控制（默认每10个episode）

### 加载预训练模型
```bash
python scripts/train/train_dqn_jsbsim.py \
    ... 其他参数 ... \
    --model-dir=path/to/saved/model
```

## 性能优化建议

### GPU加速
- 确保安装了CUDA版本的PyTorch
- 使用`--cuda`参数启用GPU训练
- 根据GPU显存调整batch size

### 并行环境
- 增加`--n-rollout-threads`可以加快数据收集
- 建议设置为CPU核心数的一半

### 超参数调优
1. **学习率**: 从0.0001开始，观察损失曲线
2. **Batch size**: GPU允许的情况下可以增大
3. **Buffer size**: 更大的buffer可以提高样本多样性
4. **Epsilon decay**: 根据任务复杂度调整探索时长

## 常见问题

### Q1: 训练初期reward不稳定
这是正常现象，DQN在warmup阶段（前20k steps）只收集数据不训练，这段时间reward会很低。

### Q2: 内存占用过高
- 减小`--dqn-buffer-size`
- 减少`--n-rollout-threads`

### Q3: 训练速度慢
- 启用`--cuda`使用GPU
- 增加`--n-rollout-threads`
- 减少`--eval-interval`和`--save-interval`

### Q4: 性能不收敛
- 检查reward函数设计
- 尝试调整学习率
- 增加探索时长（epsilon_decay_steps）
- 调整target network更新频率

## 与PPO的对比

| 特性 | DQN | PPO |
|------|-----|-----|
| 动作空间 | 离散动作 | 离散/连续 |
| 样本效率 | 高（经验回放） | 低（on-policy） |
| 训练稳定性 | 中等 | 高 |
| 收敛速度 | 较慢 | 较快 |
| 内存占用 | 高（replay buffer） | 低 |
| 适用场景 | 复杂离散决策 | 通用强化学习 |

## 引用

如果使用此代码，请引用相关论文：

```bibtex
@article{wang2016dueling,
  title={Dueling network architectures for deep reinforcement learning},
  author={Wang, Ziyu and Schaul, Tom and Hessel, Matteo and others},
  journal={ICML},
  year={2016}
}
```

## 联系方式

如有问题或建议，请提交Issue或Pull Request。

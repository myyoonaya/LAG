# Dueling-DQN实现总结

## 概述

本次实现为LAG项目添加了完整的Dueling-DQN算法支持，用于在SingleCombat_selfplay_shoot环境中训练智能体。

## 实现的文件

### 1. 核心算法模块 (`algorithms/dqn/`)

#### `dqn_network.py` - Dueling-DQN网络
- **多头分解架构**: 针对复杂的组合动作空间，将其分解为多个独立的决策头
- **Dueling结构**: 每个头包含独立的Value流和Advantage流
- **动作空间支持**: 自动解析`Tuple([MultiDiscrete([41,41,41,30]), Discrete(2)])`为5个独立头
- **Epsilon-greedy**: 内置epsilon-greedy探索策略

#### `replay_buffer.py` - 经验回放缓冲区
- 高效的numpy数组存储，预分配内存
- 支持随机采样
- 返回PyTorch tensor格式的batch数据

#### `dqn_policy.py` - DQN策略类
- 管理Q网络和目标网络
- 提供动作选择接口
- 支持模型保存和加载

#### `dqn_trainer.py` - DQN训练器
- **经验回放warmup**: 前20,000步只收集不训练
- **更新间隔控制**: 每4个env step更新1次
- **目标网络同步**: 每10,000步同步一次
- **Epsilon衰减**: 线性衰减策略
- **梯度裁剪**: 防止梯度爆炸

### 2. 训练流程 (`runner/`)

#### `dqn_selfplay_jsbsim_runner.py` - 自博弈训练
- 支持多线程并行环境收集
- 自博弈对手池管理
- 完整的训练/评估循环
- WandB日志集成

### 3. 配置和脚本

#### `config.py` - 添加DQN配置
新增配置参数：
- `--dqn-batch-size`: 批次大小 (默认32)
- `--dqn-update-interval`: 更新间隔 (默认4)
- `--dqn-target-update-interval`: 目标网络更新间隔 (默认10000)
- `--dqn-warmup-steps`: warmup步数 (默认20000)
- `--dqn-buffer-size`: 缓冲区大小 (默认100000)
- `--epsilon-start/end/decay-steps`: epsilon参数

#### 训练脚本
- `scripts/train/train_dqn_jsbsim.py`: Python训练脚本
- `scripts/train_dqn_selfplay.sh`: Linux/Mac启动脚本
- `scripts/train_dqn_selfplay.bat`: Windows启动脚本

### 4. 测试和文档

#### `tests/test_dqn.py` - 单元测试
测试所有核心组件：
- DQN网络前向传播
- 经验回放缓冲区
- 策略动作选择
- 训练器更新逻辑

#### `docs/DQN_Training_Guide.md` - 使用文档
完整的训练指南，包括：
- 算法介绍
- 快速开始
- 参数说明
- 常见问题
- 性能优化建议

## 技术特点

### 1. 多头动作分解
针对环境的复杂动作空间`Tuple([MultiDiscrete([41,41,41,30]), Discrete(2)])`：
- 飞控动作: aileron(41) + elevator(41) + rudder(41) + throttle(30)
- 射击动作: shoot(2)

每个维度独立学习Q值，降低动作空间复杂度从 `41×41×41×30×2 = 2,068,020` 到 `41+41+41+30+2 = 155`

### 2. Dueling架构优势
$$Q(s,a) = V(s) + (A(s,a) - \frac{1}{|A|}\sum_{a'}A(s,a'))$$

- Value流: 评估状态价值
- Advantage流: 评估动作相对优势
- 更稳定的Q值估计

### 3. 经验回放机制
- **打破相关性**: 随机采样历史经验
- **提高样本效率**: 每个transition可以被多次使用
- **稳定训练**: 减少数据分布变化

### 4. 目标网络
- 使用固定参数的目标网络计算TD target
- 每10,000步同步一次，避免target快速变化
- 提高训练稳定性

## 训练参数配置

按照要求配置的参数：
- ✅ **Batch size**: 32
- ✅ **更新间隔**: 每4个env step更新1次
- ✅ **每次更新**: 1个gradient step
- ✅ **Warmup**: 20,000 steps
- ✅ **Target network更新**: 每10,000 env steps

## 使用方法

### 快速开始（Windows）
```bash
cd e:\code\flyctrl\LAG
scripts\train_dqn_selfplay.bat
```

### 自定义参数
```bash
python scripts/train/train_dqn_jsbsim.py \
    --algorithm-name=dqn \
    --scenario-name=1v1/ShootMissile/Selfplay \
    --dqn-batch-size=32 \
    --dqn-update-interval=4 \
    --dqn-target-update-interval=10000 \
    --dqn-warmup-steps=20000 \
    --cuda
```

### 运行测试
```bash
python tests/test_dqn.py
```

## 预期性能

### 训练阶段
1. **Warmup (0-20k steps)**: 随机探索，收集经验
2. **Early training (20k-100k steps)**: 高探索率，快速学习
3. **Mid training (100k-500k steps)**: 探索率降低，策略改进
4. **Late training (500k+ steps)**: 策略收敛，性能稳定

### 监控指标
- **Loss**: 应逐渐下降并稳定
- **Q_mean**: 反映策略价值，应逐渐上升
- **Episode_reward**: 任务性能，应逐渐提高
- **Epsilon**: 线性衰减到0.05

## 与现有PPO的对比

| 特性 | DQN | PPO |
|------|-----|-----|
| 样本效率 | ⬆️ 高 (经验回放) | ⬇️ 低 (on-policy) |
| 内存占用 | ⬆️ 高 (replay buffer) | ⬇️ 低 |
| 训练稳定性 | ➡️ 中等 | ⬆️ 高 |
| 离散动作 | ⬆️ 原生支持 | ➡️ 支持 |
| 连续动作 | ⬇️ 不支持 | ⬆️ 支持 |
| 并行化 | ➡️ 环境并行 | ⬆️ 环境+更新并行 |

## 扩展建议

### 1. 算法改进
- [ ] 实现Double DQN减少过估计
- [ ] 实现Prioritized Experience Replay
- [ ] 实现Noisy Networks替代epsilon-greedy
- [ ] 实现Rainbow DQN集成多种改进

### 2. 工程优化
- [ ] 添加分布式训练支持
- [ ] 优化经验回放的内存管理
- [ ] 添加checkpoint自动恢复
- [ ] 实现训练可视化界面

### 3. 应用扩展
- [ ] 支持更多JSBSim任务
- [ ] 实现curriculum learning
- [ ] 添加对手建模
- [ ] 实现迁移学习

## 文件清单

```
新增/修改的文件：

algorithms/dqn/
├── __init__.py                     ✨ 新增
├── dqn_network.py                  ✨ 新增
├── dqn_policy.py                   ✨ 新增
├── dqn_trainer.py                  ✨ 新增
└── replay_buffer.py                ✨ 新增

runner/
└── dqn_selfplay_jsbsim_runner.py   ✨ 新增

scripts/
├── train/
│   └── train_dqn_jsbsim.py         ✨ 新增
├── train_dqn_selfplay.sh           ✨ 新增
└── train_dqn_selfplay.bat          ✨ 新增

tests/
└── test_dqn.py                     ✨ 新增

docs/
└── DQN_Training_Guide.md           ✨ 新增

config.py                           🔧 修改 (添加DQN配置)
```

## 总结

本次实现完整地为LAG项目添加了Dueling-DQN算法支持，包括：
- ✅ 核心算法实现（网络、策略、训练器）
- ✅ 自博弈训练流程
- ✅ 完整的配置系统
- ✅ 便捷的训练脚本
- ✅ 单元测试
- ✅ 详细文档

所有配置参数严格按照要求设置，代码结构清晰，易于扩展和维护。

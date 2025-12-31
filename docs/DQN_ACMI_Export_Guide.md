# DQN模型导出ACMI文件教程

本文档介绍如何将训练好的DQN模型导出为ACMI文件，以便在Tacview中可视化空战回放。

## 什么是ACMI文件？

ACMI (Air Combat Maneuvering Instrumentation) 是Tacview软件使用的标准格式，用于记录和回放飞行战斗数据。通过生成ACMI文件，您可以：

- 3D可视化观察飞机的机动轨迹
- 分析战斗态势和决策过程
- 评估模型的战术行为
- 制作演示视频

## 快速开始

### 1. 准备工作

确保您已经训练好了DQN模型，并知道模型保存的路径。例如：
```
results/SingleCombat/1v1/ShootMissile/Selfplay/dqn/dqn_selfplay/
```

### 2. 修改渲染脚本参数

编辑 `scripts/render_dqn_selfplay.bat` (Windows) 或 `scripts/render_dqn_selfplay.sh` (Linux)：

```bash
# 设置模型目录（必须修改为您的实际路径）
set model_dir=results/SingleCombat/1v1/ShootMissile/Selfplay/dqn/dqn_selfplay

# 选择要渲染的模型检查点
set render_index=100000          # 主策略的步数/episode
set render_opponent_index=100000  # 对手策略的步数/episode
```

**检查点命名规则：**
- 如果保存时使用 `policy_{step}.pt` 格式，设置为对应的step数
- 如果保存在 `step_{step}/policy.pt` 目录，设置为对应的step数
- 脚本会自动尝试两种命名方式

### 3. 运行渲染脚本

**Windows:**
```bash
scripts\render_dqn_selfplay.bat
```

**Linux:**
```bash
bash scripts/render_dqn_selfplay.sh
```

### 4. 查找生成的ACMI文件

渲染完成后，ACMI文件会保存在：
```
results/SingleCombat/1v1/ShootMissile/Selfplay/dqn/dqn_selfplay/dqn_selfplay.txt.acmi
```

文件名格式：`{experiment_name}.txt.acmi`

## 在Tacview中打开ACMI文件

### 安装Tacview

1. 下载Tacview：https://www.tacview.net/
2. 安装（支持Windows, Linux, macOS）
3. 免费版本足够用于基本可视化

### 打开ACMI文件

1. 启动Tacview
2. File → Open
3. 选择生成的 `.txt.acmi` 文件
4. 即可观看3D回放

### Tacview操作技巧

- **鼠标拖动**：旋转视角
- **滚轮**：缩放
- **空格键**：播放/暂停
- **←/→**：前进/后退
- **数字键1-9**：切换视角（驾驶舱视角、跟随视角等）
- **F2**：显示HUD信息
- **F3**：显示轨迹线

## 高级用法

### 命令行参数

您也可以直接使用Python脚本，自定义更多参数：

```bash
python scripts/render/render_dqn_jsbsim.py \
    --env-name SingleCombat \
    --scenario-name "1v1/ShootMissile/Selfplay" \
    --algorithm-name dqn \
    --experiment-name dqn_selfplay \
    --model-dir results/SingleCombat/1v1/ShootMissile/Selfplay/dqn/dqn_selfplay \
    --render-index 100000 \
    --render-opponent-index 80000 \
    --seed 1 \
    --render-mode txt
```

### 常用参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model-dir` | 模型保存目录（必需） | 无 |
| `--render-index` | 主策略检查点索引 | 0 |
| `--render-opponent-index` | 对手策略检查点索引 | 0 |
| `--scenario-name` | 场景名称 | 1v1/ShootMissile/Selfplay |
| `--seed` | 随机种子 | 1 |
| `--render-mode` | 渲染模式 | txt |

### 渲染不同检查点对比

您可以渲染不同训练阶段的模型进行对比：

```bash
# 早期模型 (step 20000)
set render_index=20000
set render_opponent_index=20000
scripts\render_dqn_selfplay.bat

# 中期模型 (step 50000)
set render_index=50000
set render_opponent_index=50000
scripts\render_dqn_selfplay.bat

# 最终模型 (step 100000)
set render_index=100000
set render_opponent_index=100000
scripts\render_dqn_selfplay.bat
```

### 渲染自对弈（不同版本对战）

让早期版本对战最新版本：

```bash
set render_index=100000      # 最新版本
set render_opponent_index=50000  # 中期版本
scripts\render_dqn_selfplay.bat
```

## 常见问题

### Q1: 找不到模型文件

**错误信息：** `Ego policy not found: ...`

**解决方法：**
1. 检查 `model_dir` 路径是否正确
2. 确认训练时确实保存了模型
3. 检查 `render_index` 是否对应实际存在的检查点

### Q2: ACMI文件无法在Tacview中打开

**可能原因：**
- 渲染过程中断，文件不完整
- 文件格式错误

**解决方法：**
1. 重新运行渲染脚本
2. 检查渲染日志 `render.log` 是否有错误
3. 确保环境正常运行（`env.render()` 功能正常）

### Q3: 渲染速度很慢

**优化方法：**
- 使用GPU：确保 `--cuda` 开启且CUDA可用
- 减少episode长度：修改环境配置
- 使用确定性策略：已默认设置 `epsilon=0`

### Q4: 如何选择合适的检查点？

**建议：**
1. 查看训练日志，找到平均奖励最高的步数
2. 使用 `policy_latest.pt` 作为最新模型
3. 对于selfplay，建议ego和opponent使用相同或接近的检查点

## 文件结构

渲染相关文件：

```
LAG/
├── scripts/
│   ├── render/
│   │   └── render_dqn_jsbsim.py      # 核心渲染脚本
│   ├── render_dqn_selfplay.sh         # Linux启动脚本
│   └── render_dqn_selfplay.bat        # Windows启动脚本
├── runner/
│   └── dqn_selfplay_jsbsim_runner.py  # DQN runner (包含render方法)
└── docs/
    └── DQN_ACMI_Export_Guide.md       # 本文档
```

## 工作原理

渲染过程：

1. **加载模型**：从指定路径加载训练好的DQN策略
2. **初始化环境**：创建JSBSim战斗环境
3. **确定性执行**：使用 `epsilon=0` 的贪婪策略
4. **记录轨迹**：每步调用 `env.render(mode='txt', filepath='...')` 写入ACMI
5. **生成文件**：episode结束后保存完整ACMI文件

## 示例输出

成功渲染后的日志：

```
==========================================
Rendering DQN on JSBSim Environment
==========================================
Algorithm: dqn
Scenario: 1v1/ShootMissile/Selfplay
Ego policy index: 100000
Opponent policy index: 100000
==========================================

Start rendering to ACMI file...
Output file: results/dqn_selfplay.txt.acmi
Rendered 100 steps...
Rendered 200 steps...
...

Rendering completed!
Total steps: 456
Episode reward: [235.67]
ACMI file saved to: results/dqn_selfplay.txt.acmi

You can now open this file in Tacview for visualization.
```

## 进阶：实时Tacview连接

除了生成ACMI文件，您还可以使用实时模式：

```bash
# 修改render_mode为real_time
set render_mode=real_time
```

这将启动Tacview服务器，您可以实时观看训练过程（需要额外配置，详见 `runner/tacview.py`）。

## 总结

通过ACMI文件导出功能，您可以：

✅ 可视化DQN智能体的飞行轨迹  
✅ 分析战术决策和行为模式  
✅ 对比不同训练阶段的性能  
✅ 制作演示和分析报告  

有任何问题，请查看日志文件或提issue！

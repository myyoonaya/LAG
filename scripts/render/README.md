# DQN模型导出ACMI文件 - 快速指南

## 什么是ACMI？

ACMI (Air Combat Maneuvering Instrumentation) 是Tacview用于3D可视化飞行战斗的标准格式。

## 快速使用

### 1. 修改配置

编辑 `scripts/render_dqn_selfplay.bat` (Windows) 或 `.sh` (Linux):

```batch
REM 设置您的模型路径
set model_dir=results/SingleCombat/1v1/ShootMissile/Selfplay/dqn/dqn_selfplay

REM 选择要渲染的检查点
set render_index=100000          # 主策略步数
set render_opponent_index=100000  # 对手策略步数
```

### 2. 运行渲染

```bash
# Windows
scripts\render_dqn_selfplay.bat

# Linux  
bash scripts/render_dqn_selfplay.sh
```

### 3. 查看结果

生成的ACMI文件位置：
```
{model_dir}/{experiment_name}.txt.acmi
```

在Tacview中打开此文件即可3D回放战斗过程。

## 渲染脚本

| 文件 | 说明 |
|------|------|
| `scripts/render/render_dqn_jsbsim.py` | 核心渲染脚本 |
| `scripts/render_dqn_selfplay.bat` | Windows启动脚本 |
| `scripts/render_dqn_selfplay.sh` | Linux启动脚本 |

## 关键参数

- `--model-dir`: 模型保存目录（**必需**）
- `--render-index`: 主策略检查点索引
- `--render-opponent-index`: 对手策略检查点索引
- `--scenario-name`: 场景名称

## 示例

### 渲染最新模型

```bash
python scripts/render/render_dqn_jsbsim.py \
    --model-dir results/SingleCombat/1v1/ShootMissile/Selfplay/dqn/exp1 \
    --render-index 100000 \
    --render-opponent-index 100000
```

### 对比不同版本

```bash
# 新模型 vs 旧模型
--render-index 100000 \
--render-opponent-index 50000
```

## Tacview操作

- **拖动鼠标**: 旋转视角
- **滚轮**: 缩放
- **空格**: 播放/暂停
- **←/→**: 前进/后退
- **1-9**: 切换视角

## 详细文档

完整教程请查看: [docs/DQN_ACMI_Export_Guide.md](../docs/DQN_ACMI_Export_Guide.md)

## 常见问题

**Q: 找不到模型文件？**
检查 `model_dir` 和 `render_index` 是否正确。

**Q: Tacview无法打开？**  
重新运行渲染，确保episode完整结束。

**Q: 如何选择检查点？**
查看训练日志，选择奖励最高的步数。

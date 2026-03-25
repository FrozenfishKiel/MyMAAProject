# 项目进度与待办事项 (Project Progress & Todo)

## 🎯 核心目标
将传统的固定脚本自动化（RPA）重构为真正的端到端强化学习（End-to-End RL）模型，使 AI 能够通过视觉画面自主学习明日方舟的塔防部署策略（类似 `DQN_play_sekiro` 的设计模式）。

---

## ✅ 已完成进度 (Completed)

### 1. 架构方案重构设计
- [x] 确立了新的 Action Space：从“宏操作触发器”转变为 **多维离散网格坐标 (MultiDiscrete)**。AI 现在输出 `[选卡(0-9), 网格X(0-9), 网格Y(0-4), 朝向(0-3)]`。
- [x] 确立了新的 Observation Space：从灰度图改为 **RGB 彩色图像 (3, 72, 128)**，保留重要的敌我颜色特征。
- [x] 确立了新的 Reward Function：废弃了 RPA 脚本的执行状态，改为 **视觉反馈 (YOLO 识别血条增减 + 图像 MSE 残差计算)**，准确判定部署成功、非法操作与存活。

### 2. 核心代码重写
- [x] **`src/rl-environment/actions.py`**：彻底剥离 OpenCV 找高亮/找返回界面的逻辑。重写 `execute_deployment` 方法，专门负责将网格坐标映射为真实的屏幕像素坐标，并调用 MAA 执行原生的 `post_click` 和 `post_swipe`。
- [x] **`src/rl-environment/game_env.py`**：完成 Gym 环境重构。整合了新的状态空间、动作空间，并实现了“动作执行前后对比血条”的奖励逻辑，以及基于步数和画面亮度的 Done (回合结束) 判定机制。
- [x] **`src/rl-training/train.py`**：更新训练脚本以适配新的环境结构。移除了废弃的 `TemplateMatcher`，配置了 Stable Baselines3 的 `CnnPolicy`，并开启了 Tensorboard 日志记录功能。

---

## 🚀 待办事项 (Todo / Next Steps)

### 1. 运行前置准备
- [ ] **准备 YOLO 模型**：确认 `models/yolo/best.pt` 文件存在。即使目前没有完美训练好的模型，也需要有一个能运行的 `.pt` 文件（哪怕是临时的 Mock 模型），否则 `train.py` 会在初始化时报错。
- [ ] **模拟器与游戏状态**：启动 MuMu 模拟器，打开明日方舟，进入任意一个战斗关卡，并保持在可以操作的战斗界面。

### 2. 连通性测试与运行
- [ ] **连通性校验**：测试 Python 脚本能否正常调用 MAA 接口并连接到 MuMu 模拟器的 ADB 端口。
- [ ] **首次 Dry Run 测试**：运行 `train.py`，观察控制台输出。重点检查：
  - CNN 模型是否能正常处理 `(3, 72, 128)` 的图像张量（Tensor Shape 是否匹配）。
  - MAA 是否能根据 AI 输出的随机网格坐标，在模拟器上做出正确的“点击底栏 -> 拖拽到屏幕”动作。
  - YOLO 奖励逻辑是否会因为张量计算或截屏延迟抛出异常。

### 3. 后续优化 (Optional)
- [ ] 收集数百张游戏截图，针对 `operator_hp_bar`（干员血条）训练一个基础但准确的 YOLOv8 模型，以提供最核心的 Reward 信号。
- [ ] 在 `game_env.py` 中加入 `Frame Stacking`（帧堆叠）技术，让 AI 能“看懂”敌人的移动速度和方向。
- [ ] 调优 PPO 超参数（如学习率、entropy coefficient），引导 AI 更好地探索地图网格。

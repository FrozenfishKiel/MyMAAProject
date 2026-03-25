from __future__ import annotations

import sys
import time
import cv2
import numpy as np
from pathlib import Path
from typing import Any, Dict, Tuple

from gymnasium import Env
from gymnasium.spaces import Box, MultiDiscrete

# 添加项目路径到sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ai-plugins"))
sys.path.insert(0, str(ROOT / "rl-environment"))

from yolo_recognizer import YoloRecognizer
from actions import DeployOperatorActions


class GameEnv(Env):
    """
    网格化的明日方舟 RL 部署环境
    完全移除了固定流程，交由模型输出多维坐标
    """

    def __init__(self, controller: Any, yolo_recognizer: YoloRecognizer, template_matcher: Any = None) -> None:
        """
        初始化RL环境

        Args:
            controller: MaaFramework控制器
            yolo_recognizer: YOLO识别器
            template_matcher: 预留接口，已弃用
        """
        super().__init__()

        self._controller = controller
        self._yolo_recognizer = yolo_recognizer
        self._deploy_actions = DeployOperatorActions(controller)

        # 1. 新的状态空间 (Observation Space) - 彩色图像 RGB
        # 宽 128, 高 72 (16:9比例), 通道数 3
        self.WIDTH = 128
        self.HEIGHT = 72
        self.CHANNELS = 3

        self.observation_space = Box(
            low=0, high=255,
            shape=(self.CHANNELS, self.HEIGHT, self.WIDTH),  # SB3 CNN 要求的形状是 Channel-First
            dtype=np.uint8
        )

        # 2. 新的动作空间 (Action Space) - MultiDiscrete
        # [选干员位(0-9), 网格X(0-9), 网格Y(0-4), 朝向(0-3)]
        self.action_space = MultiDiscrete([10, 10, 5, 4])

        self.time_step = 0
        self.max_time_steps = 30 # 最大部署步数

    def _get_state_and_raw_image(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取缩小后的状态输入，以及用于检测血条的原图
        """
        try:
            if self._controller is None:
                return np.zeros((self.CHANNELS, self.HEIGHT, self.WIDTH), dtype=np.uint8), None

            # 获取原图
            raw_image = self._controller.post_screencap().wait().get()

            # 缩小尺寸，用于 RL 模型输入
            state = cv2.resize(raw_image, (self.WIDTH, self.HEIGHT))

            # 颜色转换：MAA 截图默认可能是 BGR，转换为 RGB 以获得真实色彩特征
            state = cv2.cvtColor(state, cv2.COLOR_BGR2RGB)

            # 转换为 SB3 要求的格式 (Channels, Height, Width)
            state = np.transpose(state, (2, 0, 1))

            return state, raw_image
        except Exception as e:
            print(f"[ERROR] 获取状态失败: {e}")
            return np.zeros((self.CHANNELS, self.HEIGHT, self.WIDTH), dtype=np.uint8), None

    def reset(self, seed: int = None, options: Dict[str, Any] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        self.time_step = 0
        state, _ = self._get_state_and_raw_image()
        return state, {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self.time_step += 1
        terminated = False
        truncated = False
        info = {}

        print(f"\n[STEP {self.time_step}] 开始执行动作...")

        # 1. 动作执行前：截图并识别血条数量
        state_before, img_before = self._get_state_and_raw_image()
        hp_bars_before = 0
        if img_before is not None:
            # 寻找当前屏幕上有多少个干员的血条
            detections = self._yolo_recognizer.detect(img_before, conf=0.25)
            hp_bars_before = sum(1 for d in detections if d.label == "operator_hp_bar")

        # 2. 调用 actions.py，把离散的动作数组变成 MAA 的物理操作
        self._deploy_actions.execute_deployment(action)

        # 3. 等待游戏响应（拖拽动画和特效时间）
        time.sleep(1.5)

        # 4. 动作执行后：截图并计算奖励
        state_after, img_after = self._get_state_and_raw_image()
        hp_bars_after = 0
        if img_after is not None:
            detections = self._yolo_recognizer.detect(img_after, conf=0.25)
            hp_bars_after = sum(1 for d in detections if d.label == "operator_hp_bar")

        # 5. 核心逻辑：YOLO 血条判定法 + 像素残差判别非法操作
        reward = 0.0
        print(f"[REWARD] 执行前血条数: {hp_bars_before} -> 执行后血条数: {hp_bars_after}")

        if hp_bars_after > hp_bars_before:
            # 【奖励情况 1：干员部署成功】
            # 屏幕上的干员血条增多了，说明模型成功地将可用干员拖到了合法高亮格子上
            reward = 10.0
            print("[REWARD] 恭喜！成功部署干员。奖励 +10.0")
        else:
            # 如果血条没变，说明部署失败。我们计算一下动作前后的图像差异 (MSE)
            if img_before is not None and img_after is not None:
                # 转灰度算残差，加速计算
                gray_before = cv2.cvtColor(img_before, cv2.COLOR_BGR2GRAY)
                gray_after = cv2.cvtColor(img_after, cv2.COLOR_BGR2GRAY)
                mse = np.mean((gray_after.astype("float") - gray_before.astype("float")) ** 2)

                print(f"[REWARD] 部署失败，计算前后帧 MSE (均方误差) = {mse:.2f}")

                # 如果画面差异极小（比如 < 15.0），说明拖拽后弹回去了，没有任何反应
                if mse < 15.0:
                    # 【奖励情况 2：非法操作/无脑乱点】
                    reward = -1.0
                    print("[REWARD] 画面未发生明显变化(非法部署)。奖励 -1.0")
                else:
                    # 【奖励情况 3：未部署成功，但有特效在动 (正常战斗状态)】
                    reward = 0.1
                    print("[REWARD] 游戏继续，常规存活。奖励 +0.1")

        # 6. 锚点检测（判断战斗是否结束，比如跳转到了结算界面）
        # 如果获取到了原图，我们截取右上角固定位置的一小块作为“战斗内标志位”
        if img_after is not None:
            # 截取右上角 (比如设置按钮周围区域)
            h, w, _ = img_after.shape
            anchor_region = img_after[0:40, w-40:w]  # 40x40区域
            avg_brightness = np.mean(anchor_region)

            # 如果亮度低于极低阈值（黑屏或者暗场），判定结束
            if avg_brightness < 10.0:
                print(f"[ENV] 战斗画面锚点亮度过低 ({avg_brightness:.1f})，判定战斗结束。")
                terminated = True

        # 强制截断保护
        if self.time_step >= self.max_time_steps:
            truncated = True
            print(f"[ENV] 达到最大步数 ({self.max_time_steps})，终止回合。")

        info = {
            "hp_bars": hp_bars_after,
            "reward_given": reward
        }

        return state_after, reward, terminated, truncated, info

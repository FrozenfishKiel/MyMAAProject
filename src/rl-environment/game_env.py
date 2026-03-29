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
        # 将 template_matcher 存下来用于 reset 时的自动化点击
        self._template_matcher = template_matcher
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
        # 添加一个连续被 CV 阻断的计数器，用于判断战斗是否其实已经结束了
        self.consecutive_fast_fails = 0
        self.consecutive_missing_gear = 0

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

    def _click_template(self, image: np.ndarray, template_name: str, threshold: float = 0.6, roi: Tuple[int, int, int, int] = None, check_only: bool = False) -> bool:
        """
        在画面中寻找指定的模板图片，并点击它的中心点。
        """
        if self._template_matcher is None:
            print("[WARN] TemplateMatcher 未初始化，无法执行图片点击！")
            return False

        template_path = str(ROOT.parent / "data" / "templates" / f"{template_name}.png")
        if not Path(template_path).exists():
            print(f"[WARN] 找不到重启模板文件: {template_path}，跳过点击。")
            return False

        result = self._template_matcher.match(image, template_path, threshold=threshold, roi=roi)
        if result is not None:
            # 计算中心点坐标
            x1, y1, x2, y2 = result.box_xyxy
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            if check_only:
                print(f"[RESET] 找到 [{template_name}] (置信度 {result.confidence:.2f} >= {threshold})，(仅检测，不点击)")
            else:
                print(f"[RESET] 找到 [{template_name}] (置信度 {result.confidence:.2f} >= {threshold})，点击 ({cx}, {cy})")
                self._controller.post_click(cx, cy).wait()
            return True
        else:
            # 开启这行调试信息，让我们可以看到它确实在找，只是没找到（或者置信度太低）
            print(f"[DEBUG] 未匹配 [{template_name}] (要求阈值 {threshold})")
            pass
        return False

    def _check_battle_end(self, image: np.ndarray) -> bool:
        """
        检查画面是否出现了战斗结束的标志（胜利或失败）。
        需要在 data/templates/ 下提供:
        - battle_win.png (例如行动结束的文字/图标)
        - battle_lose.png (例如任务失败的红色大字/图标)
        """
        if self._template_matcher is None:
            return False

        # 检测胜利结算界面
        win_template = str(ROOT.parent / "data" / "templates" / "battle_win.png")
        if Path(win_template).exists():
            result = self._template_matcher.match(image, win_template, threshold=0.7)
            if result is not None:
                print("[ENV] 检测到战斗胜利结算画面！")
                return True

        # 检测失败结算界面
        lose_template = str(ROOT.parent / "data" / "templates" / "battle_lose.png")
        if Path(lose_template).exists():
            result = self._template_matcher.match(image, lose_template, threshold=0.7)
            if result is not None:
                print("[ENV] 检测到战斗失败结算画面！")
                return True

        return False

    def _auto_restart_battle(self) -> None:
        """
        使用状态机循环识别画面并点击，直到成功进入战斗部署界面。
        你需要在 data/templates/ 下准备好：
        1. stage_1_7.png (地图上的关卡图标 "1-7")
        2. practice.png (演习按钮)
        3. start_action.png (开始行动/确认编队界面的大红色按钮)
        4. pause_gear.png (战斗画面右上角的暂停齿轮，作为进入战斗的标志)
        """
        print("\n================= 回合结束，开始自动重启关卡 =================")
        max_attempts = 100 # 防止无限死循环，适当调大，因为结算动画和加载较慢
        attempts = 0
        in_battle = False

        while not in_battle and attempts < max_attempts:
            attempts += 1
            image = self._controller.post_screencap().wait().get()
            if image is None:
                time.sleep(1)
                continue

            # --- 逆向优先级状态机 ---
            # 逻辑：倒序检查，从最接近“进入战斗”的界面开始匹配。
            # 如果在编队界面，由于匹配了 start_action，它就不会去试图寻找 stage_1_7，解决了你的死循环问题。

            # 动态获取画面尺寸以计算 ROI (Region of Interest)
            h, w = image.shape[:2]

            # 状态1：检测是否已经进入战斗（看到齿轮标志）
            # ROI: 屏幕右上角区域 [宽 70%~100%, 高 0%~30%]
            gear_roi = (int(w * 0.7), 0, w, int(h * 0.3))

            # 使用 check_only=True 仅仅检测，不点击它
            if self._click_template(image, "pause_gear", threshold=0.60, roi=gear_roi, check_only=True):
                print("[RESET] ✅ 检测到战斗画面右上角齿轮，重启成功！")
                in_battle = True
                # 进入战斗后，稍微等一等动画播完再交还控制权给 AI
                time.sleep(4)
                break

            # 状态2：如果在干员编队页，点击红色的“开始行动”
            # ROI: 使用动态比例，限制在右下角 [宽 50%~100%, 高 40%~100%]，和演习区域完全对齐
            start_roi = (int(w * 0.75), int(h * 0.4), w, h)
            if self._click_template(image, "start_action", threshold=0.55, roi=start_roi):
                print("[RESET] 点击开始行动，进入加载...")
                time.sleep(5) # 点击开始行动后，加载时间较长
                continue

            # 状态3：在关卡详情页，点击“演习”
            # ROI: 屏幕右半边偏下区域 [宽 50%~100%, 高 40%~100%]
            practice_roi = (int(w * 0.3), int(h * 0.75), w, h)
            if self._click_template(image, "practice", threshold=0.42, roi=practice_roi):
                print("[RESET] 点击演习按钮")
                time.sleep(2)
                continue

            # 状态4：如果在地图选关界面，点击 "1-7" 关卡图标
            # ROI: 听你的！限制在屏幕中间区域，避开边缘的UI杂色干扰。
            # [宽 20%~80%, 高 20%~80%]，这样就屏蔽了左右两侧的干员立绘。
            # 并且把阈值降回 0.48，这样它就能轻松识别上去了！
            stage_roi = (int(w * 0.2), int(h * 0.2), int(w * 0.8), int(h * 0.8))
            if self._click_template(image, "stage_1_7", threshold=0.48, roi=stage_roi):
                print("[RESET] 点击关卡 1-7 图标")
                time.sleep(2)
                continue

            # 状态5：如果在战斗中，由于某种原因触发了 reset（比如达到了强制的最大步数）
            # 我们需要主动退出战斗（放弃），才能回到外面的界面去重启。
            # 这是极其关键的错误恢复机制。
            # （但因为现在有了精准的结算识别，理论上只有超时才会触发这里）

            # 状态6：如果什么都没匹配上，很有可能是在战斗结算界面的动画中，或者是弹出了物资掉落
            # 我们点击屏幕一个绝对安全的空白处（比如中下部），跳过这些过场动画
            print(f"[RESET] 尝试跳过动画/寻找入口... (尝试 {attempts}/{max_attempts})")
            self._controller.post_click(640, 600).wait()
            time.sleep(1.5)


        if attempts >= max_attempts:
            print("[ERROR] 自动重启关卡失败！陷入死循环，请检查模拟器界面是否异常！")
        print("================= 关卡重启流程结束 =================\n")

    def reset(self, seed: int = None, options: Dict[str, Any] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)

        # 核心：如果是被 done=True 触发进来的 reset，执行全自动重启流程！
        if self.time_step > 0:
            self._auto_restart_battle()

        self.time_step = 0
        state, _ = self._get_state_and_raw_image()
        return state, {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self.time_step += 1
        terminated = False
        truncated = False
        info = {}

        print(f"\n[STEP {self.time_step}] 开始执行动作...")

        # 1. 动作执行前：截图并记录所有血条的中心点坐标
        state_before, img_before = self._get_state_and_raw_image()
        hp_bar_centers_before = []
        if img_before is not None:
            detections = self._yolo_recognizer.detect(img_before, conf=0.25)
            for d in detections:
                if d.label == "operator_hp_bar":
                    # 计算血条中心点 (cx, cy)
                    cx = (d.box_xyxy[0] + d.box_xyxy[2]) / 2
                    cy = (d.box_xyxy[1] + d.box_xyxy[3]) / 2
                    hp_bar_centers_before.append((cx, cy))

        # 2. 调用 actions.py，把离散的动作数组变成 MAA 的物理操作
        # 并接收 CV Fast-Fail 阻断标志
        executed_swipe, target_gx, target_gy = self._deploy_actions.execute_deployment(action)

        # 3. 如果成功拖拽，等待游戏响应；如果是被 CV 阻断的，根本不需要等！
        if executed_swipe:
            time.sleep(1.5)
        else:
            time.sleep(0.1) # 被阻断了，直接跳过漫长的等待，进入下一轮算分

        # 4. 动作执行后：截图并记录所有血条的中心点坐标
        state_after, img_after = self._get_state_and_raw_image()
        hp_bar_centers_after = []
        if img_after is not None:
            detections = self._yolo_recognizer.detect(img_after, conf=0.25)
            for d in detections:
                if d.label == "operator_hp_bar":
                    cx = (d.box_xyxy[0] + d.box_xyxy[2]) / 2
                    cy = (d.box_xyxy[1] + d.box_xyxy[3]) / 2
                    hp_bar_centers_after.append((cx, cy))

        # 5. 计算全局 MSE (用于判断画面大变动/转场)
        mse = 0.0
        if img_before is not None and img_after is not None:
            gray_before = cv2.cvtColor(img_before, cv2.COLOR_BGR2GRAY)
            gray_after = cv2.cvtColor(img_after, cv2.COLOR_BGR2GRAY)
            mse = np.mean((gray_after.astype("float") - gray_before.astype("float")) ** 2)

        # 6. 核心逻辑：YOLO 局部坐标校验法 (解决血条闪烁和全局数量欺骗)
        reward = 0.0
        print(f"[REWARD] 执行前血条数: {len(hp_bar_centers_before)} -> 执行后血条数: {len(hp_bar_centers_after)}")

        deployed_success = False
        if executed_swipe and img_after is not None:
            # 遍历部署后的每一个血条
            for cx_after, cy_after in hp_bar_centers_after:
                is_new_hp_bar = True
                # 检查它是否和部署前的某个血条极其接近 (欧氏距离 < 50 像素)
                for cx_before, cy_before in hp_bar_centers_before:
                    dist = np.sqrt((cx_after - cx_before)**2 + (cy_after - cy_before)**2)
                    if dist < 50.0:
                        is_new_hp_bar = False
                        break

                # 如果这是一个全新的血条，我们还要检查它是不是在 AI 刚才拖拽的目标网格附近！
                # (容差 120 像素，因为血条可能在角色头顶偏移)
                if is_new_hp_bar:
                    dist_to_target = np.sqrt((cx_after - target_gx)**2 + (cy_after - target_gy)**2)
                    if dist_to_target < 120.0:
                        deployed_success = True
                        break

        if not executed_swipe:
            # 【奖励情况 0：被 CV Fast-Fail 机制直接阻断】
            reward = -1.0
            self.consecutive_fast_fails += 1
            print(f"[REWARD] ⛔ 非法区域/无费用，已被 CV 阻断。奖励 -1.0 (连续无作为: {self.consecutive_fast_fails}次)")
        elif deployed_success:
            # 【奖励情况 1：干员部署成功】
            reward = 10.0
            self.consecutive_fast_fails = 0
            print("[REWARD] 恭喜！局部坐标校验成功，干员部署在目标区域。奖励 +10.0")
        else:
            print(f"[REWARD] 部署失败，计算前后帧 MSE (均方误差) = {mse:.2f}")
            if mse < 300.0:
                reward = -1.0
                self.consecutive_fast_fails += 1
                print(f"[REWARD] 画面未发生明显变化(非法部署)。奖励 -1.0 (连续无作为: {self.consecutive_fast_fails}次)")
            else:
                reward = -0.5
                # 这是最关键的修复：只要画面有明显变化（说明是正常的战斗特效，游戏还在进行），
                # 哪怕 AI 只是瞎划了一下，我们也必须打断“连续失能”的计数器！
                self.consecutive_fast_fails = 0
                print("[REWARD] 动作无效，但游戏仍在进行(画面有变化)。奖励 -0.5 (已清空无作为计数器)")

        # 7. 多重战斗结束判定 (Priority 1 -> 2 -> 3)
        # 优先级 1：MSE 剧烈变化，判定为转场（如跳出结算界面）
        if mse > 8000.0:
            print(f"[ENV] 🚨 优先级 1 触发：检测到画面剧烈变化 (MSE={mse:.2f} > 8000)，判定为场景切换/结算，强制终止回合！")
            terminated = True

        # 优先级 2：连续找不到右上角的齿轮图标（UI元素消失）
        elif img_after is not None and not terminated:
            template_path = str(ROOT.parent / "data" / "templates" / "pause_gear.png")
            if Path(template_path).exists() and self._template_matcher is not None:
                # 仅检测不点击，阈值稍微放宽
                result = self._template_matcher.match(img_after, template_path, threshold=0.5)
                if result is None:
                    self.consecutive_missing_gear += 1
                    print(f"[ENV] ⚠️ 警告：未检测到右上角齿轮图标！(连续 {self.consecutive_missing_gear}/3 次)")
                    if self.consecutive_missing_gear >= 3:
                        print(f"[ENV] 🚨 优先级 2 触发：连续 {self.consecutive_missing_gear} 次未检测到战斗界面UI元素，判定游戏已结束！")
                        terminated = True
                else:
                    self.consecutive_missing_gear = 0

 

        # 强制截断保护 (已移除步数限制)
        # if self.time_step >= self.max_time_steps:
        #     truncated = True
        #     print(f"[ENV] 达到最大步数 ({self.max_time_steps})，终止回合。")

        info = {
            "hp_bars": len(hp_bar_centers_after),
            "reward_given": reward
        }

        return state_after, reward, terminated, truncated, info

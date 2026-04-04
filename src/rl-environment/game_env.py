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
        # [选干员位(0-9，10代表挂机/空过), 网格X(0-9), 网格Y(0-4), 朝向(0-3)]
        self.action_space = MultiDiscrete([11, 10, 5, 4])

        self.time_step = 0
        # 添加一个连续被 CV 阻断的计数器，用于判断战斗是否其实已经结束了
        self.consecutive_fast_fails = 0
        self.consecutive_missing_gear = 0

        # === 初始化雷达视觉引擎 (Radar Vision) ===
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=120, detectShadows=False)
        self.kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        self.kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))

    def close(self):
        super().close()

    def _get_state_and_raw_image(self) -> Tuple[np.ndarray, np.ndarray, list]:
        """
        获取 3 通道雷达状态输入，用于检测血条的原图，以及雷达锁定的敌人目标坐标。
        返回: (state_tensor, raw_image, enemy_targets_list)
        """
        try:
            if self._controller is None:
                return np.zeros((self.CHANNELS, self.HEIGHT, self.WIDTH), dtype=np.uint8), None, []

            # 获取原图
            raw_image = self._controller.post_screencap().wait().get()

            # --- 制作通道 1：灰度原图 ---
            gray = cv2.cvtColor(raw_image, cv2.COLOR_BGR2GRAY)

            # --- 制作通道 2：绿色可部署区域 Mask ---
            hsv = cv2.cvtColor(raw_image, cv2.COLOR_BGR2HSV)
            lower_green = np.array([35, 43, 46])
            upper_green = np.array([90, 255, 255])
            grid_mask = cv2.inRange(hsv, lower_green, upper_green)
            grid_mask = cv2.morphologyEx(grid_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

            # --- 制作通道 3：敌人动态雷达 Mask ---
            fg_mask = self.bg_subtractor.apply(raw_image)
            fg_mask_clean = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self.kernel_open)
            fg_mask_clean = cv2.morphologyEx(fg_mask_clean, cv2.MORPH_CLOSE, self.kernel_close)

            # --- 提取敌人坐标靶心 ---
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(fg_mask_clean, connectivity=8)
            enemy_targets = []
            for i in range(1, num_labels):
                area = stats[i, cv2.CC_STAT_AREA]
                if 300 < area < 4000:
                    width = stats[i, cv2.CC_STAT_WIDTH]
                    height = stats[i, cv2.CC_STAT_HEIGHT]
                    if 0.25 < width/height < 4.0:
                        enemy_targets.append((int(centroids[i][0]), int(centroids[i][1])))

            # --- 组合为 3 通道 RL 输入 ---
            # OpenCV merge 需要同样的尺寸
            stacked_img = cv2.merge([gray, grid_mask, fg_mask_clean])

            # 缩小尺寸，用于 RL 模型输入
            state = cv2.resize(stacked_img, (self.WIDTH, self.HEIGHT))

            # 转换为 SB3 要求的格式 (Channels, Height, Width)
            state = np.transpose(state, (2, 0, 1))

            return state, raw_image, enemy_targets
        except Exception as e:
            print(f"[ERROR] 获取状态失败: {e}")
            return np.zeros((self.CHANNELS, self.HEIGHT, self.WIDTH), dtype=np.uint8), None, []

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

        h, w = image.shape[:2]
        # 【关键优化】方舟打完出星级的地方是在左上角。
        # 我们把扫描区域 (ROI) 死死限制在左上角的狭小范围内，并向下偏移
        # 对应测试脚本中调校好的坐标
        star_roi = (int(w * 0.05), int(h * 0.35), int(w * 0.30), int(h * 0.55))

        # 1. 检测 3 星完美通关 (最高奖励)
        template_3star = str(ROOT.parent / "data" / "templates" / "battle_3star.png")
        if Path(template_3star).exists():
            res_3 = self._template_matcher.match(image, template_3star, threshold=0.45, roi=star_roi, silent=True, exact_scale=True)
            if res_3 is not None:
                # 把当前的置信度强行打印到屏幕（用来在实战中排错）
                print(f"\n[ENV] 🎉 检测到 3 星完美通关！(置信度: {res_3.confidence:.3f}) 发放巨额奖励 +100.0")
                return True, 100.0

        # 2. 检测 2 星漏怪通关 (中等奖励，虽然通关了但防守有漏洞)
        template_2star = str(ROOT.parent / "data" / "templates" / "battle_2star.png")
        if Path(template_2star).exists():
            res_2 = self._template_matcher.match(image, template_2star, threshold=0.45, roi=star_roi, silent=True, exact_scale=True)
            if res_2 is not None:
                print(f"\n[ENV] ⚠️ 检测到 2 星瑕疵通关！(置信度: {res_2.confidence:.3f}) 发放惩罚 -15.0")
                return True, -15.0

        # 3. 检测 0 星任务失败 (漏怪/全灭)
        template_0star = str(ROOT.parent / "data" / "templates" / "battle_0star.png")
        if Path(template_0star).exists():
            res_0 = self._template_matcher.match(image, template_0star, threshold=0.45, roi=star_roi, silent=True, exact_scale=True)
            if res_0 is not None:
                print(f"\n[ENV] 💀 检测到 0 星任务失败！(置信度: {res_0.confidence:.3f}) 发放惩罚 -50.0")
                return True, -50.0

        # 3. 检测 0 星任务失败 (漏怪/全灭)
        template_0star = str(ROOT.parent / "data" / "templates" / "battle_0star.png")
        if Path(template_0star).exists():
            if self._template_matcher.match(image, template_0star, threshold=0.5, roi=star_roi, silent=True, exact_scale=True):
                print("\n[ENV] 💀 检测到 0 星任务失败 (漏怪/全灭)。发放惩罚 -50.0")
                return True, -50.0

        return False, 0.0

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
            if self._click_template(image, "pause_gear", threshold=0.70, roi=gear_roi, check_only=True):
                print("[RESET] ✅ 检测到战斗画面右上角齿轮，重启成功！")
                in_battle = True
                # 进入战斗后，稍微等一等动画播完再交还控制权给 AI
                time.sleep(4)
                break

            # 状态2：如果在干员编队页，点击红色的“开始行动”
            # ROI: 使用动态比例，限制在右下角 [宽 50%~100%, 高 40%~100%]，和演习区域完全对齐
            start_roi = (int(w * 0.75), int(h * 0.4), w, h)
            if self._click_template(image, "start_action", threshold=0.43, roi=start_roi):
                print("[RESET] 点击开始行动，进入加载...")
                time.sleep(5) # 点击开始行动后，加载时间较长
                continue

            # 状态3：在关卡详情页，点击“演习”
            # ROI: 屏幕右半边偏下区域 [宽 50%~100%, 高 40%~100%]
            practice_roi = (int(w * 0.3), int(h * 0.75), w, h)
            if self._click_template(image, "practice", threshold=0.5, roi=practice_roi):
                print("[RESET] 点击演习按钮")
                time.sleep(2)
                continue

            # 状态4：如果在地图选关界面，点击 "1-7" 关卡图标
            # ROI: 听你的！限制在屏幕中间区域，避开边缘的UI杂色干扰。
            # [宽 20%~80%, 高 20%~80%]，这样就屏蔽了左右两侧的干员立绘。
            # 并且把阈值降回 0.48，这样它就能轻松识别上去了！
            stage_roi = (int(w * 0.2), int(h * 0.2), int(w * 0.8), int(h * 0.8))
            if self._click_template(image, "stage_1_7", threshold=0.5, roi=stage_roi):
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
        state, _, _ = self._get_state_and_raw_image()
        return state, {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self.time_step += 1
        terminated = False
        truncated = False
        info = {}

        print(f"\n[STEP {self.time_step}] 开始执行动作...")

        # 1. 动作执行前：截图并记录所有血条的中心点坐标
        state_before, img_before, enemy_targets_before = self._get_state_and_raw_image()
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
        executed_swipe, target_gx, target_gy, direction = self._deploy_actions.execute_deployment(action)

        # 3. 如果成功拖拽，采用多次校验机制防漏检（检查 3 次，间隔 0.2s）
        deployed_success = False
        hp_bar_centers_after = []

        if executed_swipe:
            print(f"[REWARD] 动作已执行，开始 3 次心跳检测 (间隔 0.2s)...")
            # 移除 1.0 秒的初始死板等待，立刻开始心跳检测，加快训练节奏

            for attempt in range(3):
                time.sleep(0.2)
                state_after, img_after, current_enemy_targets = self._get_state_and_raw_image()
                current_hp_bars = []

                if img_after is not None:
                    detections = self._yolo_recognizer.detect(img_after, conf=0.25)
                    for d in detections:
                        if d.label == "operator_hp_bar":
                            cx = (d.box_xyxy[0] + d.box_xyxy[2]) / 2
                            cy = (d.box_xyxy[1] + d.box_xyxy[3]) / 2
                            current_hp_bars.append((cx, cy))

                hp_bar_centers_after = current_hp_bars

                # 校验是否成功
                for cx_after, cy_after in current_hp_bars:
                    is_new_hp_bar = True
                    for cx_before, cy_before in hp_bar_centers_before:
                        if np.sqrt((cx_after - cx_before)**2 + (cy_after - cy_before)**2) < 50.0:
                            is_new_hp_bar = False
                            break

                    if is_new_hp_bar:
                        dist_to_target = np.sqrt((cx_after - target_gx)**2 + (cy_after - target_gy)**2)
                        # 容差扩大到 250 像素，涵盖干员下方的血条
                        if dist_to_target < 250.0:
                            deployed_success = True
                            break

                if deployed_success:
                    print(f"[REWARD] ✅ 第 {attempt+1} 次检测命中目标干员血条！")
                    break
                else:
                    print(f"[REWARD] ❌ 第 {attempt+1} 次未检测到目标血条...")
        else:
            time.sleep(0.1) # 被阻断了，直接跳过漫长的等待
            state_after, img_after, current_enemy_targets = self._get_state_and_raw_image()

        # 5. 计算全局 MSE (用于判断画面大变动/转场)
        mse = 0.0
        if img_before is not None and img_after is not None:
            gray_before = cv2.cvtColor(img_before, cv2.COLOR_BGR2GRAY)
            gray_after = cv2.cvtColor(img_after, cv2.COLOR_BGR2GRAY)
            mse = np.mean((gray_after.astype("float") - gray_before.astype("float")) ** 2)

        # 6. 核心逻辑：结算奖励
        reward = 0.0
        print(f"[REWARD] 执行前血条数: {len(hp_bar_centers_before)} -> 最终确认血条数: {len(hp_bar_centers_after)}")

        # ================== 核心雷达空间奖励结算 (朝向/距离) ==================
        if executed_swipe == False and target_gx == -1 and target_gy == -1:
            # 【情况 0：AI 主动选择挂机等费用】
            reward = -0.01
            self.consecutive_fast_fails = 0
            print("[REWARD] 💤 AI 主动挂机等费用，只扣除微小时间惩罚 -0.01")
        elif deployed_success:
            # 基础分：只要部署成功就给 +1.0 (大幅降低底薪，逼迫它追求完美操作)
            reward = 1.0
            self.consecutive_fast_fails = 0

            # 强制重置 MOG2 消除落地瞬间大面积的白光鬼影
            self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=120, detectShadows=False)

            # !!! 极其关键的修正：必须使用 AI 决策“前”的敌人坐标 (enemy_targets_before) !!!
            # 因为拖拽+动画要耗费 2~3 秒，如果用落地后的坐标，敌人可能已经走过干员了，
            # 这会导致 AI 原本正确的“迎击”预判，在结算时被误判为“背对”而冤枉扣分！
            if len(enemy_targets_before) > 0:
                # 寻找离干员最近的敌人
                closest_enemy = min(enemy_targets_before, key=lambda e: (e[0]-target_gx)**2 + (e[1]-target_gy)**2)
                enemy_x, enemy_y = closest_enemy

                # 计算距离 (等距视角下，Y轴像素差距权重算大一点)
                dist = np.sqrt((enemy_x - target_gx)**2 + (1.5 * (enemy_y - target_gy))**2)

                # 1. 距离奖励 (是否贴脸抗敌)
                if dist < 300.0:  # 距离很近，比如盾卫顶在前排
                    reward += 2.0
                    print(f"[REWARD] 🎯 完美落子！部署点靠近敌人 (距离: {dist:.1f})，奖励 +2.0")
                elif dist > 800.0: # 放到了完全打不到的地方
                    reward -= 1.0
                    print(f"[REWARD] ⚠️ 无效部署！距离敌人太远 (距离: {dist:.1f})，惩罚 -1.0")

                # 2. 朝向奖励 (是否正对敌人)
                import math
                # 计算敌人相对于干员的真实角度
                angle = math.degrees(math.atan2(enemy_y - target_gy, enemy_x - target_gx))
                if angle < 0: angle += 360

                # 将 0:上, 1:下, 2:左, 3:右 映射为角度的扇区
                # 0(上): 225~315, 1(下): 45~135, 2(左): 135~225, 3(右): 315~45
                is_facing = False
                if direction == 0 and 225 <= angle <= 315: is_facing = True
                elif direction == 1 and 45 <= angle <= 135: is_facing = True
                elif direction == 2 and 135 <= angle <= 225: is_facing = True
                elif direction == 3 and (angle >= 315 or angle <= 45): is_facing = True

                if is_facing:
                    reward += 8.0
                    print(f"[REWARD] ⚔️ 完美朝向！正对敌人 (角度: {angle:.1f}°)，巨额奖励 +8.0")
                else:
                    reward -= 1.0
                    print(f"[REWARD] 🛡️ 背对敌人！(角度: {angle:.1f}°)，严厉惩罚 -1.0")
            else:
                # 场上暂时没敌人的情况 (静态兜底)
                print("[REWARD] 场上暂无移动目标，仅发放基础部署奖励 +1.0")

        elif not executed_swipe:
            # 【情况 2：被 CV Fast-Fail 机制直接阻断】
            reward = -15.0 # 【路线B：极大增强非法拦截的惩罚】让它产生对非法区域的恐惧
            self.consecutive_fast_fails += 1
            print(f"[REWARD] ⛔ 非法区域/无费用，已被 CV 阻断。极度严厉惩罚 -15.0 (连续无作为: {self.consecutive_fast_fails}次)")
        else:
            print(f"[REWARD] 部署失败，计算前后帧 MSE (均方误差) = {mse:.2f}")
            if mse < 300.0:
                reward = -15.0 # 【路线B：极大增强画面无变化的惩罚】
                self.consecutive_fast_fails += 1
                print(f"[REWARD] 画面未发生明显变化(非法部署)。极度严厉惩罚 -15.0 (连续无作为: {self.consecutive_fast_fails}次)")
            else:
                reward = -0.2
                self.consecutive_fast_fails = 0
                print("[REWARD] 动作无效，但游戏仍在进行(画面有变化)。奖励 -0.2 (已清空无作为计数器)")

        # 7. 多重战斗结束判定 (Priority 1 -> 2 -> 3)

        # ================== 兜底保护判定 ==================
        # 如果游戏确实结束了，但由于某种原因没有进入上面的结算分支（比如小弟的线程挂了，或者截图失败），
        # 还是需要一个粗暴的方法结束游戏，避免死循环。
        if mse > 8000.0:
            print(f"[ENV] 🚨 优先级 2 触发：画面剧变 (MSE={mse:.2f} > 8000)，判定为转场/断线，结束回合！")
            terminated = True
        elif img_after is not None and not terminated:
            template_path = str(ROOT.parent / "data" / "templates" / "pause_gear.png")
            if Path(template_path).exists() and self._template_matcher is not None:
                result = self._template_matcher.match(img_after, template_path, threshold=0.5)
                if result is None:
                    self.consecutive_missing_gear += 1
                    print(f"[ENV] ⚠️ 未检测到右上角齿轮图标！(连续 {self.consecutive_missing_gear}/3 次)")
                    if self.consecutive_missing_gear >= 3:
                        print(f"[ENV] 🚨 优先级 3 触发：齿轮彻底消失，战斗已结束！")
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

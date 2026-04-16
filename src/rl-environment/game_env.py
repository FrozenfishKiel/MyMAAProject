from __future__ import annotations

import sys
import time
import threading
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

        # 1. 新的状态空间 (Observation Space) - 彩色图像 RGB + 1个职业通道
        # 宽 128, 高 72 (16:9比例), 通道数 4
        self.WIDTH = 128
        self.HEIGHT = 72
        self.CHANNELS = 4

        self.observation_space = Box(
            low=0, high=255,
            shape=(self.CHANNELS, self.HEIGHT, self.WIDTH),  # SB3 CNN 要求的形状是 Channel-First
            dtype=np.uint8
        )

        # 2. 新的动作空间 (Action Space) - MultiDiscrete
        # 完全删除挂机选项！
        # [选干员位(0-11共12张卡), 网格X(0-9), 网格Y(0-4), 朝向(0-3)]
        self.action_space = MultiDiscrete([12, 10, 5, 4])

        self.time_step = 0
        # 添加一个连续被 CV 阻断的计数器，用于判断战斗是否其实已经结束了
        self.consecutive_fast_fails = 0
        self.consecutive_missing_gear = 0

        # --- 预加载职业模板 ---
        self._class_templates = {}
        self._class_names = {
            "Caster": "术士", "Medic": "医疗", "Pioneer": "先锋", "Sniper": "狙击",
            "Special": "特种", "Support": "辅助", "Tank": "重装", "Warrior": "近卫"
        }
        self._class_id_map = {
            "Pioneer": 0, "Warrior": 1, "Sniper": 2, "Caster": 3,
            "Tank": 4, "Medic": 5, "Support": 6, "Special": 7
        }
        self._load_class_templates()

        # --- 异步全局监视器 (小弟打饭) ---
        self._battle_active = False
        self._monitor_thread = None
        self._stop_monitor_event = threading.Event()
        self._emergency_stop_flag = False

        # === 初始化雷达视觉引擎 (Radar Vision) ===
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=120, detectShadows=False)
        self.kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        self.kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        self._init_level_prior("1-7")

    def _load_class_templates(self):
        template_dir = ROOT.parent / "data" / "templates"
        for role_key in self._class_names.keys():
            tmpl_path = template_dir / f"BattleOperRole{role_key}.png"
            if tmpl_path.exists():
                tmpl_img = cv2.imread(str(tmpl_path), cv2.IMREAD_GRAYSCALE)
                self._class_templates[role_key] = tmpl_img
            else:
                print(f"[WARNING] 找不到模板文件: {tmpl_path}")

    def _init_level_prior(self, stage_code="1-7"):
        import json
        self.playable_grid = np.zeros((5, 10), dtype=np.uint8)
        try:
            json_path = ROOT.parent / "Arknights-Tile-Pos-main" / "Arknights-Tile-Pos-main" / "levels.json"
            with open(json_path, 'r', encoding='utf-8') as f:
                levels = json.load(f)
            target_level = next((l for l in levels if l.get("code") == stage_code), None)
            if target_level:
                tiles = target_level["tiles"]
                for r in range(5):
                    for c in range(10):
                        jr = r + 1
                        jc = c + 1
                        if jr < len(tiles) and jc < len(tiles[jr]):
                            self.playable_grid[r, c] = tiles[jr][jc]["buildableType"]
            print(f"[ENV] 成功加载 {stage_code} 关卡地形先验数据(黑白灰矩阵)!")
        except Exception as e:
            print(f"[ENV ERROR] 地形加载失败: {e}")
            self.playable_grid.fill(1)

        self.ch2_map = np.zeros((self.HEIGHT, self.WIDTH), dtype=np.uint8)
        cell_w = self.WIDTH / 10.0
        cell_h = self.HEIGHT / 5.0
        for r in range(5):
            for c in range(10):
                val = self.playable_grid[r, c]
                color = 0
                if val == 1: color = 255 # 低台白
                elif val == 2: color = 128 # 高台灰
                x1, y1 = int(c * cell_w), int(r * cell_h)
                x2, y2 = int((c+1) * cell_w), int((r+1) * cell_h)
                import cv2
                cv2.rectangle(self.ch2_map, (x1, y1), (x2, y2), color, -1)

    def close(self):
        super().close()

    def _get_state_and_raw_image(self) -> Tuple[np.ndarray, np.ndarray, list]:
        """
        获取 4 通道状态输入，用于检测血条的原图，以及雷达锁定的敌人目标坐标。
        返回: (state_tensor, raw_image, enemy_targets_list)
        """
        try:
            if self._controller is None:
                return np.zeros((self.CHANNELS, self.HEIGHT, self.WIDTH), dtype=np.uint8), None, []

            # 获取原图
            raw_image = self._controller.post_screencap().wait().get()

            # --- 制作通道 1：灰度原图 ---
            gray = cv2.cvtColor(raw_image, cv2.COLOR_BGR2GRAY)

            # --- 制作通道 2：先验地图地形层 (白=低台, 灰=高台, 黑=不可部署) ---
            grid_mask = cv2.resize(self.ch2_map, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_NEAREST)

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

            # --- 制作通道 4：手牌区职业映射图 ---
            # 创建一个全黑的单通道图像，尺寸与原图一致
            ch4_map = np.zeros((gray.shape[0], gray.shape[1]), dtype=np.uint8)

            # 复用我们在 test_live_brightness 中的完美切割逻辑
            h_raw, w_raw = raw_image.shape[:2]
            y_start, y_end = int(h_raw * 0.75), int(h_raw * 0.98)
            x_start, x_end = int(w_raw * 0.13), int(w_raw * 0.97)

            num_cards = 12
            total_width = x_end - x_start
            slot_width = total_width / num_cards
            card_actual_width = slot_width * 0.80

            for i in range(num_cards):
                slot_center_x = x_start + int((i + 0.5) * slot_width)
                card_x1 = int(slot_center_x - card_actual_width / 2)
                card_x2 = int(slot_center_x + card_actual_width / 2)

                card_x1 = max(0, card_x1)
                card_x2 = min(w_raw, card_x2)

                single_card_roi = raw_image[y_start:y_end, card_x1:card_x2]
                single_card_gray = cv2.cvtColor(single_card_roi, cv2.COLOR_BGR2GRAY)

                avg_brightness = np.mean(single_card_gray)

                # 只有亮着的卡，我们才去匹配职业并在第四通道给它“上色”
                if avg_brightness > 90 and single_card_gray.shape[0] > 10 and single_card_gray.shape[1] > 10:
                    best_val = -1.0
                    detected_id = -1

                    for role_key, tmpl_img in self._class_templates.items():
                        th, tw = tmpl_img.shape[:2]
                        sh, sw = single_card_gray.shape[:2]

                        if th <= sh and tw <= sw:
                            res = cv2.matchTemplate(single_card_gray, tmpl_img, cv2.TM_CCOEFF_NORMED)
                            _, max_val, _, _ = cv2.minMaxLoc(res)

                            if max_val > best_val:
                                best_val = max_val
                                detected_id = self._class_id_map[role_key]

                    if best_val > 0.5 and detected_id != -1:
                        # 给这个职业分配一个亮度值 (0最暗, 255最亮)，为了区分度，用 (id+1)*30
                        # ID 0 (先锋) = 30, ID 7 (特种) = 240
                        color_val = (detected_id + 1) * 30

                        # 在通道4中把这张卡的区域涂上对应颜色的色块，这就是喂给AI的“职业知识”！
                        cv2.rectangle(ch4_map, (card_x1, y_start), (card_x2, y_end), color_val, -1)


            # --- 组合为 4 通道 RL 输入 ---
            # 注意：cv2.merge 会返回一个多通道数组，我们可以自由组合
            stacked_img = cv2.merge([gray, grid_mask, fg_mask_clean, ch4_map])

            # 缩小尺寸，用于 RL 模型输入
            state = cv2.resize(stacked_img, (self.WIDTH, self.HEIGHT), interpolation=cv2.INTER_NEAREST)

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

    def _battle_monitor_worker(self):
        """
        后台高频监视线程：专职盯着右上角的齿轮和画面 MSE 剧变。
        一旦发现游戏结束（齿轮消失或剧变），立刻置位应急停止标志。
        """
        template_path = str(ROOT.parent / "data" / "templates" / "pause_gear.png")
        if not Path(template_path).exists() or self._template_matcher is None:
            return

        print("[ENV] 🛡️ 战斗全局异步监视器已启动...")
        missing_count = 0
        last_img = None
        
        while not self._stop_monitor_event.is_set():
            if not self._battle_active:
                time.sleep(0.5)
                continue
                
            # 获取画面，不阻塞主线程
            future = self._controller.post_screencap()
            if future is None:
                time.sleep(0.2)
                continue
                
            img = future.wait().get()
            if img is None:
                time.sleep(0.2)
                continue
                
            h, w = img.shape[:2]
            
            # 1. 画面剧变检测 (快速MSE)
            if last_img is not None:
                try:
                    gray_cur = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    gray_last = cv2.cvtColor(last_img, cv2.COLOR_BGR2GRAY)
                    mse = np.mean((gray_cur.astype("float") - gray_last.astype("float")) ** 2)
                    if mse > 8000.0:
                        print(f"\n[MONITOR] 🚨 后台紧急熔断：画面剧变 (MSE={mse:.2f} > 8000)！")
                        self._emergency_stop_flag = True
                        break
                except:
                    pass
            last_img = img

            # 2. 齿轮极速检测
            gear_roi = (int(w * 0.7), 0, w, int(h * 0.3))
            result = self._template_matcher.match(img, template_path, threshold=0.55, roi=gear_roi, silent=True)
            
            if result is None:
                missing_count += 1
                if missing_count >= 2:
                    print(f"\n[MONITOR] 🚨 后台紧急熔断：连续 {missing_count} 次未见右上角齿轮，战斗结束！")
                    self._emergency_stop_flag = True
                    break
            else:
                missing_count = 0
                
            time.sleep(0.3)  # 高频扫描 (约 3Hz)

    def _battle_monitor_worker(self):
        """
        后台高频监视线程：专职盯着右上角的齿轮和画面 MSE 剧变。
        一旦发现游戏结束（齿轮消失或剧变），立刻置位应急停止标志。
        """
        import time
        import cv2
        template_path = str(ROOT.parent / "data" / "templates" / "pause_gear.png")
        if not Path(template_path).exists() or getattr(self, '_template_matcher', None) is None:
            return

        print("[ENV] 🛡️ 战斗全局异步监视器已启动...")
        missing_count = 0
        last_img = None
        
        while not self._stop_monitor_event.is_set():
            if not self._battle_active:
                time.sleep(0.5)
                continue
                
            # 获取画面，不阻塞主线程
            future = getattr(self, '_controller', None).post_screencap()
            if future is None:
                time.sleep(0.2)
                continue
                
            img = future.wait().get()
            if img is None:
                time.sleep(0.2)
                continue
                
            h, w = img.shape[:2]
            
            # 1. 画面剧变检测 (快速MSE)
            if last_img is not None:
                try:
                    gray_cur = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    gray_last = cv2.cvtColor(last_img, cv2.COLOR_BGR2GRAY)
                    mse = np.mean((gray_cur.astype("float") - gray_last.astype("float")) ** 2)
                    if mse > 8000.0:
                        print(f"\n[MONITOR] 🚨 后台紧急熔断：画面剧变 (MSE={mse:.2f} > 8000)！")
                        self._emergency_stop_flag = True
                        break
                except:
                    pass
            last_img = img

            # 2. 齿轮极速检测
            gear_roi = (int(w * 0.7), 0, w, int(h * 0.3))
            result = self._template_matcher.match(img, template_path, threshold=0.55, roi=gear_roi, silent=True)
            
            if result is None:
                missing_count += 1
                if missing_count >= 2:
                    print(f"\n[MONITOR] 🚨 后台紧急熔断：连续 {missing_count} 次未见右上角齿轮，战斗结束！")
                    self._emergency_stop_flag = True
                    break
            else:
                missing_count = 0
                
            time.sleep(0.3)  # 高频扫描 (约 3Hz)

    def reset(self, seed: int = None, options: Dict[str, Any] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)

        # 关停上一局的监视器
        self._battle_active = False
        if getattr(self, '_monitor_thread', None) is not None:
            self._stop_monitor_event.set()
            self._monitor_thread.join(timeout=1.0)

        # 核心：如果是被 done=True 触发进来的 reset，执行全自动重启流程！
        if self.time_step > 0:
            self._auto_restart_battle()

        self.time_step = 0
        
        # 开启新一局的监视器
        self._emergency_stop_flag = False
        self._stop_monitor_event.clear()
        self._battle_active = True
        self._monitor_thread = __import__('threading').Thread(target=self._battle_monitor_worker, daemon=True)
        self._monitor_thread.start()

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
                    cx = (d.box_xyxy[0] + d.box_xyxy[2]) / 2
                    cy = (d.box_xyxy[1] + d.box_xyxy[3]) / 2
                    hp_bar_centers_before.append((cx, cy))

        # ========================================================
        # 终极形态：混合先验规则与视觉掩码 (Prior Masking & Blind Execution)
        # ========================================================
        card_idx = int(action[0])
        grid_x = int(action[1])
        grid_y = int(action[2])
        direction = int(action[3])

        executed_swipe = False
        target_gx, target_gy = -1, -1
        deployed_success = False
        hp_bar_centers_after = []

        # 获取目标网格的真实屏幕坐标 (这非常重要，后面判断血条有没有落在这个格子里要用)
        GRID_START_X, GRID_START_Y = 150, 150
        CELL_W, CELL_H = 100, 80
        target_gx = GRID_START_X + grid_x * CELL_W + CELL_W // 2
        target_gy = GRID_START_Y + grid_y * CELL_H + CELL_H // 2

        # ========================================================
        # 终极形态：混合先验规则与视觉掩码 (Prior Masking & Blind Execution)
        # 引入 OpenCV 亮度检测，前置拦截“没费”和“冷却”！
        # ========================================================
        card_is_playable = True
        if card_idx < 10 and img_before is not None:
            try:
                # 按照 test_live_brightness.py 调校好的比例截取并计算亮度
                h_raw, w_raw = img_before.shape[:2]

                y_start, y_end = int(h_raw * 0.75), int(h_raw * 0.98)
                x_start, x_end = int(w_raw * 0.13), int(w_raw * 0.97)

                num_cards = 12
                total_width = x_end - x_start
                slot_width = total_width / num_cards
                card_actual_width = slot_width * 0.80

                # 防御性代码：如果模型输出了 10/11，直接判定为不可用（10我们后面会单独处理为挂机，11直接拦截）
                if card_idx >= num_cards:
                    card_is_playable = False
                else:
                    slot_center_x = x_start + int((card_idx + 0.5) * slot_width)
                    card_x1 = int(slot_center_x - card_actual_width / 2)
                    card_x2 = int(slot_center_x + card_actual_width / 2)

                    card_x1 = max(0, card_x1)
                    card_x2 = min(w_raw, card_x2)

                    single_card_roi = img_before[y_start:y_end, card_x1:card_x2]
                    single_card_gray = cv2.cvtColor(single_card_roi, cv2.COLOR_BGR2GRAY)

                    avg_brightness = np.mean(single_card_gray)

                    # 亮度低于 90 认为手牌不可用
                    if avg_brightness <= 90:
                        card_is_playable = False

            except Exception as e:
                print(f"[DEBUG] 亮度检测失败: {e}")

        # 既然已经没有挂机选项了，我们直接移除对挂机的拦截处理。
        # 如果模型输出异常的索引（>= 12），前面的 card_is_playable 已经被设为 False 了，会在后面被拦截。

        # 【占用检测】通过 YOLO 提前判断格子上是否已经有血条了
        is_occupied = False

        for hp_cx, hp_cy in hp_bar_centers_before:
            # 血条通常在干员头顶，稍微把血条的Y坐标往下补一点(比如+30像素)来和格子中心对齐
            dist = np.sqrt((hp_cx - target_gx)**2 + (hp_cy + 30 - target_gy)**2)
            if dist < 70.0:  # 如果距离小于70像素，认为该格子已经被占用了
                is_occupied = True
                break

        if not card_is_playable:
            # 【亮度规则拦截】如果卡牌灰暗，直接拦截！
            print(f"[ACTION MASKING] ⛔ 警告：手牌 {card_idx} 亮度过低(无费/CD中)！代码直接拦截，不执行拖拽！")
            time.sleep(0.1) # 加速回合
            state_after, img_after, current_enemy_targets = self._get_state_and_raw_image()
        elif self.playable_grid[grid_y, grid_x] == 0:
            # 【地形先验拦截】如果丢进了黑洞，拦截！
            print(f"[ACTION MASKING] ⛔ 警告：目标网格({grid_x},{grid_y})不可部署区域！代码直接拦截！")
            time.sleep(0.1)
            state_after, img_after, current_enemy_targets = self._get_state_and_raw_image()
        elif is_occupied:
            # 【驻守拦截】格子已经有干员了！
            print(f"[ACTION MASKING] ⛔ 警告：目标网格({grid_x},{grid_y})已被友方干员驻守！代码直接拦截！")
            time.sleep(0.1)
            state_after, img_after, current_enemy_targets = self._get_state_and_raw_image()
        else:
            # 盲狙拖拽（不查绿光，直接拖）
            executed_swipe, target_gx, target_gy, direction = self._deploy_actions.execute_deployment_blind(action)

            print(f"[REWARD] 动作已执行，开始 3 次心跳检测血条 (间隔 0.2s)...")
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
                for cx_after, cy_after in current_hp_bars:
                    is_new = True
                    for cx_before, cy_before in hp_bar_centers_before:
                        if np.sqrt((cx_after - cx_before)**2 + (cy_after - cy_before)**2) < 50.0:
                            is_new = False; break
                    if is_new:
                        # 只有在确认为是“新血条”时，我们才去判断它是不是落在我们刚才扔的地方
                        # 这里非常关键：我们必须检查这个“新血条”是不是离我们 target_gx, target_gy 足够近
                        # 否则如果有别的干员（比如怪掉血了或者我们放了召唤物）产生了新血条，会被误判
                        dist_to_target = np.sqrt((cx_after - target_gx)**2 + (cy_after - target_gy)**2)
                        if dist_to_target < 250.0:
                            deployed_success = True; break
                if deployed_success:
                    print(f"[REWARD] ✅ 第 {attempt+1} 次检测命中目标干员血条！")
                    break

            if not deployed_success:
                print("[REWARD] ❌ 未检测到新血条！判定为【费用不足或CD中】，强制挂机 2.0 秒等待费用恢复！")
                time.sleep(2.0)
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
        if deployed_success:
            # 基础分：只要部署成功就给大分，并且针对职业特性给予额外加成
            reward = 1.0  # 奖励缩放：从 100.0 缩小到 1.0
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
                    reward += 0.5  # 从 50.0 缩小到 0.5
                    print(f"[REWARD] 🎯 完美落子！部署点靠近敌人 (距离: {dist:.1f})，奖励 +0.5")
                elif dist > 800.0: # 放到了完全打不到的地方
                    reward -= 0.2  # 从 -20.0 缩小到 -0.2
                    print(f"[REWARD] ⚠️ 无效部署！距离敌人太远 (距离: {dist:.1f})，惩罚 -0.2")

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
                    reward += 0.3  # 从 30.0 缩小到 0.3
                    print(f"[REWARD] ⚔️ 完美朝向！正对敌人 (角度: {angle:.1f}°)，额外奖励 +0.3")
                else:
                    reward -= 0.1  # 从 -10.0 缩小到 -0.1
                    print(f"[REWARD] 🛡️ 偏离/背对敌人！(角度: {angle:.1f}°)，微小惩罚 -0.1")
            else:
                # 场上暂时没敌人的情况 (静态兜底)
                print("[REWARD] 场上暂无移动目标，仅发放基础部署奖励 +1.0")

        elif not executed_swipe:
            # 【情况 2：被 Action Masking 直接阻断】
            # 必须给出明确的负反馈，否则 AI 会陷入零代价死循环
            reward = -0.05  # 从 -5.0 缩小到 -0.05
            self.consecutive_fast_fails += 1
            print(f"[REWARD] ⛔ 动作被拦截 (非法地形/没钱/有人)。惩罚 -0.05 (连续无作为: {self.consecutive_fast_fails}次)")
        elif not deployed_success:
            # 【情况 3：游戏拒绝了部署 (高低台放错) 或 模拟器手滑】
            # 极大概率是高低台干员类型不匹配导致游戏拒收！必须惩罚，逼迫它学习 Channel 4 和 Channel 2 的对应关系
            reward = -0.05  # 从 -5.0 缩小到 -0.05
            self.consecutive_fast_fails += 1
            print(f"[REWARD] ❌ 未产生血条 (游戏拒收/高低台错位/手滑)。惩罚 -0.05")
        else:
            print(f"[REWARD] 部署失败，计算前后帧 MSE (均方误差) = {mse:.2f}")
            reward = -0.02  # 从 -2.0 缩小到 -0.02
            self.consecutive_fast_fails += 1
            print(f"[REWARD] 画面未发生明显变化。惩罚 -0.02 (连续无作为: {self.consecutive_fast_fails}次)")

        # 7. 多重战斗结束判定 (Priority 1 -> 2 -> 3)

        # 异步监视器拦截
        if getattr(self, '_emergency_stop_flag', False):
            print(f"[ENV] 🚨 后台监视器触发熔断，本回合正式结束！")
            terminated = True
            self._battle_active = False

        # 强制截断保护
        if self.time_step >= 50:
             truncated = True
             print(f"[ENV] 达到最大步数 (50)，终止回合。")

        info = {
            "hp_bars": len(hp_bar_centers_after),
            "reward_given": reward
        }

        return state_after, reward, terminated, truncated, info

from __future__ import annotations

import sys
import time
import cv2
import numpy as np
from pathlib import Path

class DeployOperatorActions:
    """
    负责将 RL 模型的离散网格动作，映射为 MAA 控制器的底层坐标操作 (点击/滑动)
    """

    def __init__(self, controller: Any) -> None:
        """
        初始化部署执行器

        Args:
            controller: MaaFramework控制器
        """
        self._controller = controller

        # 屏幕分辨率假设为标准的 1280x720
        # 手牌区 (10个干员位) 的坐标范围
        self.CARD_START_X = 250
        self.CARD_END_X = 1150
        self.CARD_Y = 650

        # 战斗网格区 (10列 x 5行) 的坐标范围
        self.GRID_START_X = 150
        self.GRID_END_X = 1150
        self.GRID_START_Y = 150
        self.GRID_END_Y = 550

        self.GRID_COLS = 10
        self.GRID_ROWS = 5

        # 计算每个格子的宽高
        self.CELL_W = (self.GRID_END_X - self.GRID_START_X) // self.GRID_COLS
        self.CELL_H = (self.GRID_END_Y - self.GRID_START_Y) // self.GRID_ROWS

    def execute_deployment_blind(self, action: np.ndarray):
        card_idx = int(action[0])
        grid_x = int(action[1])
        grid_y = int(action[2])
        direction = int(action[3])

        # 获取干员卡片的中心点坐标
        card_step = (self.CARD_END_X - self.CARD_START_X) // 12
        cx = self.CARD_START_X + card_idx * card_step + card_step // 2
        cy = self.CARD_Y

        gx = self.GRID_START_X + grid_x * self.CELL_W + self.CELL_W // 2
        gy = self.GRID_START_Y + grid_y * self.CELL_H + self.CELL_H // 2

        swipe_offset = 150
        end_x, end_y = gx, gy
        if direction == 0: end_y -= swipe_offset
        elif direction == 1: end_y += swipe_offset
        elif direction == 2: end_x -= swipe_offset
        elif direction == 3: end_x += swipe_offset

        print(f"[ACTION] 盲狙执行: 选卡{card_idx} -> 拖拽至({grid_x},{grid_y}) 朝向{direction}")

        try:
            # 还原最原汁原味的操作：
            # 1. 拖动干员到目标网格并松手（此操作在方舟中如果成功，会立刻进入一个悬停等待方向的子弹时间）
            # 注意：此处 swipe 时间调慢一些，确保模拟器能稳稳抓住卡牌拖到目标位。
            # 而且由于有些网格非常远（比如最上排的 y=0），滑动时间不能太短，也不能太长导致判定为停止。
            # 我们给第一段一个恒定的 1200ms 的沉稳拖拽。
            self._controller.post_swipe(cx, cy, gx, gy, duration=1200).wait()

            # 给游戏时间弹出那个蓝色的方向选择圈！
            # 这个停顿时间非常关键，必须等拖拽动画完全结束且子弹时间界面出来！
            # 我把这个停顿拉长到了 2.0 秒，确保哪怕模拟器掉帧也能稳稳停在那里。
            time.sleep(0.5)

            # 2. 从目标网格滑动出方向来确认部署
            # 这也是一次独立的滑动
            swipe_offset = 150
            end_x, end_y = gx, gy
            if direction == 0: end_y -= swipe_offset    # 上
            elif direction == 1: end_y += swipe_offset  # 下
            elif direction == 2: end_x -= swipe_offset  # 左
            elif direction == 3: end_x += swipe_offset  # 右

            self._controller.post_swipe(gx, gy, end_x, end_y, duration=800).wait()

            # 拖拽后稍微等一下动画落地
            time.sleep(1.5)

            return True, gx, gy, direction
        except Exception as e:
            print(f"[ACTION ERROR] 盲狙失败: {e}")
            return False, gx, gy, direction
        """
        Phase 1: 选卡阶段。
        只负责点击干员卡牌（或者挂机）。点击后游戏进入子弹时间，显示特定绿光。
        返回 False 代表选择挂机（不需要执行后续 Phase 2）。
        """
        card_idx = int(action[0])
        if card_idx == 10:
            print("[ACTION] AI决定: 💤 挂机 (Skip / Wait for Cost) - 屏幕中上部防误触点击")
            self._controller.post_click(self.CARD_START_X + (self.CARD_END_X - self.CARD_START_X) // 2, 50).wait()
            time.sleep(2.0)
            return False

        card_step = (self.CARD_END_X - self.CARD_START_X) // 10
        cx = self.CARD_START_X + card_idx * card_step + card_step // 2
        cy = self.CARD_Y

        print(f"[ACTION Phase-1] AI决定: 选卡 {card_idx}，点击坐标({cx},{cy})")
        self._controller.post_click(cx, cy).wait()

        # 核心等待：这里是从点击卡片到绿光出现的唯一渲染时间窗口！
        # 如果你看到一片漆黑，说明这里等得不够久！
        # 我们把 0.8 延长到 2.5 秒，强制等绿光完全亮起。
        time.sleep(2.5)
        return True

    def execute_deployment_phase_2(self, action: np.ndarray, last_action: np.ndarray) -> Tuple[bool, int, int, int]:
        """
        Phase 2: 拖拽阶段。
        AI 看着带有绿光的新画面，决定把刚刚选中的卡放到哪里。
        """
        card_idx = int(last_action[0]) # 从上一个动作获取卡片索引
        grid_x = int(action[1])
        grid_y = int(action[2])
        direction = int(action[3])

        # 重新计算坐标
        card_step = (self.CARD_END_X - self.CARD_START_X) // 10
        cx = self.CARD_START_X + card_idx * card_step + card_step // 2
        cy = self.CARD_Y

        gx = self.GRID_START_X + grid_x * self.CELL_W + self.CELL_W // 2
        gy = self.GRID_START_Y + grid_y * self.CELL_H + self.CELL_H // 2

        swipe_offset = 150
        end_x, end_y = gx, gy
        if direction == 0: end_y -= swipe_offset
        elif direction == 1: end_y += swipe_offset
        elif direction == 2: end_x -= swipe_offset
        elif direction == 3: end_x += swipe_offset

        print(f"[ACTION Phase-2] AI决定: 放入网格({grid_x},{grid_y}) -> 朝向 {direction}")

        try:
            # --- 【CV 物理外挂：Fast-Fail 快速阻断机制】 ---
            image = self._controller.post_screencap().wait().get()
            if image is not None:
                hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                # 【HSV 微调优化】：针对明日方舟浅绿色地砖特效优化，避开星熊等深青绿色的干扰
                # 色相(H)卡得更死在黄绿~浅绿之间 (40~80)
                # 饱和度(S)卡在偏低~中等之间 (40~160)，因为特效偏白发亮，不是高饱和的纯绿
                # 亮度(V)卡在偏高，滤掉环境暗色
                lower_green = np.array([40, 40, 100])
                upper_green = np.array([80, 160, 255])
                mask = cv2.inRange(hsv, lower_green, upper_green)

                y1, y2 = max(0, gy-15), min(mask.shape[0], gy+15)
                x1, x2 = max(0, gx-15), min(mask.shape[1], gx+15)
                roi = mask[y1:y2, x1:x2]
                green_pixels = cv2.countNonZero(roi)

                # --- 增加极其直观的 DEBUG 窗口，把我们检测的格子画面展示出来 ---
                # 用同一个名字的窗口持续更新，不会消失，方便你一直看
                debug_vis = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
                debug_vis = cv2.resize(debug_vis, (300, 300), interpolation=cv2.INTER_NEAREST)

                # 画一个红色的准星，表示我们检测的正中心
                cv2.line(debug_vis, (150, 0), (150, 300), (0, 0, 255), 2)
                cv2.line(debug_vis, (0, 150), (300, 150), (0, 0, 255), 2)

                # 把当前的绿像素数量写在图片上
                cv2.putText(debug_vis, f"Green: {green_pixels} (Needs 20)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                cv2.imshow("Green Detection ROI", debug_vis)
                cv2.waitKey(1)  # 只刷新，不阻塞游戏进程
                # -------------------------------------------------------------

                if green_pixels < 20:
                    print(f"[ACTION] ⛔ Fast-Fail! 目标网格({grid_x},{grid_y})非绿色高亮(绿像素:{green_pixels})。取消拖拽。")
                    self._controller.post_click(cx, cy).wait() # 取消选中
                    return False, gx, gy, direction
                else:
                    print(f"[ACTION] ✅ CV校验通过！具有高亮绿色(像素:{green_pixels})，准许拖拽。")

            # 拖拽
            self._controller.post_swipe(cx, cy, gx, gy, duration=500).wait()
            time.sleep(0.5)
            self._controller.post_swipe(gx, gy, end_x, end_y, duration=300).wait()

            return True, gx, gy, direction

        except Exception as e:
            print(f"[ACTION ERROR] 动作执行异常: {e}")
            return False, gx, gy, direction

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

    def execute_deployment(self, action: np.ndarray) -> Tuple[bool, int, int]:
        """
        执行部署动作
        action: [card_idx, grid_x, grid_y, direction]

        Returns:
            (是否执行滑动, 目标像素x, 目标像素y)
        """
        card_idx = int(action[0])    # 0~9
        grid_x = int(action[1])      # 0~9
        grid_y = int(action[2])      # 0~4
        direction = int(action[3])   # 0:上, 1:下, 2:左, 3:右

        # 1. 计算手牌干员坐标 (平均分布)
        card_step = (self.CARD_END_X - self.CARD_START_X) // 10
        cx = self.CARD_START_X + card_idx * card_step + card_step // 2
        cy = self.CARD_Y

        # 2. 计算目标网格的中心坐标
        gx = self.GRID_START_X + grid_x * self.CELL_W + self.CELL_W // 2
        gy = self.GRID_START_Y + grid_y * self.CELL_H + self.CELL_H // 2

        # 3. 计算划定朝向的终点坐标
        swipe_offset = 150
        end_x, end_y = gx, gy
        if direction == 0:   # 上
            end_y -= swipe_offset
        elif direction == 1: # 下
            end_y += swipe_offset
        elif direction == 2: # 左
            end_x -= swipe_offset
        elif direction == 3: # 右
            end_x += swipe_offset

        print(f"[ACTION] AI决定: 选卡 {card_idx} -> 放入网格({grid_x},{grid_y}) -> 朝向 {direction}")
        print(f"         坐标映射: 拖拽从({cx},{cy})至({gx},{gy}) -> 方向滑动至({end_x},{end_y})")

        # === 执行 MAA 指令 ===
        try:
            # 步骤A：点击底部的干员卡牌（进入待部署状态）
            self._controller.post_click(cx, cy).wait()

            # 等待极短时间让游戏出现地形高亮 (子弹时间)
            time.sleep(0.4)

            # --- 【CV 物理外挂：Fast-Fail 快速阻断机制】 ---
            # 获取当前屏幕
            image = self._controller.post_screencap().wait().get()
            if image is not None:
                # 转换到 HSV 颜色空间寻找绿色高亮
                hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                # 明日方舟部署时的绿色高亮大概在这个 HSV 范围内
                lower_green = np.array([35, 43, 46])
                upper_green = np.array([90, 255, 255])
                mask = cv2.inRange(hsv, lower_green, upper_green)

                # 切割出目标网格 (gx, gy) 附近的一个小区域 (30x30像素)
                y1, y2 = max(0, gy-15), min(mask.shape[0], gy+15)
                x1, x2 = max(0, gx-15), min(mask.shape[1], gx+15)
                roi = mask[y1:y2, x1:x2]

                # 统计这块区域里的绿色像素数量
                green_pixels = cv2.countNonZero(roi)

                if green_pixels < 20:
                    # 如果几乎没有绿色，说明：
                    # 1. 根本没点中干员（或者干员在CD/费用不够）
                    # 2. 目标网格是不可部署的黑洞/高台受限
                    print(f"[ACTION] ⛔ Fast-Fail! 目标网格({grid_x},{grid_y})非绿色高亮(绿像素:{green_pixels})。取消拖拽。")
                    # 取消选中状态的最佳方法：重新点击一次刚才选中的那张底部的卡牌
                    self._controller.post_click(cx, cy).wait()
                    return False, gx, gy
                else:
                    print(f"[ACTION] ✅ CV校验通过！网格({grid_x},{grid_y})具有高亮绿色(像素:{green_pixels})，准许拖拽。")

            # 步骤B：将底部干员卡牌拖拽到目标网格（释放后进入慢动作方向选择阶段）
            self._controller.post_swipe(cx, cy, gx, gy, duration=500).wait()

            # 等待极短时间让游戏出现方向选择的慢动作 UI
            time.sleep(0.5)

            # 步骤C：在目标网格上按住，并划向指定方向松手（完成部署）
            self._controller.post_swipe(gx, gy, end_x, end_y, duration=300).wait()

            return True, gx, gy

        except Exception as e:
            print(f"[ACTION ERROR] 动作执行异常: {e}")
            return False, gx, gy

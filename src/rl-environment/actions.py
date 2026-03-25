from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any
import numpy as np

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

    def execute_deployment(self, action: np.ndarray) -> None:
        """
        执行部署动作
        action: [card_idx, grid_x, grid_y, direction]
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
            # 步骤A：将底部干员卡牌拖拽到目标网格（释放后进入慢动作方向选择阶段）
            self._controller.post_swipe(cx, cy, gx, gy, duration=500).wait()

            # 等待极短时间让游戏出现方向选择的慢动作 UI
            time.sleep(0.5)

            # 步骤B：在目标网格上按住，并划向指定方向松手（完成部署）
            self._controller.post_swipe(gx, gy, end_x, end_y, duration=300).wait()

        except Exception as e:
            print(f"[ACTION ERROR] 动作执行异常: {e}")

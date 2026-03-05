from __future__ import annotations

import numpy as np
from gymnasium import Env
from gymnasium.spaces import Box, Discrete
from typing import Any, Dict, Tuple

from src.ai_plugins.yolo_recognizer import YoloRecognizer
from src.rl_environment.actions import DeployOperatorActions


class GameEnv(Env):
    """
    部署干员的RL环境
    
    实现Gymnasium接口，用于训练RL模型
    """
    
    def __init__(self, controller: Any, yolo_recognizer: YoloRecognizer) -> None:
        """
        初始化RL环境
        
        Args:
            controller: MaaFramework控制器
            yolo_recognizer: YOLO识别器
        """
        super().__init__()
        
        self._controller = controller
        self._yolo_recognizer = yolo_recognizer
        
        # 创建部署干员动作
        self._deploy_actions = DeployOperatorActions(controller, yolo_recognizer)
        
        # 定义状态空间（observation space）
        # - 0: 是否成功点击干员头像（0/1）
        # - 1: 是否成功拖拽到放置区域（0/1）
        # - 2: 是否成功部署（0/1）
        # - 3: 当前部署干员的进展（0-3）
        # - 4: 当前时间步（0-∞）
        self.observation_space = Box(
            low=np.array([0, 0, 0, 0, 0], dtype=np.float32),
            high=np.array([1, 1, 1, 3, np.inf], dtype=np.float32),
            dtype=np.float32
        )
        
        # 定义动作空间（action space）
        # - 0: 点击干员头像
        # - 1: 拖拽到放置区域
        # - 2: 调整方向
        # - 3: 松手完成部署
        self.action_space = Discrete(4)
        
        # 初始化状态
        self.state = np.zeros(5, dtype=np.float32)
        self.time_step = 0
        
        # 记录部署干员的进展
        self._deployment_progress = 0  # 0: 未开始, 1: 已点击干员头像, 2: 已拖拽到放置区域, 3: 已成功部署
        self._action1_success = False  # 是否成功点击干员头像
        self._action2_success = False  # 是否成功拖拽到放置区域
        self._action3_direction = None  # 动作3选择的方向
        self._action4_success = False  # 是否成功部署
        
        # 记录位置
        self._operator_position = None  # 干员位置
        self._deployment_position = None  # 放置区域中心点
    
    def reset(self, seed: int = None, options: Dict[str, Any] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        重置环境
        
        Returns:
            observation: 初始状态
            info: 额外信息
        """
        super().reset(seed=seed)
        
        # 重置状态
        self.state = np.zeros(5, dtype=np.float32)
        self.time_step = 0
        
        # 重置部署干员的进展
        self._deployment_progress = 0
        self._action1_success = False
        self._action2_success = False
        self._action3_direction = None
        self._action4_success = False
        
        return self.state, {}
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        执行动作
        
        Args:
            action: 动作（0-3）
                0: 点击干员头像
                1: 拖拽到放置区域
                2: 调整方向
                3: 松手完成部署
        
        Returns:
            observation: 新状态
            reward: 奖励
            terminated: 是否终止
            truncated: 是否截断
            info: 额外信息
        """
        self.time_step += 1
        self.state[4] = self.time_step
        
        reward = 0.0
        terminated = False
        truncated = False
        info = {}
        
        # 执行动作
        if action == 0:  # 点击干员头像
            success, position = self._deploy_actions.action1_click_operator_avatar()
            self._action1_success = success
            self.state[0] = 1.0 if success else 0.0
            
            if success:
                # 成功点击干员头像，更新部署进展
                self._deployment_progress = 1
                self.state[3] = 1.0
                reward = 1.0
                # 记录干员位置
                self._operator_position = position
            else:
                # 失败点击干员头像
                reward = -0.1
        
        elif action == 1:  # 拖拽到放置区域
            if self._deployment_progress < 1:
                # 还未点击干员头像，不能拖拽
                reward = -0.1
            else:
                # 获取干员位置
                operator_position = self._get_operator_position()
                if operator_position:
                    success, position = self._deploy_actions.action2_drag_to_deployment_area(operator_position)
                    self._action2_success = success
                    self.state[1] = 1.0 if success else 0.0
                    
                    if success:
                        # 成功拖拽到放置区域，更新部署进展
                        self._deployment_progress = 2
                        self.state[3] = 2.0
                        reward = 1.0
                        # 记录放置区域中心点
                        self._deployment_position = position
                    else:
                        # 失败拖拽到放置区域
                        reward = -0.1
                else:
                    # 未获取到干员位置
                    reward = -0.1
        
        elif action == 2:  # 调整方向
            if self._deployment_progress < 2:
                # 还未拖拽到放置区域，不能调整方向
                reward = -0.1
            else:
                # 获取放置区域中心点
                deployment_position = self._get_deployment_position()
                if deployment_position:
                    direction, _ = self._deploy_actions.action3_adjust_direction(deployment_position)
                    self._action3_direction = direction
                    # 调整方向不影响奖励
                    reward = 0.0
                else:
                    # 未获取到放置区域中心点
                    reward = -0.1
        
        elif action == 3:  # 松手完成部署
            if self._deployment_progress < 2:
                # 还未拖拽到放置区域，不能松手完成部署
                reward = -0.1
            else:
                # 获取放置区域中心点
                deployment_position = self._get_deployment_position()
                if deployment_position:
                    success = self._deploy_actions.action4_release_to_deploy(deployment_position)
                    self._action4_success = success
                    self.state[2] = 1.0 if success else 0.0
                    
                    if success:
                        # 成功部署，更新部署进展
                        self._deployment_progress = 3
                        self.state[3] = 3.0
                        reward = 1.0
                        terminated = True  # 成功部署，终止
                    else:
                        # 失败部署
                        reward = -0.1
                else:
                    # 未获取到放置区域中心点
                    reward = -0.1
        
        else:
            # 未知动作
            reward = -0.1
        
        # 更新info
        info = {
            "deployment_progress": self._deployment_progress,
            "action1_success": self._action1_success,
            "action2_success": self._action2_success,
            "action3_direction": self._action3_direction,
            "action4_success": self._action4_success,
            "time_step": self.time_step
        }
        
        # 判断是否截断（超过最大时间步）
        max_time_steps = 100
        if self.time_step >= max_time_steps:
            truncated = True
        
        return self.state, reward, terminated, truncated, info
    
    def _get_operator_position(self) -> Tuple[int, int] | None:
        """
        获取干员位置
        
        Returns:
            (x, y): 干员位置，如果未获取到则返回None
        """
        return self._operator_position
    
    def _get_deployment_position(self) -> Tuple[int, int] | None:
        """
        获取放置区域中心点
        
        Returns:
            (x, y): 放置区域中心点，如果未获取到则返回None
        """
        return self._deployment_position

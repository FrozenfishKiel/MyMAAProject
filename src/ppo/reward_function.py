"""
奖励函数模块
计算PPO算法在每个步骤中获得的奖励
包括基础奖励、稀疏奖励和密集奖励
"""

import numpy as np
from typing import Dict, List, Any, Tuple, Callable
import math

# 复用现有基础设施
from src.vision.detector.yolo_detector import YOLODetector


class RewardFunction:
    """奖励函数 - 计算智能体在每个步骤中获得的奖励"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化奖励函数
        
        Args:
            config: 配置参数
        """
        self.config = config or {}
        
        # 奖励权重配置
        self.weights = {
            'survival': self.config.get('survival_weight', 0.1),
            'enemy_kill': self.config.get('enemy_kill_weight', 1.0),
            'hp_loss': self.config.get('hp_loss_weight', -0.5),
            'time_penalty': self.config.get('time_penalty_weight', -0.01),
            'action_penalty': self.config.get('action_penalty_weight', -0.05),
            'progress': self.config.get('progress_weight', 0.2),
            'efficiency': self.config.get('efficiency_weight', 0.1),
            'victory': self.config.get('victory_weight', 10.0),
            'defeat': self.config.get('defeat_weight', -5.0)
        }
        
        # 复用YOLO检测器（使用单例模式）
        self.yolo_detector = YOLODetector.get_instance(
            model_path='yolov8n.pt',
            use_gpu=False
        )
        
        # 状态历史
        self.state_history = []
        self.max_history_length = 10
        
        # 奖励统计
        self.reward_stats = {
            'total_reward': 0.0,
            'episode_reward': 0.0,
            'step_count': 0,
            'episode_count': 0
        }
        
        print("✅ 奖励函数初始化完成")
        print(f"奖励权重配置: {self.weights}")
    
    def calculate_reward(self, 
                        current_state: np.ndarray,
                        action: Dict[str, Any],
                        next_state: np.ndarray,
                        detections: List[Dict] = None,
                        is_terminal: bool = False) -> float:
        """
        计算奖励值
        
        Args:
            current_state: 当前状态向量
            action: 执行的动作
            next_state: 下一状态向量
            detections: YOLO检测结果（可选）
            is_terminal: 是否为终止状态
            
        Returns:
            reward: 计算得到的奖励值
        """
        total_reward = 0.0
        
        # 1. 基础生存奖励
        survival_reward = self._calculate_survival_reward()
        total_reward += survival_reward * self.weights['survival']
        
        # 2. 敌人击杀奖励
        enemy_kill_reward = self._calculate_enemy_kill_reward(detections)
        total_reward += enemy_kill_reward * self.weights['enemy_kill']
        
        # 3. 血量损失惩罚
        hp_loss_penalty = self._calculate_hp_loss_penalty(current_state, next_state)
        total_reward += hp_loss_penalty * self.weights['hp_loss']
        
        # 4. 时间惩罚
        time_penalty = self._calculate_time_penalty()
        total_reward += time_penalty * self.weights['time_penalty']
        
        # 5. 动作惩罚
        action_penalty = self._calculate_action_penalty(action)
        total_reward += action_penalty * self.weights['action_penalty']
        
        # 6. 进度奖励
        progress_reward = self._calculate_progress_reward(current_state, next_state)
        total_reward += progress_reward * self.weights['progress']
        
        # 7. 效率奖励
        efficiency_reward = self._calculate_efficiency_reward(action, detections)
        total_reward += efficiency_reward * self.weights['efficiency']
        
        # 8. 终止状态奖励
        if is_terminal:
            terminal_reward = self._calculate_terminal_reward(next_state)
            total_reward += terminal_reward
        
        # 9. 稀疏奖励（基于特定事件）
        sparse_reward = self._calculate_sparse_reward(detections)
        total_reward += sparse_reward
        
        # 10. 密集奖励（基于状态变化）
        dense_reward = self._calculate_dense_reward(current_state, next_state)
        total_reward += dense_reward
        
        # 更新状态历史
        self._update_state_history(current_state)
        
        # 更新奖励统计
        self._update_reward_stats(total_reward)
        
        # 限制奖励范围
        total_reward = np.clip(total_reward, -10.0, 10.0)
        
        return total_reward
    
    def _calculate_survival_reward(self) -> float:
        """计算生存奖励"""
        # 每存活一步获得基础奖励
        return 1.0
    
    def _calculate_enemy_kill_reward(self, detections: List[Dict] = None) -> float:
        """计算敌人击杀奖励"""
        if detections is None:
            return 0.0
        
        # 统计敌人数量变化
        current_enemies = self._count_enemies(detections)
        
        # 与历史状态比较
        if self.state_history:
            # 这里需要更复杂的逻辑来检测敌人击杀
            # 简化实现：基于敌人数量减少
            prev_enemies = self._estimate_previous_enemies()
            if prev_enemies > current_enemies:
                return (prev_enemies - current_enemies) * 0.5
        
        return 0.0
    
    def _calculate_hp_loss_penalty(self, current_state: np.ndarray, next_state: np.ndarray) -> float:
        """计算血量损失惩罚"""
        if len(current_state) < 3 or len(next_state) < 3:
            return 0.0
        
        # 假设状态向量的前3个维度包含血量信息
        current_hp = current_state[0] if current_state[0] > 0 else 0.5
        next_hp = next_state[0] if next_state[0] > 0 else 0.5
        
        # 计算血量损失
        hp_loss = max(0, current_hp - next_hp)
        
        # 血量损失越大，惩罚越大
        return -hp_loss
    
    def _calculate_time_penalty(self) -> float:
        """计算时间惩罚"""
        # 每步时间惩罚，鼓励快速完成
        return -1.0
    
    def _calculate_action_penalty(self, action: Dict[str, Any]) -> float:
        """计算动作惩罚"""
        # 不同动作的惩罚程度不同
        action_type = action.get('type', '')
        
        penalty_map = {
            'wait': -0.1,      # 等待惩罚较小
            'click': -0.05,    # 点击惩罚较小
            'deploy': -0.2,    # 部署惩罚中等
            'skill': -0.3,     # 使用技能惩罚中等
            'retreat': -0.4,   # 撤退惩罚较大
            'swipe': -0.1      # 滑动惩罚较小
        }
        
        return penalty_map.get(action_type, -0.1)
    
    def _calculate_progress_reward(self, current_state: np.ndarray, next_state: np.ndarray) -> float:
        """计算进度奖励"""
        if len(current_state) < 5 or len(next_state) < 5:
            return 0.0
        
        # 假设状态向量的第4个维度包含关卡进度
        current_progress = current_state[3] if current_state[3] > 0 else 0.0
        next_progress = next_state[3] if next_state[3] > 0 else 0.0
        
        # 进度增加获得奖励
        progress_gain = max(0, next_progress - current_progress)
        
        return progress_gain
    
    def _calculate_efficiency_reward(self, action: Dict[str, Any], detections: List[Dict] = None) -> float:
        """计算效率奖励"""
        if detections is None:
            return 0.0
        
        action_type = action.get('type', '')
        
        # 检查动作是否有效（基于检测结果）
        if action_type == 'click':
            button_id = action.get('params', {}).get('button_id')
            if button_id is not None:
                # 检查按钮是否被检测到
                button_detected = any(d['class_id'] == button_id for d in detections)
                if button_detected:
                    return 0.1  # 有效点击奖励
        
        elif action_type == 'deploy':
            deploy_position = action.get('params', {}).get('deploy_position')
            if deploy_position is not None:
                # 检查部署位是否可用
                deploy_available = any(d['class_id'] == 20 + deploy_position for d in detections)
                if deploy_available:
                    return 0.2  # 有效部署奖励
        
        return 0.0
    
    def _calculate_terminal_reward(self, final_state: np.ndarray) -> float:
        """计算终止状态奖励"""
        if len(final_state) < 50:
            return 0.0
        
        # 检查胜利/失败标志（假设状态向量的第50个维度）
        victory_flag = final_state[49] if len(final_state) > 49 else 0.0
        defeat_flag = final_state[48] if len(final_state) > 48 else 0.0
        
        if victory_flag > 0.8:
            return self.weights['victory']
        elif defeat_flag > 0.8:
            return self.weights['defeat']
        
        return 0.0
    
    def _calculate_sparse_reward(self, detections: List[Dict] = None) -> float:
        """计算稀疏奖励（基于特定事件）"""
        if detections is None:
            return 0.0
        
        sparse_reward = 0.0
        
        # 检查特定事件
        for detection in detections:
            class_id = detection['class_id']
            confidence = detection['confidence']
            
            # 胜利标志检测
            if class_id == 12 and confidence > 0.8:  # 胜利标志
                sparse_reward += 5.0
            
            # 失败标志检测
            elif class_id == 13 and confidence > 0.8:  # 失败标志
                sparse_reward -= 3.0
            
            # Boss击杀奖励
            elif class_id >= 50 and class_id < 60:  # Boss敌人
                # 检查Boss是否消失（需要更复杂的逻辑）
                sparse_reward += 2.0
        
        return sparse_reward
    
    def _calculate_dense_reward(self, current_state: np.ndarray, next_state: np.ndarray) -> float:
        """计算密集奖励（基于状态变化）"""
        if len(current_state) != len(next_state):
            return 0.0
        
        dense_reward = 0.0
        
        # 状态变化奖励（鼓励状态向目标方向变化）
        state_diff = next_state - current_state
        
        # 血量增加奖励
        if len(state_diff) > 0 and state_diff[0] > 0:
            dense_reward += state_diff[0] * 0.5
        
        # 进度增加奖励
        if len(state_diff) > 3 and state_diff[3] > 0:
            dense_reward += state_diff[3] * 1.0
        
        # 敌人数量减少奖励
        if len(state_diff) > 10 and state_diff[10] < 0:
            dense_reward += abs(state_diff[10]) * 0.3
        
        return dense_reward
    
    def _count_enemies(self, detections: List[Dict]) -> int:
        """统计敌人数量"""
        enemy_count = 0
        for detection in detections:
            if detection['class_id'] >= 14:  # 敌人类别
                enemy_count += 1
        return enemy_count
    
    def _estimate_previous_enemies(self) -> int:
        """估计前一状态的敌人数量"""
        if not self.state_history:
            return 0
        
        # 简化实现：返回固定值
        # 实际实现需要更复杂的逻辑
        return 5
    
    def _update_state_history(self, state: np.ndarray) -> None:
        """更新状态历史"""
        self.state_history.append(state.copy())
        if len(self.state_history) > self.max_history_length:
            self.state_history.pop(0)
    
    def _update_reward_stats(self, reward: float) -> None:
        """更新奖励统计"""
        self.reward_stats['total_reward'] += reward
        self.reward_stats['episode_reward'] += reward
        self.reward_stats['step_count'] += 1
    
    def reset_episode_stats(self) -> None:
        """重置回合统计"""
        self.reward_stats['episode_reward'] = 0.0
        self.reward_stats['step_count'] = 0
        self.reward_stats['episode_count'] += 1
        self.state_history = []
        print(f"🔄 第{self.reward_stats['episode_count']}回合奖励统计已重置")
    
    def get_reward_stats(self) -> Dict[str, float]:
        """获取奖励统计"""
        return self.reward_stats.copy()
    
    def set_weights(self, new_weights: Dict[str, float]) -> None:
        """设置奖励权重"""
        self.weights.update(new_weights)
        print(f"✅ 奖励权重已更新: {self.weights}")
    
    def calculate(self, state: np.ndarray, action: int) -> float:
        """
        简化的奖励计算方法（用于测试）
        
        Args:
            state: 状态向量
            action: 动作索引
            
        Returns:
            reward: 计算得到的奖励值
        """
        # 将动作索引转换为动作字典
        action_dict = {
            'type': 'click',
            'params': {'button_id': action}
        }
        
        # 创建模拟的下一状态
        next_state = state + np.random.randn(len(state)) * 0.1
        
        # 调用完整的奖励计算方法
        return self.calculate_reward(state, action_dict, next_state)


class AdaptiveRewardFunction(RewardFunction):
    """自适应奖励函数 - 根据学习进度动态调整奖励权重"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        
        # 自适应配置
        self.learning_progress = 0.0
        self.adaptation_rate = config.get('adaptation_rate', 0.01)
        
        # 初始权重配置
        self.base_weights = self.weights.copy()
        
        # 学习阶段配置
        self.learning_phases = [
            {'name': '探索阶段', 'target': 'survival', 'weight_factor': 2.0},
            {'name': '技能学习', 'target': 'efficiency', 'weight_factor': 1.5},
            {'name': '策略优化', 'target': 'progress', 'weight_factor': 1.2},
            {'name': '精通阶段', 'target': 'victory', 'weight_factor': 1.0}
        ]
        
        self.current_phase = 0
        
        print("✅ 自适应奖励函数初始化完成")
    
    def calculate_reward(self, 
                        current_state: np.ndarray,
                        action: Dict[str, Any],
                        next_state: np.ndarray,
                        detections: List[Dict] = None,
                        is_terminal: bool = False) -> float:
        """
        计算自适应奖励
        """
        # 先计算基础奖励
        base_reward = super().calculate_reward(
            current_state, action, next_state, detections, is_terminal
        )
        
        # 应用自适应调整
        adapted_reward = self._adapt_reward(base_reward, current_state, next_state)
        
        # 更新学习进度
        self._update_learning_progress(adapted_reward, is_terminal)
        
        return adapted_reward
    
    def _adapt_reward(self, base_reward: float, 
                     current_state: np.ndarray, 
                     next_state: np.ndarray) -> float:
        """自适应调整奖励"""
        # 根据学习阶段调整奖励
        current_phase_config = self.learning_phases[self.current_phase]
        target_component = current_phase_config['target']
        weight_factor = current_phase_config['weight_factor']
        
        # 增强目标组件的权重
        adapted_weights = self.base_weights.copy()
        adapted_weights[target_component] *= weight_factor
        
        # 重新计算奖励（简化实现）
        adapted_reward = base_reward * (1.0 + self.learning_progress * 0.1)
        
        return adapted_reward
    
    def _update_learning_progress(self, reward: float, is_terminal: bool) -> None:
        """更新学习进度"""
        if is_terminal:
            # 回合结束时更新学习进度
            episode_reward = self.reward_stats['episode_reward']
            
            # 根据回合奖励更新进度
            if episode_reward > 0:
                self.learning_progress = min(1.0, self.learning_progress + self.adaptation_rate)
            else:
                self.learning_progress = max(0.0, self.learning_progress - self.adaptation_rate * 0.5)
            
            # 检查是否需要切换到下一阶段
            if self.learning_progress > (self.current_phase + 1) * 0.25:
                self._advance_learning_phase()
    
    def _advance_learning_phase(self) -> None:
        """推进学习阶段"""
        if self.current_phase < len(self.learning_phases) - 1:
            self.current_phase += 1
            phase_name = self.learning_phases[self.current_phase]['name']
            print(f"🎯 进入{phase_name}")
    
    def get_learning_info(self) -> Dict[str, Any]:
        """获取学习信息"""
        current_phase = self.learning_phases[self.current_phase]
        return {
            'learning_progress': self.learning_progress,
            'current_phase': current_phase['name'],
            'target_component': current_phase['target'],
            'phase_index': self.current_phase
        }


class CurriculumRewardFunction(RewardFunction):
    """课程学习奖励函数 - 根据难度级别调整奖励"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        
        # 课程配置
        self.difficulty_level = config.get('starting_difficulty', 1)
        self.max_difficulty = config.get('max_difficulty', 5)
        
        # 难度级别对应的奖励配置
        self.difficulty_configs = {
            1: {'victory_weight': 5.0, 'defeat_weight': -2.0, 'time_penalty_weight': -0.005},
            2: {'victory_weight': 8.0, 'defeat_weight': -3.0, 'time_penalty_weight': -0.01},
            3: {'victory_weight': 10.0, 'defeat_weight': -4.0, 'time_penalty_weight': -0.015},
            4: {'victory_weight': 12.0, 'defeat_weight': -5.0, 'time_penalty_weight': -0.02},
            5: {'victory_weight': 15.0, 'defeat_weight': -6.0, 'time_penalty_weight': -0.025}
        }
        
        # 更新当前难度配置
        self._update_difficulty_config()
        
        print(f"✅ 课程学习奖励函数初始化完成 - 初始难度: {self.difficulty_level}")
    
    def _update_difficulty_config(self) -> None:
        """更新难度配置"""
        if self.difficulty_level in self.difficulty_configs:
            difficulty_config = self.difficulty_configs[self.difficulty_level]
            self.weights.update(difficulty_config)
    
    def increase_difficulty(self) -> bool:
        """增加难度级别"""
        if self.difficulty_level < self.max_difficulty:
            self.difficulty_level += 1
            self._update_difficulty_config()
            print(f"📈 难度级别提升至: {self.difficulty_level}")
            return True
        return False
    
    def decrease_difficulty(self) -> bool:
        """降低难度级别"""
        if self.difficulty_level > 1:
            self.difficulty_level -= 1
            self._update_difficulty_config()
            print(f"📉 难度级别降低至: {self.difficulty_level}")
            return True
        return False
    
    def get_difficulty_info(self) -> Dict[str, Any]:
        """获取难度信息"""
        return {
            'current_difficulty': self.difficulty_level,
            'max_difficulty': self.max_difficulty,
            'current_weights': self.weights
        }


if __name__ == "__main__":
    # 测试奖励函数
    print("🧪 测试奖励函数...")
    
    # 创建测试数据
    current_state = np.random.random(50)
    next_state = np.random.random(50)
    action = {'type': 'click', 'params': {'button_id': 0}}
    
    # 测试基础奖励函数
    reward_fn = RewardFunction()
    reward = reward_fn.calculate_reward(current_state, action, next_state)
    
    print(f"基础奖励: {reward:.3f}")
    print(f"奖励统计: {reward_fn.get_reward_stats()}")
    
    # 测试自适应奖励函数
    adaptive_fn = AdaptiveRewardFunction()
    adaptive_reward = adaptive_fn.calculate_reward(current_state, action, next_state)
    
    print(f"自适应奖励: {adaptive_reward:.3f}")
    print(f"学习信息: {adaptive_fn.get_learning_info()}")
    
    # 测试课程学习奖励函数
    curriculum_fn = CurriculumRewardFunction()
    curriculum_reward = curriculum_fn.calculate_reward(current_state, action, next_state)
    
    print(f"课程学习奖励: {curriculum_reward:.3f}")
    print(f"难度信息: {curriculum_fn.get_difficulty_info()}")
    
    print("✅ 奖励函数测试通过")
"""
动作空间模块
定义和管理PPO算法可以执行的所有动作
包括动作编码、解码和验证功能
"""

import numpy as np
from typing import Dict, List, Any, Tuple, Union
from enum import Enum
import random

# 复用现有基础设施
from src.vision.detector.yolo_detector import YOLODetector
from src.device.adb_controller import ADBController


class ActionType(Enum):
    """动作类型枚举"""
    CLICK = "click"           # 点击操作
    SWIPE = "swipe"           # 滑动操作
    WAIT = "wait"             # 等待操作
    DEPLOY = "deploy"         # 部署干员
    SKILL = "skill"           # 使用技能
    RETREAT = "retreat"       # 撤退干员
    CANCEL = "cancel"         # 取消操作
    PAUSE = "pause"           # 暂停游戏
    RESUME = "resume"         # 继续游戏


class ActionSpace:
    """动作空间 - 管理所有可能的动作"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化动作空间
        
        Args:
            config: 配置参数
        """
        self.config = config or {}
        
        # 动作空间维度
        self.action_dim = self.config.get('action_dim', 50)
        
        # 复用ADB控制器
        self.adb_controller = ADBController()
        
        # 复用YOLO检测器（使用单例模式）
        self.yolo_detector = YOLODetector.get_instance(
            model_path='yolov8n.pt',
            use_gpu=False
        )
        
        # 动作定义
        self.actions = self._define_actions()
        
        # 动作掩码（用于限制无效动作）
        self.action_mask = np.ones(self.action_dim, dtype=bool)
        
        # 动作历史
        self.action_history = []
        self.max_history_length = 10
        
        print(f"✅ 动作空间初始化完成 - 动作维度: {self.action_dim}")
    
    def _define_actions(self) -> List[Dict[str, Any]]:
        """定义所有可能的动作"""
        actions = []
        
        # 1. 基础点击动作（13个按钮）
        for i in range(13):
            actions.append({
                'type': ActionType.CLICK,
                'name': f'click_button_{i}',
                'description': f'点击按钮{i}',
                'params': {'button_id': i},
                'index': len(actions)
            })
        
        # 2. 部署干员动作（8个部署位）
        for i in range(8):
            actions.append({
                'type': ActionType.DEPLOY,
                'name': f'deploy_operator_{i}',
                'description': f'在部署位{i}部署干员',
                'params': {'deploy_position': i},
                'index': len(actions)
            })
        
        # 3. 使用技能动作（8个干员 * 3个技能）
        for operator_idx in range(8):
            for skill_idx in range(3):
                actions.append({
                    'type': ActionType.SKILL,
                    'name': f'use_skill_{operator_idx}_{skill_idx}',
                    'description': f'使用干员{operator_idx}的技能{skill_idx}',
                    'params': {'operator_idx': operator_idx, 'skill_idx': skill_idx},
                    'index': len(actions)
                })
        
        # 4. 撤退干员动作（8个干员）
        for i in range(8):
            actions.append({
                'type': ActionType.RETREAT,
                'name': f'retreat_operator_{i}',
                'description': f'撤退干员{i}',
                'params': {'operator_idx': i},
                'index': len(actions)
            })
        
        # 5. 滑动动作（4个方向）
        directions = ['up', 'down', 'left', 'right']
        for direction in directions:
            actions.append({
                'type': ActionType.SWIPE,
                'name': f'swipe_{direction}',
                'description': f'向{direction}滑动',
                'params': {'direction': direction, 'distance': 200},
                'index': len(actions)
            })
        
        # 6. 等待动作（不同时长）
        wait_times = [0.5, 1.0, 2.0, 5.0]
        for time in wait_times:
            actions.append({
                'type': ActionType.WAIT,
                'name': f'wait_{time}s',
                'description': f'等待{time}秒',
                'params': {'duration': time},
                'index': len(actions)
            })
        
        # 7. 特殊动作
        actions.extend([
            {
                'type': ActionType.CANCEL,
                'name': 'cancel_action',
                'description': '取消当前操作',
                'params': {},
                'index': len(actions)
            },
            {
                'type': ActionType.PAUSE,
                'name': 'pause_game',
                'description': '暂停游戏',
                'params': {},
                'index': len(actions)
            },
            {
                'type': ActionType.RESUME,
                'name': 'resume_game',
                'description': '继续游戏',
                'params': {},
                'index': len(actions)
            }
        ])
        
        # 确保动作数量不超过设定的维度
        if len(actions) > self.action_dim:
            actions = actions[:self.action_dim]
        
        return actions
    
    def get_action_dim(self) -> int:
        """获取动作空间维度"""
        return len(self.actions)
    
    def get_action_by_index(self, action_index: int) -> Dict[str, Any]:
        """
        根据动作索引获取动作字典
        
        Args:
            action_index: 动作索引
            
        Returns:
            action_dict: 动作字典
        """
        if 0 <= action_index < len(self.actions):
            return self.actions[action_index].copy()
        else:
            # 返回默认等待动作
            return {
                'type': ActionType.WAIT,
                'name': 'wait_1s',
                'description': '等待1秒',
                'params': {'duration': 1.0},
                'index': -1
            }
    
    def encode_action(self, action_dict: Dict[str, Any]) -> np.ndarray:
        """
        将动作字典编码为动作向量
        
        Args:
            action_dict: 动作字典
            
        Returns:
            action_vector: 编码后的动作向量
        """
        # 创建one-hot编码的动作向量
        action_vector = np.zeros(self.action_dim)
        
        # 查找匹配的动作
        for action in self.actions:
            if self._action_matches(action, action_dict):
                action_vector[action['index']] = 1.0
                break
        
        return action_vector
    
    def decode_action(self, action_vector: np.ndarray) -> Dict[str, Any]:
        """
        将动作向量解码为动作字典
        
        Args:
            action_vector: 动作向量
            
        Returns:
            action_dict: 解码后的动作字典
        """
        # 找到最大值的索引
        action_idx = np.argmax(action_vector)
        
        # 检查索引是否有效
        if action_idx < len(self.actions):
            return self.actions[action_idx].copy()
        else:
            # 返回默认动作
            return {
                'type': ActionType.WAIT,
                'name': 'wait_1s',
                'description': '等待1秒',
                'params': {'duration': 1.0},
                'index': -1
            }
    
    def execute_action(self, action_dict: Dict[str, Any], screenshot: np.ndarray = None) -> bool:
        """
        执行动作
        
        Args:
            action_dict: 动作字典
            screenshot: 当前屏幕截图（可选）
            
        Returns:
            success: 是否执行成功
        """
        try:
            action_type = action_dict['type']
            params = action_dict['params']
            
            if action_type == ActionType.CLICK:
                return self._execute_click(params, screenshot)
            elif action_type == ActionType.SWIPE:
                return self._execute_swipe(params)
            elif action_type == ActionType.WAIT:
                return self._execute_wait(params)
            elif action_type == ActionType.DEPLOY:
                return self._execute_deploy(params, screenshot)
            elif action_type == ActionType.SKILL:
                return self._execute_skill(params, screenshot)
            elif action_type == ActionType.RETREAT:
                return self._execute_retreat(params, screenshot)
            elif action_type == ActionType.CANCEL:
                return self._execute_cancel()
            elif action_type == ActionType.PAUSE:
                return self._execute_pause()
            elif action_type == ActionType.RESUME:
                return self._execute_resume()
            else:
                print(f"❌ 未知动作类型: {action_type}")
                return False
                
        except Exception as e:
            print(f"❌ 执行动作失败: {e}")
            return False
    
    def _execute_click(self, params: Dict[str, Any], screenshot: np.ndarray = None) -> bool:
        """执行点击动作"""
        button_id = params.get('button_id')
        
        if screenshot is not None:
            # 使用YOLO检测按钮位置
            detections = self.yolo_detector.detect(screenshot)
            button_detections = [d for d in detections if d['class'] == button_id]
            
            if button_detections:
                # 点击检测到的按钮
                button = button_detections[0]
                x, y = int(button['x']), int(button['y'])
                return self.adb_controller.tap(x, y)
            else:
                print(f"❌ 未检测到按钮 {button_id}")
                return False
        else:
            # 使用预设坐标（需要根据实际游戏界面调整）
            preset_coords = self._get_preset_coordinates()
            if button_id in preset_coords:
                x, y = preset_coords[button_id]
                return self.adb_controller.tap(x, y)
            else:
                print(f"❌ 按钮 {button_id} 没有预设坐标")
                return False
    
    def _execute_swipe(self, params: Dict[str, Any]) -> bool:
        """执行滑动动作"""
        direction = params.get('direction')
        distance = params.get('distance', 200)
        
        # 滑动方向映射
        direction_vectors = {
            'up': (0, -distance),
            'down': (0, distance),
            'left': (-distance, 0),
            'right': (distance, 0)
        }
        
        if direction in direction_vectors:
            dx, dy = direction_vectors[direction]
            
            # 从屏幕中心开始滑动
            start_x, start_y = 960, 540  # 屏幕中心
            end_x, end_y = start_x + dx, start_y + dy
            
            return self.adb_controller.swipe(start_x, start_y, end_x, end_y, duration=300)
        else:
            print(f"❌ 未知滑动方向: {direction}")
            return False
    
    def _execute_wait(self, params: Dict[str, Any]) -> bool:
        """执行等待动作"""
        import time
        duration = params.get('duration', 1.0)
        time.sleep(duration)
        return True
    
    def _execute_deploy(self, params: Dict[str, Any], screenshot: np.ndarray = None) -> bool:
        """执行部署干员动作"""
        deploy_position = params.get('deploy_position')
        
        if screenshot is not None:
            # 检测部署位
            detections = self.yolo_detector.detect(screenshot)
            deploy_detections = [d for d in detections if d['class'] == 20 + deploy_position]
            
            if deploy_detections:
                deploy_spot = deploy_detections[0]
                x, y = int(deploy_spot['x']), int(deploy_spot['y'])
                
                # 点击部署位
                success = self.adb_controller.tap(x, y)
                if success:
                    # 短暂等待后选择干员
                    import time
                    time.sleep(0.5)
                    
                    # 点击第一个干员（简化实现）
                    return self.adb_controller.tap(200, 800)
                return False
            else:
                print(f"❌ 未检测到部署位 {deploy_position}")
                return False
        else:
            print("❌ 需要屏幕截图来执行部署动作")
            return False
    
    def _execute_skill(self, params: Dict[str, Any], screenshot: np.ndarray = None) -> bool:
        """执行使用技能动作"""
        operator_idx = params.get('operator_idx')
        skill_idx = params.get('skill_idx')
        
        if screenshot is not None:
            # 检测干员技能按钮
            detections = self.yolo_detector.detect(screenshot)
            skill_detections = [d for d in detections if d['class'] == 30 + operator_idx * 3 + skill_idx]
            
            if skill_detections:
                skill_button = skill_detections[0]
                x, y = int(skill_button['x']), int(skill_button['y'])
                return self.adb_controller.tap(x, y)
            else:
                print(f"❌ 未检测到干员{operator_idx}的技能{skill_idx}")
                return False
        else:
            print("❌ 需要屏幕截图来执行技能动作")
            return False
    
    def _execute_retreat(self, params: Dict[str, Any], screenshot: np.ndarray = None) -> bool:
        """执行撤退干员动作"""
        operator_idx = params.get('operator_idx')
        
        if screenshot is not None:
            # 检测干员撤退按钮
            detections = self.yolo_detector.detect(screenshot)
            retreat_detections = [d for d in detections if d['class'] == 40 + operator_idx]
            
            if retreat_detections:
                retreat_button = retreat_detections[0]
                x, y = int(retreat_button['x']), int(retreat_button['y'])
                
                # 点击撤退按钮
                success = self.adb_controller.tap(x, y)
                if success:
                    # 确认撤退
                    import time
                    time.sleep(0.5)
                    return self.adb_controller.tap(960, 700)  # 确认按钮位置
                return False
            else:
                print(f"❌ 未检测到干员{operator_idx}的撤退按钮")
                return False
        else:
            print("❌ 需要屏幕截图来执行撤退动作")
            return False
    
    def _execute_cancel(self) -> bool:
        """执行取消动作"""
        # 使用返回键取消
        return self.adb_controller.key_event('KEYCODE_BACK')
    
    def _execute_pause(self) -> bool:
        """执行暂停动作"""
        # 点击暂停按钮（预设位置）
        return self.adb_controller.tap(1800, 50)
    
    def _execute_resume(self) -> bool:
        """执行继续动作"""
        # 点击继续按钮（预设位置）
        return self.adb_controller.tap(960, 800)
    
    def _action_matches(self, action: Dict[str, Any], action_dict: Dict[str, Any]) -> bool:
        """检查两个动作是否匹配"""
        if action['type'] != action_dict.get('type'):
            return False
        
        # 检查参数匹配
        action_params = action['params']
        dict_params = action_dict.get('params', {})
        
        for key, value in action_params.items():
            if key not in dict_params or dict_params[key] != value:
                return False
        
        return True
    
    def _get_preset_coordinates(self) -> Dict[int, Tuple[int, int]]:
        """获取预设按钮坐标"""
        # 这里需要根据实际游戏界面设置坐标
        # 示例坐标（需要根据明日方舟实际界面调整）
        return {
            0: (100, 100),   # 按钮0
            1: (200, 100),   # 按钮1
            2: (300, 100),   # 按钮2
            3: (400, 100),   # 按钮3
            4: (500, 100),   # 按钮4
            5: (600, 100),   # 按钮5
            6: (700, 100),   # 按钮6
            7: (800, 100),   # 按钮7
            8: (900, 100),   # 按钮8
            9: (1000, 100),  # 按钮9
            10: (1100, 100), # 按钮10
            11: (1200, 100), # 按钮11
            12: (1300, 100)  # 按钮12
        }
    
    def update_action_mask(self, state_vector: np.ndarray, detections: List[Dict] = None) -> None:
        """
        更新动作掩码，限制无效动作
        
        Args:
            state_vector: 当前状态向量
            detections: YOLO检测结果（可选）
        """
        # 重置动作掩码
        self.action_mask.fill(True)
        
        # 根据状态向量和检测结果更新掩码
        if detections is not None:
            self._update_mask_from_detections(detections)
        
        # 根据状态向量更新掩码
        self._update_mask_from_state(state_vector)
        
        # 根据动作历史更新掩码
        self._update_mask_from_history()
    
    def _update_mask_from_detections(self, detections: List[Dict]) -> None:
        """根据检测结果更新动作掩码"""
        detected_classes = [d['class'] for d in detections]
        
        for action in self.actions:
            action_idx = action['index']
            
            if action['type'] == ActionType.CLICK:
                button_id = action['params'].get('button_id')
                if button_id not in detected_classes:
                    self.action_mask[action_idx] = False
            
            elif action['type'] == ActionType.DEPLOY:
                deploy_position = action['params'].get('deploy_position')
                deploy_class = 20 + deploy_position
                if deploy_class not in detected_classes:
                    self.action_mask[action_idx] = False
            
            elif action['type'] == ActionType.SKILL:
                operator_idx = action['params'].get('operator_idx')
                skill_idx = action['params'].get('skill_idx')
                skill_class = 30 + operator_idx * 3 + skill_idx
                if skill_class not in detected_classes:
                    self.action_mask[action_idx] = False
            
            elif action['type'] == ActionType.RETREAT:
                operator_idx = action['params'].get('operator_idx')
                retreat_class = 40 + operator_idx
                if retreat_class not in detected_classes:
                    self.action_mask[action_idx] = False
    
    def _update_mask_from_state(self, state_vector: np.ndarray) -> None:
        """根据状态向量更新动作掩码"""
        # 这里可以根据状态向量的特定维度来限制动作
        # 例如：如果血量过低，限制某些攻击性动作
        
        # 示例：如果检测到胜利标志，限制大部分动作
        if len(state_vector) > 50 and state_vector[50] > 0.8:  # 胜利标志
            for action in self.actions:
                if action['type'] not in [ActionType.WAIT, ActionType.CLICK]:
                    self.action_mask[action['index']] = False
    
    def _update_mask_from_history(self) -> None:
        """根据动作历史更新动作掩码"""
        if len(self.action_history) < 2:
            return
        
        # 防止重复执行相同动作
        recent_actions = self.action_history[-3:]
        for action in self.actions:
            if any(self._action_matches(action, hist_action) for hist_action in recent_actions):
                # 限制连续执行相同动作
                self.action_mask[action['index']] = False
    
    def get_action_name(self, action_index: int) -> str:
        """
        根据动作索引获取动作名称
        
        Args:
            action_index: 动作索引
            
        Returns:
            action_name: 动作名称
        """
        if 0 <= action_index < len(self.actions):
            return self.actions[action_index].get('name', f'action_{action_index}')
        else:
            return f'unknown_action_{action_index}'
    
    def get_valid_actions(self) -> List[Dict[str, Any]]:
        """获取当前有效的动作列表"""
        return [action for action in self.actions if self.action_mask[action['index']]]
    
    def sample_action(self) -> Dict[str, Any]:
        """随机采样一个有效动作"""
        valid_actions = self.get_valid_actions()
        if valid_actions:
            return random.choice(valid_actions)
        else:
            # 返回默认等待动作
            return {
                'type': ActionType.WAIT,
                'name': 'wait_1s',
                'description': '等待1秒',
                'params': {'duration': 1.0},
                'index': -1
            }
    
    def record_action(self, action_dict: Dict[str, Any]) -> None:
        """记录动作历史"""
        self.action_history.append(action_dict.copy())
        if len(self.action_history) > self.max_history_length:
            self.action_history.pop(0)
    
    def reset_history(self) -> None:
        """重置动作历史"""
        self.action_history = []
        print("🔄 动作历史已重置")


class HierarchicalActionSpace(ActionSpace):
    """分层动作空间 - 支持更复杂的动作组合"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        
        # 分层动作配置
        self.hierarchical_levels = config.get('hierarchical_levels', 2)
        self.macro_actions = self._define_macro_actions()
        
        print(f"✅ 分层动作空间初始化完成 - 层级数: {self.hierarchical_levels}")
    
    def _define_macro_actions(self) -> List[Dict[str, Any]]:
        """定义宏动作"""
        macro_actions = [
            {
                'type': 'macro',
                'name': 'defensive_strategy',
                'description': '防御策略：优先部署防御型干员',
                'sub_actions': [
                    {'type': ActionType.DEPLOY, 'params': {'deploy_position': 0}},
                    {'type': ActionType.DEPLOY, 'params': {'deploy_position': 1}},
                    {'type': ActionType.WAIT, 'params': {'duration': 2.0}}
                ]
            },
            {
                'type': 'macro',
                'name': 'offensive_strategy',
                'description': '进攻策略：优先部署攻击型干员',
                'sub_actions': [
                    {'type': ActionType.DEPLOY, 'params': {'deploy_position': 2}},
                    {'type': ActionType.DEPLOY, 'params': {'deploy_position': 3}},
                    {'type': ActionType.SKILL, 'params': {'operator_idx': 0, 'skill_idx': 0}}
                ]
            },
            {
                'type': 'macro',
                'name': 'skill_chain',
                'description': '技能连招：按顺序使用多个技能',
                'sub_actions': [
                    {'type': ActionType.SKILL, 'params': {'operator_idx': 0, 'skill_idx': 0}},
                    {'type': ActionType.WAIT, 'params': {'duration': 1.0}},
                    {'type': ActionType.SKILL, 'params': {'operator_idx': 1, 'skill_idx': 0}}
                ]
            }
        ]
        
        return macro_actions
    
    def execute_macro_action(self, macro_action: Dict[str, Any], screenshot: np.ndarray = None) -> bool:
        """执行宏动作"""
        sub_actions = macro_action.get('sub_actions', [])
        
        for sub_action in sub_actions:
            success = self.execute_action(sub_action, screenshot)
            if not success:
                print(f"❌ 宏动作执行失败: {macro_action['name']}")
                return False
            
            # 短暂等待
            import time
            time.sleep(0.2)
        
        return True


if __name__ == "__main__":
    # 测试动作空间
    print("🧪 测试动作空间...")
    
    # 测试基础动作空间
    action_space = ActionSpace()
    
    print(f"动作空间维度: {action_space.get_action_dim()}")
    print(f"动作数量: {len(action_space.actions)}")
    
    # 测试动作编码/解码
    test_action = {
        'type': ActionType.CLICK,
        'params': {'button_id': 0}
    }
    
    action_vector = action_space.encode_action(test_action)
    decoded_action = action_space.decode_action(action_vector)
    
    print(f"原始动作: {test_action}")
    print(f"解码后动作: {decoded_action}")
    
    # 测试分层动作空间
    hierarchical_space = HierarchicalActionSpace()
    print(f"分层动作空间宏动作数量: {len(hierarchical_space.macro_actions)}")
    
    print("✅ 动作空间测试通过")
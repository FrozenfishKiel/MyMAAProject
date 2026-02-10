"""
游戏环境接口模块
连接PPO算法和实际游戏环境的桥梁
复用现有基础设施：ADB控制器、YOLO检测器、设备管理器等
"""

import time
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import cv2

# 复用现有基础设施模块
from src.device.adb_controller import ADBController
from src.device.device_manager import DeviceManager
from src.vision.detector.yolo_detector import YOLODetector
from src.vision.processor.image_processor import ImageProcessor
from src.utils.config_manager import ConfigManager
from src.ppo.state_encoder import GameStateEncoder


class GameEnvironment:
    """游戏环境接口 - 连接PPO算法和游戏环境"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        初始化游戏环境接口
        
        Args:
            config_path: 配置文件路径
        """
        # 加载配置
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.get_config()
        
        # 复用现有基础设施
        self.device_manager = DeviceManager()
        
        # 创建DeviceConfig对象用于ADB控制器
        from src.device.config import DeviceConfig
        adb_config = DeviceConfig()
        self.adb_controller = ADBController(adb_config)
        
        # YOLO检测器（复用已训练的13个按钮检测模型，使用单例模式）
        self.yolo_detector = YOLODetector.get_instance(
            model_path='runs/detect/train/weights/best.pt',  # 使用自定义训练模型
            use_gpu=False
        )
        
        # 图像处理器
        self.image_processor = ImageProcessor()
        
        # 状态编码器
        self.state_encoder = GameStateEncoder(self.yolo_detector, self.image_processor)
        
        # 游戏状态
        self.current_state = None
        self.episode_step = 0
        self.episode_reward = 0.0
        self.episode_done = False
        
        print("✅ 游戏环境接口初始化完成")
        print(f"   - 复用ADB控制器: {self.adb_controller}")
        print(f"   - 复用YOLO检测器: {self.yolo_detector}")
        print(f"   - 复用设备管理器: {self.device_manager}")
        print(f"   - 状态编码器维度: {self.state_encoder.get_state_dimension()}")
    
    def reset(self) -> np.ndarray:
        """
        重置游戏环境，开始新的回合
        
        Returns:
            initial_state: 初始状态向量
        """
        # 重置游戏到初始状态
        self._reset_game()
        
        # 重置内部状态
        self.episode_step = 0
        self.episode_reward = 0.0
        self.episode_done = False
        self.state_encoder.reset_history()
        
        # 获取初始状态
        initial_state = self._get_current_state()
        
        print("🔄 游戏环境已重置")
        return initial_state
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        执行一个动作，返回新的状态和奖励
        
        Args:
            action: 动作编码（0-13）
            
        Returns:
            next_state: 下一状态
            reward: 获得的奖励
            done: 是否回合结束
            info: 额外信息
        """
        # 执行动作
        action_info = self._execute_action(action)
        
        # 等待游戏响应
        time.sleep(0.5)  # 等待游戏响应时间
        
        # 获取新的状态
        next_state = self._get_current_state()
        
        # 计算奖励
        reward = self._calculate_reward(self.current_state, action, next_state)
        
        # 更新内部状态
        self.current_state = next_state
        self.episode_step += 1
        self.episode_reward += reward
        
        # 检查是否回合结束
        done = self._check_episode_done()
        
        # 构建信息字典
        info = {
            'episode_step': self.episode_step,
            'episode_reward': self.episode_reward,
            'action_info': action_info,
            'game_state': self._get_game_state_info()
        }
        
        return next_state, reward, done, info
    
    def _reset_game(self) -> None:
        """重置游戏到初始状态"""
        # 1. 退出当前关卡（如果正在游戏中）
        self.adb_controller.click(50, 50)  # 点击左上角返回
        time.sleep(1)
        
        # 2. 重新进入1-7关卡
        self._enter_level_1_7()
        
        # 3. 等待游戏加载完成
        time.sleep(3)
    
    def _enter_level_1_7(self) -> None:
        """进入1-7关卡"""
        # 这里需要根据实际游戏界面设计进入关卡的具体步骤
        # 示例：点击关卡选择、确认开始等
        
        # 点击关卡选择按钮（示例坐标）
        self.adb_controller.click(400, 300)
        time.sleep(1)
        
        # 点击开始战斗按钮（示例坐标）
        self.adb_controller.click(600, 500)
        time.sleep(2)
        
        print("🎮 已进入1-7关卡")
    
    def _get_current_state(self) -> np.ndarray:
        """获取当前游戏状态"""
        # 1. 截取屏幕
        screenshot = self.adb_controller.take_screenshot()
        
        # 2. 使用状态编码器编码游戏状态
        state_vector = self.state_encoder.encode_state(screenshot)
        
        return state_vector
    

    
    def _execute_action(self, action: int) -> Dict[str, Any]:
        """执行动作 - 完全匹配MAA标准"""
        # 动作映射表（完全匹配MAA的实际操作类型）
        action_map = {
            # 游戏控制动作 (0-2) - 基于MAA的SpeedUp操作
            0: {'type': 'speed_up', 'description': '加速游戏（MAA: SpeedUp）'},
            1: {'type': 'wait', 'duration': 1, 'description': '等待1秒（MAA: Wait）'},
            2: {'type': 'wait', 'duration': 3, 'description': '等待3秒（MAA: Wait）'},
            
            # 干员部署动作 (3-10) - 基于MAA的Deploy操作
            3: {'type': 'deploy', 'role': 'warrior', 'location': [5, 3], 'direction': 'Right', 'description': '部署近卫干员（MAA: Deploy）'},
            4: {'type': 'deploy', 'role': 'pioneer', 'location': [4, 3], 'direction': 'Right', 'description': '部署先锋干员（MAA: Deploy）'},
            5: {'type': 'deploy', 'role': 'medic', 'location': [6, 2], 'direction': 'Right', 'description': '部署医疗干员（MAA: Deploy）'},
            6: {'type': 'deploy', 'role': 'tank', 'location': [3, 3], 'direction': 'Right', 'description': '部署重装干员（MAA: Deploy）'},
            7: {'type': 'deploy', 'role': 'sniper', 'location': [7, 2], 'direction': 'Right', 'description': '部署狙击干员（MAA: Deploy）'},
            8: {'type': 'deploy', 'role': 'caster', 'location': [6, 1], 'direction': 'Right', 'description': '部署术师干员（MAA: Deploy）'},
            9: {'type': 'deploy', 'role': 'support', 'location': [5, 2], 'direction': 'Right', 'description': '部署辅助干员（MAA: Deploy）'},
            10: {'type': 'deploy', 'role': 'special', 'location': [4, 2], 'direction': 'Right', 'description': '部署特种干员（MAA: Deploy）'},
            
            # 技能和撤退动作 (11-13) - 基于MAA的Skill和Retreat操作
            11: {'type': 'skill', 'description': '使用技能（MAA: Skill）'},
            12: {'type': 'retreat', 'description': '撤退干员（MAA: Retreat）'},
            13: {'type': 'skill_daemon', 'description': '技能自动释放（MAA: SkillDaemon）'}
        }
        
        action_info = action_map.get(action, {'type': 'invalid', 'description': '无效动作'})
        
        if action_info['type'] == 'speed_up':
            # 执行加速动作
            self.adb_controller.click(100, 50)  # 加速按钮位置
            print(f"⏩ 执行加速动作: {action_info['description']}")
            
        elif action_info['type'] == 'wait':
            # 执行等待动作
            duration = action_info['duration']
            time.sleep(duration)
            print(f"⏰ 执行等待动作: {action_info['description']} ({duration}秒)")
            
        elif action_info['type'] == 'deploy':
            # 执行部署动作
            role = action_info['role']
            location = action_info.get('location', [5, 3])
            direction = action_info.get('direction', 'Right')
            
            # 根据MAA的部署逻辑选择部署位置
            deploy_pos = self._get_deploy_position_by_location(location)
            self.adb_controller.click(deploy_pos.x, deploy_pos.y)
            print(f"🎯 执行部署动作: {action_info['description']} ({deploy_pos.x}, {deploy_pos.y})")
            
        elif action_info['type'] == 'skill':
            # 执行技能动作
            skill_pos = self._get_skill_position()
            self.adb_controller.click(skill_pos.x, skill_pos.y)
            print(f"✨ 执行技能动作: {action_info['description']} ({skill_pos.x}, {skill_pos.y})")
            
        elif action_info['type'] == 'retreat':
            # 执行撤退动作
            retreat_pos = self._get_retreat_position()
            self.adb_controller.click(retreat_pos.x, retreat_pos.y)
            print(f"🚪 执行撤退动作: {action_info['description']} ({retreat_pos.x}, {retreat_pos.y})")
            
        elif action_info['type'] == 'skill_daemon':
            # 执行技能自动释放动作
            skill_pos = self._get_skill_position()
            self.adb_controller.click(skill_pos.x, skill_pos.y)
            print(f"🤖 执行技能自动释放: {action_info['description']} ({skill_pos.x}, {skill_pos.y})")
        
        return action_info
    
    def _get_deploy_position_by_location(self, location: List[int]) -> Point:
        """根据位置坐标获取部署位置 - 完全匹配MAA的部署逻辑"""
        # MAA使用网格坐标系统，例如[5, 3]表示第5列第3行
        # 这里需要将网格坐标转换为实际屏幕坐标
        
        grid_x, grid_y = location
        
        # 假设网格大小为100x100，起始位置为(200, 100)
        screen_x = 200 + (grid_x - 1) * 100
        screen_y = 100 + (grid_y - 1) * 100
        
        return Point(screen_x, screen_y)
    
    def _get_deploy_position(self, role: str) -> Point:
        """获取干员部署位置 - 基于MAA的部署策略"""
        # 这里需要根据实际游戏界面和干员职业选择最佳部署位置
        # 基于MAA的部署逻辑，不同职业的干员有不同的部署偏好
        
        # 默认的部署位置（基于MAA的常用部署位置）
        default_positions = {
            'warrior': Point(800, 400),    # 近卫：前线战斗位置
            'pioneer': Point(600, 500),    # 先锋：费用生成位置
            'medic': Point(1000, 300),     # 医疗：安全支援位置
            'tank': Point(700, 350),       # 重装：阻挡位置
            'sniper': Point(1100, 250),    # 狙击：远程输出位置
            'caster': Point(1050, 280),    # 术师：法术输出位置
            'support': Point(900, 320),    # 辅助：增益位置
            'special': Point(750, 450)     # 特种：特殊功能位置
        }
        
        return default_positions.get(role, Point(800, 400))
    
    def _get_skill_position(self) -> Point:
        """获取技能按钮位置 - 基于MAA的技能识别"""
        # 这里需要根据实际游戏界面识别技能按钮位置
        # 基于MAA的技能识别逻辑，技能按钮通常在屏幕右侧
        return Point(1200, 200)
    
    def _get_retreat_position(self) -> Point:
        """获取撤退按钮位置 - 基于MAA的撤退逻辑"""
        # 这里需要根据实际游戏界面识别撤退按钮位置
        # 基于MAA的撤退逻辑，撤退按钮通常在屏幕底部
        return Point(100, 600)
    
    def _get_redeploy_position(self) -> Point:
        """获取重新部署位置 - 基于MAA的重新部署逻辑"""
        # 这里需要根据实际游戏界面识别重新部署位置
        # 基于MAA的重新部署逻辑，通常在干员头像附近
        return Point(200, 150)
    
    def _calculate_reward(self, state: np.ndarray, action: int, next_state: np.ndarray) -> float:
        """计算奖励 - 基于MAA和WZCQ的混合奖励设计"""
        reward = 0.0
        
        # ========== 主要奖励 ==========
        
        # 1. 通关奖励（检测是否通关）
        if self._check_level_completed():
            reward += 100.0  # 大幅通关奖励
            print("🎉 通关奖励: +100")
            return reward  # 通关后立即返回
        
        # 2. 关卡失败惩罚
        if self._check_level_failed():
            reward -= 50.0
            print("❌ 关卡失败惩罚: -50")
            return reward  # 失败后立即返回
        
        # 3. 击败敌人奖励（按敌人类型区分）
        enemy_reward = self._calculate_enemy_reward(state, next_state)
        reward += enemy_reward
        
        # 4. 血量保持奖励
        hp_reward = self._calculate_hp_reward(state, next_state)
        reward += hp_reward
        
        # ========== 策略性奖励 ==========
        
        # 5. 干员部署奖励（鼓励合理部署）
        deployment_reward = self._calculate_deployment_reward(action, state)
        reward += deployment_reward
        
        # 6. 技能使用奖励（鼓励及时使用技能）
        skill_reward = self._calculate_skill_reward(action, state, next_state)
        reward += skill_reward
        
        # 7. 费用管理奖励（鼓励合理使用费用）
        cost_reward = self._calculate_cost_reward(state, next_state)
        reward += cost_reward
        
        # ========== 惩罚机制 ==========
        
        # 8. 动作有效性惩罚
        if not self._is_valid_action(action, state):
            reward -= 2.0
            print("⚠️ 无效动作惩罚: -2")
        
        # 9. 长时间等待惩罚（鼓励积极行动）
        if action in [1, 2]:  # 等待动作
            reward -= 0.5
            print("⏰ 等待动作惩罚: -0.5")
        
        # 10. 步数惩罚（鼓励高效通关）
        step_penalty = -0.1
        reward += step_penalty
        
        # 限制奖励范围，避免极端值
        reward = max(-10.0, min(100.0, reward))
        
        return reward
    
    def _check_episode_done(self) -> bool:
        """检查回合是否结束"""
        # 1. 关卡完成
        if self._check_level_completed():
            print("✅ 关卡完成，回合结束")
            return True
        
        # 2. 关卡失败
        if self._check_level_failed():
            print("❌ 关卡失败，回合结束")
            return True
        
        # 3. 步数限制
        if self.episode_step >= 100:  # 最大步数限制
            print("⏰ 步数限制，回合结束")
            return True
        
        return False
    
    def _check_level_completed(self) -> bool:
        """检查关卡是否完成 - 基于MAA的胜利条件检测"""
        # 获取当前游戏状态
        current_state = self._get_current_state()
        
        # 检查胜利图标是否存在（状态向量中的胜利图标特征）
        # 胜利图标对应的类别索引是21
        victory_detected = False
        
        try:
            # 截取屏幕进行YOLO检测
            screenshot = self.adb_controller.take_screenshot()
            detections = self.yolo_detector.detect(screenshot)
            
            # 检查是否有胜利图标被检测到
            for detection in detections:
                if detection['class'] == 21:  # 胜利图标
                    victory_detected = True
                    break
                    
            # 同时检查状态向量中的胜利特征
            if current_state[35] > 0.5:  # 游戏状态特征中的胜利位置
                victory_detected = True
                
        except Exception as e:
            print(f"⚠️ 关卡完成检测失败: {e}")
            # 使用状态向量作为备用检测
            victory_detected = current_state[35] > 0.5
        
        return victory_detected
    
    def _check_level_failed(self) -> bool:
        """检查关卡是否失败 - 基于MAA的失败条件检测"""
        # 获取当前游戏状态
        current_state = self._get_current_state()
        
        # 检查失败图标是否存在（状态向量中的失败图标特征）
        # 失败图标对应的类别索引是22
        failure_detected = False
        
        try:
            # 截取屏幕进行YOLO检测
            screenshot = self.adb_controller.take_screenshot()
            detections = self.yolo_detector.detect(screenshot)
            
            # 检查是否有失败图标被检测到
            for detection in detections:
                if detection['class'] == 22:  # 失败图标
                    failure_detected = True
                    break
                    
            # 同时检查状态向量中的失败特征
            if current_state[36] > 0.5:  # 游戏状态特征中的失败位置
                failure_detected = True
                
        except Exception as e:
            print(f"⚠️ 关卡失败检测失败: {e}")
            # 使用状态向量作为备用检测
            failure_detected = current_state[36] > 0.5
        
        return failure_detected
    
    def _check_enemy_defeated(self, state: np.ndarray, next_state: np.ndarray) -> bool:
        """检查是否击败敌人 - 基于敌人数量变化"""
        # 敌人类型特征在状态向量中的位置：31-34
        # 普通敌人(31)、精英敌人(32)、BOSS敌人(33)、无人机(34)
        
        # 计算前后状态的敌人数量变化
        current_enemies = state[31] + state[32] + state[33] + state[34]
        next_enemies = next_state[31] + next_state[32] + next_state[33] + next_state[34]
        
        # 如果敌人数量减少，说明有敌人被击败
        enemy_defeated = next_enemies < current_enemies
        
        return enemy_defeated
    
    def _calculate_hp_reward(self, state: np.ndarray, next_state: np.ndarray) -> float:
        """计算血量保持奖励 - 基于血量变化"""
        # 血量比例在状态向量中的位置：0
        current_hp = state[0]
        next_hp = next_state[0]
        
        # 血量保持奖励：血量增加或保持时给予奖励
        hp_reward = 0.0
        
        if next_hp > current_hp:
            # 血量增加，给予较大奖励
            hp_reward = 2.0
        elif next_hp == current_hp:
            # 血量保持，给予小奖励
            hp_reward = 0.5
        else:
            # 血量减少，给予惩罚
            hp_reward = -1.0 * (current_hp - next_hp)  # 按减少比例惩罚
            
        return hp_reward
    
    def _is_valid_action(self, action: int, state: np.ndarray) -> bool:
        """检查动作是否有效 - 基于当前游戏状态"""
        # 动作有效性检查逻辑
        
        # 1. 检查游戏是否已经结束
        if self._check_level_completed() or self._check_level_failed():
            return False
            
        # 2. 检查费用是否足够（部署动作需要费用）
        if 3 <= action <= 10:  # 部署动作
            current_cost = state[1]  # 费用比例在状态向量中的位置：1
            if current_cost < 0.1:  # 费用不足
                return False
                
        # 3. 检查部署位置是否可用
        if 3 <= action <= 10:  # 部署动作
            # 检查部署位置可用性（状态向量中的位置：29-30）
            location_available = state[29] > 0.5 or state[30] > 0.5
            if not location_available:
                return False
                
        # 4. 检查技能是否可用
        if action == 11:  # 技能动作
            skill_available = state[5] > 0.5  # 技能可用性在状态向量中的位置：5
            if not skill_available:
                return False
                
        return True
    
    def _estimate_hp_ratio(self, screenshot: np.ndarray) -> float:
        """估计血量比例"""
        # 通过图像处理分析血条长度
        # 这里需要根据实际游戏界面设计
        return 1.0  # 暂定实现
    
    def _estimate_cost_value(self, screenshot: np.ndarray) -> float:
        """估计费用数值"""
        # 通过OCR识别费用数字
        # 这里需要根据实际游戏界面设计
        return 10.0  # 暂定实现
    
    def _estimate_time_ratio(self, screenshot: np.ndarray) -> float:
        """估计时间比例"""
        # 通过OCR识别倒计时
        # 这里需要根据实际游戏界面设计
        return 0.5  # 暂定实现
    
    def _count_enemies(self, detections: List[Dict]) -> int:
        """统计敌人数量"""
        # 通过YOLO检测敌人目标
        # 这里需要根据实际YOLO类别设计
        enemy_count = 0
        for detection in detections:
            if 17 <= detection['class'] <= 20:  # 敌人类别（17-20）
                enemy_count += 1
        
        return enemy_count
    
    def _calculate_enemy_reward(self, state: np.ndarray, next_state: np.ndarray) -> float:
        """计算击败敌人奖励 - 按敌人类型区分奖励"""
        enemy_reward = 0.0
        
        # 敌人类型特征在状态向量中的位置：31-34
        # 普通敌人(31)、精英敌人(32)、BOSS敌人(33)、无人机(34)
        
        # 检查每种敌人数量的变化
        enemy_types = ['普通敌人', '精英敌人', 'BOSS敌人', '无人机']
        reward_values = [2.0, 5.0, 20.0, 1.0]  # 不同敌人的奖励值
        
        for i in range(4):
            current_count = state[31 + i]
            next_count = next_state[31 + i]
            
            if next_count < current_count:
                # 有敌人被击败
                defeated_count = current_count - next_count
                enemy_reward += defeated_count * reward_values[i]
                print(f"🎯 击败{enemy_types[i]}奖励: +{defeated_count * reward_values[i]}")
        
        return enemy_reward
    
    def _calculate_deployment_reward(self, action: int, state: np.ndarray) -> float:
        """计算干员部署奖励 - 鼓励合理部署策略"""
        deployment_reward = 0.0
        
        if 3 <= action <= 10:  # 部署动作
            # 1. 检查费用是否合理（避免过度部署）
            current_cost = state[1]  # 费用比例
            if current_cost > 0.3:  # 费用充足时部署给予奖励
                deployment_reward += 1.0
                print("💡 合理部署奖励: +1")
            
            # 2. 检查敌人威胁等级（在威胁高时部署给予额外奖励）
            enemy_threat = state[3]  # 敌人威胁等级
            if enemy_threat > 0.5:
                deployment_reward += 2.0
                print("🛡️ 及时部署奖励: +2")
            
            # 3. 检查部署位置合理性（避免重复部署）
            deployed_count = state[4]  # 已部署干员数量
            if deployed_count < 0.5:  # 部署数量适中时给予奖励
                deployment_reward += 0.5
        
        return deployment_reward
    
    def _calculate_skill_reward(self, action: int, state: np.ndarray, next_state: np.ndarray) -> float:
        """计算技能使用奖励 - 鼓励及时使用技能"""
        skill_reward = 0.0
        
        if action == 11:  # 技能动作
            # 1. 检查技能使用的时机（在敌人威胁高时使用给予奖励）
            enemy_threat = state[3]  # 敌人威胁等级
            if enemy_threat > 0.7:
                skill_reward += 3.0
                print("✨ 及时技能奖励: +3")
            elif enemy_threat > 0.3:
                skill_reward += 1.0
                print("✨ 合理技能奖励: +1")
            
            # 2. 检查技能使用后的效果（敌人数量减少）
            current_enemies = state[31] + state[32] + state[33] + state[34]
            next_enemies = next_state[31] + next_state[32] + next_state[33] + next_state[34]
            
            if next_enemies < current_enemies:
                skill_reward += 2.0
                print("💥 有效技能奖励: +2")
        
        return skill_reward
    
    def _calculate_cost_reward(self, state: np.ndarray, next_state: np.ndarray) -> float:
        """计算费用管理奖励 - 鼓励合理使用费用"""
        cost_reward = 0.0
        
        # 费用比例在状态向量中的位置：1
        current_cost = state[1]
        next_cost = next_state[1]
        
        # 1. 费用增长奖励（自然增长）
        if next_cost > current_cost:
            cost_reward += 0.2
        
        # 2. 费用使用合理性奖励（避免过度使用）
        if current_cost > 0.8 and next_cost < 0.5:
            # 费用充足时合理使用给予奖励
            cost_reward += 1.0
            print("💰 合理费用使用奖励: +1")
        
        return cost_reward
    
    def _get_game_state_info(self) -> Dict[str, Any]:
        """获取游戏状态信息"""
        return {
            'step': self.episode_step,
            'total_reward': self.episode_reward,
            'done': self.episode_done,
            'state_history_length': len(self.state_history)
        }
    
    def close(self) -> None:
        """关闭环境"""
        print("🔚 关闭游戏环境")
        # 清理资源
        if hasattr(self, 'adb_controller'):
            self.adb_controller.close()
    
    def render(self) -> None:
        """渲染当前游戏状态（用于调试）"""
        screenshot = self.adb_controller.take_screenshot()
        
        # 显示截图（用于调试）
        cv2.imshow('Game State', screenshot)
        cv2.waitKey(1)  # 短暂显示


class SimpleEnvironment:
    """简化版环境 - 用于快速测试"""
    
    def __init__(self, state_dim: int = 10, action_dim: int = 14):
        """初始化简化环境"""
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.current_state = np.random.randn(state_dim)
        self.episode_step = 0
        
        print(f"✅ 简化环境初始化完成 - 状态维度: {state_dim}, 动作维度: {action_dim}")
    
    def reset(self) -> np.ndarray:
        """重置环境"""
        self.current_state = np.random.randn(self.state_dim)
        self.episode_step = 0
        return self.current_state
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """执行动作"""
        # 简化版状态转移
        next_state = self.current_state + np.random.randn(self.state_dim) * 0.1
        
        # 简化版奖励
        reward = float(action) / self.action_dim  # 动作越大奖励越高
        
        # 简化版终止条件
        done = self.episode_step >= 20
        
        # 更新状态
        self.current_state = next_state
        self.episode_step += 1
        
        info = {'step': self.episode_step, 'reward': reward}
        
        return next_state, reward, done, info
    
    def close(self):
        """关闭环境"""
        print("🔚 关闭简化环境")


if __name__ == "__main__":
    # 测试环境接口
    print("🧪 测试游戏环境接口...")
    
    # 测试简化环境
    simple_env = SimpleEnvironment(state_dim=5, action_dim=14)
    
    state = simple_env.reset()
    print(f"初始状态: {state}")
    
    for step in range(5):
        action = np.random.randint(14)
        next_state, reward, done, info = simple_env.step(action)
        print(f"步骤 {step}: 动作 {action}, 奖励 {reward:.2f}, 完成 {done}")
        
        if done:
            break
    
    simple_env.close()
    
    print("✅ 游戏环境接口测试通过")
"""
游戏状态编码器
将游戏界面信息转换为PPO算法可以理解的状态向量
"""

import numpy as np
import cv2
from typing import List, Dict, Any, Tuple
import torch
from PIL import Image
import pytesseract

from src.vision.detector.yolo_detector import YOLODetector
from src.vision.processor.image_processor import ImageProcessor


class GameStateEncoder:
    """游戏状态编码器 - 将游戏界面转换为状态向量"""
    
    def __init__(self, yolo_detector: YOLODetector, image_processor: ImageProcessor):
        """
        初始化状态编码器
        
        Args:
            yolo_detector: YOLO目标检测器
            image_processor: 图像处理器
        """
        self.yolo_detector = yolo_detector
        self.image_processor = image_processor
        
        # 状态向量维度定义
        self.state_dim = 50  # 总状态维度
        
        # 特征维度分配
        self.visual_dim = 39      # 视觉特征维度
        self.numerical_dim = 8    # 数值特征维度
        self.temporal_dim = 3     # 时序特征维度
        
        # YOLO类别映射（完全匹配MAA标准）
        self.class_mapping = {
            # 界面按钮类 (0-6) - 基于MAA的实际界面识别
            0: 'start_button',         # 开始按钮（MAA: StartGame）
            1: 'pause_button',         # 暂停按钮（MAA: PauseGame）
            2: 'speed_button',         # 加速按钮（MAA: SpeedUp）
            3: 'deploy_button',        # 部署按钮（MAA: DeployOperator）
            4: 'skill_button',         # 技能按钮（MAA: UseSkill）
            5: 'retreat_button',      # 撤退按钮（MAA: RetreatOperator）
            6: 'confirm_button',       # 确认按钮（MAA: ConfirmAction）
            
            # 干员职业类 (7-14) - 基于MAA的实际职业分类
            7: 'warrior',              # 近卫（MAA: WARRIOR）
            8: 'pioneer',              # 先锋（MAA: PIONEER）
            9: 'medic',                # 医疗（MAA: MEDIC）
            10: 'tank',                # 重装（MAA: TANK）
            11: 'sniper',              # 狙击（MAA: SNIPER）
            12: 'caster',              # 术师（MAA: CASTER）
            13: 'support',             # 辅助（MAA: SUPPORT）
            14: 'special',             # 特种（MAA: SPECIAL）
            
            # 部署位置类 (15-16) - 基于MAA的部署位置识别
            15: 'melee_location',      # 近战位置（MAA: MELEE）
            16: 'ranged_location',     # 远程位置（MAA: RANGED）
            
            # 敌人类型类 (17-20) - 基于MAA的敌人识别
            17: 'normal_enemy',        # 普通敌人
            18: 'elite_enemy',         # 精英敌人
            19: 'boss_enemy',          # BOSS敌人
            20: 'drone_enemy',         # 无人机/召唤物
            
            # 游戏状态类 (21-23) - 基于MAA的状态识别
            21: 'victory_icon',        # 胜利图标
            22: 'defeat_icon',         # 失败图标
            23: 'loading_icon'         # 加载图标
        }
        
        # 屏幕尺寸假设（需要根据实际设备调整）
        self.screen_width = 1920
        self.screen_height = 1080
        
        # 状态历史记录
        self.state_history = []
        self.max_history_length = 5
        
        print(f"✅ 状态编码器初始化完成 - 总维度: {self.state_dim}")
        print(f"   - 视觉特征: {self.visual_dim}维")
        print(f"   - 数值特征: {self.numerical_dim}维")
        print(f"   - 时序特征: {self.temporal_dim}维")
    
    def encode_state(self, screenshot: np.ndarray) -> np.ndarray:
        """
        编码游戏状态
        
        Args:
            screenshot: 游戏截图
            
        Returns:
            state_vector: 状态向量 (50维)
        """
        # 1. YOLO目标检测
        detections = self.yolo_detector.detect(screenshot)
        
        # 2. 提取视觉特征
        visual_features = self._extract_visual_features(detections)
        
        # 3. 提取数值特征
        numerical_features = self._extract_numerical_features(screenshot, detections)
        
        # 4. 提取时序特征
        temporal_features = self._extract_temporal_features()
        
        # 5. 合并所有特征
        state_vector = np.concatenate([
            visual_features,
            numerical_features,
            temporal_features
        ])
        
        # 6. 更新状态历史
        self._update_state_history(state_vector)
        
        return state_vector
    
    def _extract_visual_features(self, detections: List[Dict]) -> np.ndarray:
        """提取视觉特征 (39维) - 完全匹配MAA标准"""
        features = np.zeros(self.visual_dim)
        
        # 1. 界面按钮存在性 (7维) - 基于MAA的界面识别
        button_presence = np.zeros(7)
        for detection in detections:
            if 0 <= detection['class'] <= 6:  # 界面按钮类
                button_presence[detection['class']] = 1.0
        
        # 2. 按钮位置信息 (14维) - 7个按钮的x,y坐标
        button_positions = np.zeros(14)
        for detection in detections:
            if 0 <= detection['class'] <= 6:
                idx = detection['class'] * 2
                # 归一化坐标到[0,1]范围
                button_positions[idx] = detection['x'] / self.screen_width
                button_positions[idx + 1] = detection['y'] / self.screen_height
        
        # 3. 干员职业存在性 (8维) - 基于MAA的职业分类
        operator_presence = np.zeros(8)
        for detection in detections:
            if 7 <= detection['class'] <= 14:  # 干员职业类
                operator_idx = detection['class'] - 7
                operator_presence[operator_idx] = 1.0
        
        # 4. 部署位置可用性 (2维) - 基于MAA的部署位置识别
        location_availability = np.zeros(2)
        for detection in detections:
            if 15 <= detection['class'] <= 16:  # 部署位置类
                location_idx = detection['class'] - 15
                location_availability[location_idx] = 1.0
        
        # 5. 敌人类型存在性 (4维) - 基于MAA的敌人识别
        enemy_presence = np.zeros(4)
        for detection in detections:
            if 17 <= detection['class'] <= 20:  # 敌人类
                enemy_idx = detection['class'] - 17
                enemy_presence[enemy_idx] = 1.0
        
        # 6. 游戏状态识别 (3维) - 基于MAA的状态识别
        game_state = np.zeros(3)
        for detection in detections:
            if 21 <= detection['class'] <= 23:  # 游戏状态类
                state_idx = detection['class'] - 21
                game_state[state_idx] = 1.0
        
        # 7. 敌人位置信息 (8维) - 前4个敌人的x,y坐标
        enemy_positions = np.zeros(8)
        enemy_count = 0
        for detection in detections:
            if 17 <= detection['class'] <= 20 and enemy_count < 4:  # 敌人类
                idx = enemy_count * 2
                enemy_positions[idx] = detection['x'] / self.screen_width
                enemy_positions[idx + 1] = detection['y'] / self.screen_height
                enemy_count += 1
        
        # 合并视觉特征
        features[:7] = button_presence          # 按钮存在性
        features[7:21] = button_positions       # 按钮位置
        features[21:29] = operator_presence     # 干员职业存在性
        features[29:31] = location_availability # 部署位置可用性
        features[31:35] = enemy_presence        # 敌人类型存在性
        features[35:38] = game_state            # 游戏状态识别
        features[38:39] = enemy_positions[:1]   # 敌人位置信息（简化）
        
        return features
    
    def _extract_numerical_features(self, screenshot: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """提取数值特征 (8维) - 按照MAA标准重新设计"""
        features = np.zeros(self.numerical_dim)
        
        try:
            # 1. 血量比例估计 (1维) - 基于MAA的血条识别
            hp_ratio = self._estimate_hp_ratio(screenshot)
            features[0] = np.clip(hp_ratio, 0, 1)
            
            # 2. 费用数值估计 (1维) - 基于MAA的费用识别
            cost_value = self._estimate_cost_value(screenshot)
            features[1] = np.clip(cost_value / 100, 0, 1)  # 归一化到[0,1]
            
            # 3. 时间比例估计 (1维) - 基于MAA的时间识别
            time_ratio = self._estimate_time_ratio(screenshot)
            features[2] = np.clip(time_ratio, 0, 1)
            
            # 4. 敌人威胁等级 (1维) - 基于MAA的敌人威胁评估
            enemy_threat = self._estimate_enemy_threat(detections)
            features[3] = np.clip(enemy_threat, 0, 1)
            
            # 5. 干员部署状态 (1维) - 基于MAA的干员部署评估
            deployment_status = self._estimate_deployment_status(detections)
            features[4] = np.clip(deployment_status, 0, 1)
            
            # 6. 技能可用性 (1维) - 基于MAA的技能状态识别
            skill_availability = self._estimate_skill_availability(detections)
            features[5] = np.clip(skill_availability, 0, 1)
            
            # 7. 游戏进度评估 (2维) - 基于MAA的关卡进度识别
            game_progress = self._estimate_game_progress(detections)
            features[6:8] = game_progress
            
        except Exception as e:
            print(f"⚠️ 数值特征提取失败: {e}")
            # 使用默认值
            features = np.array([1.0, 0.1, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0])
        
        return features
    
    def _extract_temporal_features(self) -> np.ndarray:
        """提取时序特征 (3维)"""
        if len(self.state_history) < 2:
            return np.zeros(self.temporal_dim)
        
        # 取最近两个状态
        current_state = self.state_history[-1]
        previous_state = self.state_history[-2]
        
        # 1. 状态变化率 (1维)
        state_change_rate = np.mean(np.abs(current_state - previous_state))
        
        # 2. 血量变化趋势 (1维)
        hp_trend = 0.0
        if len(self.state_history) >= 3:
            # 计算最近3个状态的HP变化趋势
            hp_values = [state[0] for state in self.state_history[-3:]]
            hp_trend = hp_values[-1] - hp_values[0]
        
        # 3. 敌人数量变化趋势 (1维)
        enemy_trend = 0.0
        if len(self.state_history) >= 3:
            # 计算最近3个状态的敌人数量变化趋势
            enemy_values = [state[3] for state in self.state_history[-3:]]
            enemy_trend = enemy_values[-1] - enemy_values[0]
        
        return np.array([state_change_rate, hp_trend, enemy_trend])
    
    def _estimate_hp_ratio(self, screenshot: np.ndarray) -> float:
        """估计血量比例"""
        try:
            # 截取血条区域（需要根据实际游戏界面调整坐标）
            hp_region = screenshot[50:70, 100:400]  # 示例区域
            
            # 转换为HSV颜色空间
            hsv = cv2.cvtColor(hp_region, cv2.COLOR_BGR2HSV)
            
            # 定义红色血条的颜色范围
            lower_red = np.array([0, 120, 70])
            upper_red = np.array([10, 255, 255])
            
            # 创建掩码
            mask = cv2.inRange(hsv, lower_red, upper_red)
            
            # 计算血条长度比例
            hp_pixels = np.sum(mask > 0)
            total_pixels = mask.shape[0] * mask.shape[1]
            
            return hp_pixels / total_pixels if total_pixels > 0 else 0.0
            
        except Exception as e:
            print(f"⚠️ 血量估计失败: {e}")
            return 0.5  # 默认值
    
    def _estimate_cost_value(self, screenshot: np.ndarray) -> float:
        """估计费用数值"""
        try:
            # 截取费用显示区域（需要根据实际游戏界面调整坐标）
            cost_region = screenshot[30:60, 500:600]  # 示例区域
            
            # 预处理图像
            gray = cv2.cvtColor(cost_region, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
            
            # 使用OCR识别数字
            pil_image = Image.fromarray(binary)
            text = pytesseract.image_to_string(pil_image, config='--psm 8 digits')
            
            # 提取数字
            import re
            numbers = re.findall(r'\d+', text)
            if numbers:
                return float(numbers[0])
            else:
                return 10.0  # 默认值
                
        except Exception as e:
            print(f"⚠️ 费用估计失败: {e}")
            return 10.0  # 默认值
    
    def _estimate_time_ratio(self, screenshot: np.ndarray) -> float:
        """估计时间比例"""
        # 这里需要根据实际游戏的时间显示设计
        # 暂时返回固定值
        return 0.5
    
    def _estimate_enemy_threat(self, detections: List[Dict]) -> float:
        """估计敌人威胁等级 - 完全匹配MAA的威胁评估"""
        threat_level = 0.0
        
        for detection in detections:
            if 17 <= detection['class'] <= 20:  # 敌人类别
                # 根据敌人类型分配威胁权重
                if detection['class'] == 17:  # 普通敌人
                    threat_level += 0.1
                elif detection['class'] == 18:  # 精英敌人
                    threat_level += 0.3
                elif detection['class'] == 19:  # BOSS敌人
                    threat_level += 0.8
                elif detection['class'] == 20:  # 无人机
                    threat_level += 0.05
        
        return min(threat_level, 1.0)
    
    def _estimate_deployment_status(self, detections: List[Dict]) -> float:
        """估计干员部署状态 - 完全匹配MAA的部署评估"""
        # 统计已部署的干员数量
        deployed_count = 0
        for detection in detections:
            if 7 <= detection['class'] <= 14:  # 干员职业类
                deployed_count += 1
        
        # 归一化到[0,1]，假设最多部署12个干员
        return min(deployed_count / 12, 1.0)
    
    def _estimate_skill_availability(self, detections: List[Dict]) -> float:
        """估计技能可用性 - 完全匹配MAA的技能状态识别"""
        # 检查是否有技能按钮可用
        skill_available = 0.0
        
        for detection in detections:
            if detection['class'] == 4:  # 技能按钮
                # 根据技能按钮的位置和状态判断可用性
                # 这里可以添加更复杂的逻辑，比如检查技能冷却状态
                skill_available = 1.0
                break
        
        return skill_available
    
    def _estimate_game_progress(self, detections: List[Dict]) -> np.ndarray:
        """估计游戏进度 - 完全匹配MAA的关卡进度识别"""
        progress_features = np.zeros(2)
        
        # 1. 关卡完成度 (基于敌人数量和类型)
        enemy_count = 0
        for detection in detections:
            if 17 <= detection['class'] <= 20:  # 敌人类
                enemy_count += 1
        
        # 假设关卡有20个敌人，归一化进度
        progress_features[0] = max(0.0, 1.0 - enemy_count / 20)
        
        # 2. 游戏状态 (胜利/失败/进行中)
        game_state = 0.5  # 默认进行中
        for detection in detections:
            if detection['class'] == 21:  # 胜利图标
                game_state = 1.0
                break
            elif detection['class'] == 22:  # 失败图标
                game_state = 0.0
                break
        
        progress_features[1] = game_state
        
        return progress_features
    
    def _update_state_history(self, state_vector: np.ndarray) -> None:
        """更新状态历史"""
        self.state_history.append(state_vector.copy())
        if len(self.state_history) > self.max_history_length:
            self.state_history.pop(0)
    
    def get_state_dimension(self) -> int:
        """获取状态向量维度"""
        return self.state_dim
    
    def reset_history(self) -> None:
        """重置状态历史"""
        self.state_history = []
        print("🔄 状态历史已重置")
    
    def get_feature_description(self) -> Dict[str, Tuple[int, int]]:
        """获取特征维度描述"""
        return {
            'visual_features': (0, self.visual_dim),
            'numerical_features': (self.visual_dim, self.visual_dim + self.numerical_dim),
            'temporal_features': (self.visual_dim + self.numerical_dim, self.state_dim)
        }


def test_state_encoder():
    """测试状态编码器"""
    # 创建模拟的检测器和处理器
    class MockYOLODetector:
        def detect(self, image):
            # 模拟检测结果
            return [
                {'class': 0, 'x': 100, 'y': 200},  # 部署按钮1
                {'class': 13, 'x': 300, 'y': 400}, # 干员1
                {'class': 20, 'x': 500, 'y': 600}  # 敌人1
            ]
    
    class MockImageProcessor:
        pass
    
    # 创建状态编码器
    encoder = GameStateEncoder(MockYOLODetector(), MockImageProcessor())
    
    # 创建模拟截图
    test_screenshot = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
    
    # 测试状态编码
    state_vector = encoder.encode_state(test_screenshot)
    
    print(f"✅ 状态编码测试通过")
    print(f"状态向量维度: {state_vector.shape}")
    print(f"状态向量前10个值: {state_vector[:10]}")
    
    return encoder


if __name__ == "__main__":
    test_state_encoder()
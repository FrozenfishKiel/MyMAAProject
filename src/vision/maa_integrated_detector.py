"""
MAA集成检测器 - 整合四个检测器为PPO提供状态信息
包含：战场匹配器（MAA核心）、干员血条位置检测、干员编队分析、干员头像识别
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from .maa_battlefield_detector import MaaBattlefieldDetector
from .maa_formation_analyzer import MaaFormationAnalyzer
from .maa_operator_detector import MaaOperatorDetector
from .maa_battlefield_matcher import MaaBattlefieldMatcher


class MaaIntegratedDetector:
    """MAA集成检测器 - 为PPO提供完整的状态信息"""
    
    def __init__(self):
        """初始化集成检测器"""
        # 初始化四个检测器
        self.battlefield_matcher = MaaBattlefieldMatcher()    # 战场匹配器（MAA核心）
        self.battlefield_detector = MaaBattlefieldDetector()  # 干员血条位置检测
        self.formation_analyzer = MaaFormationAnalyzer()      # 干员编队分析
        self.operator_detector = MaaOperatorDetector()        # 干员头像识别
        
        print("🎯 MAA集成检测器初始化完成！")
    
    def detect_game_state(self, image: np.ndarray, use_enhanced_detection: bool = True) -> Dict:
        """
        检测游戏状态信息
        
        Args:
            image: 游戏截图图像
            use_enhanced_detection: 是否使用增强检测（MAA BattlefieldMatcher）
            
        Returns:
            游戏状态信息字典
        """
        
        if use_enhanced_detection:
            # 使用MAA BattlefieldMatcher进行增强检测
            return self._detect_game_state_enhanced(image)
        else:
            # 使用原有检测方法
            return self._detect_game_state_basic(image)
    
    def _detect_game_state_enhanced(self, image: np.ndarray) -> Dict:
        """
        使用MAA BattlefieldMatcher进行增强的游戏状态检测
        
        Args:
            image: 游戏截图图像
            
        Returns:
            增强的游戏状态信息字典
        """
        # 使用MAA BattlefieldMatcher构建增强的游戏状态
        enhanced_state = self.battlefield_matcher.build_enhanced_game_state(image)
        
        # 补充其他检测器的信息
        # 1. 检测可用干员列表（编队界面中的干员）
        available_operators = self.formation_analyzer.analyze_formation(image)
        enhanced_state['available_operators'] = available_operators
        
        # 2. 检测干员类别（通过头像识别）
        operator_types = self.operator_detector.detect_operators(image)
        enhanced_state['operator_types'] = operator_types
        
        # 3. 检测可部署位置（简化处理）
        enhanced_state['deployable_positions'] = self._detect_deployable_positions(enhanced_state)
        
        return enhanced_state
    
    def _detect_game_state_basic(self, image: np.ndarray) -> Dict:
        """
        使用原有方法检测游戏状态信息
        
        Args:
            image: 游戏截图图像
            
        Returns:
            基础的游戏状态信息字典
        """
        game_state = {
            'deployed_operators': [],    # 已部署干员
            'available_operators': [],   # 可用干员
            'deployable_positions': [],  # 可部署位置
            'game_status': 'unknown'     # 游戏状态
        }
        
        # 1. 检测已部署干员位置（战场上的干员）
        deployed_operators = self.battlefield_detector.detect_operators(image)
        game_state['deployed_operators'] = deployed_operators
        
        # 2. 检测可用干员列表（编队界面中的干员）
        available_operators = self.formation_analyzer.analyze_formation(image)
        game_state['available_operators'] = available_operators
        
        # 3. 检测干员类别（通过头像识别）
        operator_types = self.operator_detector.detect_operators(image)
        game_state['operator_types'] = operator_types
        
        # 4. 检测游戏状态（胜利/失败/进行中）
        game_state['game_status'] = self._detect_game_status(image)
        
        return game_state
    
    def _detect_game_status(self, image: np.ndarray) -> str:
        """
        检测游戏状态
        
        Args:
            image: 游戏截图
            
        Returns:
            游戏状态字符串
        """
        # 简化的游戏状态检测
        # 实际实现需要更复杂的检测逻辑
        
        # 转换为HSV颜色空间进行颜色检测
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # 检测胜利/失败界面的特征颜色
        # 这里只是示例，实际需要更精确的检测
        
        # 检查图像中是否有明显的胜利/失败标志
        # 可以通过模板匹配或颜色阈值来实现
        
        return 'in_progress'  # 默认返回进行中状态
    
    def _detect_deployable_positions(self, game_state: Dict) -> List[int]:
        """
        检测可部署位置
        
        Args:
            game_state: 游戏状态信息
            
        Returns:
            可部署位置列表
        """
        # 简化处理：假设有8个可部署位置
        # 实际应该根据游戏状态和已部署干员来判断
        deployable_positions = []
        
        # 获取已部署干员的位置
        deployed_positions = [op.get('position', -1) for op in game_state.get('deployed_operators', [])]
        
        # 检查8个位置是否可部署
        for i in range(8):
            if i not in deployed_positions:
                deployable_positions.append(i)
        
        return deployable_positions
    
    def build_ppo_state_vector(self, game_state: Dict, use_enhanced_state: bool = True) -> List[float]:
        """
        构建PPO状态向量
        
        Args:
            game_state: 游戏状态信息
            use_enhanced_state: 是否使用增强状态信息
            
        Returns:
            PPO状态向量
        """
        
        if use_enhanced_state:
            # 使用增强的状态向量构建
            return self._build_enhanced_ppo_state_vector(game_state)
        else:
            # 使用基础的状态向量构建
            return self._build_basic_ppo_state_vector(game_state)
    
    def _build_enhanced_ppo_state_vector(self, game_state: Dict) -> List[float]:
        """
        构建增强的PPO状态向量 - 利用MAA BattlefieldMatcher的详细信息
        
        Args:
            game_state: 增强的游戏状态信息
            
        Returns:
            增强的PPO状态向量
        """
        state_vector = []
        
        # 1. 已部署干员状态（8个位置 × 4个维度 = 32个维度）
        for i in range(8):
            operator = self._get_operator_at_position(game_state, i)
            if operator:
                # 位置被占用
                state_vector.append(1)
                # 干员类型编码
                state_vector.append(self._encode_operator_type(operator.get('type', 'unknown')))
                # 是否可用（借鉴MAA的可用性检测）
                state_vector.append(1.0 if operator.get('available', True) else 0.0)
                # 是否在冷却中（借鉴MAA的冷却检测）
                state_vector.append(1.0 if operator.get('cooling', False) else 0.0)
            else:
                # 空位置
                state_vector.extend([0, 0, 0, 0])
        
        # 2. 游戏资源状态（3个维度）
        # 费用（归一化到0-1）
        costs = game_state.get('costs', 0)
        state_vector.append(min(costs / 100.0, 1.0))
        
        # 击杀数（归一化到0-1）
        kills = game_state.get('kills', 0)
        total_kills = game_state.get('total_kills', 10)
        state_vector.append(min(kills / total_kills, 1.0))
        
        # 游戏进度（基于击杀数）
        state_vector.append(kills / total_kills)
        
        # 3. 游戏状态（3个维度）
        game_status = game_state.get('game_status', 'unknown')
        state_vector.extend(self._encode_enhanced_game_status(game_status))
        
        # 4. 界面状态（2个维度）
        state_vector.append(1.0 if game_state.get('speed_button', False) else 0.0)
        state_vector.append(1.0 if game_state.get('pause_button', False) else 0.0)
        
        return state_vector
    
    def _build_basic_ppo_state_vector(self, game_state: Dict) -> List[float]:
        """
        构建基础的PPO状态向量
        
        Args:
            game_state: 基础的游戏状态信息
            
        Returns:
            基础的PPO状态向量
        """
        state_vector = []
        
        # 1. 已部署干员状态（8个维度）
        deployed_status = [0] * 8  # 假设最多8个部署位置
        for i, operator in enumerate(game_state['deployed_operators']):
            if i < 8:  # 只考虑前8个位置
                deployed_status[i] = 1
        state_vector.extend(deployed_status)
        
        # 2. 可用干员列表（8个维度）- 8类干员
        available_operators = self._get_available_operator_types(game_state)
        state_vector.extend(available_operators)
        
        # 3. 可部署位置（8个维度）- 假设8个可部署格子
        deployable_positions = [1] * 8  # 简化处理，假设所有位置都可部署
        state_vector.extend(deployable_positions)
        
        # 4. 游戏状态（2个维度）- 胜利/失败
        game_status_vector = self._encode_game_status(game_state['game_status'])
        state_vector.extend(game_status_vector)
        
        # 5. 时间/回合信息（1个维度）- 简化处理
        state_vector.append(0)  # 占位符，实际需要根据时间步长计算
        
        return state_vector
    
    def _get_operator_at_position(self, game_state: Dict, position: int) -> Optional[Dict]:
        """
        获取指定位置的干员信息
        
        Args:
            game_state: 游戏状态信息
            position: 位置索引
            
        Returns:
            干员信息字典，如果位置为空则返回None
        """
        for operator in game_state.get('deployed_operators', []):
            if operator.get('position') == position:
                return operator
        return None
    
    def _encode_operator_type(self, operator_type: str) -> float:
        """
        编码干员类型
        
        Args:
            operator_type: 干员类型字符串
            
        Returns:
            干员类型编码（0-1之间的浮点数）
        """
        operator_types = ['vanguard', 'guard', 'sniper', 'defender', 'medic', 'caster', 'supporter', 'specialist']
        try:
            index = operator_types.index(operator_type.lower())
            return index / (len(operator_types) - 1)  # 归一化到0-1
        except ValueError:
            return 0.0  # 未知类型
    
    def _encode_enhanced_game_status(self, status: str) -> List[float]:
        """
        编码增强的游戏状态
        
        Args:
            status: 游戏状态字符串
            
        Returns:
            游戏状态编码（3个维度）
        """
        if status == 'victory':
            return [1.0, 0.0, 0.0]  # 胜利
        elif status == 'defeat':
            return [0.0, 1.0, 0.0]  # 失败
        elif status == 'battle':
            return [0.0, 0.0, 1.0]  # 战斗中
        else:
            return [0.0, 0.0, 0.0]  # 未知
    
    def _get_available_operator_types(self, game_state: Dict) -> List[int]:
        """
        获取可用干员类型向量
        
        Args:
            game_state: 游戏状态信息
            
        Returns:
            8类干员的可用状态
        """
        # 8类干员：先锋、近卫、狙击、重装、医疗、术师、辅助、特种
        operator_types = ['vanguard', 'guard', 'sniper', 'defender', 
                         'medic', 'caster', 'supporter', 'specialist']
        
        available = [0] * 8
        
        # 简化处理：如果有可用干员，则认为所有类型都可用
        if game_state['available_operators']:
            available = [1] * 8
        
        return available
    
    def _encode_game_status(self, status: str) -> List[int]:
        """
        编码游戏状态
        
        Args:
            status: 游戏状态字符串
            
        Returns:
            游戏状态编码向量
        """
        if status == 'victory':
            return [1, 0]  # 胜利
        elif status == 'defeat':
            return [0, 1]  # 失败
        else:
            return [0, 0]  # 进行中
    
    def visualize_detection(self, image: np.ndarray, game_state: Dict) -> np.ndarray:
        """
        可视化所有检测结果
        
        Args:
            image: 原始图像
            game_state: 游戏状态信息
            
        Returns:
            可视化后的图像
        """
        vis_image = image.copy()
        
        # 1. 可视化已部署干员（红色框）
        for operator in game_state['deployed_operators']:
            x, y, w, h = operator['bbox']
            cv2.rectangle(vis_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
            label = f"Deployed: {operator['score']:.2f}"
            cv2.putText(vis_image, label, (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # 2. 可视化可用干员（绿色框）
        for operator in game_state['available_operators']:
            x, y, w, h = operator['name_flag_bbox']
            cv2.rectangle(vis_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            x, y, w, h = operator['avatar_bbox']
            cv2.rectangle(vis_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            label = f"Available: {operator['confidence']:.2f}"
            cv2.putText(vis_image, label, (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # 3. 添加游戏状态信息
        status_text = f"Status: {game_state['game_status']}"
        cv2.putText(vis_image, status_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return vis_image


def test_integrated_detector():
    """测试集成检测器"""
    detector = MaaIntegratedDetector()
    
    print("🎯 MAA集成检测器初始化完成！")
    print("📊 包含的检测器:")
    print("   1. 干员血条位置检测器 (BattlefieldDetector)")
    print("   2. 干员编队分析器 (FormationAnalyzer)")
    print("   3. 干员头像识别器 (OperatorDetector)")
    
    print("\n📝 使用方法:")
    print("   1. 准备游戏截图图像")
    print("   2. 调用 detector.detect_game_state(image)")
    print("   3. 获取完整的游戏状态信息")
    print("   4. 调用 detector.build_ppo_state_vector(game_state) 构建PPO状态向量")


if __name__ == "__main__":
    test_integrated_detector()
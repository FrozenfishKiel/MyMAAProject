"""
干员头像检测模块 - 直接使用MAA的模板图像
"""

import cv2
import numpy as np
from typing import List, Optional, Tuple, Dict
import os


class MaaOperatorDetector:
    """干员头像检测器 - 使用MAA的模板图像"""
    
    def __init__(self, maa_template_dir: str = None):
        """
        初始化干员检测器
        
        Args:
            maa_template_dir: MAA模板目录路径
        """
        if maa_template_dir is None:
            maa_template_dir = r"d:\BiShe\MaaAssistantArknights-dev\resource\template"
        
        self.maa_template_dir = maa_template_dir
        
        # 干员类别映射
        self.operator_classes = {
            "vanguard": "先锋",
            "guard": "近卫", 
            "sniper": "狙击",
            "defender": "重装",
            "medic": "医疗",
            "caster": "术师",
            "supporter": "辅助",
            "specialist": "特种"
        }
        
        # 职业图标映射
        self.role_templates = {
            "vanguard": "BattleOperRolePioneer",  # 先锋
            "guard": "BattleOperRoleWarrior",     # 近卫
            "sniper": "BattleOperRoleSniper",     # 狙击
            "defender": "BattleOperRoleTank",     # 重装
            "medic": "BattleOperRoleMedic",       # 医疗
            "caster": "BattleOperRoleCaster",     # 术师
            "supporter": "BattleOperRoleSupport", # 辅助
            "specialist": "BattleOperRoleSpecial" # 特种
        }
        
        # 模板匹配参数
        self.template_threshold = 0.7
        self.ocr_threshold = 0.6
        
        # 加载MAA模板
        self.maa_templates = self._load_maa_templates()
    
    def _load_maa_templates(self) -> Dict[str, np.ndarray]:
        """加载MAA的模板图像"""
        templates = {}
        
        # 关键模板文件列表
        key_templates = [
            "BattleFormationOCRNameFlag",  # 干员名称标志
            "BattleFormationOCRNameFlagOldVersion",  # 旧版干员名称标志
            "BattleOperRoleCaster",        # 术师
            "BattleOperRoleMedic",         # 医疗
            "BattleOperRoleSniper",        # 狙击
            "BattleOperRoleTank",          # 重装
            "BattleOperRoleWarrior",       # 近卫
            "BattleOperRolePioneer",       # 先锋
            "BattleOperRoleSupport",       # 辅助
            "BattleOperRoleSpecial",       # 特种
            "BattleOperRoleDrone",         # 无人机
            "BattleOpersFlag",             # 干员标志
        ]
        
        for template_name in key_templates:
            template_path = os.path.join(self.maa_template_dir, f"{template_name}.png")
            if os.path.exists(template_path):
                template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
                if template is not None:
                    templates[template_name] = template
                    print(f"✅ 加载模板: {template_name} - 尺寸: {template.shape}")
                else:
                    print(f"❌ 无法读取模板: {template_name}")
            else:
                print(f"⚠️ 模板文件不存在: {template_path}")
        
        return templates
    
    def detect_operator_roles(self, image: np.ndarray) -> List[Dict]:
        """
        检测干员职业图标 - 使用MAA的职业图标模板
        
        Args:
            image: 输入图像（BGR格式）
            
        Returns:
            检测到的干员职业信息列表
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        results = []
        
        # 检测所有职业图标
        for role_name, template_name in self.role_templates.items():
            if template_name in self.maa_templates:
                template = self.maa_templates[template_name]
                
                # 模板匹配
                res = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
                loc = np.where(res >= self.template_threshold)
                
                for pt in zip(*loc[::-1]):  # 切换x,y坐标
                    x, y = pt
                    w, h = template.shape[::-1]
                    
                    # 计算置信度
                    confidence = res[y, x]
                    
                    results.append({
                        'role': role_name,
                        'class_name': self.operator_classes.get(role_name, role_name),
                        'bbox': (x, y, w, h),
                        'confidence': float(confidence),
                        'template': template_name
                    })
        
        # 非极大值抑制，避免重复检测
        results = self._non_max_suppression(results)
        return results
    
    def detect_operator_names(self, image: np.ndarray) -> List[Dict]:
        """
        检测干员名称标志 - 使用MAA的名称标志模板
        
        Args:
            image: 输入图像（BGR格式）
            
        Returns:
            检测到的干员名称标志列表
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        results = []
        
        # 检测名称标志模板
        name_templates = ["BattleFormationOCRNameFlag", "BattleFormationOCRNameFlagOldVersion"]
        
        for template_name in name_templates:
            if template_name in self.maa_templates:
                template = self.maa_templates[template_name]
                
                # 模板匹配
                res = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
                loc = np.where(res >= self.template_threshold)
                
                for pt in zip(*loc[::-1]):
                    x, y = pt
                    w, h = template.shape[::-1]
                    
                    confidence = res[y, x]
                    
                    results.append({
                        'type': 'name_flag',
                        'bbox': (x, y, w, h),
                        'confidence': float(confidence),
                        'template': template_name
                    })
        
        return results
    
    def _non_max_suppression(self, detections: List[Dict], overlap_threshold: float = 0.5) -> List[Dict]:
        """非极大值抑制，去除重叠检测"""
        if len(detections) == 0:
            return []
        
        # 按置信度排序
        detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        suppressed = []
        
        while detections:
            current = detections.pop(0)
            suppressed.append(current)
            
            # 计算与当前检测框的IoU
            x1, y1, w1, h1 = current['bbox']
            area1 = w1 * h1
            
            remaining = []
            for detection in detections:
                x2, y2, w2, h2 = detection['bbox']
                area2 = w2 * h2
                
                # 计算交集
                xx1 = max(x1, x2)
                yy1 = max(y1, y2)
                xx2 = min(x1 + w1, x2 + w2)
                yy2 = min(y1 + h1, y2 + h2)
                
                w = max(0, xx2 - xx1)
                h = max(0, yy2 - yy1)
                inter = w * h
                
                # 计算IoU
                iou = inter / (area1 + area2 - inter)
                
                if iou < overlap_threshold:
                    remaining.append(detection)
            
            detections = remaining
        
        return suppressed
    
    def calculate_avatar_position(self, name_bbox: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        """
        根据干员名称标志位置计算头像位置
        
        Args:
            name_bbox: 干员名称标志边界框 (x, y, w, h)
            
        Returns:
            头像边界框 (x, y, w, h)
        """
        x, y, w, h = name_bbox
        
        # 根据MAA的实现，头像在名称标志的右侧
        avatar_x = x + w + 10  # 向右偏移10像素
        avatar_y = y - 5       # 向上偏移5像素
        avatar_w = 80          # 头像宽度
        avatar_h = 80          # 头像高度
        
        return (avatar_x, avatar_y, avatar_w, avatar_h)
    
    def detect_operators(self, image: np.ndarray) -> List[Dict]:
        """
        综合检测干员信息
        
        Args:
            image: 输入图像（BGR格式）
            
        Returns:
            检测到的干员信息列表
        """
        # 1. 检测干员名称标志
        name_detections = self.detect_operator_names(image)
        
        # 2. 检测干员职业图标
        role_detections = self.detect_operator_roles(image)
        
        operators = []
        
        # 3. 组合名称和职业信息
        for name_det in name_detections:
            # 计算头像位置
            avatar_bbox = self.calculate_avatar_position(name_det['bbox'])
            
            # 提取头像区域
            x, y, w, h = avatar_bbox
            avatar_region = image[y:y+h, x:x+w]
            
            # 查找最近的职业图标
            closest_role = self._find_closest_role(name_det['bbox'], role_detections)
            
            operators.append({
                'type': 'operator',
                'name_flag_bbox': name_det['bbox'],
                'avatar_bbox': avatar_bbox,
                'avatar_region': avatar_region,
                'role': closest_role['role'] if closest_role else 'unknown',
                'class_name': closest_role['class_name'] if closest_role else '未知',
                'confidence': name_det['confidence'],
                'role_confidence': closest_role['confidence'] if closest_role else 0.0
            })
        
        return operators
    
    def _find_closest_role(self, name_bbox: Tuple[int, int, int, int], role_detections: List[Dict]) -> Optional[Dict]:
        """查找距离名称标志最近的职业图标"""
        if not role_detections:
            return None
        
        name_x, name_y, name_w, name_h = name_bbox
        name_center_x = name_x + name_w // 2
        name_center_y = name_y + name_h // 2
        
        closest_role = None
        min_distance = float('inf')
        
        for role_det in role_detections:
            role_x, role_y, role_w, role_h = role_det['bbox']
            role_center_x = role_x + role_w // 2
            role_center_y = role_y + role_h // 2
            
            # 计算欧氏距离
            distance = np.sqrt((name_center_x - role_center_x)**2 + (name_center_y - role_center_y)**2)
            
            if distance < min_distance:
                min_distance = distance
                closest_role = role_det
        
        # 如果距离太远，则认为不是对应的职业图标
        if min_distance > 100:  # 距离阈值
            return None
        
        return closest_role
    
    def visualize_detection(self, image: np.ndarray, operators: List[Dict]) -> np.ndarray:
        """
        可视化检测结果
        
        Args:
            image: 原始图像
            operators: 检测到的干员列表
            
        Returns:
            可视化后的图像
        """
        vis_image = image.copy()
        
        for operator in operators:
            # 绘制名称标志边界框（绿色）
            x, y, w, h = operator['name_flag_bbox']
            cv2.rectangle(vis_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # 绘制头像边界框（红色）
            x, y, w, h = operator['avatar_bbox']
            cv2.rectangle(vis_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
            
            # 添加标签
            label = f"{operator['class_name']} ({operator['confidence']:.2f})"
            cv2.putText(vis_image, label, (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return vis_image


def test_detector():
    """测试干员检测器"""
    detector = MaaOperatorDetector()
    
    print("🎯 MAA干员检测器初始化完成！")
    print(f"📊 已加载模板数量: {len(detector.maa_templates)}")
    
    # 显示已加载的模板
    for template_name in detector.maa_templates.keys():
        print(f"   - {template_name}")
    
    print("\n📝 使用方法:")
    print("   1. 准备游戏截图图像")
    print("   2. 调用 detector.detect_operators(image)")
    print("   3. 获取干员位置和类别信息")


if __name__ == "__main__":
    test_detector()
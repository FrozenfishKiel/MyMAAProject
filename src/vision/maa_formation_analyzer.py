"""
干员编队分析器 - 基于MAA的BattleFormationAnalyzer移植
使用模板匹配识别编队界面中的干员名称和头像
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import os


class MaaFormationAnalyzer:
    """干员编队分析器 - 基于MAA的模板匹配实现"""
    
    def __init__(self, maa_template_dir: str = None):
        """
        初始化编队分析器
        
        Args:
            maa_template_dir: MAA模板目录路径
        """
        if maa_template_dir is None:
            maa_template_dir = r"d:\BiShe\MaaAssistantArknights-dev\resource\template"
        
        self.maa_template_dir = maa_template_dir
        self.template_threshold = 0.7  # 模板匹配阈值
        
        # 关键模板文件
        self.key_templates = [
            "BattleFormationOCRNameFlag",  # 干员名称标志（新版）
            "BattleFormationOCRNameFlagOldVersion",  # 干员名称标志（旧版）
        ]
        
        # 加载MAA模板
        self.maa_templates = self._load_maa_templates()
        
        # 头像位置偏移参数（参考MAA的实现）
        self.avatar_offset_x = 10  # 向右偏移像素
        self.avatar_offset_y = -5  # 向上偏移像素
        self.avatar_width = 80     # 头像宽度
        self.avatar_height = 80    # 头像高度
    
    def _load_maa_templates(self) -> Dict[str, np.ndarray]:
        """加载MAA的模板图像"""
        templates = {}
        
        for template_name in self.key_templates:
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
    
    def analyze_formation(self, image: np.ndarray) -> List[Dict]:
        """
        分析编队界面中的干员信息
        
        Args:
            image: 输入图像（BGR格式）
            
        Returns:
            检测到的干员信息列表
        """
        # 转换为灰度图像
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 先尝试新版UI布局
        name_detections = self._detect_name_flags(gray, "BattleFormationOCRNameFlag")
        
        # 如果新版没有检测到结果，尝试旧版UI布局
        if not name_detections:
            name_detections = self._detect_name_flags(gray, "BattleFormationOCRNameFlagOldVersion")
        
        # 处理检测结果
        operators = []
        for name_det in name_detections:
            operator_info = self._process_name_detection(name_det, image)
            if operator_info:
                operators.append(operator_info)
        
        return operators
    
    def _detect_name_flags(self, gray_image: np.ndarray, template_name: str) -> List[Dict]:
        """
        检测干员名称标志
        
        Args:
            gray_image: 灰度图像
            template_name: 模板名称
            
        Returns:
            检测到的名称标志列表
        """
        if template_name not in self.maa_templates:
            return []
        
        template = self.maa_templates[template_name]
        
        # 模板匹配
        res = cv2.matchTemplate(gray_image, template, cv2.TM_CCOEFF_NORMED)
        loc = np.where(res >= self.template_threshold)
        
        detections = []
        for pt in zip(*loc[::-1]):  # 切换x,y坐标
            x, y = pt
            w, h = template.shape[::-1]
            
            confidence = res[y, x]
            
            detections.append({
                'bbox': (x, y, w, h),
                'confidence': float(confidence),
                'template': template_name
            })
        
        # 非极大值抑制，避免重复检测
        filtered_detections = self._non_max_suppression(detections)
        
        return filtered_detections
    
    def _process_name_detection(self, name_detection: Dict, image: np.ndarray) -> Optional[Dict]:
        """
        处理名称检测结果，提取干员头像
        
        Args:
            name_detection: 名称检测结果
            image: 原始图像
            
        Returns:
            干员信息字典
        """
        name_bbox = name_detection['bbox']
        
        # 计算头像位置（参考MAA的实现）
        avatar_bbox = self._calculate_avatar_position(name_bbox)
        
        # 提取头像区域
        x, y, w, h = avatar_bbox
        
        # 检查边界是否有效
        if (x >= 0 and y >= 0 and 
            x + w <= image.shape[1] and 
            y + h <= image.shape[0]):
            
            avatar_region = image[y:y+h, x:x+w]
            
            return {
                'type': 'operator',
                'name_flag_bbox': name_bbox,
                'avatar_bbox': avatar_bbox,
                'avatar_region': avatar_region,
                'confidence': name_detection['confidence'],
                'template_used': name_detection['template']
            }
        
        return None
    
    def _calculate_avatar_position(self, name_bbox: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        """
        根据干员名称标志位置计算头像位置
        
        Args:
            name_bbox: 干员名称标志边界框 (x, y, w, h)
            
        Returns:
            头像边界框 (x, y, w, h)
        """
        x, y, w, h = name_bbox
        
        # 根据MAA的实现，头像在名称标志的右侧
        avatar_x = x + w + self.avatar_offset_x
        avatar_y = y + self.avatar_offset_y
        
        return (avatar_x, avatar_y, self.avatar_width, self.avatar_height)
    
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
    
    def visualize_formation(self, image: np.ndarray, operators: List[Dict]) -> np.ndarray:
        """
        可视化编队分析结果
        
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
            
            # 添加置信度标签
            label = f"Operator: {operator['confidence']:.2f}"
            cv2.putText(vis_image, label, (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return vis_image


def test_formation_analyzer():
    """测试编队分析器"""
    analyzer = MaaFormationAnalyzer()
    
    print("🎯 MAA编队分析器初始化完成！")
    print(f"📊 已加载模板数量: {len(analyzer.maa_templates)}")
    
    # 显示已加载的模板
    for template_name in analyzer.maa_templates.keys():
        print(f"   - {template_name}")
    
    print("\n📝 使用方法:")
    print("   1. 准备游戏编队界面截图")
    print("   2. 调用 analyzer.analyze_formation(image)")
    print("   3. 获取干员名称和头像位置信息")


if __name__ == "__main__":
    test_formation_analyzer()
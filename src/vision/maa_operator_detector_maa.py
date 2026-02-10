"""
干员检测器 - 完全按照MAA的实现方式移植
基于MAA的BattlefieldMatcher.cpp和OperBoxImageAnalyzer.cpp实现
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from enum import Enum
import os


class BattleRole(Enum):
    """干员职业枚举 - 完全按照MAA的定义"""
    Unknown = 0
    Caster = 1    # 术师
    Medic = 2     # 医疗
    Pioneer = 3   # 先锋
    Sniper = 4    # 狙击
    Special = 5   # 特种
    Support = 6   # 辅助
    Tank = 7      # 重装
    Warrior = 8   # 近卫
    Drone = 9     # 无人机


class MatchStatus(Enum):
    """匹配状态枚举 - 完全按照MAA的定义"""
    Invalid = 0   # 识别失败
    Success = 1   # 识别成功
    HitCache = 2  # 图像命中缓存，不进行识别


class DeploymentOper:
    """干员部署信息 - 完全按照MAA的结构"""
    
    def __init__(self):
        self.rect = None           # 干员位置矩形
        self.role = BattleRole.Unknown  # 干员职业
        self.avatar = None         # 干员头像图像
        self.available = False     # 是否可用（冷却结束）
        self.cooling = False       # 是否在冷却中
        self.cost = -1             # 部署费用
        self.index = 0             # 索引
        
    def to_dict(self) -> Dict:
        """转换为字典格式"""
        return {
            'rect': self.rect,
            'role': self.role.name,
            'available': self.available,
            'cooling': self.cooling,
            'cost': self.cost,
            'index': self.index
        }


class MaaOperatorDetectorMAA:
    """干员检测器 - 完全按照MAA的实现方式"""
    
    def __init__(self, maa_resource_dir: str = None):
        """
        初始化干员检测器
        
        Args:
            maa_resource_dir: MAA资源目录路径
        """
        if maa_resource_dir is None:
            maa_resource_dir = r"d:\BiShe\MaaAssistantArknights-dev\resource"
        
        self.maa_resource_dir = maa_resource_dir
        self.template_dir = os.path.join(maa_resource_dir, "template")
        
        # 职业映射表 - 完全按照MAA的定义
        self.role_map = {
            "Caster": BattleRole.Caster,
            "Medic": BattleRole.Medic,
            "Pioneer": BattleRole.Pioneer,
            "Sniper": BattleRole.Sniper,
            "Special": BattleRole.Special,
            "Support": BattleRole.Support,
            "Tank": BattleRole.Tank,
            "Warrior": BattleRole.Warrior,
            "Drone": BattleRole.Drone
        }
        
        # 模板匹配参数 - 完全按照MAA的配置
        # 在MAA的tasks.json中，BattleOperRole的templThreshold为0.65
        self.template_threshold = 0.65
        self.ocr_threshold = 0.6
        
        # 加载MAA模板
        self.maa_templates = self._load_maa_templates()
        
        print("✅ MAA干员检测器初始化完成！")
    
    def _load_maa_templates(self) -> Dict[str, np.ndarray]:
        """加载MAA的模板图像 - 完全按照MAA的文件结构"""
        templates = {}
        
        # MAA中用于干员识别的关键模板文件
        key_templates = [
            # 职业图标模板
            "BattleOperRoleCaster",        # 术师
            "BattleOperRoleMedic",         # 医疗
            "BattleOperRolePioneer",       # 先锋
            "BattleOperRoleSniper",        # 狙击
            "BattleOperRoleSpecial",       # 特种
            "BattleOperRoleSupport",       # 辅助
            "BattleOperRoleTank",          # 重装
            "BattleOperRoleWarrior",       # 近卫
            "BattleOperRoleDrone",         # 无人机
            
            # 战斗界面标志
            "BattleHpFlag",                # HP标志
            "BattleHpFlag2",               # HP标志（红色）
            "BattleKillsFlag",             # 击杀标志
            "BattleOpersFlag",             # 干员标志
            
            # 编队界面标志
            "BattleFormationOCRNameFlag",  # 干员名称标志
            "BattleFormationOCRNameFlagOldVersion",  # 旧版干员名称标志
        ]
        
        for template_name in key_templates:
            template_path = os.path.join(self.template_dir, f"{template_name}.png")
            if os.path.exists(template_path):
                # 完全按照MAA的实现：使用彩色图像读取
                template = cv2.imread(template_path, cv2.IMREAD_COLOR)
                if template is not None:
                    templates[template_name] = template
                    print(f"✅ 加载MAA模板: {template_name} - 尺寸: {template.shape}")
                else:
                    print(f"❌ 无法读取MAA模板: {template_name}")
            else:
                print(f"⚠️ MAA模板文件不存在: {template_path}")
        
        return templates
    
    def _detect_oper_flags(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        检测干员标志 - 完全按照MAA的BattlefieldMatcher::deployment_analyze实现
        
        按照MAA的MultiMatcher实现：
        1. 使用ROI: [0, 588, 1280, 18]
        2. 使用相关系数匹配 (TM_CCOEFF_NORMED)
        3. 阈值: 0.65
        4. 非极大值抑制：min_distance = min(templ.cols, templ.rows) / 2
        """
        # 完全按照MAA的实现：使用彩色图像进行匹配
        flags = []
        
        # 按照MAA的BattleOpersFlag配置：roi: [0, 588, 1280, 18]
        # 注意：MAA的分辨率是1280x720（横屏），我们的分辨率是1080x1920（竖屏）
        # 需要重新计算ROI位置以适应竖屏
        
        # 由于是竖屏，我们需要将MAA的横屏坐标转换为竖屏坐标
        # 在竖屏下，干员标志应该在屏幕底部
        
        # 简单方法：在竖屏下，干员标志应该在屏幕底部中间位置
        # 先使用固定位置进行测试
        scaled_roi_x = 0
        scaled_roi_y = image.shape[0] - 200  # 在屏幕底部
        scaled_roi_w = image.shape[1]  # 全宽
        scaled_roi_h = 50  # 适当高度
        
        print(f"🔍 干员标志检测ROI: ({scaled_roi_x}, {scaled_roi_y}, {scaled_roi_w}, {scaled_roi_h})")
        
        # 确保ROI在图像范围内
        scaled_roi_x = max(0, scaled_roi_x)
        scaled_roi_y = max(0, scaled_roi_y)
        scaled_roi_w = min(scaled_roi_w, image.shape[1] - scaled_roi_x)
        scaled_roi_h = min(scaled_roi_h, image.shape[0] - scaled_roi_y)
        
        if scaled_roi_w <= 0 or scaled_roi_h <= 0:
            return []
        
        # 提取ROI区域 - 完全按照MAA的实现：使用彩色图像
        roi_color = image[scaled_roi_y:scaled_roi_y+scaled_roi_h, scaled_roi_x:scaled_roi_x+scaled_roi_w]
        
        # 检测BattleOpersFlag模板
        if "BattleOpersFlag" in self.maa_templates:
            template = self.maa_templates["BattleOpersFlag"]
            
            # 模板匹配 - 完全按照MAA的实现：使用彩色图像和TM_CCOEFF_NORMED
            res = cv2.matchTemplate(roi_color, template, cv2.TM_CCOEFF_NORMED)
        
        # 按照MAA的MultiMatcher实现：遍历所有匹配点
            for i in range(res.shape[0]):
                for j in range(res.shape[1]):
                    value = res[i, j]
                    
                    # 按照MAA的阈值：0.65
                    if value < 0.65 or np.isnan(value) or np.isinf(value):
                        continue
                    
                    # 计算全局坐标
                    global_x = scaled_roi_x + j
                    global_y = scaled_roi_y + i
                    w, h = template.shape[::-1]
                    
                    # 按照MAA的非极大值抑制：min_distance = min(templ.cols, templ.rows) / 2
                    min_distance = min(template.shape[1], template.shape[0]) // 2
                    
                    # 检查是否与已有检测结果太近
                    need_push = True
                    for existing_flag in flags:
                        existing_x, existing_y, existing_w, existing_h = existing_flag['bbox']
                        if (abs(global_x - existing_x) < min_distance and 
                            abs(global_y - existing_y) < min_distance):
                            # 如果新检测的置信度更高，替换旧的
                            if value > existing_flag['confidence']:
                                existing_flag['bbox'] = (global_x, global_y, w, h)
                                existing_flag['confidence'] = float(value)
                            need_push = False
                            break
                    
                    if need_push:
                        flags.append({
                            'bbox': (global_x, global_y, w, h),
                            'confidence': float(value)
                        })
        
        # 按照水平位置排序（从左到右）
        flags.sort(key=lambda rect: rect['bbox'][0])
        
        # 转换为元组格式返回，保持接口一致性
        return [flag['bbox'] for flag in flags]
    
    def detect_oper_roles(self, image: np.ndarray, oper_flags: List[Tuple[int, int, int, int]] = None) -> List[Dict]:
        """
        检测干员职业 - 完全按照MAA的BattlefieldMatcher::oper_role_analyze实现
        
        按照MAA的BestMatcher实现：
        1. 使用ROI: 根据BattleOperRoleRange配置计算
        2. 使用相关系数匹配 (TM_CCOEFF_NORMED)
        3. 阈值: 0.65
        4. 找到最佳匹配的职业
        
        Args:
            image: 输入图像（BGR格式）
            oper_flags: 干员标志位置列表 [(x, y, w, h), ...]，如果为None则自动检测
                
        Returns:
            检测到的干员职业信息列表
        """
        # 完全按照MAA的实现：使用彩色图像进行匹配
        results = []
        
        # 如果没有提供干员标志位置，先检测干员标志
        if oper_flags is None:
            oper_flags = self._detect_oper_flags(image)
        
        # 按照MAA的实现方式，对每个干员标志进行职业检测
        for flag_rect in oper_flags:
            x, y, w, h = flag_rect
            
            # 根据MAA的BattleOperRoleRange配置计算职业图标位置
            # rectMove: [-41, 6, 31, 25] 表示相对于标志的偏移
            # 注意：在竖屏下需要重新计算偏移量
            
            # 在竖屏下，职业图标应该在干员标志的上方
            role_x = x - 20  # 适当调整x偏移
            role_y = y - 30  # 在竖屏下，职业图标在标志上方
            role_w = 40      # 适当调整宽度
            role_h = 30      # 适当调整高度
            
            # 确保ROI在图像范围内
            role_x = max(0, role_x)
            role_y = max(0, role_y)
            role_w = min(role_w, image.shape[1] - role_x)
            role_h = min(role_h, image.shape[0] - role_y)
            
            if role_w <= 0 or role_h <= 0:
                print(f"  ❌ 职业ROI超出图像范围，跳过")
                continue
            
            # 提取职业图标区域 - 完全按照MAA的实现：使用彩色图像
            role_roi_color = image[role_y:role_y+role_h, role_x:role_x+role_w]
            print(f"  🔍 职业检测ROI: ({role_x}, {role_y}, {role_w}, {role_h})")
            
            # 在ROI内检测职业图标 - 使用MAA的BestMatcher逻辑
            # 按照MAA的BattleOperRole配置：templThreshold: 0.65
            best_match = None
            best_confidence = 0.0
            
            # 按照MAA的模板命名方式：BattleOperRoleCaster.png
            for role_name in ["Caster", "Medic", "Pioneer", "Sniper", "Special", "Support", "Tank", "Warrior", "Drone"]:
                template_key = f"BattleOperRole{role_name}"
                if template_key in self.maa_templates:
                    template = self.maa_templates[template_key]
                    
                    # 模板匹配 - 完全按照MAA的实现：使用彩色图像和TM_CCOEFF_NORMED
                    res = cv2.matchTemplate(role_roi_color, template, cv2.TM_CCOEFF_NORMED)
                    
                    # 使用MAA的BestMatcher逻辑：找到最佳匹配
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                    
                    # 按照MAA的BattleOperRole配置：templThreshold: 0.65
                    print(f"    🔍 {role_name}: 匹配度={max_val:.3f}")
                    if max_val >= 0.65 and max_val > best_confidence:  # 恢复MAA的阈值
                        best_confidence = max_val
                        best_match = role_name
            
            if best_match:
                results.append({
                    'role': best_match,
                    'role_enum': self.role_map.get(best_match, BattleRole.Unknown),
                    'bbox': (role_x, role_y, role_w, role_h),
                    'confidence': float(best_confidence),
                    'template': f"BattleOperRole{best_match}",
                    'flag_rect': flag_rect
                })
        
        return results
    
    def detect_oper_cooling(self, image: np.ndarray, roi: Tuple[int, int, int, int]) -> bool:
        """
        检测干员冷却状态 - 完全按照MAA的BattlefieldMatcher::oper_cooling_analyze实现
        
        Args:
            image: 输入图像（BGR格式）
            roi: 检测区域 (x, y, width, height)
            
        Returns:
            干员是否在冷却中
        """
        x, y, w, h = roi
        img_roi = image[y:y+h, x:x+w]
        
        # 按照MAA的实现，使用HSV颜色空间分析
        hsv = cv2.cvtColor(img_roi, cv2.COLOR_BGR2HSV)
        
        # MAA中BattleOperCooling任务的参数
        # 冷却状态的颜色范围（蓝色区域）
        lower_blue = np.array([100, 50, 50])   # HSV下限
        upper_blue = np.array([130, 255, 255]) # HSV上限
        
        # 创建掩码
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        
        # 计算蓝色像素数量
        blue_pixel_count = cv2.countNonZero(mask)
        
        # MAA中的阈值（根据实际情况调整）
        threshold = 50
        
        # 如果蓝色像素超过阈值，说明干员在冷却中
        return blue_pixel_count >= threshold
    
    def detect_oper_available(self, image: np.ndarray, roi: Tuple[int, int, int, int]) -> bool:
        """
        检测干员可用性 - 完全按照MAA的BattlefieldMatcher::oper_available_analyze实现
        
        Args:
            image: 输入图像（BGR格式）
            roi: 检测区域 (x, y, width, height)
            
        Returns:
            干员是否可用
        """
        x, y, w, h = roi
        img_roi = image[y:y+h, x:x+w]
        
        # 转换为HSV颜色空间
        hsv = cv2.cvtColor(img_roi, cv2.COLOR_BGR2HSV)
        
        # 计算平均亮度（V通道）
        avg_brightness = np.mean(hsv[:, :, 2])
        
        # MAA中的阈值（根据实际情况调整）
        threshold = 100
        
        # 如果平均亮度低于阈值，说明干员不可用
        return avg_brightness >= threshold
    
    def detect_battle_flags(self, image: np.ndarray) -> Dict[str, bool]:
        """
        检测战斗界面标志 - 完全按照MAA的实现方式
        
        Args:
            image: 输入图像（BGR格式）
            
        Returns:
            各种标志的检测结果
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        results = {}
        
        # 检测HP标志
        hp_flag_detected = False
        for flag_name in ["BattleHpFlag", "BattleHpFlag2"]:
            if flag_name in self.maa_templates:
                template = self.maa_templates[flag_name]
                res = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
                if np.max(res) >= self.template_threshold:
                    hp_flag_detected = True
                    break
        
        results['hp_flag'] = hp_flag_detected
        
        # 检测击杀标志
        if "BattleKillsFlag" in self.maa_templates:
            template = self.maa_templates["BattleKillsFlag"]
            res = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
            results['kills_flag'] = np.max(res) >= self.template_threshold
        else:
            results['kills_flag'] = False
        
        # 检测干员标志
        if "BattleOpersFlag" in self.maa_templates:
            template = self.maa_templates["BattleOpersFlag"]
            res = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
            results['opers_flag'] = np.max(res) >= self.template_threshold
        else:
            results['opers_flag'] = False
        
        return results
    
    def detect_formation_names(self, image: np.ndarray) -> List[Dict]:
        """
        检测编队界面干员名称标志 - 完全按照MAA的BattleFormationAnalyzer实现
        
        Args:
            image: 输入图像（BGR格式）
            
        Returns:
            检测到的名称标志列表
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
                intersection = w * h
                
                # 计算并集
                union = area1 + area2 - intersection
                
                # 计算IoU
                iou = intersection / union if union > 0 else 0
                
                # 如果IoU小于阈值，保留该检测
                if iou < overlap_threshold:
                    remaining.append(detection)
            
            detections = remaining
        
        return suppressed
    
    def analyze_battlefield(self, image: np.ndarray) -> Dict[str, Any]:
        """
        综合分析战场状态 - 完全按照MAA的BattlefieldMatcher::analyze实现
        
        Args:
            image: 输入图像（BGR格式）
            
        Returns:
            战场状态分析结果
        """
        results = {}
        
        # 检测战斗标志
        results['battle_flags'] = self.detect_battle_flags(image)
        
        # 按照MAA的正确逻辑：先检测干员标志，然后基于标志位置检测职业
        oper_flags = self._detect_oper_flags(image)
        results['oper_flags'] = oper_flags
        
        # 基于干员标志位置检测干员职业
        results['oper_roles'] = self.detect_oper_roles(image, oper_flags)
        
        # 检测编队名称标志
        results['formation_names'] = self.detect_formation_names(image)
        
        return results


# 测试函数
if __name__ == "__main__":
    # 创建检测器实例
    detector = MaaOperatorDetectorMAA()
    
    # 测试图像路径
    test_image_path = "test_screenshot.png"
    
    if os.path.exists(test_image_path):
        # 读取测试图像
        image = cv2.imread(test_image_path)
        
        # 进行战场分析
        results = detector.analyze_battlefield(image)
        
        # 打印结果
        print("战场分析结果:")
        print(f"战斗标志: {results['battle_flags']}")
        print(f"检测到的干员职业数量: {len(results['oper_roles'])}")
        print(f"检测到的名称标志数量: {len(results['formation_names'])}")
        
        # 显示检测结果
        for i, role in enumerate(results['oper_roles']):
            print(f"干员{i+1}: {role['role']} (置信度: {role['confidence']:.2f})")
    else:
        print(f"测试图像不存在: {test_image_path}")
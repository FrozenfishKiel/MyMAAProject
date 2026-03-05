from __future__ import annotations

import cv2
import numpy as np
import random
from typing import Optional, Tuple


def find_green_highlight(image: np.ndarray) -> Optional[Tuple[int, int]]:
    """
    找到绿色高亮区域的中心点
    
    Args:
        image: BGR图像（MaaFramework截图）
    
    Returns:
        (x, y): 绿色高亮区域的中心点坐标，如果未找到则返回None
    """
    # 转换为HSV颜色空间
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # 提取绿色通道
    # 绿色的H值范围：45-75（进一步缩小范围，避免提取"自动回复"标签）
    # 绿色的S值范围：120-255（进一步提高饱和度下限）
    # 绿色的V值范围：120-255（进一步提高明度下限）
    lower_green = np.array([45, 120, 120])
    upper_green = np.array([75, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # 使用形态学操作连接分散的绿色像素
    # 膨胀操作：连接相邻的绿色像素
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.dilate(mask, kernel, iterations=2)
    
    # 闭运算：填充小的空洞
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # 找到所有连通区域
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    
    if num_labels > 1:  # 背景是0，所以至少要有1个前景区域
        # 计算每个连通区域的面积
        areas = []
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            areas.append((i, area))
        
        # 按面积从大到小排序
        areas.sort(key=lambda x: x[1], reverse=True)
        
        # 选择面积最大的前3个区域中的随机一个（避免总是选择同一个位置）
        top_n = min(3, len(areas))
        top_areas = areas[:top_n]
        
        # 从面积最大的前3个区域中随机选择一个
        selected_label, selected_area = random.choice(top_areas)
        
        # 计算选择的连通区域的中心点
        center_x = int(centroids[selected_label, 0])
        center_y = int(centroids[selected_label, 1])
        return (center_x, center_y)
    else:
        return None


def find_green_highlight_with_debug(image: np.ndarray) -> Tuple[Optional[Tuple[int, int]], np.ndarray]:
    """
    找到绿色高亮区域的中心点（带调试信息）
    
    Args:
        image: BGR图像（MaaFramework截图）
    
    Returns:
        ((x, y), mask): 绿色高亮区域的中心点坐标和二值化后的mask图像
    """
    # 转换为HSV颜色空间
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # 提取绿色通道
    # 绿色的H值范围：40-80（缩小范围，避免提取星熊发色）
    # 绿色的S值范围：100-255（提高饱和度下限，避免提取低饱和度的颜色）
    # 绿色的V值范围：100-255（提高明度下限，避免提取暗色）
    lower_green = np.array([40, 100, 100])
    upper_green = np.array([80, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # 使用形态学操作连接分散的绿色像素
    # 膨胀操作：连接相邻的绿色像素
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.dilate(mask, kernel, iterations=2)
    
    # 闭运算：填充小的空洞
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # 找到所有连通区域
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    
    if num_labels > 1:  # 背景是0，所以至少要有1个前景区域
        # 随机选择一个连通区域（排除背景）
        random_label = random.randint(1, num_labels - 1)
        
        # 计算随机选择的连通区域的中心点
        center_x = int(centroids[random_label, 0])
        center_y = int(centroids[random_label, 1])
        return ((center_x, center_y), mask)
    else:
        return (None, mask)

from __future__ import annotations

import cv2
import numpy as np
import random
from pathlib import Path


def find_green_highlight(image: np.ndarray):
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


def find_green_highlight_with_debug(image: np.ndarray):
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
        import random
        random_label = random.randint(1, num_labels - 1)
        
        # 计算随机选择的连通区域的中心点
        center_x = int(centroids[random_label, 0])
        center_y = int(centroids[random_label, 1])
        return ((center_x, center_y), mask)
    else:
        return (None, mask)


def test_green_highlight():
    """
    测试绿色高亮区域识别
    """
    # 读取截图
    image_path = str(Path(__file__).parent / "output" / "screenshot.png")
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"无法读取截图: {image_path}")
        return
    
    print(f"截图尺寸: {image.shape}")
    print(f"截图类型: {image.dtype}")
    
    # 找到绿色高亮区域的中心点
    print("\n正在识别绿色高亮区域...")
    green_center = find_green_highlight(image)
    
    if green_center:
        print(f"✅ 成功找到绿色高亮区域！")
        print(f"中心点坐标: {green_center}")
        print(f"X坐标: {green_center[0]}")
        print(f"Y坐标: {green_center[1]}")
    else:
        print(f"❌ 未找到绿色高亮区域！")
        print("可能的原因：")
        print("1. 截图中没有绿色高亮区域")
        print("2. 绿色高亮区域的颜色不在HSV范围内")
        print("3. 绿色高亮区域的面积太小")
    
    # 使用带调试信息的函数
    print("\n使用带调试信息的函数...")
    green_center_debug, mask = find_green_highlight_with_debug(image)
    
    if green_center_debug:
        print(f"✅ 成功找到绿色高亮区域！")
        print(f"中心点坐标: {green_center_debug}")
    else:
        print(f"❌ 未找到绿色高亮区域！")
    
    # 保存mask图像（用于调试）
    mask_path = "output/mask_debug.png"
    cv2.imwrite(mask_path, mask)
    print(f"\nMask图像已保存到: {mask_path}")
    
    # 在原图上标记绿色高亮区域的中心点
    if green_center:
        image_debug = image.copy()
        cv2.circle(image_debug, green_center, 10, (0, 0, 255), -1)
        cv2.circle(image_debug, green_center, 15, (0, 0, 255), 2)
        cv2.putText(image_debug, f"({green_center[0]}, {green_center[1]})", 
                    (green_center[0] + 20, green_center[1] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        debug_image_path = "output/debug_image.png"
        cv2.imwrite(debug_image_path, image_debug)
        print(f"调试图像已保存到: {debug_image_path}")


if __name__ == "__main__":
    test_green_highlight()

"""
诊断绿色高亮区域识别

功能：
1. 查看截图
2. 可视化绿色区域提取结果
3. 调整HSV范围
4. 检查游戏状态
"""

import sys
from pathlib import Path
import cv2
import numpy as np

# 添加项目路径到sys.path
ROOT = Path(__file__).resolve().parent.parent


def diagnose_green_highlight():
    """
    诊断绿色高亮区域识别
    """
    print("=" * 70)
    print("🔍 诊断绿色高亮区域识别")
    print("=" * 70)
    print()
    
    # 读取截图
    screenshot_path = ROOT / "tests" / "output" / "action2_drag_screenshot.png"
    
    if not screenshot_path.exists():
        print(f"❌ 截图文件不存在: {screenshot_path}")
        print()
        print("💡 请先运行 test_actions.py 生成截图")
        return
    
    image = cv2.imread(str(screenshot_path))
    if image is None:
        print(f"❌ 无法读取截图: {screenshot_path}")
        return
    
    print(f"✅ 截图读取成功，图像形状: {image.shape}")
    print()
    
    # 转换为HSV颜色空间
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # 测试不同的绿色范围
    test_ranges = [
        {
            "name": "原始范围（45-75）",
            "lower": np.array([45, 120, 120]),
            "upper": np.array([75, 255, 255])
        },
        {
            "name": "宽松范围（40-80）",
            "lower": np.array([40, 100, 100]),
            "upper": np.array([80, 255, 255])
        },
        {
            "name": "更宽松范围（35-85）",
            "lower": np.array([35, 80, 80]),
            "upper": np.array([85, 255, 255])
        },
        {
            "name": "最宽松范围（30-90）",
            "lower": np.array([30, 50, 50]),
            "upper": np.array([90, 255, 255])
        }
    ]
    
    for i, test_range in enumerate(test_ranges):
        print(f"【测试 {i + 1}】{test_range['name']}")
        print(f"   - H范围: {test_range['lower'][0]}-{test_range['upper'][0]}")
        print(f"   - S范围: {test_range['lower'][1]}-{test_range['upper'][1]}")
        print(f"   - V范围: {test_range['lower'][2]}-{test_range['upper'][2]}")
        
        # 提取绿色通道
        mask = cv2.inRange(hsv, test_range["lower"], test_range["upper"])
        
        # 使用形态学操作连接分散的绿色像素
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.dilate(mask, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # 找到所有连通区域
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        
        print(f"   - 检测到 {num_labels - 1} 个连通区域")
        
        if num_labels > 1:
            # 计算每个连通区域的面积
            areas = []
            for j in range(1, num_labels):
                area = stats[j, cv2.CC_STAT_AREA]
                areas.append((j, area))
            
            # 按面积从大到小排序
            areas.sort(key=lambda x: x[1], reverse=True)
            
            print(f"   - 前5个区域面积: {[area[1] for area in areas[:5]]}")
        
        # 可视化
        debug_image = image.copy()
        
        # 将mask转换为彩色图像（绿色）
        mask_colored = np.zeros_like(image)
        mask_colored[mask > 0] = [0, 255, 0]
        
        # 叠加到原图上（半透明）
        result = cv2.addWeighted(debug_image, 0.7, mask_colored, 0.3, 0)
        
        # 标记中心点
        if num_labels > 1:
            # 选择面积最大的区域
            max_label, max_area = areas[0]
            center_x = int(centroids[max_label, 0])
            center_y = int(centroids[max_label, 1])
            
            # 绘制中心点（红色圆圈）
            cv2.circle(result, (center_x, center_y), 10, (0, 0, 255), 2)
            cv2.circle(result, (center_x, center_y), 3, (0, 0, 255), -1)
            cv2.putText(result, f"Center: ({center_x}, {center_y})", 
                       (center_x + 15, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # 保存调试图像
        output_path = str(ROOT / "tests" / "output" / f"green_highlight_test_{i + 1}.png")
        cv2.imwrite(output_path, result)
        print(f"   - 调试图像已保存到: {output_path}")
        print()
    
    print("=" * 70)
    print("📊 诊断总结")
    print("=" * 70)
    print()
    print("💡 请查看生成的调试图像，选择最合适的绿色范围")
    print()
    print("📋 下一步:")
    print("   1. 查看调试图像，确认哪个范围最合适")
    print("   2. 如果所有范围都不合适，可能需要调整颜色空间")
    print("   3. 如果游戏没有绿色高亮区域，检查游戏状态")
    print()


def main():
    """
    主函数
    """
    print("\n" + "=" * 70)
    print("🔍 绿色高亮区域诊断工具")
    print("=" * 70)
    print()
    print("📋 功能:")
    print("   1. 查看截图")
    print("   2. 可视化绿色区域提取结果")
    print("   3. 测试不同的绿色范围")
    print("   4. 帮助找到最合适的HSV范围")
    print()
    
    # 诊断绿色高亮区域
    diagnose_green_highlight()


if __name__ == "__main__":
    main()

from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path


def template_match(
    image: np.ndarray,
    template_path: str,
    threshold: float = 0.7,
    roi: tuple = None
):
    """
    模板匹配
    
    Args:
        image: BGR图像
        template_path: 模板图像路径
        threshold: 匹配阈值（0-1），默认0.7
        roi: 感兴趣区域（x1, y1, x2, y2），默认None（全图）
    
    Returns:
        (confidence, box_xyxy): 匹配置信度和匹配框，如果未匹配到则返回None
    """
    # 读取模板图像
    template = cv2.imread(template_path, cv2.IMREAD_COLOR)
    if template is None:
        print(f"无法读取模板图像: {template_path}")
        return None
    
    # 如果指定了ROI，裁剪图像
    if roi is not None:
        x1, y1, x2, y2 = roi
        image_roi = image[y1:y2, x1:x2]
    else:
        image_roi = image
    
    # 转换为灰度图像
    image_gray = cv2.cvtColor(image_roi, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    
    # 模板匹配
    result = cv2.matchTemplate(image_gray, template_gray, cv2.TM_CCOEFF_NORMED)
    
    # 找到最佳匹配位置
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    
    # 如果匹配值小于阈值，返回None
    if max_val < threshold:
        return None
    
    # 计算匹配框的位置
    h, w = template_gray.shape
    top_left = max_loc
    
    # 如果指定了ROI，需要加上ROI的偏移
    if roi is not None:
        x1, y1, x2, y2 = roi
        top_left = (top_left[0] + x1, top_left[1] + y1)
    
    bottom_right = (top_left[0] + w, top_left[1] + h)
    
    return (float(max_val), (top_left[0], top_left[1], bottom_right[0], bottom_right[1]))


def test_template_match():
    """
    测试模板匹配
    """
    # 项目根目录
    project_root = Path(__file__).parent.parent
    
    # 模板目录
    templates_dir = project_root / "data" / "templates"
    
    # 读取截图
    image_path = str(Path(__file__).parent / "output" / "screenshot.png")
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"无法读取截图: {image_path}")
        return
    
    print(f"截图尺寸: {image.shape}")
    print(f"截图类型: {image.dtype}")
    
    # 检查模板目录
    print(f"\n模板目录: {templates_dir}")
    
    # 查找模板文件
    template_files = list(templates_dir.glob("*.png")) + list(templates_dir.glob("*.jpg"))
    
    if not template_files:
        print("❌ 模板目录中没有模板文件！")
        print("请将模板文件放入 data/templates/ 目录")
        print("需要的模板文件：")
        print("  - hp_bar.png (干员信息血条)")
        print("  - cancel_ui.png (点击取消UI)")
        return
    
    print(f"找到 {len(template_files)} 个模板文件：")
    for template_file in template_files:
        print(f"  - {template_file.name}")
    
    # 测试模板匹配
    print("\n正在测试模板匹配...")
    
    for template_file in template_files:
        print(f"\n测试模板: {template_file.name}")
        
        # 测试模板匹配
        result = template_match(image, str(template_file), threshold=0.7)
        
        if result:
            confidence, box_xyxy = result
            print(f"✅ 成功匹配！")
            print(f"匹配置信度: {confidence:.4f}")
            print(f"匹配框: {box_xyxy}")
            
            # 在原图上标记匹配框
            image_debug = image.copy()
            cv2.rectangle(image_debug, (box_xyxy[0], box_xyxy[1]), (box_xyxy[2], box_xyxy[3]), (0, 0, 255), 2)
            cv2.putText(image_debug, f"{template_file.name} ({confidence:.4f})", 
                        (box_xyxy[0], box_xyxy[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            debug_image_path = str(Path(__file__).parent / "output" / f"template_match_{template_file.stem}.png")
            cv2.imwrite(debug_image_path, image_debug)
            print(f"调试图像已保存到: {debug_image_path}")
        else:
            print(f"❌ 未匹配到模板！")
            print("可能的原因：")
            print("1. 模板图像和目标图像不匹配")
            print("2. 匹配阈值过高")
            print("3. 模板图像太小或太大")


if __name__ == "__main__":
    test_template_match()

"""
手动标记部署区范围

功能：
1. 连接设备并截图
2. 在图像上手动标记部署区的四个角点
3. 保存标记结果
4. 输出正确的部署区坐标
"""

import sys
from pathlib import Path
import cv2
import numpy as np

# 添加项目路径到sys.path
ROOT = Path(__file__).resolve().parent.parent
DEPS_ROOT = ROOT / "maa-deps" / "maafw-5.2.6-win_amd64"

if DEPS_ROOT.exists():
    sys.path.insert(0, str(DEPS_ROOT))


def connect_device():
    """
    连接设备
    
    Returns:
        controller: MaaFramework控制器，如果连接失败则返回None
    """
    print("=" * 70)
    print("📱 连接设备")
    print("=" * 70)
    print()
    
    try:
        import maa
        from maa.controller import AdbController
        from maa.toolkit import Toolkit
        
        # 初始化 MaaFramework
        Toolkit.init_option(str(ROOT))
        
        # 查找 ADB 设备
        print("⏳ 正在查找 ADB 设备...")
        adb_devices = Toolkit.find_adb_devices()
        
        if not adb_devices:
            print("❌ 未找到 ADB 设备")
            print()
            print("💡 请确保:")
            print("   1. MuMu 模拟器已启动")
            print("   2. ADB 调试已开启")
            print("   3. 端口配置正确(默认 7555)")
            print()
            return None
        else:
            print(f"✅ 找到 {len(adb_devices)} 个设备:")
            for i, device in enumerate(adb_devices):
                print(f"    设备 {i+1}: {device.name} ({device.address})")
            
            # 使用第一个设备
            device = adb_devices[0]
            print(f"\n📱 设备配置:")
            print(f"   - 类型: ADB")
            print(f"   - ADB 路径: {device.adb_path}")
            print(f"   - 设备地址: {device.address}")
            print()
            
            # 创建控制器
            print("⏳ 正在连接设备...")
            controller = AdbController(
                adb_path=device.adb_path,
                address=device.address,
                screencap_methods=device.screencap_methods,
                input_methods=device.input_methods,
                config=device.config,
            )
            
            # 连接设备
            connection_job = controller.post_connection()
            connection_job.wait()
            
            if connection_job.succeeded:
                print("✅ 设备连接成功!")
                print()
                return controller
            else:
                print("❌ 设备连接失败")
                print()
                return None
    except Exception as e:
        print(f"❌ 设备连接失败: {e}")
        import traceback
        traceback.print_exc()
        print()
        return None


def manual_mark_deployment_area(controller):
    """
    手动标记部署区范围
    
    Args:
        controller: MaaFramework控制器
    """
    print("=" * 70)
    print("🎨 手动标记部署区范围")
    print("=" * 70)
    print()
    
    # 截图
    print("⏳ 正在截图...")
    image = controller.post_screencap().wait().get()
    print(f"✅ 截图完成，图像形状: {image.shape}")
    print()
    
    # 打印图像信息
    height, width = image.shape[:2]
    print(f"📊 图像信息:")
    print(f"   - 高度 (Y轴): {height}")
    print(f"   - 宽度 (X轴): {width}")
    print(f"   - 有效X范围: 0 - {width-1}")
    print(f"   - 有效Y范围: 0 - {height-1}")
    print()
    
    # 创建窗口
    window_name = "手动标记部署区 - 点击四个角点"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, width // 2, height // 2)
    
    # 存储点击的四个角点
    corners = []
    
    def mouse_callback(event, x, y, flags, param):
        """
        鼠标回调函数
        """
        nonlocal corners
        
        if event == cv2.EVENT_LBUTTONDOWN:
            # 左键点击，记录角点
            if len(corners) < 4:
                corners.append((x, y))
                print(f"✅ 记录角点 {len(corners)}/4: ({x}, {y})")
                
                # 在图像上绘制已记录的角点
                display_image = image.copy()
                
                # 绘制已记录的角点
                for i, (cx, cy) in enumerate(corners):
                    cv2.circle(display_image, (cx, cy), 10, (0, 0, 255), -1)
                    cv2.putText(display_image, f"{i+1}", (cx - 5, cy - 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # 绘制连接线
                if len(corners) >= 2:
                    for i in range(len(corners) - 1):
                        cv2.line(display_image, corners[i], corners[i+1], (0, 255, 0), 2)
                
                if len(corners) == 4:
                    # 闭合矩形
                    cv2.line(display_image, corners[3], corners[0], (0, 255, 0), 2)
                
                cv2.imshow(window_name, display_image)
                
                if len(corners) == 4:
                    print()
                    print("=" * 70)
                    print("✅ 四个角点已记录完成！")
                    print("=" * 70)
                    print()
                    cv2.destroyAllWindows()
    
    # 设置鼠标回调
    cv2.setMouseCallback(window_name, mouse_callback)
    
    # 显示图像
    print("🖱️  请在图像上点击四个角点（按顺序：左上、右上、左下、右下）")
    print("   - 点击顺序：左上角 → 右上角 → 左下角 → 右下角")
    print("   - 完成后会自动关闭窗口")
    print()
    
    cv2.imshow(window_name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    if len(corners) != 4:
        print("❌ 未完成四个角点的标记")
        return None
    
    # 计算部署区范围
    x_coords = [c[0] for c in corners]
    y_coords = [c[1] for c in corners]
    
    x1 = min(x_coords)
    x2 = max(x_coords)
    y1 = min(y_coords)
    y2 = max(y_coords)
    
    deployment_area = {
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2
    }
    
    print()
    print("=" * 70)
    print("📊 部署区范围")
    print("=" * 70)
    print()
    print(f"左上角: ({x1}, {y1})")
    print(f"右上角: ({x2}, {y1})")
    print(f"左下角: ({x1}, {y2})")
    print(f"右下角: ({x2}, {y2})")
    print()
    print(f"宽度: {x2 - x1}")
    print(f"高度: {y2 - y1}")
    print()
    
    # 在图像上绘制最终的部署区
    result_image = image.copy()
    cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 3)
    cv2.putText(result_image, "Deployment Area", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # 绘制四个角点
    for i, (cx, cy) in enumerate(corners):
        cv2.circle(result_image, (cx, cy), 10, (0, 0, 255), -1)
        cv2.putText(result_image, f"{i+1}", (cx - 5, cy - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # 保存结果
    output_path = str(ROOT / "tests" / "output" / "manual_marked_deployment_area.png")
    cv2.imwrite(output_path, result_image)
    print(f"✅ 标记结果已保存: {output_path}")
    print()
    
    # 输出代码
    print("=" * 70)
    print("💻 请将以下代码复制到 actions.py 中")
    print("=" * 70)
    print()
    print("```python")
    print("# 部署区范围")
    print("self._deployment_area = {")
    print(f'    "x1": {x1},')
    print(f'    "y1": {y1},')
    print(f'    "x2": {x2},')
    print(f'    "y2": {y2}')
    print("}")
    print("```")
    print()
    
    return deployment_area


def main():
    """
    主函数
    """
    print("\n" + "=" * 70)
    print("🎨 手动标记部署区范围工具")
    print("=" * 70)
    print()
    print("📋 功能:")
    print("   1. 连接设备并截图")
    print("   2. 在图像上手动标记部署区的四个角点")
    print("   3. 保存标记结果")
    print("   4. 输出正确的部署区坐标")
    print()
    
    # 连接设备
    controller = connect_device()
    
    if controller is None:
        print("\n" + "=" * 70)
        print("❌ 设备连接失败，无法继续")
        print("=" * 70)
        print()
        return
    
    # 手动标记部署区
    deployment_area = manual_mark_deployment_area(controller)
    
    if deployment_area:
        print("=" * 70)
        print("✅ 所有任务完成！")
        print("=" * 70)
        print()
        print("💡 下一步:")
        print("   1. 将上面的代码复制到 actions.py 中")
        print("   2. 替换旧的部署区坐标")
        print("   3. 重新运行测试")
        print()


if __name__ == "__main__":
    main()

"""
在截图上画出部署区范围并验证点击位置

功能：
1. 连接设备
2. 在部署区随机点击
3. 截图并标记点击位置
4. 画出部署区范围
5. 保存带标记的调试截图
"""

import sys
from pathlib import Path
import cv2
import numpy as np
import random

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


def visualize_deployment_area_with_click(controller, num_clicks=1):
    """
    在部署区随机点击并可视化
    
    Args:
        controller: MaaFramework控制器
        num_clicks: 点击次数，默认1次
    """
    print("=" * 70)
    print("🎨 在部署区随机点击并可视化")
    print("=" * 70)
    print()
    
    # 部署区范围：1272 x 116
    # 左上角坐标：(2, 598)
    # 右下角坐标：(1274, 714)
    deployment_area = {
        "x1": 2,
        "y1": 598,
        "x2": 1274,
        "y2": 714
    }
    
    print(f"📊 部署区范围:")
    print(f"   - 左上角: ({deployment_area['x1']}, {deployment_area['y1']})")
    print(f"   - 右下角: ({deployment_area['x2']}, {deployment_area['y2']})")
    print(f"   - 宽度: {deployment_area['x2'] - deployment_area['x1']}")
    print(f"   - 高度: {deployment_area['y2'] - deployment_area['y1']}")
    print()
    
    # 生成随机点击位置
    click_positions = []
    for i in range(num_clicks):
        x = random.randint(deployment_area["x1"], deployment_area["x2"])
        y = random.randint(deployment_area["y1"], deployment_area["y2"])
        click_positions.append((x, y))
        print(f"🎯 点击 {i+1}/{num_clicks}: ({x}, {y})")
    
    print()
    
    # 执行点击
    print("⏳ 正在执行点击...")
    for i, (x, y) in enumerate(click_positions):
        print(f"   - 点击 {i+1}: ({x}, {y})")
        controller.post_click(x, y).wait()
        
        # 等待一小段时间让游戏响应
        import time
        time.sleep(0.3)
    
    print("✅ 点击完成")
    print()
    
    # 截图
    print("⏳ 正在截图...")
    image = controller.post_screencap().wait().get()
    print(f"✅ 截图完成，图像形状: {image.shape}")
    print()
    
    # 在图像上绘制部署区范围
    print("🎨 正在绘制部署区范围和点击位置...")
    
    # 绘制部署区矩形框（绿色）
    cv2.rectangle(image, 
                  (deployment_area['x1'], deployment_area['y1']),
                  (deployment_area['x2'], deployment_area['y2']),
                  (0, 255, 0), 2)
    
    # 添加部署区标签
    cv2.putText(image, "Deployment Area", 
                (deployment_area['x1'], deployment_area['y1'] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # 绘制矩形框的四个角点（红色，圆点半径10）
    corners = [
        (deployment_area["x1"], deployment_area["y1"]),  # 左上角
        (deployment_area["x2"], deployment_area["y1"]),  # 右上角
        (deployment_area["x1"], deployment_area["y2"]),  # 左下角
        (deployment_area["x2"], deployment_area["y2"]),  # 右下角
    ]
    
    for i, corner in enumerate(corners):
        cv2.circle(image, corner, 10, (0, 0, 255), -1)  # BGR 格式，红色，填充
        # 添加角点标签
        corner_labels = ["左上角", "右上角", "左下角", "右下角"]
        label_position = (corner[0], corner[1] - 15)
        cv2.putText(
            image,
            corner_labels[i],
            label_position,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),  # BGR 格式，红色
            2
        )
    
    # 绘制矩形框的中心点（蓝色，圆点半径15）
    center_x = (deployment_area["x1"] + deployment_area["x2"]) // 2
    center_y = (deployment_area["y1"] + deployment_area["y2"]) // 2
    cv2.circle(image, (center_x, center_y), 15, (255, 0, 0), -1)  # BGR 格式，蓝色，填充
    
    # 添加中心点标签
    center_label_position = (center_x, center_y + 25)
    cv2.putText(
        image,
        "中心点",
        center_label_position,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 0, 0),  # BGR 格式，蓝色
        2
    )
    
    # 绘制点击位置（黄色，圆点半径20）
    for i, (x, y) in enumerate(click_positions):
        cv2.circle(image, (x, y), 20, (0, 255, 255), -1)  # BGR 格式，黄色，填充
        # 添加点击位置标签
        label_position = (x, y - 30)
        cv2.putText(
            image,
            f"点击{i+1}",
            label_position,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),  # BGR 格式，黄色
            2
        )
    
    # 绘制坐标标签（白色）
    coordinates_labels = [
        (f"({deployment_area['x1']}, {deployment_area['y1']})", (deployment_area["x1"], deployment_area["y1"] - 40)),  # 左上角
        (f"({deployment_area['x2']}, {deployment_area['y1']})", (deployment_area["x2"], deployment_area["y1"] - 40)),  # 右上角
        (f"({deployment_area['x1']}, {deployment_area['y2']})", (deployment_area["x1"], deployment_area["y2"] + 35)),  # 左下角
        (f"({deployment_area['x2']}, {deployment_area['y2']})", (deployment_area["x2"], deployment_area["y2"] + 35)),  # 右下角
        (f"({center_x}, {center_y})", (center_x, center_y + 45)),  # 中心点
    ]
    
    for label, position in coordinates_labels:
        cv2.putText(
            image,
            label,
            position,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),  # BGR 格式，白色
            2
        )
    
    print("✅ 绘制完成")
    print()
    
    # 保存图像
    output_path = str(ROOT / "tests" / "output" / "deployment_area_with_click.png")
    print(f"💾 正在保存图像: {output_path}")
    cv2.imwrite(output_path, image)
    print(f"✅ 图像已保存到: {output_path}")
    print()
    
    # 打印图像信息
    print("📊 图像信息:")
    print(f"   - 原始图像形状: {image.shape}")
    print(f"   - 部署区左上角: ({deployment_area['x1']}, {deployment_area['y1']})")
    print(f"   - 部署区右下角: ({deployment_area['x2']}, {deployment_area['y2']})")
    print(f"   - 部署区宽度: {deployment_area['x2'] - deployment_area['x1']}")
    print(f"   - 部署区高度: {deployment_area['y2'] - deployment_area['y1']}")
    print(f"   - 部署区中心: ({center_x}, {center_y})")
    print(f"   - 点击次数: {num_clicks}")
    for i, (x, y) in enumerate(click_positions):
        print(f"   - 点击{i+1}位置: ({x}, {y})")
    print()
    
    print("=" * 70)
    print("✅ 完成！")
    print("=" * 70)
    print()
    print("💡 说明:")
    print("   - 绿色矩形框：部署区范围")
    print("   - 红色圆点：矩形框的四个角点")
    print("   - 蓝色圆点：矩形框的中心点")
    print("   - 黄色圆点：点击位置")
    print("   - 白色文字：坐标标签")
    print()
    print("📸 请查看图像: " + output_path)
    print()


def main():
    """
    主函数
    """
    print("\n" + "=" * 70)
    print("🎨 部署区范围验证工具")
    print("=" * 70)
    print()
    print("📋 功能:")
    print("   1. 连接设备")
    print("   2. 在部署区随机点击")
    print("   3. 截图并标记点击位置")
    print("   4. 画出部署区范围")
    print("   5. 保存带标记的调试截图")
    print()
    
    # 连接设备
    controller = connect_device()
    
    if controller is None:
        print("\n" + "=" * 70)
        print("❌ 设备连接失败，无法继续")
        print("=" * 70)
        print()
        return
    
    # 在部署区随机点击并可视化
    visualize_deployment_area_with_click(controller, num_clicks=3)
    
    print("=" * 70)
    print("✅ 所有任务完成！")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()

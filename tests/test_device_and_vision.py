"""测试模拟器连接和视觉识别功能"""
import sys
from pathlib import Path
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "maa-deps" / "maafw-5.2.6-win_amd64"))
sys.path.insert(0, str(ROOT / "src" / "maa-wrapper"))
sys.path.insert(0, str(ROOT / "src" / "ai-plugins"))

from runtime import MaaFwAdapter
from yolo_recognizer import YoloRecognizer

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    from maa.define import MaaAdbScreencapMethodEnum
    MAA_AVAILABLE = True
except ImportError:
    MAA_AVAILABLE = False


def test_device_connection():
    """测试模拟器连接"""
    print("=" * 60)
    print("🔗 测试 1: 模拟器连接")
    print("=" * 60)
    
    # 设备配置
    device_config = {
        "type": "adb",
        "adb_path": "e:/Program Files/Netease/MuMu Player 12/nx_device/12.0/shell/adb.exe",
        "address": "127.0.0.1:7555"
    }
    
    # 启用MuMu SDK截图增强模式
    if MAA_AVAILABLE:
        device_config["screencap_methods"] = MaaAdbScreencapMethodEnum.EmulatorExtras
        device_config["config"] = {
            "extras": {
                "mumu": {
                    "enable": True,
                    "path": "e:/Program Files/Netease/MuMu Player 12",
                    "index": 0
                }
            }
        }
        print("✅ 已启用 MuMu SDK 截图增强模式")
    
    print(f"📱 设备配置:")
    print(f"   - 类型: {device_config['type']}")
    print(f"   - ADB 路径: {device_config['adb_path']}")
    print(f"   - 地址: {device_config['address']}")
    if "screencap_methods" in device_config:
        print(f"   - 截图方式: MuMu SDK 增强模式")
    print()
    
    try:
        # 创建适配器
        adapter = MaaFwAdapter(device_config)
        
        # 连接设备
        print("⏳ 正在连接设备...")
        adapter.connect()
        print("✅ 设备连接成功!")
        print(f"   - Controller: {adapter._controller}")
        print()
        
        return adapter
    
    except Exception as e:
        print(f"❌ 设备连接失败: {e}")
        print()
        return None


def test_screenshot(adapter):
    """测试截图功能"""
    print("=" * 60)
    print("📸 测试 2: 截图功能")
    print("=" * 60)
    
    if adapter is None:
        print("⚠️ 设备未连接，跳过截图测试")
        print()
        return None
    
    try:
        # 截图
        print("⏳ 正在截图...")
        job = adapter._controller.post_screencap().wait()
        
        if not job.succeeded:
            print("❌ 截图失败")
            print()
            return None
        
        # 获取截图
        image = adapter._controller.cached_image
        print(f"✅ 截图成功!")
        print(f"   - 图像形状: {image.shape}")
        print(f"   - 图像类型: {image.dtype}")
        print()
        
        # 保存截图
        output_dir = ROOT / "tests" / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        screenshot_path = output_dir / "screenshot.png"
        
        # 使用 PIL 保存图像
        try:
            from PIL import Image
            img = Image.fromarray(image)
            img.save(str(screenshot_path))
            print(f"💾 截图已保存到: {screenshot_path}")
        except ImportError:
            # 如果没有 PIL，使用 numpy 保存
            np.save(str(screenshot_path.with_suffix('.npy')), image)
            print(f"💾 截图已保存到: {screenshot_path.with_suffix('.npy')}")
        
        print()
        
        return image
    
    except Exception as e:
        print(f"❌ 截图失败: {e}")
        print()
        return None


def test_yolo_recognizer(image):
    """测试 YOLO 识别器"""
    print("=" * 60)
    print("🎯 测试 3: YOLO 识别器")
    print("=" * 60)
    
    if image is None:
        print("⚠️ 没有截图，跳过 YOLO 识别测试")
        print()
        return
    
    # 检查 YOLO 模型是否存在
    model_path = ROOT / "models" / "yolo" / "best.pt"
    if not model_path.exists():
        print(f"⚠️ YOLO 模型不存在: {model_path}")
        print("💡 提示: 请先训练 YOLO 模型，或跳过此测试")
        print()
        return
    
    try:
        # 加载 YOLO 识别器
        print(f"⏳ 正在加载 YOLO 模型: {model_path}")
        recognizer = YoloRecognizer(str(model_path))
        recognizer.load()
        print("✅ YOLO 模型加载成功!")
        print()
        
        # 识别
        print("⏳ 正在识别...")
        detections = recognizer.detect(image, conf=0.25)
        print(f"✅ 识别完成!")
        print(f"   - 检测到 {len(detections)} 个目标")
        print()
        
        if len(detections) > 0:
            print("📋 检测结果:")
            for i, detection in enumerate(detections):
                print(f"   [{i}] 标签: {detection.label}, 置信度: {detection.confidence:.2f}, 位置: {detection.box_xyxy}")
            print()
            
            # 尝试在图像上绘制检测结果（如果有 cv2）
            try:
                import cv2
                output_image = image.copy()
                for detection in detections:
                    x1, y1, x2, y2 = detection.box_xyxy
                    cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        output_image,
                        f"{detection.label} {detection.confidence:.2f}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2
                    )
                
                # 保存结果
                output_dir = ROOT / "tests" / "output"
                output_dir.mkdir(parents=True, exist_ok=True)
                result_path = output_dir / "detection_result.png"
                cv2.imwrite(str(result_path), output_image)
                print(f"💾 检测结果已保存到: {result_path}")
                print()
            except ImportError:
                print("⚠️ 未安装 cv2，无法绘制检测结果")
                print()
        else:
            print("⚠️ 未检测到任何目标")
            print("💡 提示: 可能是 YOLO 模型未训练，或图像中没有目标")
            print()
    
    except Exception as e:
        print(f"❌ YOLO 识别失败: {e}")
        print()


def test_click_action(adapter):
    """测试点击动作"""
    print("=" * 60)
    print("⚡ 测试 4: 点击动作")
    print("=" * 60)
    
    if adapter is None:
        print("⚠️ 设备未连接，跳过点击测试")
        print()
        return
    
    try:
        # 点击屏幕中心
        x, y = 640, 360
        print(f"⏳ 正在点击位置: ({x}, {y})")
        job = adapter._controller.post_click(x, y).wait()
        
        if not job.succeeded:
            print("❌ 点击失败")
            print()
            return
        
        print("✅ 点击成功!")
        print()
        
        # 等待一下
        time.sleep(1)
        
        # 再次截图，验证点击效果
        print("⏳ 正在截图验证点击效果...")
        job = adapter._controller.post_screencap().wait()
        
        if job.succeeded:
            image = adapter._controller.cached_image
            output_dir = ROOT / "tests" / "output"
            output_dir.mkdir(parents=True, exist_ok=True)
            screenshot_path = output_dir / "screenshot_after_click.png"
            
            # 使用 PIL 保存图像
            try:
                from PIL import Image
                img = Image.fromarray(image)
                img.save(str(screenshot_path))
                print(f"💾 点击后截图已保存到: {screenshot_path}")
            except ImportError:
                # 如果没有 PIL，使用 numpy 保存
                np.save(str(screenshot_path.with_suffix('.npy')), image)
                print(f"💾 点击后截图已保存到: {screenshot_path.with_suffix('.npy')}")
            
            print()
    
    except Exception as e:
        print(f"❌ 点击失败: {e}")
        print()


def test_realtime_monitoring(adapter, duration=30, fps=10):
    """
    实时监控功能
    
    Args:
        adapter: MaaFwAdapter 实例
        duration: 监控时长（秒）
        fps: 帧率（每秒截图次数）
    """
    print("=" * 60)
    print("📺 实时监控")
    print("=" * 60)
    
    if adapter is None:
        print("⚠️ 设备未连接，跳过实时监控")
        print()
        return
    
    # 检查 YOLO 模型是否存在
    model_path = ROOT / "models" / "yolo" / "best.pt"
    if not model_path.exists():
        print(f"⚠️ YOLO 模型不存在: {model_path}")
        print("💡 提示: 请先训练 YOLO 模型，或跳过此测试")
        print()
        return
    
    # 检查 cv2 是否可用
    if not CV2_AVAILABLE:
        print("⚠️ cv2 未安装，无法显示实时监控画面")
        print("💡 提示: 安装 opencv-python: pip install opencv-python")
        print()
        return
    
    try:
        # 加载 YOLO 识别器
        print(f"⏳ 正在加载 YOLO 模型: {model_path}")
        recognizer = YoloRecognizer(str(model_path))
        recognizer.load()
        print("✅ YOLO 模型加载成功!")
        print()
        
        # 计算间隔时间
        interval = 1.0 / fps
        
        print(f"📺 开始实时监控...")
        print(f"   - 监控时长: {duration} 秒")
        print(f"   - 帧率: {fps} fps")
        print(f"   - 间隔: {interval:.3f} 秒")
        print()
        print("💡 提示: 按 'q' 或 Ctrl+C 停止监控")
        print()
        
        frame_count = 0
        start_time = time.time()
        last_print_time = start_time
        
        try:
            while True:
                # 检查是否超时
                elapsed = time.time() - start_time
                if elapsed >= duration:
                    print(f"⏱️ 监控结束，共运行 {elapsed:.1f} 秒")
                    break
                
                # 记录截图开始时间
                screenshot_start = time.time()
                
                # 截图
                job = adapter._controller.post_screencap().wait()
                if not job.succeeded:
                    print(f"⚠️ 截图失败 (帧 {frame_count})")
                    time.sleep(interval)
                    continue
                
                screenshot_time = time.time() - screenshot_start
                
                # 获取截图
                image = adapter._controller.cached_image
                
                # 记录YOLO推理开始时间
                yolo_start = time.time()
                
                # YOLO 识别
                detections = recognizer.detect(image, conf=0.25)
                
                yolo_time = time.time() - yolo_start
                
                # 打印识别结果（每秒打印一次）
                frame_count += 1
                current_time = time.time()
                if current_time - last_print_time >= 1.0:
                    timestamp = time.strftime("%H:%M:%S", time.localtime())
                    print(f"[{timestamp}] 帧 {frame_count}: 检测到 {len(detections)} 个目标 | "
                          f"截图: {screenshot_time:.3f}s, YOLO: {yolo_time:.3f}s")
                    last_print_time = current_time
                
                # 绘制检测框
                output_image = image.copy()
                for detection in detections:
                    x1, y1, x2, y2 = detection.box_xyxy
                    cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        output_image,
                        f"{detection.label} {detection.confidence:.2f}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2
                    )
                
                # 添加帧信息
                cv2.putText(
                    output_image,
                    f"Frame: {frame_count} | FPS: {fps:.1f} | Time: {elapsed:.1f}s",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2
                )
                
                # 添加检测数量信息
                cv2.putText(
                    output_image,
                    f"Detected: {len(detections)}",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2
                )
                
                # 显示实时监控画面
                cv2.imshow("Real-time Monitoring - Press 'q' to quit", output_image)
                
                # 检查按键
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print(f"\n⏸️ 用户按 'q' 停止监控")
                    break
                
                # 等待下一帧
                time.sleep(interval)
        
        except KeyboardInterrupt:
            print(f"\n⏸️ 监控已停止，共运行 {time.time() - start_time:.1f} 秒")
        
        # 关闭窗口
        cv2.destroyAllWindows()
        
        # 统计信息
        print()
        print(f"📊 监控统计:")
        print(f"   - 总帧数: {frame_count}")
        print(f"   - 运行时长: {time.time() - start_time:.1f} 秒")
        print(f"   - 平均帧率: {frame_count / (time.time() - start_time):.2f} fps")
        print()
    
    except Exception as e:
        print(f"❌ 实时监控失败: {e}")
        import traceback
        traceback.print_exc()
        print()


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("🚀 开始测试模拟器连接和视觉识别功能")
    print("=" * 60 + "\n")
    
    # 测试 1: 模拟器连接
    adapter = test_device_connection()
    
    # 测试 2: 截图功能
    image = test_screenshot(adapter)
    
    # 测试 3: YOLO 识别器
    test_yolo_recognizer(image)
    
    # 测试 4: 点击动作
    test_click_action(adapter)
    
    # 测试 5: 实时监控
    test_realtime_monitoring(adapter, duration=30, fps=10)
    
    # 总结
    print("=" * 60)
    print("✅ 测试完成!")
    print("=" * 60)
    print("\n📁 输出文件位置:")
    output_dir = ROOT / "tests" / "output"
    print(f"   - {output_dir}")
    print("\n💡 提示:")
    print("   - 如果 YOLO 模型不存在，请先训练 YOLO 模型")
    print("   - 如果未检测到目标，请检查 YOLO 模型是否正确训练")
    print("   - 可以查看输出目录中的截图和检测结果")
    print()


if __name__ == "__main__":
    main()

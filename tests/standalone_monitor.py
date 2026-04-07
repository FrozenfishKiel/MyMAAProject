import sys
from pathlib import Path
import time
import argparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "maa-deps" / "maafw-5.2.6-win_amd64"))
sys.path.insert(0, str(ROOT / "src" / "maa-wrapper"))
sys.path.insert(0, str(ROOT / "src" / "ai-plugins"))

from runtime import MaaFwAdapter
from yolo_recognizer import YoloRecognizer
import cv2

try:
    from maa.define import MaaAdbScreencapMethodEnum
    MAA_AVAILABLE = True
except ImportError:
    MAA_AVAILABLE = False

def start_monitor():
    print("=" * 60)
    print("📺 启动独立实时监控进程")
    print("=" * 60)

    # 强制将 cv2 窗口放在顶层（Windows 特性）
    cv2.namedWindow("Real-time Monitoring - Press 'q' to quit", cv2.WINDOW_NORMAL)

    device_config = {
        "type": "adb",
        "adb_path": "e:/Program Files/Netease/MuMu Player 12/nx_device/12.0/shell/adb.exe",
        "address": "127.0.0.1:7555"
    }

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

    try:
        adapter = MaaFwAdapter(device_config)
        adapter.connect()
        print("✅ 监控进程已连接至设备!")
    except Exception as e:
        print(f"❌ 设备连接失败: {e}")
        return

    model_path = ROOT / "models" / "yolo" / "best.pt"
    if not model_path.exists():
        print(f"⚠️ YOLO 模型不存在: {model_path}")
        return

    try:
        recognizer = YoloRecognizer(str(model_path))
        recognizer.load()
        print("✅ YOLO 模型加载成功!")

        interval = 1.0 / 10.0 # 10 FPS
        print(f"📺 开始实时监控 (按 'q' 退出)...")

        while True:
            job = adapter._controller.post_screencap().wait()
            if not job.succeeded:
                time.sleep(interval)
                continue

            image = adapter._controller.cached_image
            detections = recognizer.detect(image, conf=0.25)

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

            cv2.putText(
                output_image,
                f"Detected: {len(detections)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

            cv2.imshow("Real-time Monitoring - Press 'q' to quit", output_image)

            # 使用 cv2.waitKey 检查按键，同时这也是刷新画面的必要操作
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print(f"\n⏸️ 监控主动退出")
                break

            time.sleep(interval)

    except KeyboardInterrupt:
        print(f"\n⏸️ 收到中断信号，监控退出")
    except Exception as e:
        print(f"❌ 监控崩溃: {e}")
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    start_monitor()

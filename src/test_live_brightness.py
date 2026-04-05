import os
import sys
import cv2
import numpy as np
from pathlib import Path

# --- 照抄项目里的环境配置，确保 MAA 能正常导入 ---
ROOT = Path(__file__).resolve().parent
DEPS_ROOT = ROOT.parent / "maa-deps" / "maafw-5.2.6-win_amd64"
if DEPS_ROOT.exists():
    sys.path.insert(0, str(DEPS_ROOT))

try:
    from maa.controller import AdbController
    from maa.toolkit import Toolkit
except ImportError as e:
    print(f"[ERROR] 导入 maa 失败，请检查路径: {e}")
    sys.exit(1)

def test_live_brightness():
    Toolkit.init_option(str(ROOT))
    print("[INFO] 正在查找 ADB 设备...")
    adb_devices = Toolkit.find_adb_devices()

    if not adb_devices:
        print("[ERROR] 未找到 ADB 设备，请确保模拟器已启动。")
        return

    device = adb_devices[0]
    print(f"[INFO] 找到设备: {device.name} ({device.address})")

    controller = AdbController(
        adb_path=device.adb_path,
        address=device.address,
        config=device.config
    )

    if controller.post_connection().wait().succeeded:
        print("[SUCCESS] 设备连接成功! 正在通过 MAA 获取实时截图...")

        # 获取原图 (非阻塞等待)
        image = controller.post_screencap().wait().get()
        if image is None:
            print("[ERROR] 截图失败！")
            return

        print(f"[SUCCESS] 截图成功，分辨率: {image.shape}")

        # --- 核心：解析手牌区亮度 ---
        h_raw, w_raw = image.shape[:2]

        # 估算手牌区 (宽 20%~90%, 高 80%~98%，适用于主流 16:9 模拟器分辨率)
        y_start, y_end = int(h_raw * 0.85), int(h_raw * 0.95)
        x_start, x_end = int(w_raw * 0.20), int(w_raw * 0.85)

        cards_roi = image[y_start:y_end, x_start:x_end]
        gray_cards = cv2.cvtColor(cards_roi, cv2.COLOR_BGR2GRAY)

        # 平分成 10 份
        card_width = gray_cards.shape[1] // 10

        print("-" * 50)
        print("卡位 | 平均亮度 | 判定状态 (阈值暂定:100)")
        print("-" * 50)

        debug_img = image.copy()
        playable_vector = []

        for i in range(10):
            single_card = gray_cards[:, i*card_width : (i+1)*card_width]
            avg_brightness = np.mean(single_card)

            # 亮度阈值
            is_playable = "✅ 可用" if avg_brightness > 100 else "❌ 不可用"
            playable_vector.append(1 if avg_brightness > 100 else 0)

            print(f"[{i:2d}]  | {avg_brightness:8.2f} | {is_playable}")

            # 在图上画绿框/红框
            color = (0, 255, 0) if avg_brightness > 100 else (0, 0, 255)
            cv2.rectangle(debug_img, (x_start + i*card_width, y_start), (x_start + (i+1)*card_width, y_end), color, 2)
            cv2.putText(debug_img, str(int(avg_brightness)), (x_start + i*card_width + 10, y_start + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        print("-" * 50)
        print(f"最终输出给 AI 的状态向量: {playable_vector}")

        save_path = ROOT.parent / "live_brightness_test.png"
        cv2.imwrite(str(save_path), debug_img)
        print(f"\n[INFO] 🎉 调试图片已保存至项目根目录: {save_path.name}")
        print("[INFO] 请打开该图片，检查我们切的框是否刚好套住了底部的干员卡牌，以及红绿框判定是否正确！")
    else:
        print("[ERROR] 设备连接失败")

if __name__ == "__main__":
    test_live_brightness()
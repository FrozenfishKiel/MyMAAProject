import sys
import time
import cv2
import numpy as np
from pathlib import Path

# ================= 配置区 =================
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "maa-deps" / "maafw-5.2.6-win_amd64"))
sys.path.insert(0, str(ROOT / "src" / "maa-wrapper"))
sys.path.insert(0, str(ROOT / "src" / "ai-plugins"))

try:
    from runtime import MaaFwAdapter
    from maa.define import MaaAdbScreencapMethodEnum
    MAA_AVAILABLE = True
except ImportError:
    MAA_AVAILABLE = False
# ==========================================

def main():
    print("=" * 60)
    print("🎯 启动 ROI (识别区域) 校验测试")
    print("=" * 60)

    # 1. 连接设备
    device_config = {
        "type": "adb",
        "adb_path": "e:/Program Files/Netease/MuMu Player 12/nx_device/12.0/shell/adb.exe",
        "address": "127.0.0.1:7555" # 或 16384，根据你的实际端口修改
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

    print("📺 开始实时显示 ROI (请在游戏中完成一次战斗，看看方框是否完美框住了左上角的蓝星)...")
    print("💡 提示：按 'q' 退出")

    interval = 1.0 / 15.0

    while True:
        job = adapter._controller.post_screencap().wait()
        if not job.succeeded:
            time.sleep(interval)
            continue

        frame = adapter._controller.cached_image
        h, w = frame.shape[:2]

        # ------------------ ROI 框选区域 ------------------
        # 1. 结算星级 ROI (左上角)
        # 彻底往下平移！(y从 15% 开始，到 45% 结束)
        # 同时缩小宽度 (w到 40%)
        sx1, sy1 = 0, int(h * 0.4)
        sx2, sy2 = int(w * 0.40), int(h * 0.5)

        # 2. 失败结算 ROI (中间)
        # 你的 game_env.py 里是: lose_roi = (int(w * 0.2), int(h * 0.2), int(w * 0.8), int(h * 0.6))
        lx1, ly1 = int(w * 0.2), int(h * 0.2)
        lx2, ly2 = int(w * 0.8), int(h * 0.6)

        display_frame = frame.copy()

        # 画个蓝框表示星级检测区域
        cv2.rectangle(display_frame, (sx1, sy1), (sx2, sy2), (255, 255, 0), 3) # 浅蓝色(青色)
        # 画个半透明的底色，方便看清楚框的范围
        overlay_star = display_frame.copy()
        cv2.rectangle(overlay_star, (sx1, sy1), (sx2, sy2), (255, 255, 0), -1)
        cv2.addWeighted(overlay_star, 0.2, display_frame, 0.8, 0, display_frame)
        cv2.putText(display_frame, "Star ROI (0-40% W, 15-45% H)", (sx1 + 10, sy2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # 画个红框表示失败检测区域
        cv2.rectangle(display_frame, (lx1, ly1), (lx2, ly2), (0, 0, 255), 3) # 红色
        # 画个半透明的底色
        overlay_lose = display_frame.copy()
        cv2.rectangle(overlay_lose, (lx1, ly1), (lx2, ly2), (0, 0, 255), -1)
        cv2.addWeighted(overlay_lose, 0.15, display_frame, 0.85, 0, display_frame)
        cv2.putText(display_frame, "Lose ROI (20-80% W, 20-60% H)", (lx1 + 10, ly1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


        cv2.imshow("ROI Verification Test", display_frame)

        # 按键检测
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print(f"\n⏸️ 测试主动退出")
            break

        time.sleep(interval)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
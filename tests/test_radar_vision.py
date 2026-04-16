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

def get_green_grid_mask(image):
    """
    提取游戏画面中绿色可部署高亮网格的二值图
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # 明日方舟部署时的绿色高亮大概 HSV 范围
    lower_green = np.array([35, 43, 46])
    upper_green = np.array([90, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # 形态学开运算去噪（消除细小的绿点）
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask

def main():
    print("=" * 60)
    print("📡 启动视觉雷达 (Radar Vision) 实时监测")
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

    # 2. 初始化背景建模器 MOG2
    # history: 历史帧数，越小则背景更新越快（“鬼影”消失越快，但移动缓慢的敌人也容易被当成背景吃掉）
    # varThreshold: 方差阈值，越大对光照和细微变化越不敏感，能过滤特效噪点。把它从 50 提高到 120！
    # detectShadows: 是否检测阴影（变灰），我们不需要，设为 False 节省算力
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=120, detectShadows=False)

    print(f"📺 开始实时监控 (请在游戏中打开 1-7 并放置干员)...")
    print("💡 提示：按 'q' 退出，按 'r' 手动重置背景建模")

    interval = 1.0 / 15.0 # 目标 15 FPS

    while True:
        job = adapter._controller.post_screencap().wait()
        if not job.succeeded:
            time.sleep(interval)
            continue

        frame = adapter._controller.cached_image

        # --- 通道 1：灰度原图（作为底图观察用） ---
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # --- 通道 2：敌人雷达图 (MOG2 运动侦测) ---
        # 提取前景 Mask（画面中动的东西会变白）
        fg_mask = bg_subtractor.apply(frame)

        # 核心去噪：用形态学运算消除细碎的技能特效和火花
        # 换用更大一点的椭圆核（对圆形特效的过滤效果更好）
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))  # 擦除小噪点，从 5x5 扩大到 7x7
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)) # 把断裂的敌人躯干连起来

        fg_mask_clean = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel_open)
        fg_mask_clean = cv2.morphologyEx(fg_mask_clean, cv2.MORPH_CLOSE, kernel_close)

        # --- 通道 3：绿格子可部署区域图 ---
        grid_mask = get_green_grid_mask(frame)

        # --- 可视化拼接 ---
        # 1. 原始灰度图转为 BGR 方便画彩色框
        display_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)

        # 2. 把清理过的敌人雷达图（动的部分）用红色半透明叠加到底图上
        red_overlay = np.zeros_like(display_frame)
        red_overlay[fg_mask_clean > 0] = [0, 0, 255] # BGR格式，红色
        cv2.addWeighted(display_frame, 1.0, red_overlay, 0.5, 0, display_frame)

        # 3. 把绿格子高亮图用绿色半透明叠加到底图上
        green_overlay = np.zeros_like(display_frame)
        green_overlay[grid_mask > 0] = [0, 255, 0] # BGR格式，绿色
        cv2.addWeighted(display_frame, 1.0, green_overlay, 0.4, 0, display_frame)

        # --- 寻找敌人的“质心”并画个准星（演示奖励函数的计算基础） ---
        # 寻找连通区域（那些大块的白色雷达点）
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(fg_mask_clean, connectivity=8)

        enemy_count = 0
        for i in range(1, num_labels): # 0是背景，跳过
            area = stats[i, cv2.CC_STAT_AREA]
            # [极其关键的面积过滤]
            # 1-7 中的虫子和干员的面积通常在 300 ~ 4000 像素之间
            # 小于 300 的多半是飞溅的火花、小烟尘
            # 大于 4000 的，要么是极其巨大的爆炸/全屏特效，要么是大面积的鬼影
            # 所以我们把上下限从 150~10000 狠狠地收紧为 300~4000！
            if 300 < area < 4000:
                # 还可以加入宽高的形状过滤，排除那些全屏飘过的细长烟雾带
                width = stats[i, cv2.CC_STAT_WIDTH]
                height = stats[i, cv2.CC_STAT_HEIGHT]
                # 方舟敌人再怎么细长，宽高比一般不会超过 1:4 或 4:1
                if 0.25 < width/height < 4.0:
                    enemy_count += 1
                    cx, cy = int(centroids[i][0]), int(centroids[i][1])
                    # 在底图上画一个蓝色的准星，这就是未来 AI 计算“朝向/距离奖励”的靶子
                    cv2.drawMarker(display_frame, (cx, cy), (255, 0, 0), cv2.MARKER_CROSS, 20, 2)
                    cv2.circle(display_frame, (cx, cy), 50, (255, 0, 0), 1) # 示意一个大概的攻击范围

        # 信息文字显示
        cv2.putText(display_frame, f"Active Enemies (Radar): {enemy_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(display_frame, "RED=Motion, GREEN=Grid, BLUE=Target", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # 把去噪前后的雷达图也缩小显示在右上角，让你直观对比“清洗特效”的效果
        h, w = frame.shape[:2]
        small_w, small_h = w // 4, h // 4

        # 右上角 1：原始 MOG2 (包含满天飞的特效碎片)
        small_fg = cv2.resize(fg_mask, (small_w, small_h))
        small_fg_bgr = cv2.cvtColor(small_fg, cv2.COLOR_GRAY2BGR)
        cv2.putText(small_fg_bgr, "Raw MOG2", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        display_frame[0:small_h, w-small_w:w] = small_fg_bgr

        # 右上角 2：清洗后的 MOG2 (干净的敌人躯干)
        small_clean = cv2.resize(fg_mask_clean, (small_w, small_h))
        small_clean_bgr = cv2.cvtColor(small_clean, cv2.COLOR_GRAY2BGR)
        cv2.putText(small_clean_bgr, "Cleaned MOG2", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        display_frame[small_h:small_h*2, w-small_w:w] = small_clean_bgr

        # 显示主画面
        cv2.imshow("AI Vision Radar Test", display_frame)

        # 按键检测
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print(f"\n⏸️ 雷达测试主动退出")
            break
        elif key == ord('r'):
            # 模拟我们在部署干员后手动重置背景，消除“鬼影”
            print(f"\n🔄 手动重置 MOG2 背景建模 (清除鬼影)...")
            bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=50, detectShadows=False)

        time.sleep(interval)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
import time
import os
import signal
import sys
from pathlib import Path

# ================= 配置区 =================
# 将 ROOT 设置为 src 文件夹的上一级 (即 MaAutomaton-main 根目录)
ROOT = Path(__file__).resolve().parent.parent

# 添加 MAA 依赖路径
DEPS_ROOT = ROOT / "maa-deps" / "maafw-5.2.6-win_amd64"
if DEPS_ROOT.exists():
    sys.path.insert(0, str(DEPS_ROOT))

# 导入maa相关库
import importlib
try:
    maa = importlib.import_module("maa")
    from maa.controller import AdbController
    from maa.toolkit import Toolkit
except ImportError as e:
    print(f"❌ 导入 maa 模块失败: {e}")
    print(f"当前 sys.path 包含: {sys.path[:3]}")
    print(f"期望的依赖路径 {DEPS_ROOT} 是否存在: {DEPS_ROOT.exists()}")
    sys.exit(1)
import cv2

# ================= 配置区 =================
# 输出目录：按照你的要求存放到 data/Source
OUTPUT_DIR = ROOT / "data" / "Source"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
# 截图间隔 (秒)
INTERVAL = 2.0
# ==========================================

print(f"==================================================")
print(f"📸 MAA 自动截图采集脚本")
print(f"📸 输出目录: {OUTPUT_DIR}")
print(f"📸 截图间隔: {INTERVAL} 秒")
print(f"📸 提示: 请在游戏中打开任意关卡并部署干员。")
print(f"📸 按下 Ctrl+C 随时停止截图。")
print(f"==================================================")

# 初始化 MAA
try:
    Toolkit.init_option(str(ROOT))
except Exception as e:
    pass

print("[1/3] 正在查找 ADB 设备...")
adb_devices = Toolkit.find_adb_devices()
if not adb_devices:
    print("❌ 错误: 未找到 ADB 设备。请确保模拟器已启动。")
    sys.exit(1)

device = adb_devices[0]
print(f"✅ 连接到设备: {device.name} ({device.address})")

controller = AdbController(
    adb_path=device.adb_path,
    address=device.address,
    screencap_methods=device.screencap_methods,
    input_methods=device.input_methods,
    config=device.config,
)

print("[2/3] 正在与设备握手...")
connection_job = controller.post_connection()
connection_job.wait()
if not connection_job.succeeded:
    print("❌ 错误: 设备连接失败。")
    sys.exit(1)
print("✅ 设备握手成功！")

# 注册信号处理 (Ctrl+C)
keep_running = True
def signal_handler(sig, frame):
    global keep_running
    print("\n🛑 收到停止信号，正在退出采集脚本...")
    keep_running = False

signal.signal(signal.SIGINT, signal_handler)

print("[3/3] 🚀 开始自动截图！(请切换到游戏画面)")
count = 0

while keep_running:
    try:
        # 1. 触发 MAA 截图
        img_job = controller.post_screencap().wait()
        img = img_job.get()
        if img is not None:
            # 2. 生成带时间戳的文件名
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"capture_{timestamp}_{count:04d}.jpg"
            save_path = OUTPUT_DIR / filename

            # 3. 保存图片
            cv2.imwrite(str(save_path), img)
            print(f"[{time.strftime('%H:%M:%S')}] 💾 已保存: {filename}")
            count += 1
        else:
            print(f"[{time.strftime('%H:%M:%S')}] ⚠️ 截图失败，跳过。")

        # 4. 等待指定间隔
        time.sleep(INTERVAL)
    except Exception as e:
        print(f"❌ 发生异常: {e}")
        time.sleep(1)

print(f"🎉 采集结束！共采集 {count} 张图片。")
print(f"📂 图片保存在: {OUTPUT_DIR}")

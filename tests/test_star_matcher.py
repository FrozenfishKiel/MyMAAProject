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

from template_matcher import TemplateMatcher
# ==========================================

def main():
    print("=" * 60)
    print("⭐ 启动结算星级深度识别测试")
    print("=" * 60)

    # 1. 连接设备
    device_config = {
        "type": "adb",
        "adb_path": "e:/Program Files/Netease/MuMu Player 12/nx_device/12.0/shell/adb.exe",
        "address": "127.0.0.1:7555" # 或 16384，根据实际端口修改
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
        print("✅ 设备连接成功!")
    except Exception as e:
        print(f"❌ 设备连接失败: {e}")
        return

    # 初始化模板匹配器
    matcher = TemplateMatcher(adapter._controller)
    matcher.load()

    # 准备模板路径
    template_3star = str(ROOT.parent / "data" / "templates" / "battle_3star.png")
    template_2star = str(ROOT.parent / "data" / "templates" / "battle_2star.png")
    template_0star = str(ROOT.parent / "data" / "templates" / "battle_0star.png")

    print("\n📺 正在实时监控画面左上角的星级变化 (请在模拟器中进入结算界面)...")
    print("💡 提示：按 'q' 退出，按 's' 截取一张带框的判定图保存在当前目录。")

    interval = 1.0 / 10.0 # 10帧/秒足以看清

    while True:
        job = adapter._controller.post_screencap().wait()
        if not job.succeeded:
            time.sleep(interval)
            continue

        frame = adapter._controller.cached_image
        h, w = frame.shape[:2]

        # 核心：game_env.py 里定义的精确 ROI
        star_roi = (0, int(h * 0.385), int(w * 0.30), int(h * 0.5))

        # --- 手动跑一次匹配拿到置信度（不设阈值，强制看最高分是谁） ---
        # 也就是不管对不对，我都要知道这个模板在这个 ROI 里最高能打几分
        import cv2
        image_bgr = frame[star_roi[1]:star_roi[3], star_roi[0]:star_roi[2]]
        image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

        def get_raw_confidence(tmpl_path):
            if not Path(tmpl_path).exists():
                print(f"⚠️ 模板文件不存在: {tmpl_path}")
                return -1.0, None
            tmpl = cv2.imread(tmpl_path, cv2.IMREAD_GRAYSCALE)

            # 异常检查：如果模板被裁得比整个 ROI 框还要大，OpenCV 会直接报错！
            if image_gray.shape[0] < tmpl.shape[0] or image_gray.shape[1] < tmpl.shape[1]:
                print(f"❌ 严重错误: 模板({tmpl.shape[1]}x{tmpl.shape[0]})比青色的搜索框({image_gray.shape[1]}x{image_gray.shape[0]})还要大，无法进行匹配！请把青色搜索框调大，或者把模板图裁小一点！")
                return -1.0, None

            res = cv2.matchTemplate(image_gray, tmpl, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            # 返回相对于全图的坐标
            return max_val, (max_loc[0] + star_roi[0], max_loc[1] + star_roi[1], tmpl.shape[1], tmpl.shape[0])

        conf_3, loc_3 = get_raw_confidence(template_3star)
        conf_2, loc_2 = get_raw_confidence(template_2star)
        conf_0, loc_0 = get_raw_confidence(template_0star)

        # 组装展示画面
        display_frame = frame.copy()

        # 画出 ROI 大框（青色）
        cv2.rectangle(display_frame, (star_roi[0], star_roi[1]), (star_roi[2], star_roi[3]), (255, 255, 0), 2)
        cv2.putText(display_frame, "Star ROI", (star_roi[0] + 5, star_roi[1] + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        # 在屏幕左下角打印实时的 3 个置信度
        text_y = h - 100
        # 如果置信度大于 0.75，则标为绿色（判定成功），否则为红色（判定失败）
        color_3 = (0, 255, 0) if conf_3 >= 0.75 else (0, 0, 255)
        color_2 = (0, 255, 0) if conf_2 >= 0.75 else (0, 0, 255)
        color_0 = (0, 255, 0) if conf_0 >= 0.75 else (0, 0, 255)

        cv2.putText(display_frame, f"3-Star Conf: {conf_3:.3f}", (20, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_3, 2)
        cv2.putText(display_frame, f"2-Star Conf: {conf_2:.3f}", (20, text_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_2, 2)
        cv2.putText(display_frame, f"0-Star Conf: {conf_0:.3f}", (20, text_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_0, 2)

        # 找出当前三者中得分最高的一个，如果它 >= 0.75，就把它的框画出来
        max_conf = max(conf_3, conf_2, conf_0)
        best_loc = None
        best_name = ""

        if max_conf >= 0.75:
            if max_conf == conf_3:
                best_loc = loc_3
                best_name = "3-Star MATCHED!"
            elif max_conf == conf_2:
                best_loc = loc_2
                best_name = "2-Star MATCHED!"
            elif max_conf == conf_0:
                best_loc = loc_0
                best_name = "0-Star MATCHED!"

            if best_loc is not None:
                # 画出最终匹配命中的框（粗绿框）
                x, y, tw, th = best_loc
                cv2.rectangle(display_frame, (x, y), (x + tw, y + th), (0, 255, 0), 3)
                cv2.putText(display_frame, best_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)


        cv2.imshow("Star Matching Test", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\n⏸️ 退出测试")
            break
        elif key == ord('s'):
            # 存图调试
            save_path = f"star_debug_{int(time.time())}.png"
            cv2.imwrite(save_path, display_frame)
            print(f"\n📸 截图已保存至: {save_path}")
            print(f"当前置信度记录：3星({conf_3:.3f}) | 2星({conf_2:.3f}) | 0星({conf_0:.3f})")

        time.sleep(interval)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
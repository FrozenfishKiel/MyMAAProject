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

# 输出目录
OUTPUT_DIR = ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

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

        # 保存原始截图
        raw_screenshot_path = OUTPUT_DIR / "raw_screenshot.png"
        cv2.imwrite(str(raw_screenshot_path), image)
        print(f"[INFO] 原始截图已保存至: {raw_screenshot_path.name}")

        # --- 核心：解析手牌区亮度 ---
        h_raw, w_raw = image.shape[:2]

        # 【1. 上下长度大一点】：扩大 Y 轴的检测范围 (比如从 75% 到 98%)
        y_start, y_end = int(h_raw * 0.8), int(h_raw * 0.99)

        # 【2. 左右总体跨度宽一点】：从更靠左的地方开始，到更靠右的地方结束
        x_start, x_end = int(w_raw * 0.001), int(w_raw * 0.97)

        # 保存手牌区域标记图
        hand_area_img = image.copy()
        cv2.rectangle(hand_area_img, (x_start, y_start), (x_end, y_end), (255, 0, 0), 3)
        hand_area_path = OUTPUT_DIR / "hand_area_marked.png"
        cv2.imwrite(str(hand_area_path), hand_area_img)
        print(f"[INFO] 手牌区域标记已保存至: {hand_area_path.name}")

        # 总宽度和卡牌数量
        num_cards = 12
        total_width = x_end - x_start

        # 算出每一个“卡位”（包含卡牌+间隔）占的理论总宽度
        slot_width = total_width / num_cards

        # 【3. 彼此之间有点间隔】：定义单张卡牌的实际宽度
        # 假设每张卡占据它卡位空间的 80%，剩下的 20% 就是卡与卡之间的物理间隔
        card_actual_width = slot_width * 0.80

        # --- 预加载 8 种职业模板 ---
        template_dir = ROOT.parent / "data" / "templates"
        class_templates = {}
        class_names = {
            "Caster": "术士",
            "Medic": "医疗",
            "Pioneer": "先锋",
            "Sniper": "狙击",
            "Special": "特种",
            "Support": "辅助",
            "Tank": "重装",
            "Warrior": "近卫"
        }

        # 职业到 Channel 4 的索引映射 (0-7)
        # 根据我们之前的讨论，我们需要给每个职业分配一个固定的数字ID
        class_id_map = {
            "Pioneer": 0, "Warrior": 1, "Sniper": 2, "Caster": 3,
            "Tank": 4, "Medic": 5, "Support": 6, "Special": 7
        }

        for role_key, role_name in class_names.items():
            tmpl_path = template_dir / f"BattleOperRole{role_key}.png"
            if tmpl_path.exists():
                # 以灰度图模式读取模板，匹配时更稳定
                tmpl_img = cv2.imread(str(tmpl_path), cv2.IMREAD_GRAYSCALE)
                class_templates[role_key] = tmpl_img
            else:
                print(f"[WARNING] 找不到模板文件: {tmpl_path}")

        print("-" * 80)
        print("卡位 | 平均亮度 | 判定状态 | 识别职业 (匹配度)")
        print("-" * 80)

        debug_img = image.copy()
        playable_vector = []
        class_id_vector = [] # 新增：存放识别出的职业ID

        for i in range(num_cards):
            # 计算这个卡槽的正中心 X 坐标
            slot_center_x = x_start + int((i + 0.5) * slot_width)

            # 以中心点为基准，向左右延伸真实卡牌宽度的对应距离
            card_x1 = int(slot_center_x - card_actual_width / 2)
            card_x2 = int(slot_center_x + card_actual_width / 2)

            # 确保不会数组越界
            card_x1 = max(0, card_x1)
            card_x2 = min(w_raw, card_x2)

            # 直接从原图截取这单一一张卡牌的图像
            single_card_roi = image[y_start:y_end, card_x1:card_x2]
            single_card_gray = cv2.cvtColor(single_card_roi, cv2.COLOR_BGR2GRAY)

            avg_brightness = np.mean(single_card_gray)
            
            # 在调试图像上添加亮度值和处理信息
            brightness_text = f"亮度: {avg_brightness:.1f}"
            cv2.putText(debug_img, brightness_text,
                        (card_x1 + 5, y_start + 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # 亮度阈值
            is_playable = "✅ 可用" if avg_brightness > 90 else "❌ 不可用/费用不足"
            playable_vector.append(1 if avg_brightness > 90 else 0)

            # --- 模板匹配职业 ---
            detected_class = "未知"
            detected_id = -1
            best_val = -1.0

            # 只有卡牌亮着的时候，才去匹配它是什么职业
            if avg_brightness > 90 and single_card_gray.shape[0] > 10 and single_card_gray.shape[1] > 10:
                for role_key, tmpl_img in class_templates.items():
                    # 确保模板尺寸不大于被搜索的图像尺寸
                    th, tw = tmpl_img.shape[:2]
                    sh, sw = single_card_gray.shape[:2]

                    if th <= sh and tw <= sw:
                        res = cv2.matchTemplate(single_card_gray, tmpl_img, cv2.TM_CCOEFF_NORMED)
                        _, max_val, _, _ = cv2.minMaxLoc(res)

                        if max_val > best_val:
                            best_val = max_val
                            detected_class = class_names[role_key]
                            detected_id = class_id_map[role_key]

            # 如果最高匹配度不到 0.5，大概率是误判，或者那里根本没放干员
            if best_val < 0.5:
                 detected_class = "未知"
                 detected_id = -1

            class_id_vector.append(detected_id)

            print(f"[{i:2d}]  | {avg_brightness:8.2f} | {is_playable:12s} | {detected_class} ({best_val:.2f})")

            # 画框，直接画在 debug_img 上
            color = (0, 255, 0) if avg_brightness > 90 else (0, 0, 255)
            cv2.rectangle(debug_img, (card_x1, y_start), (card_x2, y_end), color, 2)

            # 在原有的亮度文字旁边加上职业名称
            text_to_put = f"{int(avg_brightness)}"
            if detected_id != -1:
                # opencv 默认不支持中文输出，所以我们用拼音首字母/英文缩写来显示，避免乱码
                text_to_put += f" {detected_class}"

            # 为了能在图上显示中文，我们需要用到PIL，但为了保持测试脚本简单，图上暂时显示英文/数字
            cv2.putText(debug_img, text_to_put,
                        (card_x1 + 5, y_start + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        print("-" * 80)
        print(f"状态向量 (亮度): {playable_vector}")
        print(f"职业ID向量 (-1=未知,0=先锋..): {class_id_vector}")

        # 创建亮度热力图
        hand_area = image[y_start:y_end, x_start:x_end]
        hand_area_gray = cv2.cvtColor(hand_area, cv2.COLOR_BGR2GRAY)
        # 应用热力图颜色映射
        heatmap = cv2.applyColorMap(hand_area_gray, cv2.COLORMAP_JET)
        # 将热力图放回原图像位置
        heatmap_overlay = image.copy()
        heatmap_overlay[y_start:y_end, x_start:x_end] = heatmap
        # 保存热力图
        heatmap_path = OUTPUT_DIR / "brightness_heatmap.png"
        cv2.imwrite(str(heatmap_path), heatmap_overlay)
        print(f"[INFO] 亮度热力图已保存至: {heatmap_path.name}")

        # 保存最终调试图像
        save_path = OUTPUT_DIR / "live_brightness_test.png"
        cv2.imwrite(str(save_path), debug_img)
        print(f"\n[INFO] 🎉 调试图片已保存至 output 目录: {save_path.name}")
        print("[INFO] 请打开该图片，检查我们切的框是否刚好套住了底部的干员卡牌，以及红绿框判定是否正确！")
        print("[INFO] 亮度热力图展示了卡牌的亮度分布，红色表示亮度高，蓝色表示亮度低。")
    else:
        print("[ERROR] 设备连接失败")

if __name__ == "__main__":
    test_live_brightness()
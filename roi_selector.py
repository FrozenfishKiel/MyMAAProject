import cv2
import sys
from pathlib import Path

def main():
    # 默认读取 screen.png，如果通过命令行传入了路径则使用传入的路径
    image_path = sys.argv[1] if len(sys.argv) > 1 else "screen.png"

    if not Path(image_path).exists():
        print(f"❌ 找不到图片: {image_path}")
        print("请用模拟器自带的截图功能，截一张全屏图并命名为 screen.png 放在当前目录下。")
        print("或者在命令行指定图片路径: python roi_selector.py <你的图片路径>")
        return

    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ 读取图片失败，请检查文件是否损坏: {image_path}")
        return

    print("\n=== 🎯 MAA ROI 框选工具 ===")
    print("操作说明：")
    print("1. 鼠标左键拖动，框住你想要的区域（比如红色的'开始行动'按钮周边一点点）")
    print("2. 按下 SPACE(空格) 或 ENTER(回车) 确认选中区域")
    print("3. 按下 c 取消并重新框选")
    print("4. 按下 ESC 退出工具\n")

    # 弹出窗口让用户框选
    window_name = "Select ROI (Drag mouse, press ENTER to confirm)"
    roi = cv2.selectROI(window_name, img, showCrosshair=True, fromCenter=False)
    cv2.destroyAllWindows()

    if roi == (0, 0, 0, 0):
        print("⚠️ 未选择任何区域，程序退出。")
    else:
        # cv2.selectROI 返回的是 (x, y, w, h)
        x, y, w, h = roi
        # 我们需要转换成 (x1, y1, x2, y2) 格式
        x1, y1 = x, y
        x2, y2 = x + w, y + h

        print("\n==========================================")
        print("✅ 框选成功！请将以下坐标复制到 game_env.py 中替换旧的 ROI：")
        print(f"坐标格式 (x1, y1, x2, y2) -> ({x1}, {y1}, {x2}, {y2})")
        print("例如：")
        print(f"start_roi = ({x1}, {y1}, {x2}, {y2})")
        print("==========================================\n")

if __name__ == "__main__":
    main()

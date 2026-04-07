import torch
import sys

def main():
    print("="*40)
    print("🔍 环境检查报告 (YOLO 训练前置检查)")
    print("="*40)

    # 检查 Python
    print(f"✅ Python 版本: {sys.version.split()[0]}")

    # 检查 PyTorch 和 CUDA
    print(f"✅ PyTorch 版本: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"✅ CUDA 状态: 可用 (GPU 训练已开启)")
        print(f"✅ 显卡型号: {torch.cuda.get_device_name(0)}")
    else:
        print(f"❌ CUDA 状态: 不可用 (将使用 CPU 训练，速度极慢)")
        print("   建议: 如果您有 N 卡，请安装对应版本的 CUDA 和 PyTorch。")

    # 检查 Ultralytics
    try:
        import ultralytics
        print(f"✅ Ultralytics (YOLO) 状态: 已安装 (版本 {ultralytics.__version__})")
    except ImportError:
        print(f"❌ Ultralytics (YOLO) 状态: 未安装")
        print("   解决: 请运行命令 `pip install ultralytics`")

    print("="*40)

if __name__ == "__main__":
    main()

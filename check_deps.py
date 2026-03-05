import sys
sys.path.insert(0, 'maa-deps/maafw-5.2.6-win_amd64')

print("=== 检查 maa-auto 环境依赖 ===")

try:
    import torch
    print(f"✓ torch 版本: {torch.__version__}")
except ImportError:
    print("✗ torch 未安装")

try:
    import torchvision
    print(f"✓ torchvision 版本: {torchvision.__version__}")
except ImportError:
    print("✗ torchvision 未安装")

try:
    import scipy
    print(f"✓ scipy 版本: {scipy.__version__}")
except ImportError:
    print("✗ scipy 未安装")

try:
    import numpy
    print(f"✓ numpy 版本: {numpy.__version__}")
except ImportError:
    print("✗ numpy 未安装")

try:
    import cv2
    print(f"✓ opencv-python 版本: {cv2.__version__}")
except ImportError:
    print("✗ opencv-python 未安装")

try:
    import maa
    print("✓ MaaFramework 已导入")
except ImportError as e:
    print(f"✗ MaaFramework 导入失败: {e}")

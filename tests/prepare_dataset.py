
import os
import shutil
import random
from pathlib import Path

# ================= 配置区 =================
ROOT = Path(r"D:\BiShe\MaAutomaton-main\MaAutomaton-main")
# 原始截图和标注的存放地
SOURCE_DIR = ROOT / "data" / "Source"
SOURCE_LABELS_DIR = SOURCE_DIR / "labels"

# YOLO 标准数据集结构
DATASET_DIR = ROOT / "data" / "dataset"
TRAIN_IMAGES = DATASET_DIR / "images" / "train"
VAL_IMAGES = DATASET_DIR / "images" / "val"
TRAIN_LABELS = DATASET_DIR / "labels" / "train"
VAL_LABELS = DATASET_DIR / "labels" / "val"

# 验证集比例 (20%)
VAL_RATIO = 0.2
# ==========================================

def main():
    print("="*50)
    print("📦 YOLO 数据集自动整理脚本")
    print(f"📥 数据源: {SOURCE_DIR}")
    print(f"📤 目标数据集: {DATASET_DIR}")
    print("="*50)

    # 1. 确保所有目标目录存在并清空旧数据(防止多次运行文件堆积)
    for dir_path in [TRAIN_IMAGES, VAL_IMAGES, TRAIN_LABELS, VAL_LABELS]:
        dir_path.mkdir(parents=True, exist_ok=True)
        # 清空已有文件
        for f in dir_path.glob("*"):
            if f.is_file():
                f.unlink()

    # 2. 从 Source 目录扫描所有标注好的数据
    if not SOURCE_LABELS_DIR.exists():
        print(f"❌ 错误: 标签目录 {SOURCE_LABELS_DIR} 不存在！")
        return

    # 找到所有 txt 文件 (除了 classes.txt)
    label_files = [f for f in SOURCE_LABELS_DIR.glob("*.txt") if f.name != "classes.txt"]

    if not label_files:
        print(f"⚠️ 警告: 在 {SOURCE_LABELS_DIR} 中没有找到任何 .txt 标注文件！")
        print("请确保你已经在 labelImg 中完成了标注，并且将格式选为了 YOLO。")
        return

    # 检查对应的图片文件是否存在
    valid_pairs = []
    for txt_path in label_files:
        # 尝试去 Source 目录找同名的 .jpg 或 .png
        img_path_jpg = SOURCE_DIR / txt_path.with_suffix('.jpg').name
        img_path_png = SOURCE_DIR / txt_path.with_suffix('.png').name

        if img_path_jpg.exists():
            valid_pairs.append((img_path_jpg, txt_path))
        elif img_path_png.exists():
            valid_pairs.append((img_path_png, txt_path))
        else:
            print(f"⚠️ 警告: 找到标注文件 {txt_path.name} 但在 {SOURCE_DIR} 中没有找到对应的图片，已跳过。")

    total_pairs = len(valid_pairs)
    print(f"✅ 找到 {total_pairs} 组有效的数据 (图片+标注)")

    if total_pairs == 0:
        return

    # 3. 随机划分训练集和验证集
    random.shuffle(valid_pairs)
    val_count = max(1, int(total_pairs * VAL_RATIO))
    val_pairs = valid_pairs[:val_count]
    train_pairs = valid_pairs[val_count:]

    print(f"📊 数据划分: {len(train_pairs)} 训练集, {len(val_pairs)} 验证集 (比例: {VAL_RATIO*100}%)")

    # 4. 执行复制
    # 复制验证集
    for img_path, txt_path in val_pairs:
        shutil.copy2(img_path, VAL_IMAGES / img_path.name)
        shutil.copy2(txt_path, VAL_LABELS / txt_path.name)

    # 复制训练集
    for img_path, txt_path in train_pairs:
        shutil.copy2(img_path, TRAIN_IMAGES / img_path.name)
        shutil.copy2(txt_path, TRAIN_LABELS / txt_path.name)

    print(f"✅ 文件已成功分发到 dataset/ 的子目录中！")

    # 5. 自动生成 data.yaml
    yaml_path = DATASET_DIR / "data.yaml"

    # 注意：YOLO 的 yaml 配置文件中，路径最好使用正斜杠
    dataset_posix_path = DATASET_DIR.as_posix()

    yaml_content = f"""path: {dataset_posix_path}
train: images/train
val: images/val
nc: 1
names: ['operator_hp_bar']
"""
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)

    print(f"✅ 已生成 YOLO 训练配置文件: {yaml_path}")
    print("="*50)
    print("🚀 准备就绪！你可以使用以下命令开始训练:")
    print(f"yolo task=detect mode=train model=yolov8n.pt data={dataset_posix_path}/data.yaml epochs=100 imgsz=640 batch=16")
    print("="*50)

if __name__ == "__main__":
    main()

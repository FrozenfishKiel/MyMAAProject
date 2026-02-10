# 数据目录说明

## 目录结构
```
data/
├── raw/                    # 原始数据
│   ├── images/             # 原始图像文件
│   └── videos/             # 原始视频文件
├── processed/              # 处理后的数据
│   ├── images/             # 预处理后的图像
│   └── annotations/        # 处理后的标注文件
├── datasets/               # 数据集
│   ├── train/              # 训练集
│   │   ├── images/         # 训练图像
│   │   └── labels/         # 训练标注（YOLO格式）
│   ├── val/                # 验证集
│   │   ├── images/         # 验证图像
│   │   └── labels/         # 验证标注
│   └── test/               # 测试集
│       ├── images/         # 测试图像
│       └── labels/         # 测试标注
├── models/                 # 训练好的模型
│   ├── yolo/               # YOLO模型
│   └── ocr/                # OCR模型
├── temp/                   # 临时文件
└── logs/                   # 训练日志
```

## 使用说明

### 1. 数据准备流程
1. 将原始截图放入 `raw/images/`
2. 使用labelImg进行标注，标注文件保存到 `processed/annotations/`
3. 将标注好的数据按比例划分到 `datasets/train/` 和 `datasets/val/`

### 2. YOLO格式要求
- 图像格式：JPG/PNG/BMP
- 标注格式：每个图像对应一个.txt文件
- 标注内容：`class_id x_center y_center width height`（归一化坐标）

### 3. labelImg使用
在项目根目录运行：
```bash
D:\Anaconda3\envs\maa-project\Scripts\labelImg.exe
```

### 4. 数据集配置文件
创建 `data.yaml` 文件：
```yaml
path: ../data/datasets/
train: train/images/
val: val/images/

nc: 10  # 类别数量
names: ['start_button', 'pause_button', 'settings_button', 'level_select', 'character_select', 'skill_icon', 'health_bar', 'mana_bar', 'dialog_box', 'mission_complete']
```
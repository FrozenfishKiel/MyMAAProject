# 操作手册

## 1. 环境准备

### 1.1 Python

- 建议 Python 3.10+

安装依赖：

```bash
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

### 1.2 Maa 运行时依赖

本仓库默认携带 Windows x64 运行时：`maa-deps/maafw-5.2.6-win_amd64/`。

`src/main.py` 会在启动时把该目录加入 `sys.path`，优先使用离线依赖。

### 1.3 设备/游戏

#### Windows 端游（Win32）

- 建议使用窗口模式，固定分辨率，避免窗口大小变化导致坐标与 ROI 偏移
- 需要获得目标窗口句柄 `HWND`

#### 手游（ADB）

- 确保 `adb` 可用
- 模拟器/真机地址可通过 `adb devices` 获取（例如 `127.0.0.1:5555`）

## 2. 运行方式

### 2.1 Dry Run（不操作设备）

用于验证任务链语法、报告生成与回归：

```bash
python src/main.py --config src/task-config/sample.yaml --dry-run
```

### 2.2 Win32 真机运行

创建一个配置文件（示例）：

```yaml
project:
  name: win32-demo
device:
  type: win32
  hwnd: 123456
tasks:
  - type: tap
    x: 100
    y: 200
```

运行：

```bash
python src/main.py --config path/to/win32.yaml
```

### 2.3 ADB 真机/模拟器运行

配置示例：

```yaml
project:
  name: adb-demo
device:
  type: adb
  adb_path: C:/Android/platform-tools/adb.exe
  address: 127.0.0.1:5555
tasks:
  - type: tap
    x: 540
    y: 960
```

运行：

```bash
python src/main.py --config path/to/adb.yaml
```

## 3. 输出与交付

默认输出目录：`report/out/`

- `report.html`：步骤级别执行结果
- `summary.xlsx`：汇总表
- `run.json`：结构化数据

交付建议：

- 按“用例集/版本号/设备环境”归档输出目录
- 与 `src/task-config` 一起打包，确保可复现

## 4. 常见问题排查

### 4.1 连接失败

- Win32：确认 `hwnd` 有效，目标窗口存在且未最小化
- ADB：确认 `adb_path` 正确、`address` 可连通、设备已授权

### 4.2 点击无效

- 检查坐标系：分辨率变化会导致点击偏移
- 检查窗口是否置顶/可交互

### 4.3 报告未生成

- 确认有写入权限
- 检查 `--out` 指定目录是否存在（不存在会自动创建）

## 5. 合规与风险提示

- 仅在获得授权的测试环境中使用自动化控制与采集能力
- 不将该项目用于破坏、绕过或干扰目标软件的安全机制


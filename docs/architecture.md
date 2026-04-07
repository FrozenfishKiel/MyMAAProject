# 项目架构

## 目标与边界

- 输入：游戏可视化画面（截图/视频帧）、外部设备控制能力（窗口/ADB）
- 输出：任务执行日志、用例通过率、HTML/Excel 报告
- 约束：不依赖游戏源码接口；以 MaaFramework 的控制/采集/基础识别为核心，AI 仅用于补齐复杂识别与自主决策

## 分层架构

### 运行时主链路

`src/main.py` → 加载配置 → 选择适配器（DryRun/Maa）→ 运行任务链 → 收集 StepResult → 生成报告

### 模块层次

1) 接入层（入口/编排）
- `src/main.py`：配置加载、运行模式选择、报告落盘

2) Maa 封装层（设备/动作）
- `src/maa-wrapper/runtime.py`：
  - `BotRuntime`：执行任务链，生成结构化结果
  - `BaseAdapter`：动作抽象（点击/等待等）
  - `DryRunAdapter`：不触发真实操作，用于回归与 CI
  - `MaaFwAdapter`：基于 Maa 的 Controller 进行连接与点击

3) 任务配置层（低代码任务链）
- `src/task-config/*.yaml`：用 YAML/JSON 描述任务步骤（可扩展条件分支/循环/失败恢复）

4) AI 插件层（增强识别/决策）
- `src/ai-plugins/yolo_recognizer.py`：目标检测结果标准化（label/conf/box）
- `src/ai-plugins/rl_policy.py`：策略模型输出标准化（action/params）

5) 决策层（规则 + AI 混合）
- `src/decision/hybrid_decision.py`：按模式选择“固定任务链”或“AI 生成任务链”（待接入）

6) 报告层（可视化与归档）
- `src/report/report_generator.py`：输出 `run.json`、`report.html`、`summary.xlsx`

## 数据流

1) 配置流
- `task-config` → `main.py` 解析 → `BotRuntime.run_tasks()`

2) 执行动作流
- `BotRuntime` → `Adapter.tap/sleep` →
  - DryRun：记录动作
  - Maa：`MaaFwAdapter` → `maa.controller.*Controller` → `post_*().wait()`

3) 结果与报告流
- 每一步生成 `StepResult`（耗时/成功/错误）→ `write_reports()` → HTML/Excel/JSON

## 关键扩展点

### 1) 任务链 DSL 扩展

- 在 `src/maa-wrapper/runtime.py` 的 `_run_step()` 增加新 step 类型
- 推荐演进方向：
  - 条件：基于识别结果/计数器/超时
  - 循环：直到某个 UI 状态出现
  - 恢复：失败时回退/重试/切换识别策略

### 2) 识别策略扩展

- 基础识别：优先走 Maa 内置模板匹配/OCR
- 复杂识别：将 Maa 采集到的图像输入 YOLO，输出坐标/置信度，转换为任务可用的点击目标

### 3) 决策策略扩展

- 固定流程：`task-config` 驱动（适合新手引导/固定 UI 流程）
- 自主探索：RL 输出“下一步动作”，再映射为可执行的任务步骤（点击/滑动/等待）

## 配置约定（当前实现）

`src/task-config/sample.yaml` 展示了最小结构：

- `project.name`：项目名
- `device.mode`：`dry_run` 时强制不连设备
- `device.type`：`win32` 或 `adb`（真实运行时必填）
- `device.hwnd`：Win32 模式下必填
- `device.adb_path` / `device.address`：ADB 模式下必填
- `tasks[]`：步骤列表（当前支持 `wait/tap/assert`）

## 报告产物

- `report/out/run.json`：结构化执行结果（便于二次分析/可视化）
- `report/out/report.html`：人类可读的步骤表
- `report/out/summary.xlsx`：适合汇总与交付


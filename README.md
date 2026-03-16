# maa-game-test-ai-bot

基于 MaaFramework 的无源码游戏自动化测试 Bot（Maa 负责设备/采集/基础识别/动作，AI 负责复杂识别与自主决策增强）。

## 目录结构

```
maa-game-test-ai-bot/
├── docs/
├── src/
│   ├── maa-wrapper/
│   ├── ai-plugins/
│   ├── task-config/
│   ├── decision/
│   ├── report/
│   └── main.py
├── models/
├── data/
├── maa-deps/
├── tests/
├── requirements.txt
└── Dockerfile
```

## 快速启动

1) 创建虚拟环境并安装依赖：

```bash
python -m venv .venv
.venv\\Scripts\\python -m pip install -r requirements.txt
```

2) 运行（默认 dry-run，不会真实操作设备）：

```bash
python src/main.py --config src/task-config/sample.yaml --dry-run
```

3) 输出：

- `report/out/report.html`
- `report/out/summary.xlsx`

## 文档

- 项目架构：`docs/architecture.md`
- 操作手册：`docs/operations-manual.md`
- 训练手册：`docs/training-manual.md`
- Maa 学习分析：`docs/maa-analysis.md`
- Maa 编译部署：`docs/maa-compile.md`

## 许可证

本项目依赖 MaaFramework，使用时需遵循其上游仓库的许可证要求并保留版权与许可声明。

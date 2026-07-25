# VRP 项目每日进度记录

本文件用于按日期简短记录 ZJU/ZJUI VRP assessment 项目的推进情况。

## 2026-07-25

### 今日进展

- 完成中文技术学习笔记，并生成 Markdown 与 PDF。
- 解析官方 `.pkl` 数据，生成数据预览和样本 0 完整预览。
- 梳理并解释 CVRP 数据格式、可行性规则、评价指标和常见方法。
- 确定下一阶段先做 CPU 可运行的容量感知最近邻 baseline，之后再加 2-opt。
- 编写并提交 baseline solver 中文设计规格。

### 关键决策

- 先保证可行、可提交、可验证，再追求更低 cost。
- 第一版 solver 不引入 OR-Tools、不训练神经网络、不依赖 CUDA；当前电脑以 CPU 为准，后续训练可转到 RTX 4060 笔记本。
- `solve.py` 保留 `--device` 参数；当前 heuristic baseline 可忽略 GPU。
- JSON 输出使用 1-based customer 编号，且不写入 depot。
- 原始数据目录 `VRP_project/` 不提交到 git。
- README 最适合在 `solve.py`、测试命令和 smoke test 稳定后再补充。

### 今日产出文件

- `docs/vrp_project_technical_learning_note_zh.md`
- `docs/vrp_project_technical_learning_note_zh.pdf`
- `docs/data_preview/vrp_pkl_preview.md`
- `docs/data_preview/train_data_sample0_full_preview.md`
- `docs/superpowers/specs/2026-07-25-vrp-baseline-solver-design.zh.md`
- `docs/progress/daily-progress.md`

### 今日提交

- `3b3457d docs: add VRP technical learning note`
- `0875e6d docs: add readable VRP pickle preview`
- `51e4fdd docs: improve VRP pickle preview readability`
- `974879d docs: clarify VRP preview truncation`
- `10b100e docs: add VRP baseline solver design`

### 状态与下一步

- 已完成技术学习和 baseline 设计准备，尚未开始实现代码。
- 等待 baseline solver 规格 review；通过后生成 implementation plan，并实现数据读取、评估工具、最近邻 baseline、`solve.py` 和测试。

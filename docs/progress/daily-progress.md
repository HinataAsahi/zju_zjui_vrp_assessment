# VRP 项目每日进度记录

本文件用于按日期记录 ZJU/ZJUI VRP assessment 项目的推进情况。每天更新时建议追加新的日期小节，记录当天完成事项、关键决策、产出文件、提交记录和下一步。

## 2026-07-25

### 今日目标

从项目理解阶段推进到 baseline 实现准备阶段：先完成技术学习笔记和数据理解，再明确下一阶段的 CPU baseline solver 设计。

### 已完成事项

- 生成并整理了中文技术学习笔记，帮助理解 CVRP 任务、数据结构、可行性规则、评价指标、baseline 思路和后续 AI 方法。
- 为学习笔记生成了 Markdown 和 PDF 两种格式。
- 解析并预览了官方 `.pkl` 数据，确认 VSCode 不能直接阅读 `.pkl` 是因为它是 Python pickle 二进制文件。
- 生成了数据预览文档，并补充了样本 0 的完整 customer 预览，便于核对 50 个 customer 的坐标、demand、reference routes 和 cost。
- 解释并确认了几个核心概念：
  - `loc_preview` / `demand_preview` 默认只是截断预览，不代表全部 customer。
  - `capacity` 使用浮点数主要是因为数据加载后可能表现为数值张量或浮点类型，容量约束仍按数值比较处理。
  - 非学习式 CVRP 方法不需要使用 reference labels 进行训练。
  - 贪心插入、route split、OR-Tools、POMO-style 强化学习、supervised imitation 的基本含义。
- 明确了下一阶段优先做“方案 1”：CPU 可运行、确定性、容量感知最近邻 baseline。
- 明确了后续再尝试加入 2-opt，而不是在第一版 baseline 中同时实现。
- 明确了当前工作电脑使用 CPU；后续训练可迁移到带 RTX 4060 的拯救者笔记本上运行。
- 编写并提交了 baseline solver 的中文设计规格，作为后续 implementation plan 的依据。

### 关键决策

- 先保证可行、可提交、可验证，再追求更低 cost。
- 第一版 solver 不引入 OR-Tools、不训练神经网络、不依赖 CUDA。
- `solve.py` 仍保留 `--device` 参数，以兼容官方接口和后续扩展；当前 heuristic baseline 会忽略 GPU。
- 内部可以使用 0-based 下标，但 JSON 输出必须使用 1-based customer 编号。
- depot 是隐含起点和终点，不写入输出 route。
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

### 当前状态

- 项目已经完成技术学习和 baseline 设计准备。
- 当前还未开始实现 `src/vrp_io.py`、`src/vrp_eval.py`、`src/heuristics.py` 和 `solve.py`。
- `main` 分支已与 GitHub 远端同步。
- 工作区仍有未跟踪的原始数据目录 `VRP_project/`，它应继续保持不入库。

### 下一步

等待 baseline solver 中文设计规格被 review。通过后，按流程调用 `superpowers:writing-plans`，生成 implementation plan，然后开始实现：

- 数据读取与内部实例表示。
- cost 计算与可行性检查。
- 容量感知最近邻 baseline。
- 官方入口 `solve.py`。
- 单元测试和 CLI 集成测试。
- 本地 smoke test。

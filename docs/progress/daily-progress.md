# VRP 项目每日进度记录

本文件用于按日期简短记录 ZJU/ZJUI VRP assessment 项目的推进情况。

## 2026-08-02

- 回收 RTX 4060 上的 priority RL 正式微调结果，训练约 13 小时。
- 规范化复制回来的 checkpoint 文件名，并完成本机 full validation 评估。
- RL full validation gap 为 0.158813，check 预测 1500/1500 可行。
- 新增中文最终技术报告和答辩提纲，明确默认提交方案与学习路线结论。
- 按官方 outline 补充提交清单，并准备 CVRP 路线与方法对比可视化。

### 状态与下一步

- RL 较 `mse_pairwise` priority 有小幅提升，但明显弱于默认 heuristic。
- 默认提交继续使用 `nearest_2opt_relocate_limited_swap`。
- 后续可基于最终报告、提交清单和可视化图整理 PPT。

## 2026-07-31

- 设计并实现 priority RL 微调入口，完成本地 smoke 验证。

### 状态与下一步

- 下一步在 RTX 4060 上从 pairwise imitation checkpoint 继续正式 RL 微调；默认提交仍使用 heuristic。

## 2026-07-30

### 今日进展

- 将 priority 训练日志升级为实时 batch 进度条，保留 `stdout` 最终 JSON 输出。
- 更新 `AGENTS.md`：后续 AI training 代码默认需要终端进度日志和进度条。
- 回收第二版 `mse_pairwise` 训练结果：略优于 MSE 第一版，但仍明显弱于 heuristic 默认。

### 状态与下一步

- 默认提交继续使用 heuristic；priority 学习路线暂不替代默认方法。

## 2026-07-29

### 今日进展

- 完成客户优先级 oracle 测试：参考优先级 + 容量切分可保证 1000/1000 validation 可行。
- 确认客户优先级必须接完整 `2-opt + relocate + swap` 后处理，oracle 平均 cost 可到 10.995458085453048。
- 落地 supervised imitation 第一版：优先级标签、特征、模型、训练 CLI、评估 CLI 和测试。
- 编写 GPU 迁移运行指令，覆盖拉取仓库、检查数据、训练、评估和生成预测结果。
- 完成 review 修正：训练验证集限制必须为正，补充 unlabeled 评估测试，并收紧迁移文档格式。
- 回收第一版 MSE priority 训练结果：可行但弱于当前 heuristic，不能替换默认提交。
- 增加可选 `mse_pairwise` 训练目标，准备在 4060 上做第二版 priority 实验。

### 状态与下一步

- 当前 heuristic 默认仍保持 `nearest_2opt_relocate_limited_swap` 5/2。
- 客户优先级模型仍是实验线；下一步训练 `priority_mse_pairwise_rank.pt` 后再对比默认 heuristic。

## 2026-07-28

### 今日进展

- 审核通过 route 间 best relocate 设计，并生成 implementation plan。
- 实现可选方法 `nearest_2opt_relocate_best`，默认方法仍保持 `nearest_2opt`。
- 完成测试、官方 check 数据 smoke test 和完整验证集评估。
- 审核通过 limited relocate 默认候选设计，目标公开集总耗时约 180 秒。
- 实现可选方法 `nearest_2opt_relocate_limited`，默认方法暂时仍保持 `nearest_2opt`。
- 完成 validation 评估和公开 check 数据计时，记录是否进入默认切换 gate。
- 确认将 `nearest_2opt_relocate_limited` 升级为默认提交方法。
- 尝试 `nearest_2opt_relocate_candidate_limited`：保留 8/3 轮数限制，并将候选 target route 数量调到 2/2。
- 补测 candidate-limited 的 3/2 与 1/1 参数，确认 2/2 是目前较合适的速度-质量折中点。
- 实现可选方法 `nearest_2opt_relocate_limited_swap`，在 limited relocate 后做有限轮 route 间 swap。
- 确认将 `nearest_2opt_relocate_limited_swap` 升级为默认提交方法。
- 生成正式 `outputs/predictions.json`，并检查 1500/1500 个公开 check 实例可行。
- 微调 limited-swap 参数，确认 5/2 比 4/2、4/3、3/2 更适合作为默认。

### 状态与下一步

- 当前默认方法为 `nearest_2opt_relocate_limited_swap`。
- limited 方法通过默认切换 gate：validation 1000/1000 可行，平均 cost 低于 `nearest_2opt`，公开 check 耗时 76.35 秒。
- candidate-limited 2/2 方法 validation 平均 cost 为 11.677200534322004，公开 check 耗时 63.43 秒；速度更好但质量略差，默认方法暂不切换。
- candidate-limited 补测：3/2 公开 check 75.08 秒，1/1 公开 check 50.33 秒但 validation 平均 cost 退到 11.700284874443323。
- limited-swap 5/2 方法 validation 平均 cost 为 11.513862926474488，公开 check 耗时 135.13 秒；正式 `predictions.json` 生成耗时 148.00 秒。

## 2026-07-26

### 今日进展

- baseline solver 规格已审核通过。
- 生成并执行 baseline implementation plan。
- 实现 CPU 最近邻 baseline、官方 `solve.py`、评估脚本、测试和 README。
- 完整验证集 baseline 评估完成：1000/1000 可行，平均 gap 约 31.89%。
- baseline 相关提交已合并到 `main` 并推送到 GitHub。

### 状态与下一步

- 第一版可提交 baseline 已完成本地验证。
- 明天继续 2-opt 改进设计，优先考虑只做 route 内 2-opt。

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

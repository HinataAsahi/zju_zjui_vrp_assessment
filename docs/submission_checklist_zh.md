# VRP Assessment 提交清单

本文根据 `VRP_project/vrp_project_outline.docx` 整理，用于最终提交前核对。

## 项目要求覆盖情况

| Outline 要求 | 当前状态 |
| --- | --- |
| Data loading / preprocessing | 已实现：`src/vrp_io.py`, `src/priority_data.py` |
| Model training | 已实现：`scripts/train_priority_model.py`, `scripts/train_priority_rl.py` |
| Validation | 已实现：`scripts/evaluate_baseline.py`, `scripts/evaluate_priority_model.py` |
| Solution generation | 已实现：`solve.py` |
| Official JSON format | 已实现：`format_version` 为 `cvrp_v1`，customer 使用 1-based 编号 |
| Feasibility checking | 已实现：`src/vrp_eval.py` 和测试 |
| Report | 已准备：`docs/final_report_zh.md` |
| Presentation | 已准备提纲：`docs/presentation_outline_zh.md` |
| Public-set output | 已生成：`outputs/heuristic/predictions.json` |
| Visualization | 已准备：`docs/assets/` 下两张 PNG |

## 必须提交

| 提交项 | 当前项目对应文件 | 说明 |
| --- | --- | --- |
| Code | `solve.py`, `src/`, `scripts/`, `tests/`, `README.md` | 隐藏集评测会运行 `solve.py`。当前默认方法为 `nearest_2opt_relocate_limited_swap`。 |
| Public-set output | `outputs/heuristic/predictions.json` | 对 `check_data_to_students.pkl` 生成的预测结果，格式为 `cvrp_v1`。 |
| Report | `docs/final_report_zh.md` | 覆盖问题背景、方法、实验结果、AI 学习路线和最终决策。 |
| Presentation | `docs/presentation_outline_zh.md` | 可直接改写成 PPT。 |

官方要求邮件提交 code、public-set output 和 report，截止时间为 8 月 9 日 23:59。

## 建议一并提交或展示

| 文件 | 用途 |
| --- | --- |
| `docs/assets/cvrp_sample_solution.png` | 展示一组 CVRP 路线如何从 depot 出发并回到 depot。 |
| `docs/assets/method_comparison_gap.png` | 展示主要方法的 validation gap 对比。 |
| `docs/experiments/baseline_results.md` | 作为完整实验记录，支撑报告中的数值。 |
| `docs/vrp_project_technical_learning_note_zh.md` | 作为学习与方法理解材料，可不放入正式压缩包。 |
| `docs/vrp_project_technical_learning_note_zh.pdf` | 学习笔记 PDF 版本，可按需要附加。 |
| `checkpoints/priority_imitation/priority_mse_pairwise_rank.pt` | 若老师特别要求“提交训练模型”，可作为 AI priority 模型 checkpoint 附加。 |
| `checkpoints/priority_rl/priority_rl_finetune.pt` | 若需要展示 RL 微调模型，可作为实验 checkpoint 附加。 |

注意：当前最终默认 `solve.py` 不依赖 checkpoint；checkpoint 主要用于说明 AI 学习路线已经实现并测试过，但没有替换默认 heuristic。

## 不建议提交

| 文件或目录 | 原因 |
| --- | --- |
| `VRP_project/` | 原始数据目录，不应提交到代码仓库或重复打包。 |
| `VRP_project.zip` | 原始数据压缩包，不应重复提交。 |
| `.firecrawl/` | DOCX 解析中间文件。 |
| `.pytest_cache/`, `__pycache__/` | 本地缓存。 |
| `outputs/priority_imitation/`, `outputs/priority_rl/` | 实验输出较多，除非报告需要单独附加。 |
| `docs/remote_4060_tailscale_ssh.md` | 本地远程连接说明，不属于考核提交材料。 |
| `exports/` | 远程服务器结果导出中间目录，本次考核未使用。 |

## 最终提交前命令

```bash
# 1. 运行测试
python3 -m pytest tests -v

# 2. 重新生成 public-set heuristic 输出
python3 solve.py --input VRP_project/VRPData/check_data_to_students.pkl --output outputs/heuristic/predictions.json --device cuda:0 --seed 2026

# 3. 检查输出格式和可行性
python3 scripts/evaluate_baseline.py --input VRP_project/VRPData/validation_data.pkl --method nearest_2opt_relocate_limited_swap
```

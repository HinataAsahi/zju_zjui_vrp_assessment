# ZJU/ZJUI VRP Assessment 最终技术报告

## 1. 项目目标

本项目面向 ZJU/ZJUI VRP assessment，目标是在给定 CVRP 实例后输出可行车辆路径方案，并尽量降低总路径长度。项目最终提交入口为 `solve.py`，默认方法需要在 CPU 环境下稳定运行，同时保留 `--device` 参数以兼容官方调用格式。

本项目的实际策略是：先保证所有实例可行，再逐步降低 cost，并用 validation 数据和 public check 数据同时约束质量与运行时间。

## 2. 数据与评价

数据为 CVRP 实例，每个实例包含 depot 坐标、customer 坐标、customer demand、车辆容量，以及部分数据中的参考解标签。输出 JSON 中 route 使用 1-based customer 编号，不包含 depot。

核心评价指标如下：

| 指标 | 含义 |
| --- | --- |
| Feasible count | 满足容量约束且每个客户恰好访问一次的实例数量 |
| Average cost | 平均总路径长度，越低越好 |
| Average gap | 与参考标签相比的平均差距，越低越好 |
| Inference time | 单实例或整批推理时间，需要满足考核时间要求 |

## 3. 最终默认方法

最终默认方法为：

```text
nearest_2opt_relocate_limited_swap
```

该方法由四个阶段组成：

1. 容量感知最近邻：从 depot 出发，每次选择当前容量还能装下且距离最近的未访问客户。
2. Route 内 2-opt：在单条 route 内尝试反转一段访问顺序，减少交叉和绕路。
3. 有限轮 route 间 relocate：把某个客户从一条 route 移动到另一条 route 的更合适位置，只接受能降低 cost 且不违反容量的移动。
4. 有限轮 route 间 swap：在两条 route 间交换两个客户，只接受可行且能降低 cost 的交换。

默认参数为：

| 参数 | CVRP-50 | CVRP-100 |
| --- | ---: | ---: |
| Relocate pass budget | 8 | 3 |
| Swap pass budget | 5 | 2 |

这些限制用于控制运行时间，避免 full relocate/swap 在 public check 数据上过慢。

## 4. 主要实验结果

完整 validation 数据包含 1000 个实例。主要方法对比如下：

| 方法 | Feasible | Average cost | Average gap | Average inference time |
| --- | ---: | ---: | ---: | ---: |
| Nearest neighbor | 1000/1000 | 13.934951704742822 | 0.31894172859263836 | 0.00029439034105962494 |
| Nearest + 2-opt | 1000/1000 | 13.410121525149266 | 0.2688820821667571 | 0.000574803089039051 |
| Nearest + 2-opt + best relocate | 1000/1000 | 11.412079047307257 | 0.07837888104759934 | 0.1047532973489142 |
| Nearest + 2-opt + limited relocate | 1000/1000 | 11.675451253965944 | 0.10360446980377136 | 0.0543102543517598 |
| Nearest + 2-opt + limited relocate + limited swap | 1000/1000 | 11.513862926474488 | 0.08817281823708832 | 0.08248868217083509 |

Public check 数据上，最终默认方法生成 `outputs/heuristic/predictions.json` 时 1500/1500 个实例可行，平均 cost 为 `13.976807065714844`，正式生成耗时约 `148.00` 秒，满足约 180 秒的时间目标。

## 5. 为什么选择该默认方案

`nearest_2opt_relocate_best` 在 validation 上平均 gap 最低，但 public check 规模更大，完整 best relocate 的时间风险更高，不适合作为默认提交方法。

`nearest_2opt_relocate_limited` 速度较好，但质量明显弱于加入 limited swap 后的版本。

`nearest_2opt_relocate_candidate_limited` 进一步限制候选 route，运行更快，但 validation 平均 cost 略差。由于最终默认方法仍能满足时间目标，因此没有牺牲质量换更快速度。

因此，`nearest_2opt_relocate_limited_swap` 是当前最合适的默认提交方案：可行性稳定、质量明显优于早期 baseline，并且 public check 运行时间仍在目标范围内。

## 6. 学习方法探索

项目中也尝试了客户优先级学习路线。该路线先预测客户访问优先级，再按容量切分 route，并接同样的后处理。

主要学习实验如下：

| 方法 | Validation average cost | Validation average gap | Check average cost | 结论 |
| --- | ---: | ---: | ---: | --- |
| MSE priority imitation | 约 12.22 | 约 0.161 | 15.618431050502089 | 可行，但明显弱于 heuristic |
| MSE + pairwise priority imitation | 12.274470971742279 | 0.1609528098566702 | 15.552168765777415 | 略优于 MSE，但仍弱 |
| Priority RL finetuning | 12.255131216816354 | 0.158813498232984 | 15.544025451493914 | 较 imitation 小幅提升，但仍弱于默认 heuristic |

学习方法没有作为默认提交方案，主要原因是当前模型只学习“客户优先级”，并不是完整 VRP 决策策略；RL 微调也只是轻量 REINFORCE 微调，不等同于完整 POMO 这类端到端强化学习框架。

## 7. 工程实现

项目实现包含：

| 模块 | 作用 |
| --- | --- |
| `solve.py` | 官方提交入口，读取 `.pkl` 并输出 JSON |
| `src/heuristics.py` | 最近邻、2-opt、relocate、swap 等 heuristic 方法 |
| `src/vrp_io.py` | 数据读取与预测 JSON 写入 |
| `src/vrp_eval.py` | 可行性检查、cost、gap 计算 |
| `scripts/evaluate_baseline.py` | heuristic validation 评估 |
| `scripts/train_priority_model.py` | supervised imitation 训练 |
| `scripts/train_priority_rl.py` | priority RL 微调 |
| `scripts/evaluate_priority_model.py` | priority 模型评估和预测生成 |

训练脚本默认提供终端进度日志和 batch 进度条，便于在 RTX 4060 笔记本或后续远程服务器上观察训练状态。

## 8. 最终提交建议

当前最终建议是：

1. 继续使用 `solve.py` 默认方法 `nearest_2opt_relocate_limited_swap`。
2. 提交预测文件时使用 heuristic 输出，而不是 priority imitation 或 priority RL 输出。
3. 在报告中把学习方法作为实验探索与负结果分析，而不是作为失败项隐藏。
4. 若后续还有较多时间，可把完整 POMO 或更强的神经组合优化方法作为未来工作，不建议在本次考核主线中继续投入大量训练时间。

最终生成 heuristic 预测的命令为：

```bash
python3 solve.py --input VRP_project/VRPData/check_data_to_students.pkl --output outputs/heuristic/predictions.json --device cuda:0 --seed 2026
```

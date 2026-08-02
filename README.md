# ZJU/ZJUI VRP Assessment

本仓库是一个面向 CVRP 校园包裹配送任务的完整项目。当前最终提交方案使用
CPU 可运行的启发式算法作为默认 `solve.py` 方法，同时保留 supervised
imitation 与 priority RL 作为 AI 学习路线探索。

## 当前结论

| 项目 | 当前选择 |
| --- | --- |
| 官方入口 | `solve.py` |
| 默认方法 | `nearest_2opt_relocate_limited_swap` |
| 是否依赖 GPU | 默认提交方法不依赖 GPU，`--device` 仅用于兼容官方接口 |
| Public check 输出 | `outputs/heuristic/predictions.json` |
| 展示稿 | `docs/vrp_presentation_zh.html` |
| 最终报告 | `docs/final_report_zh.md` |

默认 heuristic 在完整 validation 上达到 `1000/1000` 可行，平均 gap 为
`0.08817281823708832`。在 public check 数据上生成的 heuristic 输出为
`1500/1500` 可行，平均 cost 为 `13.976807065714844`，生成耗时约 `148`
秒。

## 快速开始

原始数据应放在本地 `VRP_project/` 目录下，不要提交到 git。目录结构应类似：

```text
VRP_project/
  VRPData/
    train_data.pkl
    validation_data.pkl
    check_data_to_students.pkl
```

生成 public check 预测结果：

```bash
python3 solve.py --input VRP_project/VRPData/check_data_to_students.pkl --output outputs/heuristic/predictions.json --device cuda:0 --seed 2026
```

评估默认 heuristic：

```bash
python3 scripts/evaluate_baseline.py --input VRP_project/VRPData/validation_data.pkl --method nearest_2opt_relocate_limited_swap
```

运行测试：

```bash
python3 -m pytest tests -v
```

## 方法说明

最终默认方法 `nearest_2opt_relocate_limited_swap` 由四步组成：

1. 容量感知最近邻：快速构造可行初始解。
2. Route 内 2-opt：优化单条 route 内部访问顺序。
3. 有限轮 route 间 relocate：把客户移动到更合适的 route。
4. 有限轮 route 间 swap：交换不同 route 中的客户，继续降低 cost。

这种方法的目标不是追求理论最优，而是在考核时间限制内稳定输出高质量可行解。

## 主要结果

| 方法 | Feasible | Validation average cost | Validation average gap |
| --- | ---: | ---: | ---: |
| Nearest | 1000/1000 | 13.934951704742822 | 0.31894172859263836 |
| Nearest + 2-opt | 1000/1000 | 13.410121525149266 | 0.2688820821667571 |
| Limited relocate | 1000/1000 | 11.675451253965944 | 0.10360446980377136 |
| Limited relocate + limited swap | 1000/1000 | 11.513862926474488 | 0.08817281823708832 |
| Pairwise priority imitation | 1000/1000 | 12.274470971742279 | 0.1609528098566702 |
| Priority RL finetuning | 1000/1000 | 12.255131216816354 | 0.158813498232984 |

结论：priority imitation 和 RL 微调都能生成可行解，但当前效果弱于默认 heuristic。
因此最终提交仍使用 heuristic，学习路线作为实验探索写入报告。

## 展示与文档

建议优先阅读：

| 文件 | 用途 |
| --- | --- |
| `docs/vrp_presentation_zh.html` | 中文 HTML 展示稿，可直接浏览器演示或打印成 PDF |
| `docs/final_report_zh.md` | 中文最终技术报告 |
| `docs/submission_checklist_zh.md` | 最终提交清单 |
| `docs/presentation_outline_zh.md` | PPT 提纲 |
| `docs/experiments/baseline_results.md` | 完整实验记录 |
| `docs/gpu_training.md` | RTX 4060/GPU 训练运行指令 |
| `docs/vrp_project_technical_learning_note_zh.md` | 中文技术学习笔记 |

本地预览 HTML 展示稿：

```bash
python3 -m http.server 8000
```

然后打开：

```text
http://127.0.0.1:8000/docs/vrp_presentation_zh.html
```

## 可视化

报告和展示稿使用了两张可视化图：

| 文件 | 含义 |
| --- | --- |
| `docs/assets/cvrp_sample_solution.png` | CVRP 样例路线图 |
| `docs/assets/method_comparison_gap.png` | 方法 validation gap 对比图 |

重新生成某个实例的路线图：

```bash
python3 scripts/visualize_cvrp_solution.py --input VRP_project/VRPData/check_data_to_students.pkl --solutions outputs/heuristic/predictions.json --instance-id 0 --output docs/assets/cvrp_sample_solution.png --title "CVRP sample heuristic solution"
```

## 可选方法命令

运行更快的 `nearest` baseline：

```bash
python3 solve.py --input VRP_project/VRPData/check_data_to_students.pkl --output outputs/heuristic/predictions_nearest.json --method nearest --device cuda:0 --seed 2026
```

运行 route 内 2-opt baseline：

```bash
python3 solve.py --input VRP_project/VRPData/check_data_to_students.pkl --output outputs/heuristic/predictions_2opt.json --method nearest_2opt --device cuda:0 --seed 2026
```

运行完整 best relocate 方法：

```bash
python3 solve.py --input VRP_project/VRPData/check_data_to_students.pkl --output outputs/heuristic/predictions_relocate_best.json --method nearest_2opt_relocate_best --device cuda:0 --seed 2026
```

## 远程结果导出

`scripts/export_remote_results.py` 用于未来远程服务器训练时，把 priority RL 结果整理到
`exports/priority_rl_results/`。它服务于远程服务器工作流，本次 RTX 4060 笔记本训练没有使用它。

```bash
python3 scripts/export_remote_results.py --source-root . --export-dir exports/priority_rl_results
```

## 提交注意

不要提交：

- `VRP_project/` 原始数据目录
- `VRP_project.zip`
- `outputs/` 和 `checkpoints/` 中的大量本地实验输出
- `.firecrawl/`
- `.pytest_cache/`、`__pycache__/`
- `docs/remote_4060_tailscale_ssh.md`

提交材料请以 [docs/submission_checklist_zh.md](docs/submission_checklist_zh.md)
为准。

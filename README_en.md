# ZJU/ZJUI VRP Assessment

[Chinese version](README.md)

This repository contains a complete CVRP project for the campus parcel delivery
assessment. The final submission uses a CPU-ready heuristic solver as the
default `solve.py` method, while supervised imitation and priority RL are kept
as AI-based experimental tracks.

## Current Status

| Item | Current choice |
| --- | --- |
| Official entry point | `solve.py` |
| Default method | `nearest_2opt_relocate_limited_swap` |
| GPU dependency | The default submission method does not require GPU; `--device` is kept for interface compatibility |
| Public check output | `outputs/heuristic/predictions.json` |
| Presentation deck | `docs/vrp_presentation_zh.html` |
| Final report | `docs/final_report_zh.md` |

On the full validation set, the default heuristic reaches `1000/1000`
feasible solutions with an average gap of `0.08817281823708832`. On the
released public check set, the generated heuristic output is `1500/1500`
feasible with an average cost of `13.976807065714844`, and the full generation
time is about `148` seconds.

## Quick Start

The raw data should stay in the local `VRP_project/` directory and should not
be committed to git. The expected structure is:

```text
VRP_project/
  VRPData/
    train_data.pkl
    validation_data.pkl
    check_data_to_students.pkl
```

Generate predictions for the public check set:

```bash
python3 solve.py --input VRP_project/VRPData/check_data_to_students.pkl --output outputs/heuristic/predictions.json --device cuda:0 --seed 2026
```

Evaluate the default heuristic:

```bash
python3 scripts/evaluate_baseline.py --input VRP_project/VRPData/validation_data.pkl --method nearest_2opt_relocate_limited_swap
```

Run tests:

```bash
python3 -m pytest tests -v
```

## Method

The final default method, `nearest_2opt_relocate_limited_swap`, has four stages:

1. Capacity-aware nearest neighbor: quickly builds a feasible initial solution.
2. Intra-route 2-opt: improves the visit order inside each route.
3. Limited inter-route relocate: moves customers to better routes when feasible.
4. Limited inter-route swap: exchanges customers between routes to further reduce cost.

The goal is not to prove theoretical optimality. The goal is to produce stable,
high-quality feasible solutions under the assessment time limit.

## Main Results

| Method | Feasible | Validation average cost | Validation average gap |
| --- | ---: | ---: | ---: |
| Nearest | 1000/1000 | 13.934951704742822 | 0.31894172859263836 |
| Nearest + 2-opt | 1000/1000 | 13.410121525149266 | 0.2688820821667571 |
| Limited relocate | 1000/1000 | 11.675451253965944 | 0.10360446980377136 |
| Limited relocate + limited swap | 1000/1000 | 11.513862926474488 | 0.08817281823708832 |
| Pairwise priority imitation | 1000/1000 | 12.274470971742279 | 0.1609528098566702 |
| Priority RL finetuning | 1000/1000 | 12.255131216816354 | 0.158813498232984 |

Priority imitation and RL finetuning both produce feasible solutions, but their
current performance is weaker than the default heuristic. Therefore, the final
submission keeps the heuristic as the default solver and documents the learning
methods as experimental analysis.

## Documentation

Recommended reading order:

| File | Purpose |
| --- | --- |
| `docs/vrp_presentation_zh.html` | Chinese HTML presentation deck; can be shown in a browser or printed to PDF |
| `docs/final_report_zh.md` | Chinese final technical report |
| `docs/submission_checklist_zh.md` | Final submission checklist |
| `docs/presentation_outline_zh.md` | PPT outline |
| `docs/experiments/baseline_results.md` | Full experiment record |
| `docs/gpu_training.md` | RTX 4060/GPU training commands |
| `docs/vrp_project_technical_learning_note_zh.md` | Chinese technical learning note |

Preview the HTML presentation locally:

```bash
python3 -m http.server 8000
```

Then open:

```text
http://127.0.0.1:8000/docs/vrp_presentation_zh.html
```

## Visualization

The report and presentation use two visualization assets:

| File | Meaning |
| --- | --- |
| `docs/assets/cvrp_sample_solution.png` | A sample CVRP route visualization |
| `docs/assets/method_comparison_gap.png` | Validation gap comparison across methods |

Regenerate a route visualization for one instance:

```bash
python3 scripts/visualize_cvrp_solution.py --input VRP_project/VRPData/check_data_to_students.pkl --solutions outputs/heuristic/predictions.json --instance-id 0 --output docs/assets/cvrp_sample_solution.png --title "CVRP sample heuristic solution"
```

## Optional Solver Commands

Run the faster nearest-neighbor baseline:

```bash
python3 solve.py --input VRP_project/VRPData/check_data_to_students.pkl --output outputs/heuristic/predictions_nearest.json --method nearest --device cuda:0 --seed 2026
```

Run the intra-route 2-opt baseline:

```bash
python3 solve.py --input VRP_project/VRPData/check_data_to_students.pkl --output outputs/heuristic/predictions_2opt.json --method nearest_2opt --device cuda:0 --seed 2026
```

Run the full best-relocate method:

```bash
python3 solve.py --input VRP_project/VRPData/check_data_to_students.pkl --output outputs/heuristic/predictions_relocate_best.json --method nearest_2opt_relocate_best --device cuda:0 --seed 2026
```

## Remote Result Export

`scripts/export_remote_results.py` is for future remote-server workflows. It
collects priority-RL result files into `exports/priority_rl_results/`. This
helper was not used for the current RTX 4060 laptop training run.

```bash
python3 scripts/export_remote_results.py --source-root . --export-dir exports/priority_rl_results
```

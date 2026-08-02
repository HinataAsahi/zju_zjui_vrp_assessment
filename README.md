# ZJU/ZJUI VRP Assessment

This repository contains a CPU-ready CVRP baseline for the ZJU/ZJUI VRP assessment.

## Data

Raw project data should stay in the local `VRP_project/` directory and should not be committed.

## Run Solver

```bash
python3 solve.py --input VRP_project/VRPData/check_data_to_students.pkl --output outputs/heuristic/predictions.json --device cuda:0 --seed 2026
```

The default method is `nearest_2opt_relocate_limited_swap`: deterministic capacity-aware nearest neighbor plus route-inner 2-opt, fixed-budget inter-route relocate, and limited inter-route swap. It accepts `--device` for interface compatibility, but does not require CUDA. `nearest_2opt` remains a faster route-inner 2-opt baseline, `nearest_2opt_relocate_best` is an optional stronger CPU heuristic that performs full inter-route best relocate, `nearest_2opt_relocate_limited` keeps only fixed-budget relocate, and `nearest_2opt_relocate_candidate_limited` additionally limits candidate target routes during relocate search.

```bash
python3 solve.py --input VRP_project/VRPData/check_data_to_students.pkl --output outputs/heuristic/predictions_relocate.json --method nearest_2opt_relocate_best --device cuda:0 --seed 2026
```

```bash
python3 solve.py --input VRP_project/VRPData/check_data_to_students.pkl --output outputs/heuristic/predictions_limited.json --method nearest_2opt_relocate_limited --device cuda:0 --seed 2026
```

```bash
python3 solve.py --input VRP_project/VRPData/check_data_to_students.pkl --output outputs/heuristic/predictions_candidate_limited.json --method nearest_2opt_relocate_candidate_limited --device cuda:0 --seed 2026
```

```bash
python3 solve.py --input VRP_project/VRPData/check_data_to_students.pkl --output outputs/heuristic/predictions_limited_swap.json --method nearest_2opt_relocate_limited_swap --device cuda:0 --seed 2026
```

Use the original nearest-neighbor method for comparison:

```bash
python3 solve.py --input VRP_project/VRPData/check_data_to_students.pkl --output outputs/heuristic/predictions_nearest.json --method nearest --device cuda:0 --seed 2026
```

## Evaluate Baseline

```bash
python3 scripts/evaluate_baseline.py --input VRP_project/VRPData/validation_data.pkl --method nearest_2opt
python3 scripts/evaluate_baseline.py --input VRP_project/VRPData/validation_data.pkl --method nearest_2opt_relocate_best
python3 scripts/evaluate_baseline.py --input VRP_project/VRPData/validation_data.pkl --method nearest_2opt_relocate_limited
python3 scripts/evaluate_baseline.py --input VRP_project/VRPData/validation_data.pkl --method nearest_2opt_relocate_candidate_limited
python3 scripts/evaluate_baseline.py --input VRP_project/VRPData/validation_data.pkl --method nearest_2opt_relocate_limited_swap
python3 scripts/evaluate_baseline.py --input VRP_project/VRPData/validation_data.pkl --method nearest
```

## Tests

```bash
python3 -m pytest tests -v
```

## Documentation

- [中文最终技术报告](docs/final_report_zh.md)
- [中文答辩提纲](docs/presentation_outline_zh.md)
- [中文技术学习笔记](docs/vrp_project_technical_learning_note_zh.md)
- [实验结果记录](docs/experiments/baseline_results.md)
- [GPU 迁移运行指令](docs/gpu_training.md)

## Optional Remote Export

`scripts/export_remote_results.py` collects final priority-RL result files into
`exports/priority_rl_results/` while preserving their relative paths. This
serves future remote-server workflows where downloading one export directory is
more convenient; it was not used for the current VRP assessment training on the
RTX 4060 laptop.

```bash
python3 scripts/export_remote_results.py --source-root . --export-dir exports/priority_rl_results
```

## Roadmap

- Current: `nearest_2opt_relocate_limited_swap` is the default CPU submission method with a tuned `5/2` swap-pass budget.
- Priority-RL finetuning has been tested on an RTX 4060 laptop; it remains weaker than the heuristic default on full validation.
- Next: prepare the final report and presentation around the heuristic default, with priority learning documented as an experimental route.

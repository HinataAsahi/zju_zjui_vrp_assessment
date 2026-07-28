# ZJU/ZJUI VRP Assessment

This repository contains a CPU-ready CVRP baseline for the ZJU/ZJUI VRP assessment.

## Data

Raw project data should stay in the local `VRP_project/` directory and should not be committed.

## Run Solver

```bash
python3 solve.py --input VRP_project/VRPData/check_data_to_students.pkl --output outputs/predictions.json --device cuda:0 --seed 2026
```

The default method is `nearest_2opt_relocate_limited`: deterministic capacity-aware nearest neighbor plus route-inner 2-opt and fixed-budget inter-route relocate. It accepts `--device` for interface compatibility, but does not require CUDA. `nearest_2opt` remains a faster route-inner 2-opt baseline, `nearest_2opt_relocate_best` is an optional stronger CPU heuristic that performs full inter-route best relocate, `nearest_2opt_relocate_candidate_limited` additionally limits candidate target routes during relocate search, and `nearest_2opt_relocate_limited_swap` adds limited inter-route swap after limited relocate.

```bash
python3 solve.py --input VRP_project/VRPData/check_data_to_students.pkl --output outputs/predictions_relocate.json --method nearest_2opt_relocate_best --device cuda:0 --seed 2026
```

```bash
python3 solve.py --input VRP_project/VRPData/check_data_to_students.pkl --output outputs/predictions_limited.json --method nearest_2opt_relocate_limited --device cuda:0 --seed 2026
```

```bash
python3 solve.py --input VRP_project/VRPData/check_data_to_students.pkl --output outputs/predictions_candidate_limited.json --method nearest_2opt_relocate_candidate_limited --device cuda:0 --seed 2026
```

```bash
python3 solve.py --input VRP_project/VRPData/check_data_to_students.pkl --output outputs/predictions_limited_swap.json --method nearest_2opt_relocate_limited_swap --device cuda:0 --seed 2026
```

Use the original nearest-neighbor method for comparison:

```bash
python3 solve.py --input VRP_project/VRPData/check_data_to_students.pkl --output outputs/predictions_nearest.json --method nearest --device cuda:0 --seed 2026
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

## Roadmap

- Current: `nearest_2opt_relocate_limited` is the default CPU submission method.
- Next: decide whether to promote `nearest_2opt_relocate_limited_swap`, which improves validation quality and stays within the public timing budget but is slower.
- Later: optional AI training on a CUDA machine such as the RTX 4060 laptop.

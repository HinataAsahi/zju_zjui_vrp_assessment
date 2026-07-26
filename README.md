# ZJU/ZJUI VRP Assessment

This repository contains a CPU-ready CVRP baseline for the ZJU/ZJUI VRP assessment.

## Data

Raw project data should stay in the local `VRP_project/` directory and should not be committed.

## Run Solver

```bash
python3 solve.py --input VRP_project/VRPData/check_data_to_students.pkl --output outputs/predictions.json --device cuda:0 --seed 2026
```

The first baseline is a deterministic capacity-aware nearest neighbor heuristic. It accepts `--device` for interface compatibility, but does not require CUDA.

## Evaluate Baseline

```bash
python3 scripts/evaluate_baseline.py --input VRP_project/VRPData/validation_data.pkl --limit 20
```

## Tests

```bash
python3 -m pytest tests -v
```

## Roadmap

- Current: feasible nearest neighbor baseline.
- Next: route-level 2-opt improvement.
- Later: optional AI training on a CUDA machine such as the RTX 4060 laptop.

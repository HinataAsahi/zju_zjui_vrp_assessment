# Baseline Evaluation Results

## 2026-07-26: Capacity-Aware Nearest Neighbor

### Command

```bash
python3 scripts/evaluate_baseline.py --input VRP_project/VRPData/validation_data.pkl
```

### Dataset

- File: `VRP_project/VRPData/validation_data.pkl`
- Instances: 1000
- Problem size: CVRP-50
- Reference labels: available

### Results

| Metric | Value |
| --- | ---: |
| Instance count | 1000 |
| Feasible count | 1000 |
| Feasibility rate | 1.0 |
| Average cost | 13.934951704742822 |
| Average gap | 0.31894172859263836 |
| Average inference time | 0.00029439034105962494 |

### Notes

- This is the first CPU-only feasible baseline.
- The method is deterministic nearest neighbor with capacity awareness.
- Feasibility is already stable; the main weakness is route quality.
- The next improvement target is route-level 2-opt.

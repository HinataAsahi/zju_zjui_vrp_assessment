# Baseline Evaluation Results

## 2026-07-26: Capacity-Aware Nearest Neighbor

### Command

```bash
python3 scripts/evaluate_baseline.py --input VRP_project/VRPData/validation_data.pkl --method nearest
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

## 2026-07-28: Capacity-Aware Nearest Neighbor + Route-Inner 2-opt

### Command

```bash
python3 scripts/evaluate_baseline.py --input VRP_project/VRPData/validation_data.pkl --method nearest_2opt
```

### Results

| Metric | Value |
| --- | ---: |
| Instance count | 1000 |
| Feasible count | 1000 |
| Feasibility rate | 1.0 |
| Average cost | 13.410121525149266 |
| Average gap | 0.2688820821667571 |
| Average inference time | 0.000574803089039051 |

### Notes

- The improved method remains feasible for all 1000 validation instances.
- Its average cost is lower than the previously recorded nearest baseline average cost `13.934951704742822`.

## 2026-07-28: Nearest Neighbor + Route-Inner 2-opt + Inter-Route Best Relocate

### Command

```bash
python3 scripts/evaluate_baseline.py --input VRP_project/VRPData/validation_data.pkl --method nearest_2opt_relocate_best
```

### Results

| Metric | Value |
| --- | ---: |
| instance_count | 1000 |
| feasible_count | 1000 |
| feasibility_rate | 1.0 |
| average_cost | 11.412079047307257 |
| average_gap | 0.07837888104759934 |
| average_inference_time | 0.1047532973489142 |

### Notes

- The method remains feasible for all 1000 validation instances.
- Its average cost `11.412079047307257` is lower than the previous `nearest_2opt` average cost `13.410121525149266`.

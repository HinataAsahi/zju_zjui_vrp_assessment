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

## 2026-07-28: Nearest Neighbor + Route-Inner 2-opt + Limited Inter-Route Relocate

### Command

```bash
python3 scripts/evaluate_baseline.py --input /home/a9191/university/master_china/zhejiang_ZJUI_2/VRP_project/VRP_project/VRPData/validation_data.pkl --method nearest_2opt_relocate_limited
```

### Results

| Metric | Value |
| --- | ---: |
| instance_count | 1000 |
| feasible_count | 1000 |
| feasibility_rate | 1.0 |
| average_cost | 11.675451253965944 |
| average_gap | 0.10360446980377136 |
| average_inference_time | 0.0543102543517598 |

### Public Check Timing

```bash
/usr/bin/time -f "elapsed_seconds=%e" python3 solve.py --input /home/a9191/university/master_china/zhejiang_ZJUI_2/VRP_project/VRP_project/VRPData/check_data_to_students.pkl --output outputs/predictions_limited.json --method nearest_2opt_relocate_limited --device cuda:0 --seed 2026
```

- `elapsed_seconds`: 76.35. The approximately 180-second target was met.
- The validation result is feasible for all 1000 instances, and its `average_cost` `11.675451253965944` is below the `nearest_2opt` average cost `13.410121525149266`.
- Default-candidate gate: passed. `nearest_2opt_relocate_limited` was later upgraded to the default CPU submission method.

## 2026-07-28: Nearest Neighbor + Route-Inner 2-opt + Candidate-Limited Inter-Route Relocate

### Command

```bash
python3 scripts/evaluate_baseline.py --input VRP_project/VRPData/validation_data.pkl --method nearest_2opt_relocate_candidate_limited
```

### Results

| Metric | Value |
| --- | ---: |
| instance_count | 1000 |
| feasible_count | 1000 |
| feasibility_rate | 1.0 |
| average_cost | 11.677200534322004 |
| average_gap | 0.10377228851455313 |
| average_inference_time | 0.052491096285695676 |

### Public Check Timing

```bash
/usr/bin/time -f "elapsed_seconds=%e" python3 solve.py --input VRP_project/VRPData/check_data_to_students.pkl --output outputs/predictions_candidate_limited_2_2.json --method nearest_2opt_relocate_candidate_limited --device cuda:0 --seed 2026
```

- `elapsed_seconds`: 63.43. The approximately 180-second target was met.
- Candidate budget: CVRP-50 uses up to 2 candidate target routes per moved customer; CVRP-100 also uses up to 2. Relocate pass budget remains 8 for CVRP-50 and 3 for CVRP-100.
- Tuning note: `3/2` was also tested; it kept feasibility but had validation `average_cost=11.67604868544792`, worse than the original `4/3` candidate-limited result and not enough speed evidence to keep.
- The `2/2` validation result is feasible for all 1000 instances. Its `average_cost` `11.677200534322004` is slightly worse than the current default `nearest_2opt_relocate_limited` validation cost `11.675451253965944`, but its public check timing is faster than the latest current-default run (`73.85` seconds).
- Default status: keep `nearest_2opt_relocate_limited` for now until the speed-quality tradeoff is explicitly accepted.

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
- Tuning comparison:

| Candidate route budget | Validation average_cost | Validation average_gap | Validation average_inference_time | Public check elapsed_seconds | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| `4/3` | 11.675216058546207 | 0.1035842323611632 | 0.0588456712206098 | 82.10 | Best candidate-limited validation quality, but slower than `2/2`. |
| `3/2` | 11.67604868544792 | 0.10367660350681505 | 0.05734983776875015 | 75.08 | Middle option; quality improvement over `2/2` is small for the added time. |
| `2/2` | 11.677200534322004 | 0.10377228851455313 | 0.052491096285695676 | 63.43 | Kept as the tuned candidate-limited setting. |
| `1/1` | 11.700284874443323 | 0.10589457351556121 | 0.03589411162603937 | 50.33 | Fastest tested option, but the validation quality loss is too large. |

- The `2/2` validation result is feasible for all 1000 instances. Its `average_cost` `11.677200534322004` is slightly worse than the current default `nearest_2opt_relocate_limited` validation cost `11.675451253965944`, but its public check timing is faster than the latest current-default run (`73.85` seconds).
- Default status: keep `nearest_2opt_relocate_limited` for now until the speed-quality tradeoff is explicitly accepted.

## 2026-07-28: Nearest Neighbor + Route-Inner 2-opt + Limited Relocate + Limited Inter-Route Swap

### Command

```bash
python3 scripts/evaluate_baseline.py --input VRP_project/VRPData/validation_data.pkl --method nearest_2opt_relocate_limited_swap
```

### Results

| Metric | Value |
| --- | ---: |
| instance_count | 1000 |
| feasible_count | 1000 |
| feasibility_rate | 1.0 |
| average_cost | 11.513862926474488 |
| average_gap | 0.08817281823708832 |
| average_inference_time | 0.08248868217083509 |

### Public Check Timing

```bash
/usr/bin/time -f "elapsed_seconds=%e" python3 solve.py --input VRP_project/VRPData/check_data_to_students.pkl --output outputs/predictions_limited_swap_5_2.json --method nearest_2opt_relocate_limited_swap --device cuda:0 --seed 2026
```

- `elapsed_seconds`: 135.13. The approximately 180-second target was met.
- Swap budget: CVRP-50 uses up to 5 swap passes; CVRP-100 uses up to 2 swap passes. The method runs after `nearest_2opt_relocate_limited`.
- Tuning comparison:

| Swap pass budget | Validation average_cost | Validation average_gap | Validation average_inference_time | Public check elapsed_seconds | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| `4/2` | 11.519231481331843 | 0.0886829417106512 | 0.0912746598602098 | 144.29 | Previous default. |
| `5/2` | 11.513862926474488 | 0.08817281823708832 | 0.08248868217083509 | 135.13 | Kept as the tuned default. |
| `4/3` | 11.519231481331843 | 0.0886829417106512 | 0.08032113239067257 | 150.05 | No validation quality gain over `4/2`. |
| `3/2` | 11.528272980024548 | 0.08954072031977317 | 0.08362124218912505 | 138.06 | Faster than `4/2` in this run, but quality loss is not worth it. |

- Formal default `outputs/predictions.json` was regenerated with the `5/2` budget in `148.00` seconds and passed a local feasibility check for all 1500 public-check instances.
- The validation result is feasible for all 1000 instances. Its `average_cost` improves from the previous default `4/2` value `11.519231481331843` to `11.513862926474488`.
- Default status: `5/2` is the tuned default CPU submission method.

## 2026-07-29: Reference-Priority Oracle Check

### Purpose

This check tests whether a supervised-imitation model that predicts customer
priority is worth implementing. It uses validation reference routes as an
oracle priority order, reconstructs feasible routes by capacity splitting, and
optionally applies the existing post-processing pipeline.

### Results

| Variant | Feasible | Validation average_cost | Validation average_gap | Elapsed seconds | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| Reference priority + capacity split | 1000/1000 | 12.264334883168045 | 0.15775741444822425 | 0.54 | Feasible but too weak alone. |
| Reference priority + capacity split + 2-opt | 1000/1000 | 11.700193004258352 | 0.10528970373869408 | 1.15 | Still worse than the tuned heuristic default. |
| Reference priority + capacity split + 2-opt + limited relocate + limited swap | 1000/1000 | 10.995458085453048 | 0.0378498455595169 | 82.22 | Strong oracle upper bound; customer-priority learning is worth trying. |

### Conclusion

- Customer-priority representation is usable: oracle priorities always produced feasible routes after capacity splitting.
- The representation should not be used alone. It needs the existing `2-opt + limited relocate + limited swap` post-processing.
- The oracle result `10.995458085453048` is much better than the current tuned heuristic default `11.513862926474488`, so a supervised-imitation priority model has a meaningful quality target.

## 2026-07-29: Supervised Imitation Priority Model, MSE Rank Loss

### Purpose

This is the first learned priority model trained on the RTX 4060 laptop. It
predicts customer priority scores, reconstructs routes by capacity splitting,
and applies the same `2-opt + limited relocate + limited swap` post-processing
as the priority oracle check.

### Training Setup

```bash
python3 scripts/train_priority_model.py --train-input VRP_project/VRPData/train_data.pkl --validation-input VRP_project/VRPData/validation_data.pkl --checkpoint-output checkpoints/priority_mse_rank.pt --summary-output outputs/priority_mse_rank_summary.json --epochs 50 --batch-size 64 --hidden-dim 128 --num-heads 4 --num-layers 2 --dropout 0.1 --learning-rate 0.001 --weight-decay 0.0001 --eval-limit 100 --device cuda --postprocess-eval
```

### Training Result

| Metric | Value |
| --- | ---: |
| Train instances | 7000 |
| Validation instances available | 1000 |
| Validation instances per epoch | 100 |
| Best epoch | 29 |
| Best validation feasible count | 100/100 |
| Best validation average_cost | 12.218516753151578 |
| Best validation average_gap | 0.16157482063807882 |

### Comparison on the First 100 Validation Instances

| Method | Feasible | Average cost | Average gap |
| --- | ---: | ---: | ---: |
| Tuned heuristic default | 100/100 | 11.424134539766737 | 0.08426480064303492 |
| MSE priority model | 100/100 | 12.215777101910927 | 0.16126667122853794 |

### Check Data Sanity

| Output | Feasible | Average cost | Average gap |
| --- | ---: | ---: | ---: |
| `outputs/predictions.json` | 1500/1500 | 13.976807065714844 | N/A |
| `outputs/predictions_priority_model.json` | 1500/1500 | 15.618431050502089 | N/A |

### Conclusion

- The learned MSE priority model is feasible but clearly weaker than the tuned
  heuristic default.
- Simply training for more epochs is unlikely to close the gap; the loss curve
  improved early and then plateaued.
- The next learned-priority experiment should use a ranking-aware objective,
  starting with `mse_pairwise`, while keeping the heuristic default unchanged.

## 2026-07-30: Supervised Imitation Priority Model, MSE + Pairwise Rank Loss

### Purpose

This run tests whether adding pairwise ranking pressure improves the learned
customer-priority order compared with the first MSE-only priority model.

### Training Setup

```bash
python3 scripts/train_priority_model.py --train-input VRP_project/VRPData/train_data.pkl --validation-input VRP_project/VRPData/validation_data.pkl --checkpoint-output checkpoints/priority_mse_pairwise_rank.pt --summary-output outputs/priority_mse_pairwise_rank_summary.json --epochs 50 --batch-size 64 --hidden-dim 128 --num-heads 4 --num-layers 2 --dropout 0.1 --learning-rate 0.001 --weight-decay 0.0001 --eval-limit 100 --device cuda --postprocess-eval --loss mse_pairwise --pairwise-weight 0.5 --pairwise-margin 0.1
```

### Training Result

| Metric | Value |
| --- | ---: |
| Train instances | 7000 |
| Validation instances available | 1000 |
| Validation instances per epoch | 100 |
| Best epoch | 47 |
| Best validation feasible count | 100/100 |
| Best validation average_cost | 12.176409600231384 |
| Best validation average_gap | 0.1555420716601009 |

### Local Re-Evaluation

| Method | Validation scope | Feasible | Average cost | Average gap |
| --- | --- | ---: | ---: | ---: |
| Tuned heuristic default | first 100 | 100/100 | 11.424134539766737 | 0.08426480064303492 |
| MSE priority model | first 100 | 100/100 | 12.215777101910927 | 0.16126667122853794 |
| MSE + pairwise priority model | first 100 | 100/100 | 12.182751773028457 | 0.15613505108262565 |
| Tuned heuristic default | full 1000 | 1000/1000 | 11.513862926474488 | 0.08817281823708832 |
| MSE + pairwise priority model | full 1000 | 1000/1000 | 12.274470971742279 | 0.1609528098566702 |

### Check Data Sanity

| Output | Feasible | Average cost | Average gap |
| --- | ---: | ---: | ---: |
| `outputs/predictions.json` | 1500/1500 | 13.976807065714844 | N/A |
| `outputs/predictions_priority_model.json` | 1500/1500 | 15.618431050502089 | N/A |
| `outputs/predictions_priority_mse_pairwise.json` | 1500/1500 | 15.552168765777415 | N/A |

### Conclusion

- Pairwise ranking slightly improves the learned priority model compared with
  the first MSE-only run.
- The learned priority model is still far weaker than the tuned heuristic
  default on validation and check-data sanity metrics.
- Do not replace the default `solve.py` heuristic. Further learned-priority
  work needs a larger methodological change rather than another small loss
  adjustment.

## 2026-07-31: Priority Model REINFORCE Finetuning

### Purpose

This experiment tests whether REINFORCE finetuning can improve the
`mse_pairwise` learned-priority model by directly rewarding lower route cost.
The default `solve.py` heuristic remains unchanged.

### Local Smoke Result

| Metric | Value |
| --- | ---: |
| Train instances | 16 |
| Validation instances per epoch | 8 |
| Epochs | 1 |
| Samples per instance | 2 |
| Postprocess reward | false |
| Postprocess eval | false |
| Best epoch | 1 |
| Best validation feasible count | 8/8 |
| Best validation average_cost | 27.658878074400118 |
| Best validation average_gap | 1.6792680066323138 |

Formal finetuning:

```bash
python3 scripts/train_priority_rl.py --train-input VRP_project/VRPData/train_data.pkl --validation-input VRP_project/VRPData/validation_data.pkl --init-checkpoint checkpoints/priority_imitation/priority_mse_pairwise_rank.pt --checkpoint-output checkpoints/priority_rl/priority_rl_finetune.pt --last-checkpoint-output checkpoints/priority_rl/priority_rl_finetune_last.pt --summary-output outputs/priority_rl/rl_finetune_summary.json --epochs 20 --batch-size 32 --samples-per-instance 2 --temperature 1.0 --learning-rate 0.00001 --weight-decay 0.0001 --eval-limit 100 --device cuda --postprocess-reward --postprocess-eval
```

Formal RTX 4060 finetuning result:

| Metric | Value |
| --- | ---: |
| Wall time | ~13 hours |
| Train instances | 7000 |
| Epochs completed | 20 |
| Best epoch | 6 |
| Best per-epoch validation limit | 100 |
| Best validation average_cost | 12.148102602854195 |
| Best validation average_gap | 0.15319454020097345 |
| Full validation feasible count | 1000/1000 |
| Full validation average_cost | 12.255131216816354 |
| Full validation average_gap | 0.158813498232984 |
| Check prediction feasible count | 1500/1500 |
| Check prediction average_cost | 15.544025451493914 |

The RL result did not pass the default-method gate. It slightly improves over
the full-validation `mse_pairwise` priority model gap (`0.1609528098566702`) and
check-data average cost (`15.552168765777415`), but remains far from the tuned
heuristic default validation gap (`0.08817281823708832`) and check-data average
cost (`13.976807065714844`). The RL prediction file is feasible on all check
instances, but the default submission method remains
`nearest_2opt_relocate_limited_swap`.

### Decision Rule

- If full-validation gap remains around `0.15`, stop priority RL and move to
  AI-assisted heuristic.
- If full-validation gap clearly improves below `0.156`, continue RL tuning.
- If full-validation gap approaches the heuristic default gap around `0.088`,
  compare check-data sanity and runtime before any default-method discussion.

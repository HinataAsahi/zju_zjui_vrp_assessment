# VRP Route 间 Best Relocate 改进设计

日期：2026-07-28

## 背景

当前项目已经完成三层基础能力：

- `nearest`：容量感知最近邻 baseline。
- `nearest_2opt`：最近邻后接 route 内 2-opt，作为当前默认方法。
- 评估脚本可以通过 `--method` 对不同方法做验证集对比。

截至 2026-07-28，完整验证集结果为：

| Method | Feasible rate | Average cost | Average gap |
| --- | ---: | ---: | ---: |
| `nearest` | 1.0 | 13.934951704742822 | 0.31894172859263836 |
| `nearest_2opt` | 1.0 | 13.410121525149266 | 0.2688820821667571 |

route 内 2-opt 只能调整同一辆车内部的访问顺序，不能修正“客户分到哪辆车”这个问题。因此下一阶段需要尝试 route 间改进。

## 目标

本阶段目标是新增一个可选增强方法：

```text
nearest_2opt_relocate_best
```

它在 `nearest_2opt` 的基础上，尝试把单个客户从一条 route 移动到另一条 route 的某个插入位置。每一轮扫描所有合法候选移动，选择 cost 下降最多的一个执行。

完成后应满足：

- 保留 `nearest` 和 `nearest_2opt` 的既有行为。
- `solve.py` 默认仍使用 `nearest_2opt`。
- 新方法通过 `--method nearest_2opt_relocate_best` 显式启用。
- route 间移动必须满足容量约束。
- route 间移动不能漏客户、重复客户或产生非法客户编号。
- 每次接受的移动必须降低总路径长度。
- 完整验证集可行率保持 1.0。
- 完整验证集平均 cost 不高于 `nearest_2opt` 已记录值 `13.410121525149266`。
- 继续保持 CPU 可运行，不引入新运行时依赖。

## 非目标

本阶段不实现：

- route 间 swap。
- candidate-limited relocate。
- 多客户同时移动。
- 随机化搜索。
- OR-Tools。
- 神经网络训练。
- 默认方法升级。

如果 `nearest_2opt_relocate_best` 验证集结果稳定，再单独决定是否把默认方法从 `nearest_2opt` 升级到它。

## 方法命名

本阶段显式使用带策略强度的 method 名称：

```text
nearest_2opt_relocate_best
```

命名含义：

- `nearest`：初始解来自容量感知最近邻。
- `2opt`：每条 route 内做 2-opt。
- `relocate`：route 间移动单个客户。
- `best`：每一轮选择当前收益最大的合法移动。

这样后续可以自然扩展其他方法，例如：

- `nearest_2opt_relocate_first`
- `nearest_2opt_relocate_limited`
- `nearest_2opt_relocate_best_deep`
- `nearest_2opt_swap_best`

## 方案比较

### 方案 1：Best Relocate，推荐

每轮扫描所有“从 route A 移出一个客户，插入 route B 某个位置”的候选移动。只考虑 route A 和 route B 不同的情况。候选移动必须满足目标 route 容量不超限，并且移动后的两条 route 总 cost 降低。

优点：

- 收益判断清晰。
- 结果确定性强。
- 比 first relocate 更稳定。
- 实现复杂度仍可控。

缺点：

- 每轮扫描成本高于 first relocate。
- 对 CVRP-100 可能需要搜索轮数上限控制。

这是本阶段采用方案。

### 方案 2：First Relocate

扫描候选移动时，遇到第一个能降低 cost 的移动就执行。

优点：

- 更快。
- 实现更简单。

缺点：

- 候选遍历顺序会明显影响结果。
- 容易错过当前轮更好的移动。
- 实验记录不如 best relocate 稳定。

本阶段不采用。

### 方案 3：Relocate + Swap

同时允许单客户移动和两条 route 间互换客户。

优点：

- 潜在收益更高。
- 能修正更多客户分配错误。

缺点：

- 容量检查更复杂。
- 搜索空间更大。
- 难以在当前阶段保持设计和测试简单。

本阶段不采用，作为后续独立阶段。

## 推荐设计

在 `src/heuristics.py` 中新增 route 间 best relocate 后处理函数：

- `improve_routes_relocate_best(instance, routes, capacity_tol=1e-9, improvement_tol=1e-12, max_passes=50) -> tuple[tuple[int, ...], ...]`
- `solve_nearest_neighbor_2opt_relocate_best(instance, capacity_tol=1e-9, improvement_tol=1e-12, max_relocate_passes=50) -> tuple[tuple[int, ...], ...]`

同时更新：

- `SolverMethod`
- `SOLVER_METHODS`
- `solve_with_method`
- `solve.py` help text
- `scripts/evaluate_baseline.py` help text
- README 和实验记录

`nearest_2opt_relocate_best` 的流程为：

```text
solve_nearest_neighbor
  -> improve_routes_2opt
  -> improve_routes_relocate_best
  -> validate_solution
```

其中 `improve_routes_relocate_best` 每执行一次 route 间移动后，应对受影响 routes 再做 route 内 2-opt，使插入后的局部顺序重新优化。

## Relocate 规则

一个候选 move 包含：

- `source_route_index`
- `source_customer_position`
- `target_route_index`
- `target_insert_position`
- `customer_id`
- `cost_delta`

候选 move 的生成规则：

1. 遍历每条 source route。
2. 遍历 source route 中每个客户。
3. 遍历每条不同的 target route。
4. 如果 `target_load + customer_demand > capacity + capacity_tol`，跳过。
5. 遍历 target route 的所有插入位置，包括开头和末尾。
6. 计算移动前后 source route 与 target route 的 cost 差。
7. 只保留 `candidate_cost + improvement_tol < current_cost` 的移动。

每轮选择 `cost_delta` 最小的候选，也就是 cost 下降最多的候选。

平局处理必须确定性：

```text
(cost_delta, source_route_index, target_route_index, source_customer_position, target_insert_position, customer_id)
```

执行 move 后：

- 从 source route 删除该客户。
- 将该客户插入 target route 指定位置。
- 如果 source route 变为空，则删除空 route。
- 对受影响 route 执行 route 内 2-opt。
- 重新计算 route loads。
- 开始下一轮扫描。

最多执行 `max_passes` 轮。如果达到上限，即使仍可能有改进，也停止，以保证 CPU 时间可控。

## 容量与可行性

route 间 relocate 会改变每辆车服务的客户集合，因此容量检查是本阶段核心风险。

实现必须维护或重新计算每条 route 的 load：

```text
route_load = sum(instance.demand[customer_id - 1] for customer_id in route)
```

接受移动前必须确认目标 route 新 load 不超过 `capacity + capacity_tol`。

移动后必须保证：

- 所有客户仍出现一次。
- route 数量可以减少，但不能产生空 route。
- 每条 route load 不超过 capacity。
- 总 cost 不增加。

最终仍通过 `validate_solution(instance, routes)` 做全局验证。`solve.py` 已经有输出前验证，这个流程应保持。

## 搜索预算

默认搜索预算：

```text
max_relocate_passes = 50
```

选择 50 的原因：

- CVRP-50 上足够尝试多轮收益明显的移动。
- CPU 电脑上仍有较强可控性。
- 如果后续评估发现收益还在继续，可以再设计 `best_deep` 或暴露 CLI 参数。

本阶段不新增 CLI 参数控制 `max_relocate_passes`。先把方法固定为可复现实验方法，避免用户命令和实验记录过早复杂化。

## CLI 与评估接口

`solve.py --method` 允许值新增：

```text
nearest_2opt_relocate_best
```

默认值保持：

```text
nearest_2opt
```

示例：

```bash
python3 solve.py --input VRP_project/VRPData/check_data_to_students.pkl --output outputs/predictions_relocate.json --method nearest_2opt_relocate_best --device cuda:0 --seed 2026
```

评估示例：

```bash
python3 scripts/evaluate_baseline.py --input VRP_project/VRPData/validation_data.pkl --method nearest_2opt_relocate_best
```

## 测试策略

新增或更新测试应覆盖：

- 一个小型手工实例中，合法 relocate 能降低 total cost。
- 目标 route 容量不足时，relocate 不会执行。
- source route 只剩一个客户时，如果移动后 source route 为空，空 route 会被删除。
- `improve_routes_relocate_best` 不漏客户、不重复客户。
- `solve_nearest_neighbor_2opt_relocate_best` 输出通过 `validate_solution`。
- `solve_with_method(..., "nearest_2opt_relocate_best")` 能调用新方法。
- `solve.py --method nearest_2opt_relocate_best` 能写出合法 `cvrp_v1` JSON。
- `scripts/evaluate_baseline.py --method nearest_2opt_relocate_best` 能输出评估 summary。

测试继续使用小型手工 fixture，不依赖 raw data 入库。

## 验收标准

本阶段实现完成后应满足：

- `python3 -m pytest tests -v` 通过。
- `python3 solve.py --input VRP_project/VRPData/check_data_to_students.pkl --output outputs/predictions_relocate.json --method nearest_2opt_relocate_best --device cuda:0 --seed 2026` 能在 CPU 电脑上运行。
- `python3 scripts/evaluate_baseline.py --input VRP_project/VRPData/validation_data.pkl --method nearest_2opt_relocate_best` 完整运行成功。
- 验证集可行率为 1.0。
- 验证集平均 cost 不高于 `nearest_2opt` 的 `13.410121525149266`。
- README 记录新 method 的用途和命令。
- `docs/experiments/baseline_results.md` 记录新 method 的完整验证集结果。
- `docs/progress/daily-progress.md` 简短记录本阶段进度。

## 后续阶段

如果 `nearest_2opt_relocate_best` 收益明显且运行时间可接受，下一阶段可以考虑：

- 将默认方法从 `nearest_2opt` 升级为 `nearest_2opt_relocate_best`。
- 设计 `nearest_2opt_swap_best`。
- 设计 candidate-limited relocate，用于更大规模实例或更深搜索预算。

这些都应作为独立阶段重新设计，避免和本阶段混合。

# VRP Route 内 2-opt 改进设计

日期：2026-07-28

## 背景

当前项目已经完成第一版 CPU 最近邻 baseline。它能稳定读取官方 `.pkl` 数据、生成合法 `cvrp_v1` JSON，并在完整验证集上达到 1000/1000 可行，平均 gap 约 31.89%。

最近邻 baseline 的主要问题不是可行性，而是 route 质量：它每一步只看“当前最近客户”，可能在同一辆车的路线内部产生绕路或交叉。route 内 2-opt 正好用于修正这类问题。

## 目标

本阶段目标是在现有最近邻 baseline 后加入确定性的 route 内 2-opt 局部搜索，降低总路径长度，同时保持当前 CPU 可运行、可提交、可验证的项目状态。

完成后应满足：

- 每个实例仍先由容量感知最近邻生成可行 routes。
- 对每条 route 独立执行 2-opt，只改变同一条 route 内客户访问顺序。
- 不改变客户属于哪辆车，因此不主动触碰容量分配。
- 优化后仍强制运行可行性检查。
- `solve.py` 默认使用 2-opt 增强版，但保留参数可运行原始最近邻 baseline。
- `scripts/evaluate_baseline.py` 可以分别评估 `nearest` 与 `nearest_2opt`，便于量化收益。
- 不引入新的运行时依赖，不依赖 CUDA。

## 非目标

本阶段不实现 route 间换客户、route 间 relocate、route 间 swap、OR-Tools、神经网络训练或最终报告材料。

route 间换客户会作为 route 内 2-opt 完成后的独立阶段再设计，因为它会改变每辆车服务的客户集合，需要额外处理容量约束和更大的搜索空间。

## 方案比较

### 方案 1：最近邻后处理 + route 内 2-opt

先用现有最近邻生成 routes，然后对每条 route 独立做 2-opt。2-opt 只通过反转 route 中一段客户顺序来缩短路径，例如将 `1 -> 5 -> 2 -> 4` 中的一段翻转成更短的访问顺序。

优点：

- 改动小，贴合现有 `src/heuristics.py`。
- 不改变 route 的客户集合，容量可行性风险低。
- CPU 上运行成本可控。
- 容易通过单元测试证明 cost 不增加、客户集合不变。

缺点：

- 只能修正同一辆车内部的绕路。
- 如果最近邻一开始把客户分到不合适的车上，route 内 2-opt 无法修正。

这是本阶段推荐方案。

### 方案 2：直接做 route 间换客户

在不同 routes 之间移动客户或交换客户，以同时优化客户分配和访问顺序。

优点：

- 潜在收益更高。
- 可以修正最近邻阶段的分车错误。

缺点：

- 必须重新检查每条 route 的容量。
- 搜索空间更大，CPU 时间更难控制。
- 实现复杂度和测试复杂度明显更高。

该方案适合作为下一阶段，不与本阶段混在一起实现。

### 方案 3：引入 OR-Tools 或更强启发式

用成熟求解器或更复杂启发式替换当前 baseline。

优点：

- 可能快速获得更强结果。

缺点：

- 会引入新依赖和提交环境风险。
- 不利于保持 `solve.py` 简洁稳定。
- 当前项目还需要先建立可控的本地改进链路。

本阶段不采用。

## 推荐设计

采用方案 1：在最近邻 routes 后增加 route 内 2-opt 后处理。

核心新增函数放在 `src/heuristics.py`：

- `improve_route_2opt(instance, route, improvement_tol=1e-12) -> tuple[int, ...]`
- `improve_routes_2opt(instance, routes, improvement_tol=1e-12) -> tuple[tuple[int, ...], ...]`
- `solve_nearest_neighbor_2opt(instance, capacity_tol=1e-9, improvement_tol=1e-12) -> tuple[tuple[int, ...], ...]`

保留现有 `solve_nearest_neighbor` 行为不变，避免破坏原始 baseline 的可复现性和已有测试。

## 2-opt 规则

一条 route 只包含客户编号，不包含 depot。计算一段反转是否有收益时，把 depot 当成 route 的隐含起点和终点。

对于 route 中的片段 `route[i:j+1]`，2-opt 会尝试反转这一段。它只会改变两条边：

```text
反转前：prev -> route[i]      route[j] -> next
反转后：prev -> route[j]      route[i] -> next
```

其中：

- `prev` 是 `i == 0` 时的 depot，否则是 `route[i - 1]`。
- `next` 是 `j == len(route) - 1` 时的 depot，否则是 `route[j + 1]`。

如果反转后的两条边总长度比反转前短超过 `improvement_tol`，就接受这次反转。

遍历顺序固定为从小到大的 `i` 和 `j`，因此算法是确定性的。每接受一次反转后重新扫描，直到没有任何可改进片段。

## CLI 与评估接口

`solve.py` 新增参数：

```bash
--method nearest_2opt
```

允许值：

- `nearest_2opt`：默认值，最近邻 + route 内 2-opt。
- `nearest`：只运行原始最近邻 baseline。

示例：

```bash
python3 solve.py --input VRP_project/VRPData/check_data_to_students.pkl --output outputs/predictions.json --device cuda:0 --seed 2026
python3 solve.py --input VRP_project/VRPData/check_data_to_students.pkl --output outputs/predictions_nearest.json --method nearest
```

`scripts/evaluate_baseline.py` 同样新增 `--method`，用于分别跑：

```bash
python3 scripts/evaluate_baseline.py --input VRP_project/VRPData/validation_data.pkl --method nearest
python3 scripts/evaluate_baseline.py --input VRP_project/VRPData/validation_data.pkl --method nearest_2opt
```

## 数据流

默认求解流程变为：

```text
输入 .pkl
  -> vrp_io.load_instances
  -> heuristics.solve_nearest_neighbor
  -> heuristics.improve_routes_2opt
  -> vrp_eval.validate_solution
  -> vrp_io.write_solutions_json
```

如果 `--method nearest`，则跳过 `improve_routes_2opt`。

## 正确性要求

route 内 2-opt 必须满足：

- 不新增客户。
- 不删除客户。
- 不重复客户。
- 不改变 route 数量。
- 不改变每条 route 的客户集合。
- 不增加单条 route cost，允许浮点容差内相等。
- 不破坏 `validate_solution` 的可行性结果。
- 对空 route、单客户 route、两客户 route 能稳定返回。

## 测试策略

新增或更新测试：

- 小型几何样例中，`improve_route_2opt` 能消除明显交叉并降低 cost。
- 已经局部最优的 route 保持不变。
- route 内 2-opt 保持每条 route 的客户集合不变。
- `solve_nearest_neighbor_2opt` 输出仍通过 `validate_solution`。
- `solve.py --method nearest` 保持原始最近邻行为。
- `solve.py` 默认使用 `nearest_2opt` 并写出合法 JSON。
- `evaluate_baseline.py --method nearest_2opt` 能输出可行率、平均 cost、平均 gap 和平均推理时间。

测试继续使用小型手工 fixture，不依赖原始大数据入库。完整验证集评估作为本地 smoke test，结果记录到 `docs/experiments/baseline_results.md`。

## 验收标准

本阶段完成后应满足：

- `python3 -m pytest tests -v` 通过。
- `python3 solve.py --input VRP_project/VRPData/check_data_to_students.pkl --output outputs/predictions.json --device cuda:0 --seed 2026` 能在 CPU 电脑上运行。
- `python3 scripts/evaluate_baseline.py --input VRP_project/VRPData/validation_data.pkl --method nearest_2opt` 完整运行成功。
- 验证集 `nearest_2opt` 的可行率仍为 1.0。
- 验证集 `nearest_2opt` 的平均 cost 不高于已记录的原始最近邻 baseline 平均 cost `13.934951704742822`。
- README 和实验记录说明默认方法已升级为最近邻 + route 内 2-opt。

## 后续阶段

route 内 2-opt 完成后，再根据验证集收益决定是否启动 route 间换客户阶段。

下一阶段可以优先比较两个方向：

- route 间 relocate：把一个客户从 route A 移动到 route B。
- route 间 swap：交换 route A 与 route B 中的两个客户。

这两个方向都需要重新设计容量检查、候选筛选和搜索预算，不能直接混入本阶段。

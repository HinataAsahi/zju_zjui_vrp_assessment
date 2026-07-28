# VRP Route 间 Limited Relocate 默认候选设计

日期：2026-07-28

## 背景

当前项目已经保留三类可运行方法：

- `nearest`：容量感知最近邻，速度最快，质量最低。
- `nearest_2opt`：最近邻后接 route 内 2-opt，当前默认方法。
- `nearest_2opt_relocate_best`：route 内 2-opt 后接 route 间 best relocate，质量明显更好，但公开集含 CVRP-100 时运行时间风险较高。

截至 2026-07-28，验证集 CVRP-50 结果为：

| Method | feasible_rate | average_cost | average_gap | average_inference_time |
| --- | ---: | ---: | ---: | ---: |
| `nearest` | 1.0 | 13.934951704742822 | 0.31894172859263836 | 0.00029439034105962494 |
| `nearest_2opt` | 1.0 | 13.410121525149266 | 0.2688820821667571 | 0.000574803089039051 |
| `nearest_2opt_relocate_best` | 1.0 | 11.412079047307257 | 0.07837888104759934 | 0.1047532973489142 |

`nearest_2opt_relocate_best` 在 CVRP-50 上平均推理时间约为 `0.105s`，但公开测试集包含 `1000` 个 CVRP-50 和 `500` 个 CVRP-100。一次公开集完整评估尝试在 CPU 上超过 4 分钟仍未结束，因此不宜直接升级为默认提交方法。

官方文档没有给出 `solve.py` 的明确硬性秒数上限，但评价指标包含 `Average Inference Time`，且隐藏测试集规模与公开测试集类似。因此默认方法应优先保证能稳定跑完整个 `1500` 实例输入。

## 目标

本阶段目标是新增一个默认提交候选方法：

```text
nearest_2opt_relocate_limited
```

它复用当前 `nearest_2opt_relocate_best` 的 route 间单客户 relocate 逻辑，但将 `max_relocate_passes` 根据实例规模限制到较小值，使公开/隐藏测试集的总运行时间更可控。

完成后应满足：

- 保留 `nearest`、`nearest_2opt`、`nearest_2opt_relocate_best` 的既有行为。
- 新方法通过 `--method nearest_2opt_relocate_limited` 显式启用。
- 第一轮实现完成后，先不直接改变默认方法；只有公开集 smoke/计时达到门槛后，才将 `solve.py` 默认从 `nearest_2opt` 升级为 `nearest_2opt_relocate_limited`。
- route 间移动仍必须满足容量约束。
- route 间移动不能漏客户、重复客户或产生非法客户编号。
- 每次接受的移动仍必须降低总路径长度。
- 验证集可行率保持 `1.0`。
- 验证集平均 cost 应低于 `nearest_2opt` 的 `13.410121525149266`。
- 公开测试集 `check_data_to_students.pkl` 的 `1500` 个实例应能在约 `180` 秒目标内完成。
- 继续保持 CPU 可运行，不引入新运行时依赖。

## 非目标

本阶段不实现：

- candidate-limited relocate。
- route 间 swap。
- wall-clock 动态时间预算。
- 随机化搜索。
- OR-Tools。
- 神经网络训练。
- CUDA/GPU 加速。

如果 fixed-pass limited 版本效果好，后续再单独设计候选数量限制 + 轮数限制，也就是方案 3。

## 方案比较

### 方案 1：自适应固定轮数限制，本阶段采用

根据客户数量给不同的 relocate 最大轮数：

| 类型 | customer 数 | max_relocate_passes |
| --- | ---: | ---: |
| CVRP-50 | `<= 50` | `8` |
| CVRP-100 | `> 50` | `3` |

优点：

- 确定性强，同一输入在同一代码下输出稳定。
- 复用现有 best relocate 函数，改动范围小。
- 不依赖机器 wall-clock，因此实验结果更容易复现。
- 比 50 轮 best relocate 更适合作为默认提交候选。

缺点：

- 不能严格保证每台机器都在 180 秒内。
- 固定轮数可能对某些实例过浅，对另一些实例过深。

### 方案 2：wall-clock 时间预算，本阶段不采用

给整个输入文件或每个实例设置时间预算，到点即停止搜索。

优点：

- 与“默认提交必须跑完”的目标最直接对应。

缺点：

- 不同机器速度不同，输出结果可能不一致。
- 测试和实验记录更难复现。
- 在 Python 中做细粒度计时会让核心循环更复杂。

### 方案 3：候选数量限制 + 轮数限制，后续可能采用

只检查一部分更有希望的 candidate，例如邻近 route、近邻客户或有限插入位置，再配合较小轮数。

优点：

- 速度和质量可能比方案 1 更平衡。
- 更适合 CVRP-100 或更大实例。

缺点：

- 需要设计 candidate 选择规则。
- 测试复杂度更高。
- 过早实现会让当前默认候选阶段变复杂。

本阶段先做方案 1。如果方案 1 公开集时间接近目标且质量明显改善，再考虑是否直接升级默认；如果方案 1 质量或时间不满意，再设计方案 3。

## 推荐设计

在 `src/heuristics.py` 中新增预算选择 helper：

```python
def relocate_limited_passes(customer_count: int) -> int:
    if customer_count <= 50:
        return 8
    return 3
```

新增 solver：

```python
def solve_nearest_neighbor_2opt_relocate_limited(
    instance: CVRPInstance,
    capacity_tol: float = 1e-9,
    improvement_tol: float = 1e-12,
) -> tuple[tuple[int, ...], ...]:
    return solve_nearest_neighbor_2opt_relocate_best(
        instance,
        capacity_tol=capacity_tol,
        improvement_tol=improvement_tol,
        max_relocate_passes=relocate_limited_passes(instance.customer_count),
    )
```

同时更新：

- `SolverMethod`
- `SOLVER_METHODS`
- `solve_with_method`
- `solve.py` help text
- `scripts/evaluate_baseline.py` help text
- tests
- README
- `docs/experiments/baseline_results.md`
- `docs/progress/daily-progress.md`

## 默认方法切换规则

第一步只新增 `nearest_2opt_relocate_limited`，默认仍保持 `nearest_2opt`。

实现后运行：

```bash
python3 scripts/evaluate_baseline.py --input VRP_project/VRPData/validation_data.pkl --method nearest_2opt_relocate_limited
```

并运行公开集计时：

```bash
python3 solve.py --input VRP_project/VRPData/check_data_to_students.pkl --output outputs/predictions_limited.json --method nearest_2opt_relocate_limited --device cuda:0 --seed 2026
```

实现计划应把默认切换作为独立 review gate。只有满足下面门槛，并经用户确认后，才将 `solve.py` 默认方法升级为 `nearest_2opt_relocate_limited`：

- validation `feasibility_rate == 1.0`。
- validation `average_cost < 13.410121525149266`。
- public check `1500` 个实例生成成功。
- public check 总耗时目标约 `180` 秒以内。
- 输出文件通过现有可行性/格式检查。

如果不满足时间门槛，则不升级默认方法，并进入方案 3 设计。

## 测试策略

新增或更新测试应覆盖：

- `relocate_limited_passes(50) == 8`。
- `relocate_limited_passes(100) == 3`。
- `solve_nearest_neighbor_2opt_relocate_limited` 输出可行解。
- limited 方法的 cost 不高于同实例 `nearest_2opt`，允许浮点容差。
- `solve_with_method(..., "nearest_2opt_relocate_limited")` 能调用新方法。
- `solve.py --method nearest_2opt_relocate_limited` 能写出合法 `cvrp_v1` JSON。
- `scripts/evaluate_baseline.py --method nearest_2opt_relocate_limited` 能输出评估 summary。
- CLI help 能描述 limited 方法。

测试仍使用小型手工 fixture，不依赖 raw data 入库。公开集计时和 validation 结果写入实验记录，不写入单元测试。

## 验收标准

本阶段实现完成后应满足：

- `python3 -m pytest tests -v` 通过。
- `nearest_2opt_relocate_limited` 在 validation 上可行率为 `1.0`。
- `nearest_2opt_relocate_limited` 在 validation 上平均 cost 低于 `13.410121525149266`。
- 公开集 `check_data_to_students.pkl` 能完整生成 `outputs/predictions_limited.json`。
- 公开集总运行时间被记录，并用于决定是否升级默认方法。
- README 记录 limited 方法用途和命令。
- `docs/experiments/baseline_results.md` 记录 validation 和公开集计时结果。
- `docs/progress/daily-progress.md` 简短记录本阶段进度。

## 后续阶段

如果 fixed-pass limited 版本满足时间目标且质量明显优于 `nearest_2opt`：

- 将 `solve.py` 默认方法升级为 `nearest_2opt_relocate_limited`。
- 生成新的默认 `outputs/predictions.json`。
- 在报告中把 `nearest_2opt_relocate_best` 作为质量上限对照，把 limited 作为提交候选。

如果 fixed-pass limited 版本仍然太慢或质量损失明显：

- 保持默认 `nearest_2opt`。
- 单独设计方案 3：候选数量限制 + 轮数限制。

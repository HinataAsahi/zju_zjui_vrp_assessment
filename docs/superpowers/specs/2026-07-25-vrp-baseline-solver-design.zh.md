# VRP Baseline Solver 设计

日期：2026-07-25

## 目标

本阶段目标是实现一个稳定、可提交、CPU 即可运行的 CVRP baseline 系统，为后续 2-opt 改进和 AI 方法训练打基础。

这套 baseline 不追求一次达到最优成绩，而是优先保证：

- 能读取官方 `.pkl` 数据。
- 能处理训练/验证数据的 6 元组格式，也能处理公开测试集的 4 元组格式。
- 能为 CVRP-50 和 CVRP-100 动态生成合法路径。
- 能输出符合官方要求的 `cvrp_v1` JSON 文件。
- 能本地验证路径可行性、计算路径 cost，并在有标签数据上评估 gap。
- 能在当前 CPU 电脑上确定性运行，不依赖 CUDA。

## 非目标

本阶段不实现 2-opt、不训练神经网络、不引入 OR-Tools，也不生成最终报告或展示 slides。

2-opt 会作为 baseline 稳定后的下一阶段扩展。AI 训练会在后续阶段设计，并考虑迁移到带 RTX 4060 的拯救者笔记本上运行。

## 项目约束

### 运行环境

当前实现应以 CPU 为默认运行环境。`solve.py` 可以接受 `--device cuda:0` 这样的参数，以兼容官方或后续训练接口，但启发式 baseline 不实际依赖 GPU。

后续如果加入神经模型训练，应将训练相关依赖、CUDA 使用和模型权重管理与当前 CPU baseline 分开，避免影响官方提交脚本的稳定性。

### 数据与编号

本项目数据实例有两种结构：

```text
(depot, loc, demand, capacity, routes, cost)
(depot, loc, demand, capacity)
```

内部计算可以使用 0-based 下标，例如 `loc[0]` 表示第一个客户；但 JSON 输出中的客户编号必须是 1-based，即客户 `1` 对应 `loc[0]`。

`depot` 是隐含起点和终点，不写入输出 route。每条 route 只包含客户编号。

`capacity` 和 `demand` 在数据中可能表现为浮点数，因此内部统一按 `float` 计算容量约束；输出的客户编号仍应是整数。

### 仓库卫生

原始数据目录 `VRP_project/` 不提交到 git。若运行脚本产生预测 JSON、临时日志、实验输出或模型文件，应放在被忽略的输出目录中，或在后续实现时补充 `.gitignore` 规则。

## 系统结构

本阶段建议拆成四个核心模块和一组测试。

### `src/vrp_io.py`

负责数据读取和输出写入。

主要职责：

- 从 `.pkl` 文件读取实例列表。
- 将 4 元组和 6 元组规范化为统一的内部实例对象。
- 保留可选的参考 `routes` 与 `cost`，用于验证集评估。
- 将 solver 生成的路径写成官方 JSON。

内部实例对象建议包含：

- `instance_id`
- `depot`
- `loc`
- `demand`
- `capacity`
- `reference_routes`
- `reference_cost`

其中 `reference_routes` 和 `reference_cost` 对公开测试集为 `None`。

### `src/vrp_eval.py`

负责可行性检查和 cost 计算。

主要职责：

- 计算单条 route 的 depot-to-customers-to-depot 欧氏距离。
- 计算一个实例所有 routes 的总距离。
- 检查是否漏客户、重复客户、客户编号越界、route 中误写 depot、route 超容量。
- 在有参考 cost 的数据上计算 gap。
- 汇总平均 cost、可行率、平均 gap 和平均推理时间。

可行性检查应返回结构化结果，而不只是 `True/False`。例如返回 `is_feasible`、`errors`、`route_loads`、`visited_count`、`total_cost`，方便调试和报告。

### `src/heuristics.py`

负责本阶段的启发式 baseline。

采用“容量感知最近邻”方法：

1. 从 depot 出发，当前车辆载重为 0。
2. 在尚未访问、且加入后不超过 capacity 的客户中，选择距离当前位置最近的客户。
3. 把该客户加入当前 route，更新当前位置和载重。
4. 如果没有任何剩余客户能加入当前 route，就结束当前 route，车辆回到 depot，开启下一条 route。
5. 重复直到所有客户都被访问。

为保证可复现，遇到距离相同或非常接近的候选客户时，按客户下标从小到大打破平局。`--seed` 参数保留给后续随机化策略，本阶段默认不引入随机选择。

这个算法的优点是实现简单、速度快、很容易保证容量可行；缺点是局部贪心，整体路线质量通常不如 2-opt 或神经方法。

### `solve.py`

作为官方入口脚本。

命令格式：

```bash
python solve.py --input /path/to/check_data.pkl --output /path/to/predictions.json --device cuda:0 --seed 2026
```

主要职责：

- 解析 `--input`、`--output`、`--device`、`--seed`。
- 读取输入 `.pkl`。
- 对每个实例调用 baseline solver。
- 写出 UTF-8 JSON。
- 在写出前运行基本可行性检查，避免明显非法输出。

输出格式：

```json
{
  "format_version": "cvrp_v1",
  "solutions": [
    {
      "instance_id": 0,
      "routes": [[1, 7, 3], [2, 5, 4, 6]]
    }
  ]
}
```

## 数据流

整体流程是：

```text
输入 .pkl
  -> vrp_io.load_instances
  -> 统一 CVRPInstance 列表
  -> heuristics.solve_nearest_neighbor
  -> routes
  -> vrp_eval.validate_solution / compute_total_cost
  -> vrp_io.write_solutions_json
  -> predictions.json
```

对于训练集或验证集，可以额外运行评估流程：

```text
带标签实例
  -> baseline routes
  -> total_cost
  -> reference_cost
  -> gap
  -> 汇总指标
```

公开测试集没有 reference cost，因此只计算本地 cost、可行性和推理时间，不计算 gap。

## 错误处理

实现应明确处理以下情况：

- 输入文件不存在。
- `.pkl` 顶层对象不是 list。
- 单个实例 tuple 长度不是 4 或 6。
- `loc` 和 `demand` 长度不一致。
- route 中客户编号不是整数。
- route 中出现小于 1 或大于客户数的编号。
- 客户被重复访问或漏访问。
- route 需求总和超过 capacity。
- 输出路径的父目录不存在时自动创建。

浮点容量比较使用很小的容差，例如 `1e-9`，避免 `40.0` 这类数据在计算中出现无意义的边界误判。

## 测试策略

本阶段测试应优先覆盖行为，而不是覆盖实现细节。

建议测试：

- 4 元组公开测试格式可以被读取。
- 6 元组训练/验证格式可以被读取，并保留参考 routes/cost。
- cost 计算能处理单 route 和多 route。
- 可行性检查能识别合法解。
- 可行性检查能识别漏客户、重复客户、编号越界和超容量。
- 最近邻 baseline 对小型手工实例能生成合法解。
- `solve.py` 能读取临时 `.pkl`，写出符合 `cvrp_v1` 的 JSON。

在测试数据上使用小型手工 fixture，不依赖原始大数据入库。需要时可以另写一个人工命令，在本地 raw data 上跑 smoke test，但输出文件不提交。

## 验收标准

本阶段完成后应满足：

- `python solve.py --input ... --output ... --device cuda:0 --seed 2026` 可以在 CPU 电脑上运行。
- 输出 JSON 包含 `format_version: "cvrp_v1"` 和按输入顺序排列的 `solutions`。
- 每个 solution 的 `instance_id` 与输入序号一致。
- routes 使用 1-based 客户编号，且不包含 depot。
- 对公开测试集中的 CVRP-50 和 CVRP-100 都能生成可行解。
- 单元测试和 CLI 集成测试通过。
- 验证集上可以报告 baseline 的平均 cost、可行率和相对 reference cost 的 gap。

## README 时机

README 最适合在 `solve.py`、依赖说明、测试命令和一次本地 smoke test 都稳定后补充。原因是 README 应该记录真实可运行的命令，而不是尚未验证的预期接口。

因此，本阶段实现完成并验证通过后，应新增或更新 README，至少包含：

- 项目目标。
- 环境和依赖。
- 如何运行 `solve.py`。
- 如何运行测试。
- 数据目录不入库的说明。
- 当前 baseline 方法和后续 2-opt / AI 计划。

## 后续扩展

### 2-opt

下一阶段可以在每条 route 内加入 2-opt 局部搜索，用于减少路径交叉和无效绕路。2-opt 不改变客户分配到哪辆车，因此通常不会破坏容量约束，但仍需要在优化后重新运行可行性检查。

### AI 训练

AI 方法应独立设计，避免把训练复杂度直接塞进 `solve.py`。训练可以在 RTX 4060 笔记本上进行，推理脚本则应尽量保持可降级：没有 CUDA 或模型权重时，仍能使用当前 heuristic baseline 产出合法提交。

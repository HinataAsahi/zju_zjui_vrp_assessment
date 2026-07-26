# VRP Baseline Solver Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建一个 CPU 可运行、确定性、可提交的 CVRP baseline solver，包含数据读取、可行性/代价评估、容量感知最近邻算法、官方 `solve.py` 入口和基础文档。

**Architecture:** 使用纯 Python 标准库实现核心逻辑，按职责拆分为 `src/vrp_io.py`、`src/vrp_eval.py`、`src/heuristics.py` 和顶层 `solve.py`。所有算法函数返回 1-based customer routes，`vrp_eval` 统一负责验证和 cost 计算，CLI 在写 JSON 前强制验证输出。

**Tech Stack:** Python 3.10+ 标准库、`pickle`、`json`、`argparse`、`dataclasses`；测试使用 `pytest` 和临时文件 fixture。第一版运行时不引入 NumPy、PyTorch、OR-Tools 或 CUDA 依赖。

## Global Constraints

- 当前实现以 CPU 为默认运行环境，启发式 baseline 不实际依赖 GPU。
- `solve.py` 接受 `--input`、`--output`、`--device`、`--seed`；`--device` 保留但 heuristic baseline 忽略 GPU。
- 支持 6 元组 `(depot, loc, demand, capacity, routes, cost)` 和 4 元组 `(depot, loc, demand, capacity)`。
- 内部可以使用 0-based 下标，但 JSON 输出必须使用 1-based customer 编号。
- `depot` 是隐含起点和终点，不写入输出 route。
- `capacity` 和 `demand` 统一按 `float` 计算，容量比较使用 `1e-9` 容差。
- 输出 JSON 顶层必须包含 `format_version: "cvrp_v1"`。
- 原始数据目录 `VRP_project/` 不提交到 git。
- 本阶段不实现 2-opt、不训练神经网络、不引入 OR-Tools。

---

## File Structure

- Create: `src/__init__.py`
  - 标记 `src` 为本项目内部包，便于测试和 CLI import。
- Create: `src/vrp_io.py`
  - 定义 `CVRPInstance`、`SolutionRecord`，读取 `.pkl`，规范化 4/6 元组，写出官方 JSON。
- Create: `src/vrp_eval.py`
  - 定义 `ValidationResult`、`EvaluationSummary`，计算欧氏距离、route cost、total cost、可行性和 gap。
- Create: `src/heuristics.py`
  - 实现容量感知最近邻 baseline，输出 1-based routes。
- Create: `solve.py`
  - 官方入口脚本，解析参数，读取数据，调用 baseline，验证并写 JSON。
- Create: `scripts/evaluate_baseline.py`
  - 在训练/验证集上运行 baseline，打印平均 cost、可行率、平均 gap 和推理时间。
- Create: `tests/test_vrp_io.py`
  - 覆盖 `.pkl` 读取、元组规范化、错误处理、JSON 写出。
- Create: `tests/test_vrp_eval.py`
  - 覆盖 cost、可行性检查和指标汇总。
- Create: `tests/test_heuristics.py`
  - 覆盖最近邻 baseline 的容量分车、确定性和可行性。
- Create: `tests/test_solve_cli.py`
  - 覆盖 `solve.py` 临时 `.pkl` 输入到 JSON 输出的集成路径。
- Create: `tests/test_evaluate_baseline_cli.py`
  - 覆盖评估脚本在小型带标签 fixture 上输出指标。
- Create: `.gitignore`
  - 忽略 raw data、缓存、输出目录和模型文件。
- Create: `README.md`
  - 在实现和验证稳定后记录真实运行命令。
- Modify: `docs/progress/daily-progress.md`
  - 追加 2026-07-26 简短进度。

---

### Task 1: Data Model, Pickle I/O, and JSON Writer

**Files:**
- Create: `src/__init__.py`
- Create: `src/vrp_io.py`
- Test: `tests/test_vrp_io.py`

**Interfaces:**
- Produces:
  - `CVRPInstance`
  - `SolutionRecord`
  - `normalize_instance(instance_id: int, raw: tuple[Any, ...]) -> CVRPInstance`
  - `load_instances(path: str | Path) -> list[CVRPInstance]`
  - `write_solutions_json(path: str | Path, solutions: Sequence[SolutionRecord]) -> None`

- [ ] **Step 1: Write failing tests for tuple normalization and JSON output**

Create `tests/test_vrp_io.py`:

```python
import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.vrp_io import (  # noqa: E402
    CVRPInstance,
    SolutionRecord,
    load_instances,
    normalize_instance,
    write_solutions_json,
)


def test_normalize_labeled_instance_preserves_reference_fields():
    raw = (
        [[0.0, 0.0]],
        [[1.0, 0.0], [0.0, 1.0]],
        [3, 4],
        40.0,
        [[1, 2]],
        2.0,
    )

    instance = normalize_instance(7, raw)

    assert instance == CVRPInstance(
        instance_id=7,
        depot=(0.0, 0.0),
        loc=((1.0, 0.0), (0.0, 1.0)),
        demand=(3.0, 4.0),
        capacity=40.0,
        reference_routes=((1, 2),),
        reference_cost=2.0,
    )


def test_normalize_unlabeled_instance_sets_reference_fields_to_none():
    raw = (
        [[0.5, 0.5]],
        [[1, 0], [0, 1], [1, 1]],
        [2, 5, 7],
        50,
    )

    instance = normalize_instance(0, raw)

    assert instance.reference_routes is None
    assert instance.reference_cost is None
    assert instance.capacity == 50.0
    assert instance.demand == (2.0, 5.0, 7.0)


def test_load_instances_reads_pickle_list(tmp_path):
    data_path = tmp_path / "sample.pkl"
    raw_instances = [
        ([[0, 0]], [[1, 0]], [3], 40),
        ([[1, 1]], [[0, 1]], [4], 40, [[1]], 2.0),
    ]
    with data_path.open("wb") as handle:
        pickle.dump(raw_instances, handle)

    instances = load_instances(data_path)

    assert [instance.instance_id for instance in instances] == [0, 1]
    assert instances[0].reference_cost is None
    assert instances[1].reference_cost == 2.0


def test_load_instances_rejects_non_list_pickle(tmp_path):
    data_path = tmp_path / "bad.pkl"
    with data_path.open("wb") as handle:
        pickle.dump({"not": "a list"}, handle)

    try:
        load_instances(data_path)
    except TypeError as exc:
        assert "should contain a list" in str(exc)
    else:
        raise AssertionError("Expected TypeError")


def test_normalize_instance_rejects_bad_tuple_length():
    try:
        normalize_instance(0, ([[0, 0]], [[1, 0]], [3]))
    except ValueError as exc:
        assert "tuple length 4 or 6" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_normalize_instance_rejects_mismatched_loc_and_demand():
    try:
        normalize_instance(0, ([[0, 0]], [[1, 0], [0, 1]], [3], 40))
    except ValueError as exc:
        assert "loc and demand length mismatch" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_write_solutions_json_creates_parent_and_uses_cvrp_v1(tmp_path):
    output_path = tmp_path / "nested" / "predictions.json"
    solutions = [
        SolutionRecord(instance_id=0, routes=((1, 2), (3,))),
        SolutionRecord(instance_id=1, routes=((2,), (1,))),
    ]

    write_solutions_json(output_path, solutions)

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload == {
        "format_version": "cvrp_v1",
        "solutions": [
            {"instance_id": 0, "routes": [[1, 2], [3]]},
            {"instance_id": 1, "routes": [[2], [1]]},
        ],
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
python3 -m pytest tests/test_vrp_io.py -v
```

Expected: FAIL because `src.vrp_io` does not exist.

- [ ] **Step 3: Implement minimal I/O module**

Create `src/__init__.py` as an empty file.

Create `src/vrp_io.py`:

```python
"""Data loading and output writing for the VRP assessment."""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


Point = tuple[float, float]
Route = tuple[int, ...]


@dataclass(frozen=True)
class CVRPInstance:
    instance_id: int
    depot: Point
    loc: tuple[Point, ...]
    demand: tuple[float, ...]
    capacity: float
    reference_routes: tuple[Route, ...] | None = None
    reference_cost: float | None = None

    @property
    def customer_count(self) -> int:
        return len(self.loc)


@dataclass(frozen=True)
class SolutionRecord:
    instance_id: int
    routes: tuple[Route, ...]


def _as_float(value: Any) -> float:
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


def _as_sequence(value: Any) -> list[Any]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    return list(value)


def _as_point(value: Any) -> Point:
    seq = _as_sequence(value)
    if len(seq) == 1 and isinstance(seq[0], (list, tuple)):
        seq = _as_sequence(seq[0])
    if len(seq) != 2:
        raise ValueError(f"point should have length 2, got {len(seq)}")
    return (_as_float(seq[0]), _as_float(seq[1]))


def _as_points(values: Any) -> tuple[Point, ...]:
    return tuple(_as_point(item) for item in _as_sequence(values))


def _as_demands(values: Any) -> tuple[float, ...]:
    return tuple(_as_float(item) for item in _as_sequence(values))


def _as_routes(values: Any) -> tuple[Route, ...]:
    routes: list[Route] = []
    for route in _as_sequence(values):
        routes.append(tuple(int(customer) for customer in _as_sequence(route)))
    return tuple(routes)


def normalize_instance(instance_id: int, raw: tuple[Any, ...]) -> CVRPInstance:
    if not isinstance(raw, tuple):
        raise TypeError(f"instance {instance_id} should be a tuple")
    if len(raw) not in (4, 6):
        raise ValueError(f"instance {instance_id} should have tuple length 4 or 6")

    depot, loc, demand, capacity = raw[:4]
    loc_points = _as_points(loc)
    demands = _as_demands(demand)
    if len(loc_points) != len(demands):
        raise ValueError(
            f"instance {instance_id} loc and demand length mismatch: "
            f"{len(loc_points)} != {len(demands)}"
        )

    reference_routes = None
    reference_cost = None
    if len(raw) == 6:
        reference_routes = _as_routes(raw[4])
        reference_cost = _as_float(raw[5])

    return CVRPInstance(
        instance_id=instance_id,
        depot=_as_point(depot),
        loc=loc_points,
        demand=demands,
        capacity=_as_float(capacity),
        reference_routes=reference_routes,
        reference_cost=reference_cost,
    )


def load_instances(path: str | Path) -> list[CVRPInstance]:
    input_path = Path(path)
    with input_path.open("rb") as handle:
        data = pickle.load(handle)
    if not isinstance(data, list):
        raise TypeError(f"{input_path} should contain a list")
    return [normalize_instance(index, raw) for index, raw in enumerate(data)]


def write_solutions_json(
    path: str | Path,
    solutions: Sequence[SolutionRecord],
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": "cvrp_v1",
        "solutions": [
            {
                "instance_id": solution.instance_id,
                "routes": [list(route) for route in solution.routes],
            }
            for solution in solutions
        ],
    }
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
python3 -m pytest tests/test_vrp_io.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/__init__.py src/vrp_io.py tests/test_vrp_io.py
git commit -m "feat: add VRP data IO"
```

---

### Task 2: Cost Computation, Feasibility Validation, and Metrics

**Files:**
- Create: `src/vrp_eval.py`
- Test: `tests/test_vrp_eval.py`

**Interfaces:**
- Consumes:
  - `CVRPInstance` from `src.vrp_io`
- Produces:
  - `ValidationResult`
  - `EvaluationSummary`
  - `euclidean(a: tuple[float, float], b: tuple[float, float]) -> float`
  - `compute_route_cost(instance: CVRPInstance, route: Sequence[int]) -> float`
  - `compute_total_cost(instance: CVRPInstance, routes: Sequence[Sequence[int]]) -> float`
  - `validate_solution(instance: CVRPInstance, routes: Sequence[Sequence[int]], capacity_tol: float = 1e-9) -> ValidationResult`
  - `compute_gap(cost: float, reference_cost: float) -> float`
  - `summarize_results(instances: Sequence[CVRPInstance], routes_by_instance: Sequence[Sequence[Sequence[int]]], inference_times: Sequence[float] | None = None) -> EvaluationSummary`

- [ ] **Step 1: Write failing tests for costs, validation, and summary**

Create `tests/test_vrp_eval.py`:

```python
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.vrp_eval import (  # noqa: E402
    compute_gap,
    compute_route_cost,
    compute_total_cost,
    summarize_results,
    validate_solution,
)
from src.vrp_io import CVRPInstance  # noqa: E402


def make_instance() -> CVRPInstance:
    return CVRPInstance(
        instance_id=0,
        depot=(0.0, 0.0),
        loc=((3.0, 4.0), (6.0, 8.0), (0.0, 1.0)),
        demand=(3.0, 4.0, 2.0),
        capacity=7.0,
        reference_cost=20.0,
    )


def test_compute_route_and_total_cost():
    instance = make_instance()

    assert compute_route_cost(instance, [1]) == 10.0
    assert compute_total_cost(instance, [[1, 2], [3]]) == 22.0


def test_validate_solution_accepts_feasible_routes():
    result = validate_solution(make_instance(), [[1, 2], [3]])

    assert result.is_feasible is True
    assert result.errors == ()
    assert result.route_loads == (7.0, 2.0)
    assert result.visited_count == 3
    assert result.total_cost > 0.0


def test_validate_solution_detects_missing_duplicate_and_out_of_range():
    result = validate_solution(make_instance(), [[1, 1, 4]])

    assert result.is_feasible is False
    assert "duplicate_customer:1" in result.errors
    assert "out_of_range_customer:4" in result.errors
    assert "missing_customer:2" in result.errors
    assert "missing_customer:3" in result.errors


def test_validate_solution_detects_overload():
    result = validate_solution(make_instance(), [[1, 2, 3]])

    assert result.is_feasible is False
    assert "route_over_capacity:0:9.0>7.0" in result.errors


def test_validate_solution_rejects_non_integer_customer():
    result = validate_solution(make_instance(), [[1, "2"]])

    assert result.is_feasible is False
    assert "non_integer_customer:2" in result.errors


def test_compute_gap_uses_relative_reference_cost():
    assert compute_gap(22.0, 20.0) == 0.1


def test_summarize_results_reports_average_metrics():
    instance_a = make_instance()
    instance_b = CVRPInstance(
        instance_id=1,
        depot=(0.0, 0.0),
        loc=((1.0, 0.0),),
        demand=(1.0,),
        capacity=2.0,
        reference_cost=2.0,
    )

    summary = summarize_results(
        [instance_a, instance_b],
        [[[1, 2], [3]], [[1]]],
        inference_times=[0.01, 0.03],
    )

    assert summary.instance_count == 2
    assert summary.feasible_count == 2
    assert summary.feasibility_rate == 1.0
    assert summary.average_inference_time == 0.02
    assert summary.average_cost > 0.0
    assert summary.average_gap is not None
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
python3 -m pytest tests/test_vrp_eval.py -v
```

Expected: FAIL because `src.vrp_eval` does not exist.

- [ ] **Step 3: Implement evaluation module**

Create `src/vrp_eval.py`:

```python
"""Cost computation and feasibility checks for CVRP solutions."""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import mean
from typing import Sequence

from src.vrp_io import CVRPInstance


@dataclass(frozen=True)
class ValidationResult:
    is_feasible: bool
    errors: tuple[str, ...]
    route_loads: tuple[float, ...]
    visited_count: int
    total_cost: float


@dataclass(frozen=True)
class EvaluationSummary:
    instance_count: int
    feasible_count: int
    feasibility_rate: float
    average_cost: float
    average_gap: float | None
    average_inference_time: float | None


def euclidean(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _customer_point(instance: CVRPInstance, customer_id: int) -> tuple[float, float]:
    return instance.loc[customer_id - 1]


def compute_route_cost(instance: CVRPInstance, route: Sequence[int]) -> float:
    if not route:
        return 0.0
    total = euclidean(instance.depot, _customer_point(instance, int(route[0])))
    for prev_customer, next_customer in zip(route, route[1:]):
        total += euclidean(
            _customer_point(instance, int(prev_customer)),
            _customer_point(instance, int(next_customer)),
        )
    total += euclidean(_customer_point(instance, int(route[-1])), instance.depot)
    return total


def compute_total_cost(
    instance: CVRPInstance,
    routes: Sequence[Sequence[int]],
) -> float:
    return sum(compute_route_cost(instance, route) for route in routes)


def validate_solution(
    instance: CVRPInstance,
    routes: Sequence[Sequence[int]],
    capacity_tol: float = 1e-9,
) -> ValidationResult:
    errors: list[str] = []
    seen: set[int] = set()
    route_loads: list[float] = []

    for route_index, route in enumerate(routes):
        route_load = 0.0
        for raw_customer in route:
            if not isinstance(raw_customer, int) or isinstance(raw_customer, bool):
                errors.append(f"non_integer_customer:{raw_customer}")
                continue
            customer = raw_customer
            if customer < 1 or customer > instance.customer_count:
                errors.append(f"out_of_range_customer:{customer}")
                continue
            if customer in seen:
                errors.append(f"duplicate_customer:{customer}")
            seen.add(customer)
            route_load += instance.demand[customer - 1]
        route_loads.append(route_load)
        if route_load > instance.capacity + capacity_tol:
            errors.append(
                f"route_over_capacity:{route_index}:{route_load}>{instance.capacity}"
            )

    for customer in range(1, instance.customer_count + 1):
        if customer not in seen:
            errors.append(f"missing_customer:{customer}")

    valid_routes = [
        [
            customer
            for customer in route
            if (
                isinstance(customer, int)
                and not isinstance(customer, bool)
                and 1 <= customer <= instance.customer_count
            )
        ]
        for route in routes
    ]
    total_cost = compute_total_cost(instance, valid_routes)
    return ValidationResult(
        is_feasible=not errors,
        errors=tuple(errors),
        route_loads=tuple(route_loads),
        visited_count=len(seen),
        total_cost=total_cost,
    )


def compute_gap(cost: float, reference_cost: float) -> float:
    if reference_cost == 0:
        raise ValueError("reference_cost must be non-zero")
    return (cost - reference_cost) / reference_cost


def summarize_results(
    instances: Sequence[CVRPInstance],
    routes_by_instance: Sequence[Sequence[Sequence[int]]],
    inference_times: Sequence[float] | None = None,
) -> EvaluationSummary:
    validations = [
        validate_solution(instance, routes)
        for instance, routes in zip(instances, routes_by_instance)
    ]
    costs = [result.total_cost for result in validations]
    gaps = [
        compute_gap(result.total_cost, instance.reference_cost)
        for instance, result in zip(instances, validations)
        if instance.reference_cost is not None
    ]
    instance_count = len(instances)
    feasible_count = sum(1 for result in validations if result.is_feasible)
    return EvaluationSummary(
        instance_count=instance_count,
        feasible_count=feasible_count,
        feasibility_rate=feasible_count / instance_count if instance_count else 0.0,
        average_cost=mean(costs) if costs else 0.0,
        average_gap=mean(gaps) if gaps else None,
        average_inference_time=mean(inference_times) if inference_times else None,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
python3 -m pytest tests/test_vrp_eval.py -v
```

Expected: PASS.

- [ ] **Step 5: Run existing tests to catch import regressions**

Run:

```bash
python3 -m pytest tests/test_preview_vrp_pkl.py tests/test_vrp_io.py tests/test_vrp_eval.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/vrp_eval.py tests/test_vrp_eval.py
git commit -m "feat: add VRP solution evaluation"
```

---

### Task 3: Capacity-Aware Nearest Neighbor Heuristic

**Files:**
- Create: `src/heuristics.py`
- Test: `tests/test_heuristics.py`

**Interfaces:**
- Consumes:
  - `CVRPInstance` from `src.vrp_io`
  - `euclidean` from `src.vrp_eval`
- Produces:
  - `solve_nearest_neighbor(instance: CVRPInstance, capacity_tol: float = 1e-9) -> tuple[tuple[int, ...], ...]`

- [ ] **Step 1: Write failing tests for deterministic feasible routing**

Create `tests/test_heuristics.py`:

```python
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.heuristics import solve_nearest_neighbor  # noqa: E402
from src.vrp_eval import validate_solution  # noqa: E402
from src.vrp_io import CVRPInstance  # noqa: E402


def test_nearest_neighbor_splits_routes_by_capacity():
    instance = CVRPInstance(
        instance_id=0,
        depot=(0.0, 0.0),
        loc=((1.0, 0.0), (2.0, 0.0), (10.0, 0.0)),
        demand=(4.0, 4.0, 4.0),
        capacity=8.0,
    )

    routes = solve_nearest_neighbor(instance)

    assert routes == ((1, 2), (3,))
    assert validate_solution(instance, routes).is_feasible is True


def test_nearest_neighbor_breaks_distance_ties_by_customer_id():
    instance = CVRPInstance(
        instance_id=0,
        depot=(0.0, 0.0),
        loc=((1.0, 0.0), (-1.0, 0.0), (0.0, 2.0)),
        demand=(1.0, 1.0, 1.0),
        capacity=3.0,
    )

    routes = solve_nearest_neighbor(instance)

    assert routes[0][0] == 1
    assert validate_solution(instance, routes).is_feasible is True


def test_nearest_neighbor_rejects_customer_demand_over_capacity():
    instance = CVRPInstance(
        instance_id=0,
        depot=(0.0, 0.0),
        loc=((1.0, 0.0),),
        demand=(9.0,),
        capacity=8.0,
    )

    try:
        solve_nearest_neighbor(instance)
    except ValueError as exc:
        assert "demand exceeds capacity" in str(exc)
    else:
        raise AssertionError("Expected ValueError")
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
python3 -m pytest tests/test_heuristics.py -v
```

Expected: FAIL because `src.heuristics` does not exist.

- [ ] **Step 3: Implement nearest neighbor heuristic**

Create `src/heuristics.py`:

```python
"""Heuristic CVRP solvers."""

from __future__ import annotations

from src.vrp_eval import euclidean
from src.vrp_io import CVRPInstance


def solve_nearest_neighbor(
    instance: CVRPInstance,
    capacity_tol: float = 1e-9,
) -> tuple[tuple[int, ...], ...]:
    for customer_index, demand in enumerate(instance.demand):
        if demand > instance.capacity + capacity_tol:
            raise ValueError(
                f"customer {customer_index + 1} demand exceeds capacity: "
                f"{demand}>{instance.capacity}"
            )

    unvisited = set(range(instance.customer_count))
    routes: list[tuple[int, ...]] = []

    while unvisited:
        route: list[int] = []
        load = 0.0
        current_point = instance.depot

        while True:
            candidates = [
                customer_index
                for customer_index in unvisited
                if load + instance.demand[customer_index]
                <= instance.capacity + capacity_tol
            ]
            if not candidates:
                break

            next_customer = min(
                candidates,
                key=lambda customer_index: (
                    euclidean(current_point, instance.loc[customer_index]),
                    customer_index,
                ),
            )
            unvisited.remove(next_customer)
            route.append(next_customer + 1)
            load += instance.demand[next_customer]
            current_point = instance.loc[next_customer]

        if not route:
            raise RuntimeError("nearest neighbor made no progress")
        routes.append(tuple(route))

    return tuple(routes)
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
python3 -m pytest tests/test_heuristics.py -v
```

Expected: PASS.

- [ ] **Step 5: Run accumulated unit tests**

Run:

```bash
python3 -m pytest tests/test_preview_vrp_pkl.py tests/test_vrp_io.py tests/test_vrp_eval.py tests/test_heuristics.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/heuristics.py tests/test_heuristics.py
git commit -m "feat: add nearest neighbor baseline"
```

---

### Task 4: Official `solve.py` CLI

**Files:**
- Create: `solve.py`
- Test: `tests/test_solve_cli.py`

**Interfaces:**
- Consumes:
  - `load_instances`, `SolutionRecord`, `write_solutions_json` from `src.vrp_io`
  - `validate_solution` from `src.vrp_eval`
  - `solve_nearest_neighbor` from `src.heuristics`
- Produces:
  - `solve_instances(instances: Sequence[CVRPInstance]) -> list[SolutionRecord]`
  - `main(argv: list[str] | None = None) -> int`

- [ ] **Step 1: Write failing CLI integration test**

Create `tests/test_solve_cli.py`:

```python
import json
import pickle
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_solve_cli_writes_valid_cvrp_v1_json(tmp_path):
    input_path = tmp_path / "check.pkl"
    output_path = tmp_path / "nested" / "predictions.json"
    raw_instances = [
        ([[0, 0]], [[1, 0], [2, 0], [10, 0]], [4, 4, 4], 8),
        ([[0, 0]], [[0, 1], [0, 2]], [1, 1], 2),
    ]
    with input_path.open("wb") as handle:
        pickle.dump(raw_instances, handle)

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "solve.py"),
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--device",
            "cuda:0",
            "--seed",
            "2026",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["format_version"] == "cvrp_v1"
    assert payload["solutions"] == [
        {"instance_id": 0, "routes": [[1, 2], [3]]},
        {"instance_id": 1, "routes": [[1, 2]]},
    ]
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
python3 -m pytest tests/test_solve_cli.py -v
```

Expected: FAIL because `solve.py` does not exist.

- [ ] **Step 3: Implement `solve.py`**

Create `solve.py`:

```python
"""Official solver entry point for the VRP assessment."""

from __future__ import annotations

import argparse
import random
import sys
from typing import Sequence

from src.heuristics import solve_nearest_neighbor
from src.vrp_eval import validate_solution
from src.vrp_io import CVRPInstance, SolutionRecord, load_instances, write_solutions_json


def solve_instances(instances: Sequence[CVRPInstance]) -> list[SolutionRecord]:
    solutions: list[SolutionRecord] = []
    for instance in instances:
        routes = solve_nearest_neighbor(instance)
        validation = validate_solution(instance, routes)
        if not validation.is_feasible:
            joined_errors = ", ".join(validation.errors)
            raise ValueError(f"instance {instance.instance_id} infeasible: {joined_errors}")
        solutions.append(SolutionRecord(instance_id=instance.instance_id, routes=routes))
    return solutions


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Solve CVRP instances.")
    parser.add_argument("--input", required=True, help="Input .pkl file path.")
    parser.add_argument("--output", required=True, help="Output JSON file path.")
    parser.add_argument("--device", default="cpu", help="Accepted for compatibility.")
    parser.add_argument("--seed", type=int, default=2026, help="Deterministic seed.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    random.seed(args.seed)
    try:
        instances = load_instances(args.input)
        solutions = solve_instances(instances)
        write_solutions_json(args.output, solutions)
    except Exception as exc:
        print(f"solve.py failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run CLI test to verify it passes**

Run:

```bash
python3 -m pytest tests/test_solve_cli.py -v
```

Expected: PASS.

- [ ] **Step 5: Run all accumulated tests**

Run:

```bash
python3 -m pytest tests/test_preview_vrp_pkl.py tests/test_vrp_io.py tests/test_vrp_eval.py tests/test_heuristics.py tests/test_solve_cli.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add solve.py tests/test_solve_cli.py
git commit -m "feat: add official solve entrypoint"
```

---

### Task 5: Baseline Evaluation CLI

**Files:**
- Create: `scripts/evaluate_baseline.py`
- Test: `tests/test_evaluate_baseline_cli.py`

**Interfaces:**
- Consumes:
  - `load_instances` from `src.vrp_io`
  - `solve_nearest_neighbor` from `src.heuristics`
  - `summarize_results` from `src.vrp_eval`
- Produces:
  - `evaluate_instances(instances: Sequence[CVRPInstance], limit: int | None = None) -> EvaluationSummary`
  - CLI command `python3 scripts/evaluate_baseline.py --input <pkl> --limit <n>`

- [ ] **Step 1: Write failing evaluation CLI test**

Create `tests/test_evaluate_baseline_cli.py`:

```python
import json
import pickle
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_evaluate_baseline_cli_prints_summary_json(tmp_path):
    input_path = tmp_path / "validation.pkl"
    raw_instances = [
        ([[0, 0]], [[1, 0], [2, 0]], [1, 1], 2, [[1, 2]], 4.0),
        ([[0, 0]], [[0, 1]], [1], 2, [[1]], 2.0),
    ]
    with input_path.open("wb") as handle:
        pickle.dump(raw_instances, handle)

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "evaluate_baseline.py"),
            "--input",
            str(input_path),
            "--limit",
            "2",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["instance_count"] == 2
    assert payload["feasible_count"] == 2
    assert payload["feasibility_rate"] == 1.0
    assert payload["average_cost"] > 0.0
    assert payload["average_gap"] is not None
    assert payload["average_inference_time"] >= 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
python3 -m pytest tests/test_evaluate_baseline_cli.py -v
```

Expected: FAIL because `scripts/evaluate_baseline.py` does not exist.

- [ ] **Step 3: Implement evaluation CLI**

Create `scripts/evaluate_baseline.py`:

```python
"""Evaluate the nearest neighbor baseline on labeled CVRP data."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.heuristics import solve_nearest_neighbor  # noqa: E402
from src.vrp_eval import EvaluationSummary, summarize_results  # noqa: E402
from src.vrp_io import CVRPInstance, load_instances  # noqa: E402


def evaluate_instances(
    instances: Sequence[CVRPInstance],
    limit: int | None = None,
) -> EvaluationSummary:
    selected = list(instances[:limit]) if limit is not None else list(instances)
    routes_by_instance = []
    inference_times = []
    for instance in selected:
        start = time.perf_counter()
        routes = solve_nearest_neighbor(instance)
        inference_times.append(time.perf_counter() - start)
        routes_by_instance.append(routes)
    return summarize_results(selected, routes_by_instance, inference_times)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate nearest neighbor baseline.")
    parser.add_argument("--input", required=True, help="Input labeled .pkl file path.")
    parser.add_argument("--limit", type=int, default=None, help="Optional instance limit.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = evaluate_instances(load_instances(args.input), args.limit)
    print(
        json.dumps(
            {
                "instance_count": summary.instance_count,
                "feasible_count": summary.feasible_count,
                "feasibility_rate": summary.feasibility_rate,
                "average_cost": summary.average_cost,
                "average_gap": summary.average_gap,
                "average_inference_time": summary.average_inference_time,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run evaluation CLI test to verify it passes**

Run:

```bash
python3 -m pytest tests/test_evaluate_baseline_cli.py -v
```

Expected: PASS.

- [ ] **Step 5: Run full test suite**

Run:

```bash
python3 -m pytest tests -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add scripts/evaluate_baseline.py tests/test_evaluate_baseline_cli.py
git commit -m "feat: add baseline evaluation CLI"
```

---

### Task 6: Repository Hygiene, README, Smoke Test, and Progress Log

**Files:**
- Create: `.gitignore`
- Create: `README.md`
- Modify: `docs/progress/daily-progress.md`

**Interfaces:**
- Consumes:
  - `solve.py`
  - `scripts/evaluate_baseline.py`
  - full test suite
- Produces:
  - Clear local commands for users
  - Git ignore rules for raw data and outputs
  - 2026-07-26 progress entry

- [ ] **Step 1: Add ignore rules for raw data, caches, outputs, and model files**

Create `.gitignore`:

```gitignore
__pycache__/
*.py[cod]
.pytest_cache/
.mypy_cache/

VRP_project/
outputs/
artifacts/
checkpoints/
models/
*.pt
*.pth
*.ckpt
```

- [ ] **Step 2: Add concise README with verified commands**

Create `README.md`:

````markdown
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
python3 scripts/evaluate_baseline.py --input VRP_project/VRPData/validation_data.pkl
```

## Tests

```bash
python3 -m pytest tests -v
```

## Roadmap

- Current: feasible nearest neighbor baseline.
- Next: route-level 2-opt improvement.
- Later: optional AI training on a CUDA machine such as the RTX 4060 laptop.
````

- [ ] **Step 3: Run full unit and integration tests**

Run:

```bash
python3 -m pytest tests -v
```

Expected: PASS.

- [ ] **Step 4: Run raw-data smoke test if local data exists**

Run:

```bash
python3 solve.py --input VRP_project/VRPData/check_data_to_students.pkl --output outputs/predictions.json --device cuda:0 --seed 2026
```

Expected: exit code 0 and `outputs/predictions.json` exists. Because `outputs/` is ignored, do not commit the generated JSON.

- [ ] **Step 5: Run validation evaluation smoke test if local data exists**

Run:

```bash
python3 scripts/evaluate_baseline.py --input VRP_project/VRPData/validation_data.pkl --limit 20
```

Expected: prints JSON containing `"instance_count": 20`, `"feasibility_rate": 1.0`, and non-null `"average_gap"`.

- [ ] **Step 6: Append concise 2026-07-26 progress entry**

Modify `docs/progress/daily-progress.md` by adding this section above older date entries:

```markdown
## 2026-07-26

### 今日进展

- baseline solver 规格已审核通过。
- 生成并执行 baseline implementation plan。
- 实现 CPU 最近邻 baseline、官方 `solve.py`、评估脚本、测试和 README。

### 状态与下一步

- 第一版可提交 baseline 已完成本地验证。
- 下一阶段可 review baseline 结果，并设计 2-opt 改进。
```

- [ ] **Step 7: Verify git status excludes raw data and outputs**

Run:

```bash
git status --short
```

Expected: tracked changes include `.gitignore`, `README.md`, and `docs/progress/daily-progress.md`; raw `VRP_project/` and `outputs/` are not listed.

- [ ] **Step 8: Commit**

```bash
git add .gitignore README.md docs/progress/daily-progress.md
git commit -m "docs: document baseline workflow"
```

---

## Final Verification Before Reporting Completion

- [ ] Run full test suite:

```bash
python3 -m pytest tests -v
```

Expected: PASS.

- [ ] Run official CLI on public check data:

```bash
python3 solve.py --input VRP_project/VRPData/check_data_to_students.pkl --output outputs/predictions.json --device cuda:0 --seed 2026
```

Expected: PASS, output JSON exists, no raw output committed.

- [ ] Run validation metric smoke test:

```bash
python3 scripts/evaluate_baseline.py --input VRP_project/VRPData/validation_data.pkl --limit 20
```

Expected: PASS, feasibility rate is `1.0`.

- [ ] Check whitespace:

```bash
git diff --check
```

Expected: no output, exit code 0.

- [ ] Check branch state:

```bash
git status --short --branch
```

Expected: no uncommitted tracked changes. Ignored raw data and ignored outputs may exist locally.

- [ ] Push completed commits:

```bash
git push origin main
```

Expected: push succeeds and `main` matches `origin/main`.

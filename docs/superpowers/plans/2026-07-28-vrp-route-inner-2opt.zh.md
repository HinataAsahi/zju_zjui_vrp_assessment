# VRP Route Inner 2-opt Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在现有容量感知最近邻 baseline 后加入确定性的 route 内 2-opt 后处理，并让 `solve.py` 与评估脚本默认使用改进版。

**Architecture:** 保留 `solve_nearest_neighbor` 的原始行为不变，在 `src/heuristics.py` 增加纯函数式 2-opt 后处理和统一的 solver method 分发函数。`solve.py` 与 `scripts/evaluate_baseline.py` 都通过同一个 method 分发入口选择 `nearest` 或 `nearest_2opt`，输出前继续使用 `validate_solution` 强制验证。

**Tech Stack:** Python 3.10+ 标准库、`argparse`、`json`、`pytest`；不新增运行时依赖，不使用 CUDA、OR-Tools、PyTorch 或 NumPy。

## Global Constraints

- 每个实例仍先由容量感知最近邻生成可行 routes。
- 对每条 route 独立执行 2-opt，只改变同一条 route 内客户访问顺序。
- 不改变客户属于哪辆车，因此不主动触碰容量分配。
- 优化后仍强制运行可行性检查。
- `solve.py` 默认使用 2-opt 增强版，但保留参数可运行原始最近邻 baseline。
- `scripts/evaluate_baseline.py` 可以分别评估 `nearest` 与 `nearest_2opt`，便于量化收益。
- 不引入新的运行时依赖，不依赖 CUDA。
- 本阶段不实现 route 间换客户、route 间 relocate、route 间 swap、OR-Tools、神经网络训练或最终报告材料。
- route 内 2-opt 不新增客户、不删除客户、不重复客户、不改变 route 数量、不改变每条 route 的客户集合。
- 验证集 `nearest_2opt` 的平均 cost 不高于已记录的原始最近邻 baseline 平均 cost `13.934951704742822`。

---

## File Structure

- Modify: `src/heuristics.py`
  - 增加 `SolverMethod`、`SOLVER_METHODS`、`improve_route_2opt`、`improve_routes_2opt`、`solve_nearest_neighbor_2opt`、`solve_with_method`。
- Modify: `tests/test_heuristics.py`
  - 增加 route 内 2-opt 的行为测试和 method 分发测试。
- Modify: `solve.py`
  - 新增 `--method` 参数，默认 `nearest_2opt`，调用 `solve_with_method`。
- Modify: `tests/test_solve_cli.py`
  - 覆盖默认 method、`--method nearest` 兼容行为和非法 method。
- Modify: `scripts/evaluate_baseline.py`
  - 新增 `--method` 参数，默认 `nearest_2opt`，评估时调用 `solve_with_method`。
- Modify: `tests/test_evaluate_baseline_cli.py`
  - 覆盖评估脚本的默认 method、`--method nearest_2opt` 和 method 参数传递。
- Modify: `README.md`
  - 说明默认 solver 已升级为最近邻 + route 内 2-opt，并给出原始最近邻对照命令。
- Modify: `docs/experiments/baseline_results.md`
  - 记录完整验证集 `nearest_2opt` 结果，与已记录的 `nearest` 结果对比。
- Modify: `docs/progress/daily-progress.md`
  - 简短记录 2026-07-28 的 2-opt 设计、计划、实现与验证状态。

---

### Task 1: Core Route-Inner 2-opt Heuristic

**Files:**
- Modify: `src/heuristics.py`
- Test: `tests/test_heuristics.py`

**Interfaces:**
- Consumes:
  - `CVRPInstance` from `src.vrp_io`
  - `compute_route_cost(instance: CVRPInstance, route: Sequence[int]) -> float` from `src.vrp_eval`
  - `compute_total_cost(instance: CVRPInstance, routes: Sequence[Sequence[int]]) -> float` from `src.vrp_eval`
  - `validate_solution(instance: CVRPInstance, routes: Sequence[Sequence[int]], capacity_tol: float = 1e-9) -> ValidationResult` from `src.vrp_eval`
- Produces:
  - `SolverMethod = Literal["nearest", "nearest_2opt"]`
  - `SOLVER_METHODS: tuple[SolverMethod, ...]`
  - `improve_route_2opt(instance: CVRPInstance, route: Sequence[int], improvement_tol: float = 1e-12) -> tuple[int, ...]`
  - `improve_routes_2opt(instance: CVRPInstance, routes: Sequence[Sequence[int]], improvement_tol: float = 1e-12) -> tuple[tuple[int, ...], ...]`
  - `solve_nearest_neighbor_2opt(instance: CVRPInstance, capacity_tol: float = 1e-9, improvement_tol: float = 1e-12) -> tuple[tuple[int, ...], ...]`
  - `solve_with_method(instance: CVRPInstance, method: str = "nearest_2opt", capacity_tol: float = 1e-9, improvement_tol: float = 1e-12) -> tuple[tuple[int, ...], ...]`

- [ ] **Step 1: Extend heuristic imports in the test file**

Modify the top of `tests/test_heuristics.py` so the imports include `pytest`, the new heuristic functions, and cost helpers:

```python
import pytest

from src.heuristics import (  # noqa: E402
    improve_route_2opt,
    improve_routes_2opt,
    solve_nearest_neighbor,
    solve_nearest_neighbor_2opt,
    solve_with_method,
)
from src.vrp_eval import compute_route_cost, compute_total_cost, validate_solution  # noqa: E402
```

- [ ] **Step 2: Write failing tests for route-level 2-opt behavior**

Append these tests to `tests/test_heuristics.py`:

```python
def make_crossing_route_instance() -> CVRPInstance:
    return CVRPInstance(
        instance_id=0,
        depot=(0.5, -1.0),
        loc=(
            (0.0, 0.0),
            (1.0, 1.0),
            (0.0, 1.0),
            (1.0, 0.0),
            (2.0, 0.0),
        ),
        demand=(1.0, 1.0, 1.0, 1.0, 1.0),
        capacity=5.0,
    )


def test_improve_route_2opt_reduces_crossing_route_cost():
    instance = make_crossing_route_instance()
    route = (1, 2, 3, 4)

    improved = improve_route_2opt(instance, route)

    assert len(improved) == len(route)
    assert sorted(improved) == sorted(route)
    assert compute_route_cost(instance, improved) < compute_route_cost(instance, route)


def test_improve_route_2opt_keeps_local_optimum_route_unchanged():
    instance = make_crossing_route_instance()
    route = (1, 3, 2, 4)

    improved = improve_route_2opt(instance, route)

    assert improved == route


def test_improve_routes_2opt_preserves_route_boundaries_and_customer_sets():
    instance = make_crossing_route_instance()
    routes = ((1, 2, 3, 4), (5,))

    improved = improve_routes_2opt(instance, routes)

    assert len(improved) == len(routes)
    assert [sorted(route) for route in improved] == [sorted(route) for route in routes]
    assert compute_total_cost(instance, improved) <= compute_total_cost(instance, routes) + 1e-12


def test_solve_nearest_neighbor_2opt_keeps_solution_feasible_and_not_worse():
    instance = CVRPInstance(
        instance_id=0,
        depot=(0.0, 0.0),
        loc=((1.0, 0.0), (0.0, 1.0), (1.0, 1.0), (2.0, 0.0)),
        demand=(1.0, 1.0, 1.0, 1.0),
        capacity=4.0,
    )

    nearest_routes = solve_nearest_neighbor(instance)
    improved_routes = solve_nearest_neighbor_2opt(instance)

    assert validate_solution(instance, improved_routes).is_feasible is True
    assert compute_total_cost(instance, improved_routes) <= compute_total_cost(instance, nearest_routes) + 1e-12


def test_solve_with_method_rejects_unknown_method():
    instance = make_crossing_route_instance()

    with pytest.raises(ValueError, match="unknown solver method"):
        solve_with_method(instance, method="not_a_method")
```

- [ ] **Step 3: Run heuristic tests to verify RED**

Run:

```bash
python3 -m pytest tests/test_heuristics.py -v
```

Expected: FAIL with import errors for `improve_route_2opt`, `improve_routes_2opt`, `solve_nearest_neighbor_2opt`, or `solve_with_method`.

- [ ] **Step 4: Implement the 2-opt functions in `src/heuristics.py`**

Modify `src/heuristics.py` imports:

```python
from typing import Literal, Sequence

from src.vrp_eval import euclidean
from src.vrp_io import CVRPInstance
```

Add these definitions below the imports and above `solve_nearest_neighbor`:

```python
SolverMethod = Literal["nearest", "nearest_2opt"]
SOLVER_METHODS: tuple[SolverMethod, ...] = ("nearest", "nearest_2opt")
```

Add these functions below `solve_nearest_neighbor`:

```python
def _route_node_point(
    instance: CVRPInstance,
    customer_id: int | None,
) -> tuple[float, float]:
    if customer_id is None:
        return instance.depot
    if customer_id < 1 or customer_id > instance.customer_count:
        raise ValueError("route customer IDs should be positive 1-based integers")
    return instance.loc[customer_id - 1]


def improve_route_2opt(
    instance: CVRPInstance,
    route: Sequence[int],
    improvement_tol: float = 1e-12,
) -> tuple[int, ...]:
    best_route = tuple(route)
    if len(best_route) < 3:
        return best_route

    while True:
        route_len = len(best_route)
        improved = False
        for i in range(route_len - 1):
            for j in range(i + 1, route_len):
                prev_customer = None if i == 0 else best_route[i - 1]
                next_customer = None if j == route_len - 1 else best_route[j + 1]
                prev_point = _route_node_point(instance, prev_customer)
                first_point = _route_node_point(instance, best_route[i])
                last_point = _route_node_point(instance, best_route[j])
                next_point = _route_node_point(instance, next_customer)

                current_edges = euclidean(prev_point, first_point) + euclidean(
                    last_point,
                    next_point,
                )
                candidate_edges = euclidean(prev_point, last_point) + euclidean(
                    first_point,
                    next_point,
                )
                if candidate_edges + improvement_tol < current_edges:
                    best_route = (
                        best_route[:i]
                        + tuple(reversed(best_route[i : j + 1]))
                        + best_route[j + 1 :]
                    )
                    improved = True
                    break
            if improved:
                break
        if not improved:
            return best_route


def improve_routes_2opt(
    instance: CVRPInstance,
    routes: Sequence[Sequence[int]],
    improvement_tol: float = 1e-12,
) -> tuple[tuple[int, ...], ...]:
    return tuple(
        improve_route_2opt(instance, route, improvement_tol=improvement_tol)
        for route in routes
    )


def solve_nearest_neighbor_2opt(
    instance: CVRPInstance,
    capacity_tol: float = 1e-9,
    improvement_tol: float = 1e-12,
) -> tuple[tuple[int, ...], ...]:
    routes = solve_nearest_neighbor(instance, capacity_tol=capacity_tol)
    return improve_routes_2opt(
        instance,
        routes,
        improvement_tol=improvement_tol,
    )


def solve_with_method(
    instance: CVRPInstance,
    method: str = "nearest_2opt",
    capacity_tol: float = 1e-9,
    improvement_tol: float = 1e-12,
) -> tuple[tuple[int, ...], ...]:
    if method == "nearest":
        return solve_nearest_neighbor(instance, capacity_tol=capacity_tol)
    if method == "nearest_2opt":
        return solve_nearest_neighbor_2opt(
            instance,
            capacity_tol=capacity_tol,
            improvement_tol=improvement_tol,
        )
    raise ValueError(f"unknown solver method: {method}")
```

- [ ] **Step 5: Run heuristic tests to verify GREEN**

Run:

```bash
python3 -m pytest tests/test_heuristics.py -v
```

Expected: PASS for all tests in `tests/test_heuristics.py`.

- [ ] **Step 6: Commit Task 1**

Run:

```bash
git add src/heuristics.py tests/test_heuristics.py
git diff --cached --check
git commit -m "feat: add route inner 2-opt heuristic"
```

---

### Task 2: Official Solver Method Selection

**Files:**
- Modify: `solve.py`
- Test: `tests/test_solve_cli.py`

**Interfaces:**
- Consumes:
  - `SOLVER_METHODS` and `solve_with_method` from `src.heuristics`
  - `solve_instances(instances: Sequence[CVRPInstance], method: str = "nearest_2opt") -> list[SolutionRecord]`
- Produces:
  - `solve.py --method nearest_2opt` as default official behavior
  - `solve.py --method nearest` for original baseline comparison

- [ ] **Step 1: Write failing tests for solver method behavior**

Modify `tests/test_solve_cli.py` imports:

```python
import json
import pickle
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from solve import parse_args, solve_instances  # noqa: E402
from src.vrp_eval import validate_solution  # noqa: E402
from src.vrp_io import CVRPInstance  # noqa: E402
```

Append these tests:

```python
def test_parse_args_defaults_to_nearest_2opt():
    args = parse_args(
        [
            "--input",
            "input.pkl",
            "--output",
            "predictions.json",
        ]
    )

    assert args.method == "nearest_2opt"


def test_solve_instances_accepts_nearest_2opt_method():
    instance = CVRPInstance(
        instance_id=3,
        depot=(0.0, 0.0),
        loc=((1.0, 0.0), (0.0, 1.0), (1.0, 1.0)),
        demand=(1.0, 1.0, 1.0),
        capacity=3.0,
    )

    solutions = solve_instances([instance], method="nearest_2opt")

    assert len(solutions) == 1
    assert solutions[0].instance_id == 3
    assert validate_solution(instance, solutions[0].routes).is_feasible is True
```

Modify the existing `test_solve_cli_writes_valid_cvrp_v1_json` subprocess arguments to include:

```python
            "--method",
            "nearest",
```

Add this subprocess test:

```python
def test_solve_cli_rejects_unknown_method(tmp_path):
    input_path = tmp_path / "check.pkl"
    output_path = tmp_path / "predictions.json"
    with input_path.open("wb") as handle:
        pickle.dump([], handle)

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "solve.py"),
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--method",
            "bad_method",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "invalid choice" in result.stderr
```

- [ ] **Step 2: Run solver CLI tests to verify RED**

Run:

```bash
python3 -m pytest tests/test_solve_cli.py -v
```

Expected: FAIL because `parse_args` does not define `method`, and `solve_instances` does not accept `method`.

- [ ] **Step 3: Implement method selection in `solve.py`**

Modify the heuristic import:

```python
from src.heuristics import SOLVER_METHODS, solve_with_method
```

Change `solve_instances`:

```python
def solve_instances(
    instances: Sequence[CVRPInstance],
    method: str = "nearest_2opt",
) -> list[SolutionRecord]:
    solutions: list[SolutionRecord] = []
    for instance in instances:
        routes = solve_with_method(instance, method=method)
        validation = validate_solution(instance, routes)
        if not validation.is_feasible:
            joined_errors = ", ".join(validation.errors)
            raise ValueError(f"instance {instance.instance_id} infeasible: {joined_errors}")
        solutions.append(SolutionRecord(instance_id=instance.instance_id, routes=routes))
    return solutions
```

Add the parser argument after `--seed`:

```python
    parser.add_argument(
        "--method",
        choices=SOLVER_METHODS,
        default="nearest_2opt",
        help="Solver method: nearest_2opt uses nearest neighbor plus route-inner 2-opt.",
    )
```

Change `main` so it passes the method:

```python
        solutions = solve_instances(instances, method=args.method)
```

- [ ] **Step 4: Run solver CLI tests to verify GREEN**

Run:

```bash
python3 -m pytest tests/test_solve_cli.py -v
```

Expected: PASS for all tests in `tests/test_solve_cli.py`.

- [ ] **Step 5: Commit Task 2**

Run:

```bash
git add solve.py tests/test_solve_cli.py
git diff --cached --check
git commit -m "feat: add solver method selection"
```

---

### Task 3: Evaluation Script Method Selection

**Files:**
- Modify: `scripts/evaluate_baseline.py`
- Test: `tests/test_evaluate_baseline_cli.py`

**Interfaces:**
- Consumes:
  - `SOLVER_METHODS` and `solve_with_method` from `src.heuristics`
  - `evaluate_instances(instances: Sequence[CVRPInstance], limit: int | None = None, method: str = "nearest_2opt") -> EvaluationSummary`
- Produces:
  - `scripts/evaluate_baseline.py --method nearest_2opt`
  - `scripts/evaluate_baseline.py --method nearest`

- [ ] **Step 1: Write failing tests for evaluation method behavior**

Modify `tests/test_evaluate_baseline_cli.py` imports:

```python
from scripts.evaluate_baseline import evaluate_instances, parse_args  # noqa: E402
from src.vrp_io import CVRPInstance  # noqa: E402
```

Append these tests:

```python
def test_evaluate_parse_args_defaults_to_nearest_2opt():
    args = parse_args(["--input", "validation.pkl"])

    assert args.method == "nearest_2opt"


def test_evaluate_instances_accepts_nearest_2opt_method():
    instance = CVRPInstance(
        instance_id=0,
        depot=(0.0, 0.0),
        loc=((1.0, 0.0), (0.0, 1.0)),
        demand=(1.0, 1.0),
        capacity=2.0,
        reference_cost=4.0,
    )

    summary = evaluate_instances([instance], method="nearest_2opt")

    assert summary.instance_count == 1
    assert summary.feasible_count == 1
    assert summary.feasibility_rate == 1.0
    assert summary.average_cost > 0.0
```

Modify `test_evaluate_baseline_cli_prints_summary_json` subprocess arguments to include:

```python
            "--method",
            "nearest_2opt",
```

Add this subprocess test:

```python
def test_evaluate_baseline_cli_rejects_unknown_method(tmp_path):
    input_path = tmp_path / "validation.pkl"
    with input_path.open("wb") as handle:
        pickle.dump([], handle)

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "evaluate_baseline.py"),
            "--input",
            str(input_path),
            "--method",
            "bad_method",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "invalid choice" in result.stderr
```

- [ ] **Step 2: Run evaluation CLI tests to verify RED**

Run:

```bash
python3 -m pytest tests/test_evaluate_baseline_cli.py -v
```

Expected: FAIL because `parse_args` does not define `method`, and `evaluate_instances` does not accept `method`.

- [ ] **Step 3: Implement method selection in `scripts/evaluate_baseline.py`**

Modify the heuristic import:

```python
from src.heuristics import SOLVER_METHODS, solve_with_method  # noqa: E402
```

Change `evaluate_instances`:

```python
def evaluate_instances(
    instances: Sequence[CVRPInstance],
    limit: int | None = None,
    method: str = "nearest_2opt",
) -> EvaluationSummary:
    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative")
    selected = list(instances[:limit]) if limit is not None else list(instances)
    routes_by_instance = []
    inference_times = []
    for instance in selected:
        start = time.perf_counter()
        routes = solve_with_method(instance, method=method)
        inference_times.append(time.perf_counter() - start)
        routes_by_instance.append(routes)
    return summarize_results(selected, routes_by_instance, inference_times)
```

Add the parser argument after `--limit`:

```python
    parser.add_argument(
        "--method",
        choices=SOLVER_METHODS,
        default="nearest_2opt",
        help="Solver method to evaluate.",
    )
```

Change `main` so it passes the method:

```python
    summary = evaluate_instances(
        load_instances(args.input),
        limit=args.limit,
        method=args.method,
    )
```

- [ ] **Step 4: Run evaluation CLI tests to verify GREEN**

Run:

```bash
python3 -m pytest tests/test_evaluate_baseline_cli.py -v
```

Expected: PASS for all tests in `tests/test_evaluate_baseline_cli.py`.

- [ ] **Step 5: Commit Task 3**

Run:

```bash
git add scripts/evaluate_baseline.py tests/test_evaluate_baseline_cli.py
git diff --cached --check
git commit -m "feat: add evaluation method selection"
```

---

### Task 4: Documentation, Full Verification, and Result Recording

**Files:**
- Modify: `README.md`
- Modify: `docs/experiments/baseline_results.md`
- Modify: `docs/progress/daily-progress.md`
- Test: full project verification commands

**Interfaces:**
- Consumes:
  - `solve.py --method nearest_2opt`
  - `scripts/evaluate_baseline.py --method nearest_2opt`
  - Existing baseline metric `13.934951704742822`
- Produces:
  - Updated user-facing run commands
  - Recorded validation result for `nearest_2opt`
  - Short daily progress entry for 2026-07-28

- [ ] **Step 1: Run full unit test suite**

Run:

```bash
python3 -m pytest tests -v
```

Expected: PASS for all tests.

- [ ] **Step 2: Run official solver smoke test on check data**

Run:

```bash
python3 solve.py --input VRP_project/VRPData/check_data_to_students.pkl --output outputs/predictions.json --device cuda:0 --seed 2026
```

Expected: exit code 0 and `outputs/predictions.json` is written. The output file remains ignored by git.

- [ ] **Step 3: Run full validation evaluation for improved method**

Run:

```bash
python3 scripts/evaluate_baseline.py --input VRP_project/VRPData/validation_data.pkl --method nearest_2opt
```

Expected: JSON output where `instance_count` is `1000`, `feasible_count` is `1000`, `feasibility_rate` is `1.0`, and `average_cost` is less than or equal to `13.934951704742822` within normal floating-point tolerance.

- [ ] **Step 4: Run original nearest evaluation for comparison if needed**

Run:

```bash
python3 scripts/evaluate_baseline.py --input VRP_project/VRPData/validation_data.pkl --method nearest
```

Expected: JSON output comparable to the recorded 2026-07-26 nearest baseline. Use it only as a sanity check if the improved method result looks suspicious.

- [ ] **Step 5: Update README with method commands**

Modify `README.md` so the solver section states that the default method is deterministic nearest neighbor plus route-inner 2-opt, and add this comparison command:

```bash
python3 solve.py --input VRP_project/VRPData/check_data_to_students.pkl --output outputs/predictions_nearest.json --method nearest --device cuda:0 --seed 2026
```

Modify the evaluation section so it shows:

```bash
python3 scripts/evaluate_baseline.py --input VRP_project/VRPData/validation_data.pkl --method nearest_2opt
python3 scripts/evaluate_baseline.py --input VRP_project/VRPData/validation_data.pkl --method nearest
```

- [ ] **Step 6: Record validation results**

Append a new section to `docs/experiments/baseline_results.md` with heading:

```markdown
## 2026-07-28: Capacity-Aware Nearest Neighbor + Route-Inner 2-opt
```

The section must include the exact command from Step 3, the printed `instance_count`, `feasible_count`, `feasibility_rate`, `average_cost`, `average_gap`, and `average_inference_time`, plus a note comparing `average_cost` against the previously recorded nearest baseline average cost `13.934951704742822`.

- [ ] **Step 7: Record daily progress**

Append or update the 2026-07-28 section in `docs/progress/daily-progress.md` with concise bullets:

```markdown
## 2026-07-28

### 今日进展

- 审核通过 route 内 2-opt 设计，并生成 implementation plan。
- 实现最近邻 + route 内 2-opt，并保留原始最近邻对照方法。
- 完成测试、官方 check 数据 smoke test 和完整验证集评估。

### 状态与下一步

- 当前默认方法为 `nearest_2opt`。
- 下一阶段根据收益决定是否设计 route 间 relocate/swap。
```

- [ ] **Step 8: Run documentation and full verification checks**

Run:

```bash
git diff --check
python3 -m pytest tests -v
```

Expected: `git diff --check` exits 0, and pytest reports all tests passing.

- [ ] **Step 9: Commit Task 4**

Run:

```bash
git add README.md docs/experiments/baseline_results.md docs/progress/daily-progress.md
git diff --cached --check
git commit -m "docs: record route inner 2-opt results"
```

---

## Plan Self-Review

- Spec coverage: Task 1 covers route 内 2-opt correctness and deterministic heuristic behavior; Task 2 covers `solve.py --method`; Task 3 covers evaluation method selection; Task 4 covers README, experiment results, full verification, and progress recording.
- Placeholder scan: This plan contains no unresolved marker words, incomplete code slots, or unspecified function names.
- Type consistency: The produced signatures in Task 1 are consumed by Task 2 and Task 3 with the same names and parameter order.
- Scope check: route 间 relocate/swap is explicitly excluded from this plan and remains a future independent phase.

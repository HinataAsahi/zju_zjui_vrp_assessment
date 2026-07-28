# VRP Inter-Route Best Relocate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 新增可选方法 `nearest_2opt_relocate_best`，在当前 `nearest_2opt` 后执行 route 间单客户 best relocate，并保持默认方法仍为 `nearest_2opt`。

**Architecture:** 在 `src/heuristics.py` 内扩展现有启发式模块，先生成 `nearest_2opt` 解，再通过纯函数式 best relocate 后处理调整 route 间客户分配。CLI 和评估脚本继续复用 `SOLVER_METHODS` 与 `solve_with_method`，只新增 method 选择、测试、README、实验记录和每日进度。

**Tech Stack:** Python 3.10+ 标准库、`dataclasses`、`argparse`、`json`、`pytest`；不新增运行时依赖，不使用 CUDA、OR-Tools、PyTorch 或 NumPy。

## Global Constraints

- 保留 `nearest` 和 `nearest_2opt` 的既有行为。
- `solve.py` 默认仍使用 `nearest_2opt`。
- 新方法通过 `--method nearest_2opt_relocate_best` 显式启用。
- route 间移动必须满足容量约束。
- route 间移动不能漏客户、重复客户或产生非法客户编号。
- 每次接受的移动必须降低总路径长度。
- 完整验证集可行率保持 1.0。
- 完整验证集平均 cost 不高于 `nearest_2opt` 已记录值 `13.410121525149266`。
- 继续保持 CPU 可运行，不引入新运行时依赖。
- 本阶段不实现 route 间 swap、candidate-limited relocate、多客户同时移动、随机化搜索、OR-Tools、神经网络训练或默认方法升级。
- 默认搜索预算固定为 `max_relocate_passes = 50`；本阶段不新增 CLI 参数控制搜索预算。

---

## File Structure

- Modify: `src/heuristics.py`
  - 增加 `_RelocateMove`、route load helper、best relocate 搜索与应用函数、`solve_nearest_neighbor_2opt_relocate_best`，并把新 method 加入分发。
- Modify: `tests/test_heuristics.py`
  - 增加 route 间 relocate 降 cost、容量阻塞、空 route 删除、可行性和 method 分发测试。
- Modify: `solve.py`
  - 更新 `--method` help text，使用户能发现 `nearest_2opt_relocate_best`。
- Modify: `tests/test_solve_cli.py`
  - 覆盖 `solve.py --method nearest_2opt_relocate_best` 能写出合法 JSON。
- Modify: `scripts/evaluate_baseline.py`
  - 更新评估脚本 help text。
- Modify: `tests/test_evaluate_baseline_cli.py`
  - 覆盖评估脚本能显式评估 `nearest_2opt_relocate_best`。
- Modify: `README.md`
  - 记录新 method 的用途和运行命令，说明默认仍为 `nearest_2opt`。
- Modify: `docs/experiments/baseline_results.md`
  - 记录完整验证集 `nearest_2opt_relocate_best` 结果。
- Modify: `docs/progress/daily-progress.md`
  - 简短记录本阶段设计、实现和验证状态。

---

### Task 1: Core Best Relocate Heuristic

**Files:**
- Modify: `src/heuristics.py`
- Test: `tests/test_heuristics.py`

**Interfaces:**
- Consumes:
  - `CVRPInstance` from `src.vrp_io`
  - `compute_route_cost(instance: CVRPInstance, route: Sequence[int]) -> float` from `src.vrp_eval`
  - `compute_total_cost(instance: CVRPInstance, routes: Sequence[Sequence[int]]) -> float` from `src.vrp_eval`
  - `validate_solution(instance: CVRPInstance, routes: Sequence[Sequence[int]], capacity_tol: float = 1e-9) -> ValidationResult` from `src.vrp_eval`
  - `improve_route_2opt(instance: CVRPInstance, route: Sequence[int], improvement_tol: float = 1e-12) -> tuple[int, ...]`
  - `solve_nearest_neighbor_2opt(instance: CVRPInstance, capacity_tol: float = 1e-9, improvement_tol: float = 1e-12) -> tuple[tuple[int, ...], ...]`
- Produces:
  - `SolverMethod = Literal["nearest", "nearest_2opt", "nearest_2opt_relocate_best"]`
  - `SOLVER_METHODS: tuple[SolverMethod, ...]`
  - `improve_routes_relocate_best(instance: CVRPInstance, routes: Sequence[Sequence[int]], capacity_tol: float = 1e-9, improvement_tol: float = 1e-12, max_passes: int = 50) -> tuple[tuple[int, ...], ...]`
  - `solve_nearest_neighbor_2opt_relocate_best(instance: CVRPInstance, capacity_tol: float = 1e-9, improvement_tol: float = 1e-12, max_relocate_passes: int = 50) -> tuple[tuple[int, ...], ...]`
  - `solve_with_method(..., method="nearest_2opt_relocate_best", ...)`

- [ ] **Step 1: Write failing tests for best relocate**

Modify the import block in `tests/test_heuristics.py`:

```python
from src.heuristics import (  # noqa: E402
    improve_route_2opt,
    improve_routes_2opt,
    improve_routes_relocate_best,
    solve_nearest_neighbor,
    solve_nearest_neighbor_2opt,
    solve_nearest_neighbor_2opt_relocate_best,
    solve_with_method,
)
```

Append these tests to `tests/test_heuristics.py`:

```python
def make_relocate_instance(
    demand: tuple[float, ...] = (1.0, 1.0, 1.0),
    capacity: float = 3.0,
) -> CVRPInstance:
    return CVRPInstance(
        instance_id=0,
        depot=(0.0, 0.0),
        loc=((1.0, 0.0), (10.0, 0.0), (11.0, 0.0)),
        demand=demand,
        capacity=capacity,
    )


def test_improve_routes_relocate_best_moves_customer_between_routes_to_reduce_cost():
    instance = make_relocate_instance()
    routes = ((1, 2), (3,))

    improved = improve_routes_relocate_best(instance, routes)

    assert len(improved) == 2
    assert sorted(customer for route in improved for customer in route) == [1, 2, 3]
    assert compute_total_cost(instance, improved) < compute_total_cost(instance, routes)
    assert validate_solution(instance, improved).is_feasible is True


def test_improve_routes_relocate_best_skips_moves_that_exceed_target_capacity():
    instance = make_relocate_instance(
        demand=(1.0, 0.5, 1.5),
        capacity=1.5,
    )
    routes = ((1, 2), (3,))

    improved = improve_routes_relocate_best(instance, routes)

    assert improved == routes
    assert validate_solution(instance, improved).is_feasible is True


def test_improve_routes_relocate_best_removes_empty_source_route():
    instance = make_relocate_instance()
    routes = ((2,), (1, 3))

    improved = improve_routes_relocate_best(instance, routes)

    assert len(improved) == 1
    assert sorted(improved[0]) == [1, 2, 3]
    assert compute_total_cost(instance, improved) < compute_total_cost(instance, routes)
    assert validate_solution(instance, improved).is_feasible is True


def test_improve_routes_relocate_best_rejects_negative_max_passes():
    instance = make_relocate_instance()

    with pytest.raises(ValueError, match="max_passes must be non-negative"):
        improve_routes_relocate_best(instance, ((1, 2), (3,)), max_passes=-1)


def test_solve_nearest_neighbor_2opt_relocate_best_keeps_solution_feasible_and_not_worse():
    instance = CVRPInstance(
        instance_id=0,
        depot=(0.0, 0.0),
        loc=((1.0, 0.0), (2.0, 0.0), (10.0, 0.0), (11.0, 0.0)),
        demand=(1.0, 1.0, 1.0, 1.0),
        capacity=2.0,
    )

    base_routes = solve_nearest_neighbor_2opt(instance)
    improved_routes = solve_nearest_neighbor_2opt_relocate_best(instance)

    assert validate_solution(instance, improved_routes).is_feasible is True
    assert compute_total_cost(instance, improved_routes) <= compute_total_cost(instance, base_routes) + 1e-12


def test_solve_with_method_accepts_nearest_2opt_relocate_best():
    instance = make_relocate_instance()

    routes = solve_with_method(instance, method="nearest_2opt_relocate_best")

    assert validate_solution(instance, routes).is_feasible is True
```

- [ ] **Step 2: Run heuristic tests to verify RED**

Run:

```bash
python3 -m pytest tests/test_heuristics.py -v
```

Expected: FAIL during collection with an import error for `improve_routes_relocate_best` or `solve_nearest_neighbor_2opt_relocate_best`.

- [ ] **Step 3: Implement best relocate in `src/heuristics.py`**

Modify imports at the top of `src/heuristics.py`:

```python
from dataclasses import dataclass
from typing import Literal, Sequence

from src.vrp_eval import compute_route_cost, compute_total_cost, euclidean
from src.vrp_io import CVRPInstance
```

Replace method definitions with:

```python
SolverMethod = Literal["nearest", "nearest_2opt", "nearest_2opt_relocate_best"]
SOLVER_METHODS: tuple[SolverMethod, ...] = (
    "nearest",
    "nearest_2opt",
    "nearest_2opt_relocate_best",
)
```

Add this dataclass below `SOLVER_METHODS`:

```python
@dataclass(frozen=True)
class _RelocateMove:
    cost_delta: float
    source_route_index: int
    target_route_index: int
    source_customer_position: int
    target_insert_position: int
    customer_id: int
```

Add these helper functions below `improve_routes_2opt`:

```python
def _route_load(instance: CVRPInstance, route: Sequence[int]) -> float:
    return sum(instance.demand[customer_id - 1] for customer_id in route)


def _find_best_relocate_move(
    instance: CVRPInstance,
    routes: tuple[tuple[int, ...], ...],
    route_loads: tuple[float, ...],
    capacity_tol: float,
    improvement_tol: float,
) -> _RelocateMove | None:
    best_move: _RelocateMove | None = None

    for source_route_index, source_route in enumerate(routes):
        source_cost = compute_route_cost(instance, source_route)
        for source_customer_position, customer_id in enumerate(source_route):
            customer_demand = instance.demand[customer_id - 1]
            new_source_route = (
                source_route[:source_customer_position]
                + source_route[source_customer_position + 1 :]
            )
            new_source_cost = compute_route_cost(instance, new_source_route)

            for target_route_index, target_route in enumerate(routes):
                if source_route_index == target_route_index:
                    continue
                if (
                    route_loads[target_route_index] + customer_demand
                    > instance.capacity + capacity_tol
                ):
                    continue

                target_cost = compute_route_cost(instance, target_route)
                current_cost = source_cost + target_cost
                for target_insert_position in range(len(target_route) + 1):
                    new_target_route = (
                        target_route[:target_insert_position]
                        + (customer_id,)
                        + target_route[target_insert_position:]
                    )
                    candidate_cost = new_source_cost + compute_route_cost(
                        instance,
                        new_target_route,
                    )
                    if candidate_cost + improvement_tol >= current_cost:
                        continue

                    move = _RelocateMove(
                        cost_delta=candidate_cost - current_cost,
                        source_route_index=source_route_index,
                        target_route_index=target_route_index,
                        source_customer_position=source_customer_position,
                        target_insert_position=target_insert_position,
                        customer_id=customer_id,
                    )
                    if best_move is None or (
                        move.cost_delta,
                        move.source_route_index,
                        move.target_route_index,
                        move.source_customer_position,
                        move.target_insert_position,
                        move.customer_id,
                    ) < (
                        best_move.cost_delta,
                        best_move.source_route_index,
                        best_move.target_route_index,
                        best_move.source_customer_position,
                        best_move.target_insert_position,
                        best_move.customer_id,
                    ):
                        best_move = move

    return best_move


def _apply_relocate_move(
    instance: CVRPInstance,
    routes: tuple[tuple[int, ...], ...],
    move: _RelocateMove,
    improvement_tol: float,
) -> tuple[tuple[int, ...], ...]:
    mutable_routes = [list(route) for route in routes]
    customer_id = mutable_routes[move.source_route_index].pop(
        move.source_customer_position
    )
    if customer_id != move.customer_id:
        raise RuntimeError("relocate move customer mismatch")

    mutable_routes[move.target_route_index].insert(
        move.target_insert_position,
        customer_id,
    )

    affected_route_indexes = {move.source_route_index, move.target_route_index}
    improved_routes: list[tuple[int, ...]] = []
    for route_index, route in enumerate(mutable_routes):
        if not route:
            continue
        route_tuple = tuple(route)
        if route_index in affected_route_indexes:
            route_tuple = improve_route_2opt(
                instance,
                route_tuple,
                improvement_tol=improvement_tol,
            )
        improved_routes.append(route_tuple)
    return tuple(improved_routes)
```

Add these public functions below the helpers:

```python
def improve_routes_relocate_best(
    instance: CVRPInstance,
    routes: Sequence[Sequence[int]],
    capacity_tol: float = 1e-9,
    improvement_tol: float = 1e-12,
    max_passes: int = 50,
) -> tuple[tuple[int, ...], ...]:
    if max_passes < 0:
        raise ValueError("max_passes must be non-negative")

    best_routes = tuple(tuple(route) for route in routes)
    for _ in range(max_passes):
        route_loads = tuple(_route_load(instance, route) for route in best_routes)
        move = _find_best_relocate_move(
            instance,
            best_routes,
            route_loads,
            capacity_tol=capacity_tol,
            improvement_tol=improvement_tol,
        )
        if move is None:
            return best_routes

        current_total_cost = compute_total_cost(instance, best_routes)
        candidate_routes = _apply_relocate_move(
            instance,
            best_routes,
            move,
            improvement_tol=improvement_tol,
        )
        if (
            compute_total_cost(instance, candidate_routes) + improvement_tol
            >= current_total_cost
        ):
            return best_routes
        best_routes = candidate_routes

    return best_routes


def solve_nearest_neighbor_2opt_relocate_best(
    instance: CVRPInstance,
    capacity_tol: float = 1e-9,
    improvement_tol: float = 1e-12,
    max_relocate_passes: int = 50,
) -> tuple[tuple[int, ...], ...]:
    routes = solve_nearest_neighbor_2opt(
        instance,
        capacity_tol=capacity_tol,
        improvement_tol=improvement_tol,
    )
    return improve_routes_relocate_best(
        instance,
        routes,
        capacity_tol=capacity_tol,
        improvement_tol=improvement_tol,
        max_passes=max_relocate_passes,
    )
```

Update `solve_with_method`:

```python
    if method == "nearest_2opt_relocate_best":
        return solve_nearest_neighbor_2opt_relocate_best(
            instance,
            capacity_tol=capacity_tol,
            improvement_tol=improvement_tol,
        )
```

Place that branch before the final `raise ValueError`.

- [ ] **Step 4: Run heuristic tests to verify GREEN**

Run:

```bash
python3 -m pytest tests/test_heuristics.py -v
```

Expected: PASS for all tests in `tests/test_heuristics.py`.

- [ ] **Step 5: Run full suite for task-level safety**

Run:

```bash
python3 -m pytest tests -v
```

Expected: PASS for all tests.

- [ ] **Step 6: Commit Task 1**

Run:

```bash
git add src/heuristics.py tests/test_heuristics.py
git diff --cached --check
git commit -m "feat: add inter-route best relocate heuristic"
```

---

### Task 2: Official Solver Method Exposure

**Files:**
- Modify: `solve.py`
- Test: `tests/test_solve_cli.py`

**Interfaces:**
- Consumes:
  - `SOLVER_METHODS` from `src.heuristics`, now including `"nearest_2opt_relocate_best"`
  - `solve_with_method(instance: CVRPInstance, method: str = "nearest_2opt", capacity_tol: float = 1e-9, improvement_tol: float = 1e-12) -> tuple[tuple[int, ...], ...]`
  - `solve_instances(instances: Sequence[CVRPInstance], method: str = "nearest_2opt") -> list[SolutionRecord]`
- Produces:
  - `solve.py --method nearest_2opt_relocate_best`
  - Default `solve.py` method remains `nearest_2opt`

- [ ] **Step 1: Write failing solver CLI test for the new method**

Modify imports in `tests/test_solve_cli.py` so they include:

```python
from src.vrp_eval import validate_solution  # noqa: E402
from src.vrp_io import CVRPInstance, load_instances  # noqa: E402
```

Append this test:

```python
def test_solve_cli_accepts_nearest_2opt_relocate_best_method(tmp_path):
    input_path = tmp_path / "check.pkl"
    output_path = tmp_path / "predictions_relocate.json"
    raw_instances = [
        ([[0, 0]], [[1, 0], [10, 0], [11, 0]], [1, 1, 1], 3),
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
            "--method",
            "nearest_2opt_relocate_best",
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
    instance = load_instances(input_path)[0]
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    routes = tuple(tuple(route) for route in payload["solutions"][0]["routes"])
    assert payload["format_version"] == "cvrp_v1"
    assert payload["solutions"][0]["instance_id"] == 0
    assert validate_solution(instance, routes).is_feasible is True


def test_solve_cli_help_describes_relocate_method():
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "solve.py"),
            "--help",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert "inter-route best relocate" in result.stdout
```

- [ ] **Step 2: Run solver CLI tests to verify RED**

Run:

```bash
python3 -m pytest tests/test_solve_cli.py -v
```

Expected: FAIL because the current `solve.py --help` text does not contain `inter-route best relocate`.

- [ ] **Step 3: Update solver help text**

Modify `solve.py` method argument help:

```python
    parser.add_argument(
        "--method",
        choices=SOLVER_METHODS,
        default="nearest_2opt",
        help=(
            "Solver method. nearest_2opt is the stable default; "
            "nearest_2opt_relocate_best adds inter-route best relocate."
        ),
    )
```

- [ ] **Step 4: Run solver CLI tests to verify GREEN**

Run:

```bash
python3 -m pytest tests/test_solve_cli.py -v
```

Expected: PASS for all tests in `tests/test_solve_cli.py`.

- [ ] **Step 5: Run full suite for task-level safety**

Run:

```bash
python3 -m pytest tests -v
```

Expected: PASS for all tests.

- [ ] **Step 6: Commit Task 2**

Run:

```bash
git add solve.py tests/test_solve_cli.py
git diff --cached --check
git commit -m "feat: expose relocate solver method"
```

---

### Task 3: Evaluation Method Exposure

**Files:**
- Modify: `scripts/evaluate_baseline.py`
- Test: `tests/test_evaluate_baseline_cli.py`

**Interfaces:**
- Consumes:
  - `SOLVER_METHODS` from `src.heuristics`, including `"nearest_2opt_relocate_best"`
  - `evaluate_instances(instances: Sequence[CVRPInstance], limit: int | None = None, method: str = "nearest_2opt") -> EvaluationSummary`
- Produces:
  - `scripts/evaluate_baseline.py --method nearest_2opt_relocate_best`
  - Default evaluation method remains `nearest_2opt`

- [ ] **Step 1: Write failing evaluation tests for the new method**

Append these tests to `tests/test_evaluate_baseline_cli.py`:

```python
def test_evaluate_instances_accepts_nearest_2opt_relocate_best_method():
    instance = CVRPInstance(
        instance_id=0,
        depot=(0.0, 0.0),
        loc=((1.0, 0.0), (10.0, 0.0), (11.0, 0.0)),
        demand=(1.0, 1.0, 1.0),
        capacity=3.0,
        reference_cost=22.0,
    )

    summary = evaluate_instances(
        [instance],
        method="nearest_2opt_relocate_best",
    )

    assert summary.instance_count == 1
    assert summary.feasible_count == 1
    assert summary.feasibility_rate == 1.0
    assert summary.average_cost > 0.0


def test_evaluate_baseline_cli_accepts_nearest_2opt_relocate_best(tmp_path):
    input_path = tmp_path / "validation.pkl"
    raw_instances = [
        ([[0, 0]], [[1, 0], [10, 0], [11, 0]], [1, 1, 1], 3, [[1, 2, 3]], 22.0),
    ]
    with input_path.open("wb") as handle:
        pickle.dump(raw_instances, handle)

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "evaluate_baseline.py"),
            "--input",
            str(input_path),
            "--method",
            "nearest_2opt_relocate_best",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["instance_count"] == 1
    assert payload["feasible_count"] == 1
    assert payload["feasibility_rate"] == 1.0


def test_evaluate_baseline_cli_help_describes_relocate_method():
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "evaluate_baseline.py"),
            "--help",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert "inter-route best relocate" in result.stdout
```

- [ ] **Step 2: Run evaluation tests to verify RED**

Run:

```bash
python3 -m pytest tests/test_evaluate_baseline_cli.py -v
```

Expected: FAIL because the current evaluation script `--help` text does not contain `inter-route best relocate`.

- [ ] **Step 3: Update evaluation script help text**

Modify `scripts/evaluate_baseline.py` method argument help:

```python
    parser.add_argument(
        "--method",
        choices=SOLVER_METHODS,
        default="nearest_2opt",
        help=(
            "Solver method to evaluate. Use nearest_2opt_relocate_best "
            "to evaluate inter-route best relocate."
        ),
    )
```

- [ ] **Step 4: Run evaluation tests to verify GREEN**

Run:

```bash
python3 -m pytest tests/test_evaluate_baseline_cli.py -v
```

Expected: PASS for all tests in `tests/test_evaluate_baseline_cli.py`.

- [ ] **Step 5: Run full suite for task-level safety**

Run:

```bash
python3 -m pytest tests -v
```

Expected: PASS for all tests.

- [ ] **Step 6: Commit Task 3**

Run:

```bash
git add scripts/evaluate_baseline.py tests/test_evaluate_baseline_cli.py
git diff --cached --check
git commit -m "feat: expose relocate evaluation method"
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
  - `solve.py --method nearest_2opt_relocate_best`
  - `scripts/evaluate_baseline.py --method nearest_2opt_relocate_best`
  - Existing `nearest_2opt` metric `13.410121525149266`
- Produces:
  - Updated user-facing run commands
  - Recorded validation result for `nearest_2opt_relocate_best`
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
python3 solve.py --input VRP_project/VRPData/check_data_to_students.pkl --output outputs/predictions_relocate.json --method nearest_2opt_relocate_best --device cuda:0 --seed 2026
```

Expected: exit code 0 and `outputs/predictions_relocate.json` is written. The output file remains ignored by git.

- [ ] **Step 3: Run full validation evaluation for relocate method**

Run:

```bash
python3 scripts/evaluate_baseline.py --input VRP_project/VRPData/validation_data.pkl --method nearest_2opt_relocate_best
```

Expected: JSON output where `instance_count` is `1000`, `feasible_count` is `1000`, `feasibility_rate` is `1.0`, and `average_cost` is less than or equal to `13.410121525149266` within normal floating-point tolerance.

- [ ] **Step 4: Run current default evaluation only if comparison looks suspicious**

Run this command only if Step 3 reports a surprising regression or timing issue:

```bash
python3 scripts/evaluate_baseline.py --input VRP_project/VRPData/validation_data.pkl --method nearest_2opt
```

Expected: JSON output comparable to the recorded 2026-07-28 route-inner 2-opt baseline.

- [ ] **Step 5: Update README with relocate method commands**

Modify `README.md` so the solver section states that `nearest_2opt` remains the default and `nearest_2opt_relocate_best` is an optional stronger CPU heuristic.

Add this command:

```bash
python3 solve.py --input VRP_project/VRPData/check_data_to_students.pkl --output outputs/predictions_relocate.json --method nearest_2opt_relocate_best --device cuda:0 --seed 2026
```

Modify the evaluation section so it includes:

```bash
python3 scripts/evaluate_baseline.py --input VRP_project/VRPData/validation_data.pkl --method nearest_2opt_relocate_best
```

Modify the roadmap so the next step mentions deciding whether to make `nearest_2opt_relocate_best` the default or designing swap.

- [ ] **Step 6: Record validation results**

Append a new section to `docs/experiments/baseline_results.md` with heading:

```markdown
## 2026-07-28: Nearest Neighbor + Route-Inner 2-opt + Inter-Route Best Relocate
```

The section must include the exact command from Step 3, the printed `instance_count`, `feasible_count`, `feasibility_rate`, `average_cost`, `average_gap`, and `average_inference_time`, plus a note comparing `average_cost` against the previous `nearest_2opt` average cost `13.410121525149266`.

- [ ] **Step 7: Record daily progress**

Update the 2026-07-28 section in `docs/progress/daily-progress.md` with concise bullets:

```markdown
- 审核通过 route 间 best relocate 设计，并生成 implementation plan。
- 实现可选方法 `nearest_2opt_relocate_best`，默认方法仍保持 `nearest_2opt`。
- 完成测试、官方 check 数据 smoke test 和完整验证集评估。
```

Update the 2026-07-28 next-step bullets to mention:

```markdown
- 根据 relocate 收益决定是否升级默认方法，或继续设计 route 间 swap。
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
git commit -m "docs: record inter-route best relocate results"
```

---

## Plan Self-Review

- Spec coverage: Task 1 covers best relocate behavior, capacity checks, empty route deletion, method registration, and search budget. Task 2 covers official solver exposure while preserving the default. Task 3 covers evaluation exposure. Task 4 covers README, experiment records, progress notes, full tests, smoke test, and validation metrics.
- Marker scan: no unresolved marker words, incomplete code slots, or unspecified function names are intentionally left in this plan.
- Type consistency: Task 1 produces `improve_routes_relocate_best` and `solve_nearest_neighbor_2opt_relocate_best`; Tasks 2 and 3 consume the method through `SOLVER_METHODS` and `solve_with_method`.
- Scope check: route 间 swap、candidate-limited relocate、default method upgrade、OR-Tools and neural training are outside this implementation plan.

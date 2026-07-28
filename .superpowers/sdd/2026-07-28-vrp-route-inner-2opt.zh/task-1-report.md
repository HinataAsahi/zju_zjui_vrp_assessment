# Task 1 Report: Core Route-Inner 2-opt Heuristic

## What I implemented

- Added `SolverMethod` and `SOLVER_METHODS` with the `nearest` and `nearest_2opt` methods.
- Added route-level first-improvement 2-opt with the required improvement tolerance.
- Added route-collection 2-opt post-processing that preserves route boundaries.
- Added `solve_nearest_neighbor_2opt` and `solve_with_method` dispatch.
- Preserved the existing nearest-neighbor implementation and behavior.
- Added the specified crossing-route, local-optimum, route-boundary, feasibility, and unknown-method tests.

## What I tested and test results

- Focused command: `python3 -m pytest tests/test_heuristics.py -v`
  - Result: `8 passed in 0.02s`
- Full command: `python3 -m pytest -q`
  - Result: `43 passed in 0.24s`
- Diff hygiene: `git diff --check`
  - Result: passed with no output.

## TDD Evidence

### RED

Command:

```text
python3 -m pytest tests/test_heuristics.py -v
```

Observed result:

```text
collected 0 items / 1 error
ImportError: cannot import name 'improve_route_2opt' from 'src.heuristics'
1 error
```

The failure occurred during test collection because the new production interfaces did not yet exist.

### GREEN

Command:

```text
python3 -m pytest tests/test_heuristics.py -v
```

Observed result:

```text
collected 8 items
8 passed in 0.02s
```

## Files changed

- `src/heuristics.py`
- `tests/test_heuristics.py`
- `.superpowers/sdd/2026-07-28-vrp-route-inner-2opt.zh/task-1-report.md`

## Self-review findings

- No issues found.
- The 2-opt implementation only reverses customer subsequences within each route and accepts changes only when the edge delta exceeds `improvement_tol`.
- Existing nearest-neighbor tests remain passing.
- The full repository test suite remains passing.

## Issues or concerns

None.

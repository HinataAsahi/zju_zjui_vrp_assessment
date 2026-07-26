# Task 1 Report: Data Model, Pickle I/O, and JSON Writer

## What I implemented

- Added the `src` package and `src.vrp_io` module.
- Added immutable `CVRPInstance` and `SolutionRecord` data models.
- Added normalization for labeled 6-tuples and unlabeled 4-tuples.
- Converted coordinates, demands, and capacity to float-based internal values.
- Added length validation for locations and demands.
- Added pickle list loading with zero-based instance IDs.
- Added JSON output with parent-directory creation and exact `cvrp_v1` format version.
- Added the seven required tests from the task brief.

## TDD RED command/output and why expected

Command:

```bash
python3 -m pytest tests/test_vrp_io.py -v
```

Result:

```text
collected 0 items / 1 error
ModuleNotFoundError: No module named 'src'
EXIT_CODE=2
```

This was expected because the tests were written before the required `src` package and `src.vrp_io` module existed.

## GREEN command/output

Command:

```bash
python3 -m pytest tests/test_vrp_io.py -v
```

Result:

```text
collected 7 items
7 passed in 0.03s
EXIT_CODE=0
```

Accumulated verification:

```bash
python3 -m pytest -v
```

Result: `11 passed in 0.04s`.

`git diff --check` also completed successfully.

## Files changed

- `src/__init__.py`
- `src/vrp_io.py`
- `tests/test_vrp_io.py`
- `.superpowers/sdd/2026-07-26-vrp-baseline-solver.zh/task-1-report.md`

## Self-review notes

- Output routes are serialized as provided and do not add the implicit depot.
- The module has no GPU, CUDA, neural-model, 2-opt, or OR-Tools dependency.
- The implementation is limited to the Task 1 interfaces and behaviors.
- No raw project data or generated experiment output was added.

## Concerns

None for the scope of Task 1.

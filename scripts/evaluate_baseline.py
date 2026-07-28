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

from src.heuristics import SOLVER_METHODS, solve_with_method  # noqa: E402
from src.vrp_eval import EvaluationSummary, summarize_results  # noqa: E402
from src.vrp_io import CVRPInstance, load_instances  # noqa: E402


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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate nearest neighbor baseline.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--input", required=True, help="Input labeled .pkl file path.")
    parser.add_argument("--limit", type=int, default=None, help="Optional instance limit.")
    parser.add_argument(
        "--method",
        choices=SOLVER_METHODS,
        default="nearest_2opt",
        help=(
            "Solver method to evaluate. Use nearest_2opt_relocate_best "
            "to evaluate inter-route best relocate."
        ),
    )
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 0:
        parser.error("limit must be non-negative")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = evaluate_instances(
        load_instances(args.input),
        limit=args.limit,
        method=args.method,
    )
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

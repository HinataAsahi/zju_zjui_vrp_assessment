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
    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative")
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
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 0:
        parser.error("limit must be non-negative")
    return args


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

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

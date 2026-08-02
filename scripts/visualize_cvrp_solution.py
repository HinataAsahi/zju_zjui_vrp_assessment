"""Visualize one CVRP solution as a report-ready PNG."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.vrp_eval import validate_solution
from src.vrp_io import CVRPInstance, Route, load_instances


def _load_solution_routes(path: Path, instance_id: int) -> tuple[Route, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format_version") != "cvrp_v1":
        raise ValueError("solutions JSON format_version should be cvrp_v1")
    solutions = payload.get("solutions")
    if not isinstance(solutions, list):
        raise ValueError("solutions JSON should contain a solutions list")

    for solution in solutions:
        if solution.get("instance_id") != instance_id:
            continue
        routes = solution.get("routes")
        if not isinstance(routes, list):
            raise ValueError(f"instance_id {instance_id} routes should be a list")
        return tuple(tuple(int(customer) for customer in route) for route in routes)

    raise ValueError(f"instance_id {instance_id} not found in solutions JSON")


def _route_points(
    instance: CVRPInstance,
    route: Sequence[int],
) -> tuple[tuple[float, float], ...]:
    customer_points = tuple(instance.loc[customer_id - 1] for customer_id in route)
    return (instance.depot, *customer_points, instance.depot)


def plot_solution(
    instance: CVRPInstance,
    routes: Sequence[Sequence[int]],
    output_path: Path,
    title: str,
) -> None:
    os.environ.setdefault(
        "MPLCONFIGDIR",
        str(Path(tempfile.gettempdir()) / "vrp_matplotlib"),
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    validation = validate_solution(instance, routes)
    if not validation.is_feasible:
        raise ValueError(
            f"instance {instance.instance_id} infeasible: "
            + ", ".join(validation.errors)
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 7), dpi=160)
    colors = plt.get_cmap("tab20")

    customer_x = [point[0] for point in instance.loc]
    customer_y = [point[1] for point in instance.loc]
    ax.scatter(
        customer_x,
        customer_y,
        s=28,
        c="#2f6f9f",
        alpha=0.85,
        linewidths=0,
        label="Customers",
    )
    ax.scatter(
        [instance.depot[0]],
        [instance.depot[1]],
        marker="*",
        s=220,
        c="#d62828",
        edgecolors="#7a1111",
        linewidths=0.8,
        label="Depot",
        zorder=5,
    )

    for route_index, route in enumerate(routes):
        points = _route_points(instance, route)
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        ax.plot(
            xs,
            ys,
            color=colors(route_index % 20),
            linewidth=1.35,
            alpha=0.86,
        )
        if route:
            label_customer = route[len(route) // 2]
            label_point = instance.loc[label_customer - 1]
            load = validation.route_loads[route_index]
            ax.text(
                label_point[0],
                label_point[1],
                f"R{route_index + 1}\n{load:.0f}/{instance.capacity:.0f}",
                fontsize=6.5,
                ha="center",
                va="center",
                bbox={
                    "boxstyle": "round,pad=0.18",
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.72,
                },
            )

    subtitle = (
        f"instance={instance.instance_id} | customers={instance.customer_count} | "
        f"routes={len(routes)} | cost={validation.total_cost:.4f}"
    )
    ax.set_title(f"{title}\n{subtitle}", fontsize=11)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, linewidth=0.4, alpha=0.35)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize one CVRP solution.")
    parser.add_argument("--input", required=True, help="Input .pkl file path.")
    parser.add_argument("--solutions", required=True, help="CVRP v1 solution JSON.")
    parser.add_argument("--instance-id", type=int, required=True, help="Instance ID.")
    parser.add_argument("--output", required=True, help="Output PNG file path.")
    parser.add_argument(
        "--title",
        default="CVRP solution",
        help="Figure title.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        instances = load_instances(args.input)
        if args.instance_id < 0 or args.instance_id >= len(instances):
            raise ValueError(f"instance_id {args.instance_id} not found in input")
        routes = _load_solution_routes(Path(args.solutions), args.instance_id)
        plot_solution(
            instances[args.instance_id],
            routes,
            Path(args.output),
            args.title,
        )
    except Exception as exc:
        print(f"visualize_cvrp_solution.py failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

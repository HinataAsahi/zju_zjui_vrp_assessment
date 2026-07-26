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

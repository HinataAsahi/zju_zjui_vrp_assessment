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
    if len(instances) != len(routes_by_instance):
        raise ValueError(
            "instances and routes_by_instance must have the same length"
        )
    if inference_times is not None and len(inference_times) != len(instances):
        raise ValueError("inference_times must have the same length as instances")

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

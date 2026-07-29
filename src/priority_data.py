"""Priority-label and route-construction utilities for supervised imitation."""

from __future__ import annotations

from typing import Sequence

from src.vrp_eval import euclidean
from src.vrp_io import CVRPInstance


def reference_priority_order(instance: CVRPInstance) -> tuple[int, ...]:
    if instance.reference_routes is None:
        raise ValueError(f"instance {instance.instance_id} has no reference routes")

    order = tuple(customer for route in instance.reference_routes for customer in route)
    expected = tuple(range(1, instance.customer_count + 1))
    if tuple(sorted(order)) != expected:
        raise ValueError(
            f"instance {instance.instance_id} reference routes must cover each "
            "customer exactly once"
        )
    return order


def normalized_rank_labels(instance: CVRPInstance) -> tuple[float, ...]:
    order = reference_priority_order(instance)
    denominator = max(instance.customer_count - 1, 1)
    labels = [0.0] * instance.customer_count
    for rank, customer_id in enumerate(order):
        labels[customer_id - 1] = rank / denominator
    return tuple(labels)


def customer_feature_rows(instance: CVRPInstance) -> tuple[tuple[float, ...], ...]:
    depot_x, depot_y = instance.depot
    rows: list[tuple[float, ...]] = []
    for (x, y), demand in zip(instance.loc, instance.demand):
        rows.append(
            (
                x,
                y,
                x - depot_x,
                y - depot_y,
                demand,
                demand / instance.capacity,
                euclidean((x, y), instance.depot),
            )
        )
    return tuple(rows)


def split_priority_order_by_capacity(
    instance: CVRPInstance,
    order: Sequence[int],
    capacity_tol: float = 1e-9,
) -> tuple[tuple[int, ...], ...]:
    if tuple(sorted(order)) != tuple(range(1, instance.customer_count + 1)):
        raise ValueError(
            f"instance {instance.instance_id} priority order must cover each "
            "customer exactly once"
        )

    routes: list[tuple[int, ...]] = []
    route: list[int] = []
    load = 0.0
    for customer_id in order:
        demand = instance.demand[customer_id - 1]
        if demand > instance.capacity + capacity_tol:
            raise ValueError(
                f"instance {instance.instance_id} customer {customer_id} "
                "demand exceeds capacity"
            )
        if route and load + demand > instance.capacity + capacity_tol:
            routes.append(tuple(route))
            route = []
            load = 0.0
        route.append(customer_id)
        load += demand
    if route:
        routes.append(tuple(route))
    return tuple(routes)


def routes_from_priority_scores(
    instance: CVRPInstance,
    scores: Sequence[float],
) -> tuple[tuple[int, ...], ...]:
    if len(scores) != instance.customer_count:
        raise ValueError(
            f"instance {instance.instance_id} scores length must match "
            "customer count"
        )
    order = tuple(
        customer_id
        for customer_id, _score in sorted(
            enumerate(scores, start=1),
            key=lambda item: (item[1], item[0]),
        )
    )
    return split_priority_order_by_capacity(instance, order)

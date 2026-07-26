"""Heuristic CVRP solvers."""

from __future__ import annotations

from src.vrp_eval import euclidean
from src.vrp_io import CVRPInstance


def solve_nearest_neighbor(
    instance: CVRPInstance,
    capacity_tol: float = 1e-9,
) -> tuple[tuple[int, ...], ...]:
    for customer_index, demand in enumerate(instance.demand):
        if demand > instance.capacity + capacity_tol:
            raise ValueError(
                f"customer {customer_index + 1} demand exceeds capacity: "
                f"{demand}>{instance.capacity}"
            )

    unvisited = set(range(instance.customer_count))
    routes: list[tuple[int, ...]] = []

    while unvisited:
        route: list[int] = []
        load = 0.0
        current_point = instance.depot

        while True:
            candidates = [
                customer_index
                for customer_index in unvisited
                if load + instance.demand[customer_index]
                <= instance.capacity + capacity_tol
            ]
            if not candidates:
                break

            next_customer = min(
                candidates,
                key=lambda customer_index: (
                    euclidean(current_point, instance.loc[customer_index]),
                    customer_index,
                ),
            )
            unvisited.remove(next_customer)
            route.append(next_customer + 1)
            load += instance.demand[next_customer]
            current_point = instance.loc[next_customer]

        if not route:
            raise RuntimeError("nearest neighbor made no progress")
        routes.append(tuple(route))

    return tuple(routes)

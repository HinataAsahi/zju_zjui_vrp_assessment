"""Heuristic CVRP solvers."""

from __future__ import annotations

from typing import Literal, Sequence

from src.vrp_eval import euclidean
from src.vrp_io import CVRPInstance


SolverMethod = Literal["nearest", "nearest_2opt"]
SOLVER_METHODS: tuple[SolverMethod, ...] = ("nearest", "nearest_2opt")


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


def _route_node_point(
    instance: CVRPInstance,
    customer_id: int | None,
) -> tuple[float, float]:
    if customer_id is None:
        return instance.depot
    if customer_id < 1 or customer_id > instance.customer_count:
        raise ValueError("route customer IDs should be positive 1-based integers")
    return instance.loc[customer_id - 1]


def improve_route_2opt(
    instance: CVRPInstance,
    route: Sequence[int],
    improvement_tol: float = 1e-12,
) -> tuple[int, ...]:
    best_route = tuple(route)
    if len(best_route) < 3:
        return best_route

    while True:
        route_len = len(best_route)
        improved = False
        for i in range(route_len - 1):
            for j in range(i + 1, route_len):
                prev_customer = None if i == 0 else best_route[i - 1]
                next_customer = None if j == route_len - 1 else best_route[j + 1]
                prev_point = _route_node_point(instance, prev_customer)
                first_point = _route_node_point(instance, best_route[i])
                last_point = _route_node_point(instance, best_route[j])
                next_point = _route_node_point(instance, next_customer)

                current_edges = euclidean(prev_point, first_point) + euclidean(
                    last_point,
                    next_point,
                )
                candidate_edges = euclidean(prev_point, last_point) + euclidean(
                    first_point,
                    next_point,
                )
                if candidate_edges + improvement_tol < current_edges:
                    best_route = (
                        best_route[:i]
                        + tuple(reversed(best_route[i : j + 1]))
                        + best_route[j + 1 :]
                    )
                    improved = True
                    break
            if improved:
                break
        if not improved:
            return best_route


def improve_routes_2opt(
    instance: CVRPInstance,
    routes: Sequence[Sequence[int]],
    improvement_tol: float = 1e-12,
) -> tuple[tuple[int, ...], ...]:
    return tuple(
        improve_route_2opt(instance, route, improvement_tol=improvement_tol)
        for route in routes
    )


def solve_nearest_neighbor_2opt(
    instance: CVRPInstance,
    capacity_tol: float = 1e-9,
    improvement_tol: float = 1e-12,
) -> tuple[tuple[int, ...], ...]:
    routes = solve_nearest_neighbor(instance, capacity_tol=capacity_tol)
    return improve_routes_2opt(
        instance,
        routes,
        improvement_tol=improvement_tol,
    )


def solve_with_method(
    instance: CVRPInstance,
    method: str = "nearest_2opt",
    capacity_tol: float = 1e-9,
    improvement_tol: float = 1e-12,
) -> tuple[tuple[int, ...], ...]:
    if method == "nearest":
        return solve_nearest_neighbor(instance, capacity_tol=capacity_tol)
    if method == "nearest_2opt":
        return solve_nearest_neighbor_2opt(
            instance,
            capacity_tol=capacity_tol,
            improvement_tol=improvement_tol,
        )
    raise ValueError(f"unknown solver method: {method}")

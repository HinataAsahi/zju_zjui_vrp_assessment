"""Heuristic CVRP solvers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

from src.vrp_eval import compute_route_cost, compute_total_cost, euclidean
from src.vrp_io import CVRPInstance


SolverMethod = Literal[
    "nearest",
    "nearest_2opt",
    "nearest_2opt_relocate_best",
    "nearest_2opt_relocate_limited",
    "nearest_2opt_relocate_candidate_limited",
    "nearest_2opt_relocate_limited_swap",
]
SOLVER_METHODS: tuple[SolverMethod, ...] = (
    "nearest",
    "nearest_2opt",
    "nearest_2opt_relocate_best",
    "nearest_2opt_relocate_limited",
    "nearest_2opt_relocate_candidate_limited",
    "nearest_2opt_relocate_limited_swap",
)


@dataclass(frozen=True)
class _RelocateMove:
    cost_delta: float
    source_route_index: int
    target_route_index: int
    source_customer_position: int
    target_insert_position: int
    customer_id: int


@dataclass(frozen=True)
class _SwapMove:
    cost_delta: float
    first_route_index: int
    second_route_index: int
    first_customer_position: int
    second_customer_position: int
    first_customer_id: int
    second_customer_id: int


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


def _route_load(instance: CVRPInstance, route: Sequence[int]) -> float:
    return sum(instance.demand[customer_id - 1] for customer_id in route)


def _find_best_relocate_move(
    instance: CVRPInstance,
    routes: tuple[tuple[int, ...], ...],
    route_loads: tuple[float, ...],
    capacity_tol: float,
    improvement_tol: float,
    max_candidate_routes: int | None = None,
) -> _RelocateMove | None:
    best_move: _RelocateMove | None = None

    for source_route_index, source_route in enumerate(routes):
        source_cost = compute_route_cost(instance, source_route)
        for source_customer_position, customer_id in enumerate(source_route):
            customer_demand = instance.demand[customer_id - 1]
            new_source_route = (
                source_route[:source_customer_position]
                + source_route[source_customer_position + 1 :]
            )
            new_source_cost = compute_route_cost(instance, new_source_route)
            target_route_indexes = _candidate_target_route_indexes(
                instance,
                routes,
                route_loads,
                source_route_index=source_route_index,
                customer_id=customer_id,
                customer_demand=customer_demand,
                capacity_tol=capacity_tol,
                max_candidate_routes=max_candidate_routes,
            )

            for target_route_index in target_route_indexes:
                target_route = routes[target_route_index]

                target_cost = compute_route_cost(instance, target_route)
                current_cost = source_cost + target_cost
                for target_insert_position in range(len(target_route) + 1):
                    new_target_route = (
                        target_route[:target_insert_position]
                        + (customer_id,)
                        + target_route[target_insert_position:]
                    )
                    candidate_cost = new_source_cost + compute_route_cost(
                        instance,
                        new_target_route,
                    )
                    if candidate_cost + improvement_tol >= current_cost:
                        continue

                    move = _RelocateMove(
                        cost_delta=candidate_cost - current_cost,
                        source_route_index=source_route_index,
                        target_route_index=target_route_index,
                        source_customer_position=source_customer_position,
                        target_insert_position=target_insert_position,
                        customer_id=customer_id,
                    )
                    if best_move is None or (
                        move.cost_delta,
                        move.source_route_index,
                        move.target_route_index,
                        move.source_customer_position,
                        move.target_insert_position,
                        move.customer_id,
                    ) < (
                        best_move.cost_delta,
                        best_move.source_route_index,
                        best_move.target_route_index,
                        best_move.source_customer_position,
                        best_move.target_insert_position,
                        best_move.customer_id,
                    ):
                        best_move = move

    return best_move


def _candidate_target_route_indexes(
    instance: CVRPInstance,
    routes: tuple[tuple[int, ...], ...],
    route_loads: tuple[float, ...],
    source_route_index: int,
    customer_id: int,
    customer_demand: float,
    capacity_tol: float,
    max_candidate_routes: int | None,
) -> tuple[int, ...]:
    feasible_target_indexes = [
        target_route_index
        for target_route_index in range(len(routes))
        if target_route_index != source_route_index
        and route_loads[target_route_index] + customer_demand
        <= instance.capacity + capacity_tol
    ]
    if max_candidate_routes is None:
        return tuple(feasible_target_indexes)
    if max_candidate_routes == 0:
        return ()

    customer_point = instance.loc[customer_id - 1]
    ranked_targets = sorted(
        feasible_target_indexes,
        key=lambda target_route_index: (
            _route_distance_to_customer_point(
                instance,
                routes[target_route_index],
                customer_point,
            ),
            target_route_index,
        ),
    )
    return tuple(ranked_targets[:max_candidate_routes])


def _route_distance_to_customer_point(
    instance: CVRPInstance,
    route: Sequence[int],
    customer_point: tuple[float, float],
) -> float:
    if not route:
        return euclidean(customer_point, instance.depot)
    return min(
        euclidean(customer_point, _route_node_point(instance, target_customer_id))
        for target_customer_id in route
    )


def _apply_relocate_move(
    instance: CVRPInstance,
    routes: tuple[tuple[int, ...], ...],
    move: _RelocateMove,
    improvement_tol: float,
) -> tuple[tuple[int, ...], ...]:
    mutable_routes = [list(route) for route in routes]
    customer_id = mutable_routes[move.source_route_index].pop(
        move.source_customer_position
    )
    if customer_id != move.customer_id:
        raise RuntimeError("relocate move customer mismatch")

    mutable_routes[move.target_route_index].insert(
        move.target_insert_position,
        customer_id,
    )

    affected_route_indexes = {move.source_route_index, move.target_route_index}
    improved_routes: list[tuple[int, ...]] = []
    for route_index, route in enumerate(mutable_routes):
        if not route:
            continue
        route_tuple = tuple(route)
        if route_index in affected_route_indexes:
            route_tuple = improve_route_2opt(
                instance,
                route_tuple,
                improvement_tol=improvement_tol,
            )
        improved_routes.append(route_tuple)
    return tuple(improved_routes)


def improve_routes_relocate_best(
    instance: CVRPInstance,
    routes: Sequence[Sequence[int]],
    capacity_tol: float = 1e-9,
    improvement_tol: float = 1e-12,
    max_passes: int = 50,
    max_candidate_routes: int | None = None,
) -> tuple[tuple[int, ...], ...]:
    if max_passes < 0:
        raise ValueError("max_passes must be non-negative")
    if max_candidate_routes is not None and max_candidate_routes < 0:
        raise ValueError("max_candidate_routes must be non-negative")

    best_routes = tuple(tuple(route) for route in routes)
    for _ in range(max_passes):
        route_loads = tuple(_route_load(instance, route) for route in best_routes)
        move = _find_best_relocate_move(
            instance,
            best_routes,
            route_loads,
            capacity_tol=capacity_tol,
            improvement_tol=improvement_tol,
            max_candidate_routes=max_candidate_routes,
        )
        if move is None:
            return best_routes

        current_total_cost = compute_total_cost(instance, best_routes)
        candidate_routes = _apply_relocate_move(
            instance,
            best_routes,
            move,
            improvement_tol=improvement_tol,
        )
        if (
            compute_total_cost(instance, candidate_routes) + improvement_tol
            >= current_total_cost
        ):
            return best_routes
        best_routes = candidate_routes

    return best_routes


def _find_best_swap_move(
    instance: CVRPInstance,
    routes: tuple[tuple[int, ...], ...],
    route_loads: tuple[float, ...],
    capacity_tol: float,
    improvement_tol: float,
) -> _SwapMove | None:
    best_move: _SwapMove | None = None

    for first_route_index, first_route in enumerate(routes[:-1]):
        first_route_cost = compute_route_cost(instance, first_route)
        for second_route_index in range(first_route_index + 1, len(routes)):
            second_route = routes[second_route_index]
            second_route_cost = compute_route_cost(instance, second_route)
            current_cost = first_route_cost + second_route_cost

            for first_customer_position, first_customer_id in enumerate(first_route):
                first_customer_demand = instance.demand[first_customer_id - 1]
                for second_customer_position, second_customer_id in enumerate(second_route):
                    second_customer_demand = instance.demand[second_customer_id - 1]
                    new_first_load = (
                        route_loads[first_route_index]
                        - first_customer_demand
                        + second_customer_demand
                    )
                    new_second_load = (
                        route_loads[second_route_index]
                        - second_customer_demand
                        + first_customer_demand
                    )
                    if (
                        new_first_load > instance.capacity + capacity_tol
                        or new_second_load > instance.capacity + capacity_tol
                    ):
                        continue

                    new_first_route = (
                        first_route[:first_customer_position]
                        + (second_customer_id,)
                        + first_route[first_customer_position + 1 :]
                    )
                    new_second_route = (
                        second_route[:second_customer_position]
                        + (first_customer_id,)
                        + second_route[second_customer_position + 1 :]
                    )
                    candidate_cost = compute_route_cost(
                        instance,
                        new_first_route,
                    ) + compute_route_cost(instance, new_second_route)
                    if candidate_cost + improvement_tol >= current_cost:
                        continue

                    move = _SwapMove(
                        cost_delta=candidate_cost - current_cost,
                        first_route_index=first_route_index,
                        second_route_index=second_route_index,
                        first_customer_position=first_customer_position,
                        second_customer_position=second_customer_position,
                        first_customer_id=first_customer_id,
                        second_customer_id=second_customer_id,
                    )
                    if best_move is None or (
                        move.cost_delta,
                        move.first_route_index,
                        move.second_route_index,
                        move.first_customer_position,
                        move.second_customer_position,
                        move.first_customer_id,
                        move.second_customer_id,
                    ) < (
                        best_move.cost_delta,
                        best_move.first_route_index,
                        best_move.second_route_index,
                        best_move.first_customer_position,
                        best_move.second_customer_position,
                        best_move.first_customer_id,
                        best_move.second_customer_id,
                    ):
                        best_move = move

    return best_move


def _apply_swap_move(
    instance: CVRPInstance,
    routes: tuple[tuple[int, ...], ...],
    move: _SwapMove,
    improvement_tol: float,
) -> tuple[tuple[int, ...], ...]:
    mutable_routes = [list(route) for route in routes]
    first_customer_id = mutable_routes[move.first_route_index][
        move.first_customer_position
    ]
    second_customer_id = mutable_routes[move.second_route_index][
        move.second_customer_position
    ]
    if (
        first_customer_id != move.first_customer_id
        or second_customer_id != move.second_customer_id
    ):
        raise RuntimeError("swap move customer mismatch")

    mutable_routes[move.first_route_index][
        move.first_customer_position
    ] = second_customer_id
    mutable_routes[move.second_route_index][
        move.second_customer_position
    ] = first_customer_id

    affected_route_indexes = {move.first_route_index, move.second_route_index}
    improved_routes: list[tuple[int, ...]] = []
    for route_index, route in enumerate(mutable_routes):
        route_tuple = tuple(route)
        if route_index in affected_route_indexes:
            route_tuple = improve_route_2opt(
                instance,
                route_tuple,
                improvement_tol=improvement_tol,
            )
        improved_routes.append(route_tuple)
    return tuple(improved_routes)


def improve_routes_swap_best(
    instance: CVRPInstance,
    routes: Sequence[Sequence[int]],
    capacity_tol: float = 1e-9,
    improvement_tol: float = 1e-12,
    max_passes: int = 50,
) -> tuple[tuple[int, ...], ...]:
    if max_passes < 0:
        raise ValueError("max_passes must be non-negative")

    best_routes = tuple(tuple(route) for route in routes)
    for _ in range(max_passes):
        route_loads = tuple(_route_load(instance, route) for route in best_routes)
        move = _find_best_swap_move(
            instance,
            best_routes,
            route_loads,
            capacity_tol=capacity_tol,
            improvement_tol=improvement_tol,
        )
        if move is None:
            return best_routes

        current_total_cost = compute_total_cost(instance, best_routes)
        candidate_routes = _apply_swap_move(
            instance,
            best_routes,
            move,
            improvement_tol=improvement_tol,
        )
        if (
            compute_total_cost(instance, candidate_routes) + improvement_tol
            >= current_total_cost
        ):
            return best_routes
        best_routes = candidate_routes

    return best_routes


def relocate_limited_passes(customer_count: int) -> int:
    if customer_count <= 50:
        return 8
    return 3


def relocate_candidate_route_limit(customer_count: int) -> int:
    if customer_count <= 50:
        return 2
    return 2


def swap_limited_passes(customer_count: int) -> int:
    if customer_count <= 50:
        return 4
    return 2


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


def solve_nearest_neighbor_2opt_relocate_best(
    instance: CVRPInstance,
    capacity_tol: float = 1e-9,
    improvement_tol: float = 1e-12,
    max_relocate_passes: int = 50,
    max_candidate_routes: int | None = None,
) -> tuple[tuple[int, ...], ...]:
    routes = solve_nearest_neighbor_2opt(
        instance,
        capacity_tol=capacity_tol,
        improvement_tol=improvement_tol,
    )
    return improve_routes_relocate_best(
        instance,
        routes,
        capacity_tol=capacity_tol,
        improvement_tol=improvement_tol,
        max_passes=max_relocate_passes,
        max_candidate_routes=max_candidate_routes,
    )


def solve_nearest_neighbor_2opt_relocate_limited(
    instance: CVRPInstance,
    capacity_tol: float = 1e-9,
    improvement_tol: float = 1e-12,
) -> tuple[tuple[int, ...], ...]:
    return solve_nearest_neighbor_2opt_relocate_best(
        instance,
        capacity_tol=capacity_tol,
        improvement_tol=improvement_tol,
        max_relocate_passes=relocate_limited_passes(instance.customer_count),
    )


def solve_nearest_neighbor_2opt_relocate_candidate_limited(
    instance: CVRPInstance,
    capacity_tol: float = 1e-9,
    improvement_tol: float = 1e-12,
) -> tuple[tuple[int, ...], ...]:
    routes = solve_nearest_neighbor_2opt(
        instance,
        capacity_tol=capacity_tol,
        improvement_tol=improvement_tol,
    )
    return improve_routes_relocate_best(
        instance,
        routes,
        capacity_tol=capacity_tol,
        improvement_tol=improvement_tol,
        max_passes=relocate_limited_passes(instance.customer_count),
        max_candidate_routes=relocate_candidate_route_limit(instance.customer_count),
    )


def solve_nearest_neighbor_2opt_relocate_limited_swap(
    instance: CVRPInstance,
    capacity_tol: float = 1e-9,
    improvement_tol: float = 1e-12,
) -> tuple[tuple[int, ...], ...]:
    routes = solve_nearest_neighbor_2opt_relocate_limited(
        instance,
        capacity_tol=capacity_tol,
        improvement_tol=improvement_tol,
    )
    return improve_routes_swap_best(
        instance,
        routes,
        capacity_tol=capacity_tol,
        improvement_tol=improvement_tol,
        max_passes=swap_limited_passes(instance.customer_count),
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
    if method == "nearest_2opt_relocate_best":
        return solve_nearest_neighbor_2opt_relocate_best(
            instance,
            capacity_tol=capacity_tol,
            improvement_tol=improvement_tol,
        )
    if method == "nearest_2opt_relocate_limited":
        return solve_nearest_neighbor_2opt_relocate_limited(
            instance,
            capacity_tol=capacity_tol,
            improvement_tol=improvement_tol,
        )
    if method == "nearest_2opt_relocate_candidate_limited":
        return solve_nearest_neighbor_2opt_relocate_candidate_limited(
            instance,
            capacity_tol=capacity_tol,
            improvement_tol=improvement_tol,
        )
    if method == "nearest_2opt_relocate_limited_swap":
        return solve_nearest_neighbor_2opt_relocate_limited_swap(
            instance,
            capacity_tol=capacity_tol,
            improvement_tol=improvement_tol,
        )
    raise ValueError(f"unknown solver method: {method}")

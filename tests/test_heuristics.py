import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.heuristics import (  # noqa: E402
    improve_route_2opt,
    improve_routes_2opt,
    improve_routes_relocate_best,
    relocate_limited_passes,
    solve_nearest_neighbor,
    solve_nearest_neighbor_2opt,
    solve_nearest_neighbor_2opt_relocate_best,
    solve_nearest_neighbor_2opt_relocate_limited,
    solve_with_method,
)
from src.vrp_eval import compute_route_cost, compute_total_cost, validate_solution  # noqa: E402
from src.vrp_io import CVRPInstance  # noqa: E402


def test_nearest_neighbor_splits_routes_by_capacity():
    instance = CVRPInstance(
        instance_id=0,
        depot=(0.0, 0.0),
        loc=((1.0, 0.0), (2.0, 0.0), (10.0, 0.0)),
        demand=(4.0, 4.0, 4.0),
        capacity=8.0,
    )

    routes = solve_nearest_neighbor(instance)

    assert routes == ((1, 2), (3,))
    assert validate_solution(instance, routes).is_feasible is True


def test_nearest_neighbor_breaks_distance_ties_by_customer_id():
    instance = CVRPInstance(
        instance_id=0,
        depot=(0.0, 0.0),
        loc=((1.0, 0.0), (-1.0, 0.0), (0.0, 2.0)),
        demand=(1.0, 1.0, 1.0),
        capacity=3.0,
    )

    routes = solve_nearest_neighbor(instance)

    assert routes[0][0] == 1
    assert validate_solution(instance, routes).is_feasible is True


def test_nearest_neighbor_rejects_customer_demand_over_capacity():
    instance = CVRPInstance(
        instance_id=0,
        depot=(0.0, 0.0),
        loc=((1.0, 0.0),),
        demand=(9.0,),
        capacity=8.0,
    )

    try:
        solve_nearest_neighbor(instance)
    except ValueError as exc:
        assert "demand exceeds capacity" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def make_crossing_route_instance() -> CVRPInstance:
    return CVRPInstance(
        instance_id=0,
        depot=(0.5, -1.0),
        loc=(
            (0.0, 0.0),
            (1.0, 1.0),
            (0.0, 1.0),
            (1.0, 0.0),
            (2.0, 0.0),
        ),
        demand=(1.0, 1.0, 1.0, 1.0, 1.0),
        capacity=5.0,
    )


def test_improve_route_2opt_reduces_crossing_route_cost():
    instance = make_crossing_route_instance()
    route = (1, 2, 3, 4)

    improved = improve_route_2opt(instance, route)

    assert len(improved) == len(route)
    assert sorted(improved) == sorted(route)
    assert compute_route_cost(instance, improved) < compute_route_cost(instance, route)


def test_improve_route_2opt_keeps_local_optimum_route_unchanged():
    instance = make_crossing_route_instance()
    route = (1, 3, 2, 4)

    improved = improve_route_2opt(instance, route)

    assert improved == route


def test_improve_routes_2opt_preserves_route_boundaries_and_customer_sets():
    instance = make_crossing_route_instance()
    routes = ((1, 2, 3, 4), (5,))

    improved = improve_routes_2opt(instance, routes)

    assert len(improved) == len(routes)
    assert [sorted(route) for route in improved] == [sorted(route) for route in routes]
    assert compute_total_cost(instance, improved) <= compute_total_cost(instance, routes) + 1e-12


def test_solve_nearest_neighbor_2opt_keeps_solution_feasible_and_not_worse():
    instance = CVRPInstance(
        instance_id=0,
        depot=(0.0, 0.0),
        loc=((1.0, 0.0), (0.0, 1.0), (1.0, 1.0), (2.0, 0.0)),
        demand=(1.0, 1.0, 1.0, 1.0),
        capacity=4.0,
    )

    nearest_routes = solve_nearest_neighbor(instance)
    improved_routes = solve_nearest_neighbor_2opt(instance)

    assert validate_solution(instance, improved_routes).is_feasible is True
    assert compute_total_cost(instance, improved_routes) <= compute_total_cost(instance, nearest_routes) + 1e-12


def test_solve_with_method_rejects_unknown_method():
    instance = make_crossing_route_instance()

    with pytest.raises(ValueError, match="unknown solver method"):
        solve_with_method(instance, method="not_a_method")


def make_relocate_instance(
    demand: tuple[float, ...] = (1.0, 1.0, 1.0),
    capacity: float = 3.0,
) -> CVRPInstance:
    return CVRPInstance(
        instance_id=0,
        depot=(0.0, 0.0),
        loc=((1.0, 0.0), (10.0, 0.0), (11.0, 0.0)),
        demand=demand,
        capacity=capacity,
    )


def test_improve_routes_relocate_best_moves_customer_between_routes_to_reduce_cost():
    instance = make_relocate_instance()
    routes = ((1, 2), (3,))

    improved = improve_routes_relocate_best(instance, routes)

    assert sorted(customer for route in improved for customer in route) == [1, 2, 3]
    assert compute_total_cost(instance, improved) < compute_total_cost(instance, routes)
    assert validate_solution(instance, improved).is_feasible is True


def test_improve_routes_relocate_best_skips_moves_that_exceed_target_capacity():
    instance = make_relocate_instance(
        demand=(1.0, 0.5, 1.5),
        capacity=1.5,
    )
    routes = ((1, 2), (3,))

    improved = improve_routes_relocate_best(instance, routes)

    assert improved == routes
    assert validate_solution(instance, improved).is_feasible is True


def test_improve_routes_relocate_best_removes_empty_source_route():
    instance = make_relocate_instance()
    routes = ((2,), (1, 3))

    improved = improve_routes_relocate_best(instance, routes)

    assert len(improved) == 1
    assert sorted(improved[0]) == [1, 2, 3]
    assert compute_total_cost(instance, improved) < compute_total_cost(instance, routes)
    assert validate_solution(instance, improved).is_feasible is True


def test_improve_routes_relocate_best_rejects_negative_max_passes():
    instance = make_relocate_instance()

    with pytest.raises(ValueError, match="max_passes must be non-negative"):
        improve_routes_relocate_best(instance, ((1, 2), (3,)), max_passes=-1)


def test_solve_nearest_neighbor_2opt_relocate_best_keeps_solution_feasible_and_not_worse():
    instance = CVRPInstance(
        instance_id=0,
        depot=(0.0, 0.0),
        loc=((1.0, 0.0), (2.0, 0.0), (10.0, 0.0), (11.0, 0.0)),
        demand=(1.0, 1.0, 1.0, 1.0),
        capacity=2.0,
    )

    base_routes = solve_nearest_neighbor_2opt(instance)
    improved_routes = solve_nearest_neighbor_2opt_relocate_best(instance)

    assert validate_solution(instance, improved_routes).is_feasible is True
    assert compute_total_cost(instance, improved_routes) <= compute_total_cost(instance, base_routes) + 1e-12


def test_solve_with_method_accepts_nearest_2opt_relocate_best():
    instance = make_relocate_instance()

    routes = solve_with_method(instance, method="nearest_2opt_relocate_best")

    assert validate_solution(instance, routes).is_feasible is True


def test_relocate_limited_passes_uses_cvrp50_budget():
    assert relocate_limited_passes(1) == 8
    assert relocate_limited_passes(50) == 8


def test_relocate_limited_passes_uses_cvrp100_budget():
    assert relocate_limited_passes(51) == 3
    assert relocate_limited_passes(100) == 3


def test_solve_nearest_neighbor_2opt_relocate_limited_keeps_solution_feasible_and_not_worse():
    instance = CVRPInstance(
        instance_id=0,
        depot=(0.0, 0.0),
        loc=((1.0, 0.0), (2.0, 0.0), (10.0, 0.0), (11.0, 0.0)),
        demand=(1.0, 1.0, 1.0, 1.0),
        capacity=2.0,
    )

    base_routes = solve_nearest_neighbor_2opt(instance)
    limited_routes = solve_nearest_neighbor_2opt_relocate_limited(instance)

    assert validate_solution(instance, limited_routes).is_feasible is True
    assert compute_total_cost(instance, limited_routes) <= compute_total_cost(instance, base_routes) + 1e-12


def test_solve_with_method_accepts_nearest_2opt_relocate_limited():
    instance = make_relocate_instance()

    routes = solve_with_method(instance, method="nearest_2opt_relocate_limited")

    assert validate_solution(instance, routes).is_feasible is True

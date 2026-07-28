import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import src.heuristics as heuristics  # noqa: E402
from src.heuristics import (  # noqa: E402
    improve_route_2opt,
    improve_routes_2opt,
    improve_routes_relocate_best,
    relocate_candidate_route_limit,
    relocate_limited_passes,
    solve_nearest_neighbor,
    solve_nearest_neighbor_2opt,
    solve_nearest_neighbor_2opt_relocate_best,
    solve_nearest_neighbor_2opt_relocate_candidate_limited,
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


def test_improve_routes_relocate_best_rejects_negative_candidate_route_limit():
    instance = make_relocate_instance()

    with pytest.raises(ValueError, match="max_candidate_routes must be non-negative"):
        improve_routes_relocate_best(
            instance,
            ((1, 2), (3,)),
            max_candidate_routes=-1,
        )


def test_improve_routes_relocate_best_skips_search_when_candidate_route_limit_is_zero():
    instance = make_relocate_instance()
    routes = ((1, 2), (3,))

    improved = improve_routes_relocate_best(
        instance,
        routes,
        max_candidate_routes=0,
    )

    assert improved == routes


def test_improve_routes_relocate_best_candidate_limit_handles_empty_target_route():
    instance = make_relocate_instance()
    routes = ((1, 2), (), (3,))

    improved = improve_routes_relocate_best(
        instance,
        routes,
        max_candidate_routes=1,
    )

    assert sorted(customer for route in improved for customer in route) == [1, 2, 3]


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


def test_relocate_candidate_route_limit_uses_cvrp50_budget():
    assert relocate_candidate_route_limit(1) == 2
    assert relocate_candidate_route_limit(50) == 2


def test_relocate_candidate_route_limit_uses_cvrp100_budget():
    assert relocate_candidate_route_limit(51) == 2
    assert relocate_candidate_route_limit(100) == 2


@pytest.mark.parametrize(
    ("customer_count", "expected_passes"),
    ((50, 8), (51, 3), (100, 3)),
)
def test_solve_nearest_neighbor_2opt_relocate_limited_forwards_budget_not_hardcoded_50(
    monkeypatch, customer_count, expected_passes
):
    """Catches a wrapper mutation that hardcodes max_relocate_passes=50."""
    instance = CVRPInstance(
        instance_id=0,
        depot=(0.0, 0.0),
        loc=tuple((float(customer_id), 0.0) for customer_id in range(1, customer_count + 1)),
        demand=(1.0,) * customer_count,
        capacity=float(customer_count),
    )
    captured: dict[str, int] = {}

    def fake_solve_nearest_neighbor_2opt_relocate_best(
        instance, capacity_tol, improvement_tol, max_relocate_passes
    ):
        captured["max_relocate_passes"] = max_relocate_passes
        return ()

    monkeypatch.setattr(
        heuristics,
        "solve_nearest_neighbor_2opt_relocate_best",
        fake_solve_nearest_neighbor_2opt_relocate_best,
    )

    solve_nearest_neighbor_2opt_relocate_limited(instance)

    assert captured["max_relocate_passes"] == expected_passes


@pytest.mark.parametrize(
    ("customer_count", "expected_passes", "expected_candidate_routes"),
    ((50, 8, 2), (51, 3, 2), (100, 3, 2)),
)
def test_solve_nearest_neighbor_2opt_relocate_candidate_limited_forwards_budgets(
    monkeypatch, customer_count, expected_passes, expected_candidate_routes
):
    instance = CVRPInstance(
        instance_id=0,
        depot=(0.0, 0.0),
        loc=tuple((float(customer_id), 0.0) for customer_id in range(1, customer_count + 1)),
        demand=(1.0,) * customer_count,
        capacity=float(customer_count),
    )
    captured: dict[str, int | None] = {}

    def fake_solve_nearest_neighbor_2opt(instance, capacity_tol, improvement_tol):
        return ((1,),)

    def fake_improve_routes_relocate_best(
        instance,
        routes,
        capacity_tol,
        improvement_tol,
        max_passes,
        max_candidate_routes,
    ):
        captured["max_passes"] = max_passes
        captured["max_candidate_routes"] = max_candidate_routes
        return tuple(tuple(route) for route in routes)

    monkeypatch.setattr(
        heuristics,
        "solve_nearest_neighbor_2opt",
        fake_solve_nearest_neighbor_2opt,
    )
    monkeypatch.setattr(
        heuristics,
        "improve_routes_relocate_best",
        fake_improve_routes_relocate_best,
    )

    routes = solve_nearest_neighbor_2opt_relocate_candidate_limited(instance)

    assert routes == ((1,),)
    assert captured["max_passes"] == expected_passes
    assert captured["max_candidate_routes"] == expected_candidate_routes


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


def test_solve_nearest_neighbor_2opt_relocate_candidate_limited_keeps_solution_feasible_and_not_worse():
    instance = CVRPInstance(
        instance_id=0,
        depot=(0.0, 0.0),
        loc=((1.0, 0.0), (2.0, 0.0), (10.0, 0.0), (11.0, 0.0)),
        demand=(1.0, 1.0, 1.0, 1.0),
        capacity=2.0,
    )

    base_routes = solve_nearest_neighbor_2opt(instance)
    candidate_routes = solve_nearest_neighbor_2opt_relocate_candidate_limited(instance)

    assert validate_solution(instance, candidate_routes).is_feasible is True
    assert compute_total_cost(instance, candidate_routes) <= compute_total_cost(instance, base_routes) + 1e-12


def test_solve_with_method_accepts_nearest_2opt_relocate_limited():
    instance = make_relocate_instance()

    routes = solve_with_method(instance, method="nearest_2opt_relocate_limited")

    assert validate_solution(instance, routes).is_feasible is True


def test_solve_with_method_accepts_nearest_2opt_relocate_candidate_limited():
    instance = make_relocate_instance()

    routes = solve_with_method(instance, method="nearest_2opt_relocate_candidate_limited")

    assert validate_solution(instance, routes).is_feasible is True

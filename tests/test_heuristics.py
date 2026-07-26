import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.heuristics import solve_nearest_neighbor  # noqa: E402
from src.vrp_eval import validate_solution  # noqa: E402
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

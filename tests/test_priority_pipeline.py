import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.priority_pipeline import (  # noqa: E402
    postprocess_priority_routes,
    solve_from_priority_scores,
)
from src.vrp_eval import compute_total_cost, validate_solution  # noqa: E402
from src.vrp_io import CVRPInstance  # noqa: E402


def make_instance() -> CVRPInstance:
    return CVRPInstance(
        instance_id=0,
        depot=(0.0, 0.0),
        loc=((0.0, 1.0), (0.0, 2.0), (10.0, 1.0), (10.0, 2.0)),
        demand=(1.0, 1.0, 1.0, 1.0),
        capacity=2.0,
    )


def test_postprocess_priority_routes_keeps_solution_feasible_and_not_worse():
    instance = make_instance()
    routes = ((1, 3), (2, 4))

    improved = postprocess_priority_routes(instance, routes)

    assert validate_solution(instance, improved).is_feasible is True
    assert compute_total_cost(instance, improved) <= compute_total_cost(instance, routes) + 1e-12


def test_solve_from_priority_scores_can_skip_postprocess():
    instance = make_instance()

    routes = solve_from_priority_scores(
        instance,
        scores=(0.0, 0.1, 0.2, 0.3),
        postprocess=False,
    )

    assert routes == ((1, 2), (3, 4))
    assert validate_solution(instance, routes).is_feasible is True


def test_solve_from_priority_scores_defaults_to_postprocess():
    instance = make_instance()

    raw_routes = solve_from_priority_scores(
        instance,
        scores=(0.0, 0.2, 0.1, 0.3),
        postprocess=False,
    )
    improved = solve_from_priority_scores(
        instance,
        scores=(0.0, 0.2, 0.1, 0.3),
    )

    assert validate_solution(instance, improved).is_feasible is True
    assert compute_total_cost(instance, improved) <= compute_total_cost(instance, raw_routes) + 1e-12

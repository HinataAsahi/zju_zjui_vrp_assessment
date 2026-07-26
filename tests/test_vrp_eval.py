import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.vrp_eval import (  # noqa: E402
    compute_gap,
    compute_route_cost,
    compute_total_cost,
    summarize_results,
    validate_solution,
)
from src.vrp_io import CVRPInstance  # noqa: E402


def make_instance() -> CVRPInstance:
    return CVRPInstance(
        instance_id=0,
        depot=(0.0, 0.0),
        loc=((3.0, 4.0), (6.0, 8.0), (0.0, 1.0)),
        demand=(3.0, 4.0, 2.0),
        capacity=7.0,
        reference_cost=20.0,
    )


def test_compute_route_and_total_cost():
    instance = make_instance()

    assert compute_route_cost(instance, [1]) == 10.0
    assert compute_total_cost(instance, [[1, 2], [3]]) == 22.0


def test_validate_solution_accepts_feasible_routes():
    result = validate_solution(make_instance(), [[1, 2], [3]])

    assert result.is_feasible is True
    assert result.errors == ()
    assert result.route_loads == (7.0, 2.0)
    assert result.visited_count == 3
    assert result.total_cost > 0.0


def test_validate_solution_detects_missing_duplicate_and_out_of_range():
    result = validate_solution(make_instance(), [[1, 1, 4]])

    assert result.is_feasible is False
    assert "duplicate_customer:1" in result.errors
    assert "out_of_range_customer:4" in result.errors
    assert "missing_customer:2" in result.errors
    assert "missing_customer:3" in result.errors


def test_validate_solution_detects_overload():
    result = validate_solution(make_instance(), [[1, 2, 3]])

    assert result.is_feasible is False
    assert "route_over_capacity:0:9.0>7.0" in result.errors


def test_validate_solution_rejects_non_integer_customer():
    result = validate_solution(make_instance(), [[1, "2"]])

    assert result.is_feasible is False
    assert "non_integer_customer:2" in result.errors


def test_compute_gap_uses_relative_reference_cost():
    assert compute_gap(22.0, 20.0) == 0.1


def test_summarize_results_reports_average_metrics():
    instance_a = make_instance()
    instance_b = CVRPInstance(
        instance_id=1,
        depot=(0.0, 0.0),
        loc=((1.0, 0.0),),
        demand=(1.0,),
        capacity=2.0,
        reference_cost=2.0,
    )

    summary = summarize_results(
        [instance_a, instance_b],
        [[[1, 2], [3]], [[1]]],
        inference_times=[0.01, 0.03],
    )

    assert summary.instance_count == 2
    assert summary.feasible_count == 2
    assert summary.feasibility_rate == 1.0
    assert summary.average_inference_time == 0.02
    assert summary.average_cost > 0.0
    assert summary.average_gap is not None


def test_summarize_results_rejects_mismatched_routes():
    with pytest.raises(ValueError, match="instances and routes_by_instance"):
        summarize_results([make_instance()], [])


def test_summarize_results_rejects_mismatched_inference_times():
    with pytest.raises(ValueError, match="inference_times"):
        summarize_results([make_instance()], [[[1, 2], [3]]], inference_times=[])

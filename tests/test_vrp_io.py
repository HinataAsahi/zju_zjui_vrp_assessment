import json
import pickle
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.vrp_io import (  # noqa: E402
    CVRPInstance,
    SolutionRecord,
    load_instances,
    normalize_instance,
    write_solutions_json,
)


def test_normalize_labeled_instance_preserves_reference_fields():
    raw = (
        [[0.0, 0.0]],
        [[1.0, 0.0], [0.0, 1.0]],
        [3, 4],
        40.0,
        [[1, 2]],
        2.0,
    )

    instance = normalize_instance(7, raw)

    assert instance == CVRPInstance(
        instance_id=7,
        depot=(0.0, 0.0),
        loc=((1.0, 0.0), (0.0, 1.0)),
        demand=(3.0, 4.0),
        capacity=40.0,
        reference_routes=((1, 2),),
        reference_cost=2.0,
    )


def test_normalize_unlabeled_instance_sets_reference_fields_to_none():
    raw = (
        [[0.5, 0.5]],
        [[1, 0], [0, 1], [1, 1]],
        [2, 5, 7],
        50,
    )

    instance = normalize_instance(0, raw)

    assert instance.reference_routes is None
    assert instance.reference_cost is None
    assert instance.capacity == 50.0
    assert instance.demand == (2.0, 5.0, 7.0)


@pytest.mark.parametrize("customer", [0, 3])
def test_normalize_labeled_instance_rejects_out_of_range_reference_customer(
    customer,
):
    raw = (
        [[0.0, 0.0]],
        [[1.0, 0.0], [0.0, 1.0]],
        [3, 4],
        40.0,
        [[customer]],
        2.0,
    )

    with pytest.raises(ValueError, match="positive 1-based integer"):
        normalize_instance(7, raw)


def test_normalize_labeled_instance_rejects_fractional_reference_customer():
    raw = (
        [[0.0, 0.0]],
        [[1.0, 0.0], [0.0, 1.0]],
        [3, 4],
        40.0,
        [[1.5]],
        2.0,
    )

    with pytest.raises(TypeError, match="positive 1-based integer"):
        normalize_instance(7, raw)


def test_load_instances_reads_pickle_list(tmp_path):
    data_path = tmp_path / "sample.pkl"
    raw_instances = [
        ([[0, 0]], [[1, 0]], [3], 40),
        ([[1, 1]], [[0, 1]], [4], 40, [[1]], 2.0),
    ]
    with data_path.open("wb") as handle:
        pickle.dump(raw_instances, handle)

    instances = load_instances(data_path)

    assert [instance.instance_id for instance in instances] == [0, 1]
    assert instances[0].reference_cost is None
    assert instances[1].reference_cost == 2.0


def test_load_instances_rejects_non_list_pickle(tmp_path):
    data_path = tmp_path / "bad.pkl"
    with data_path.open("wb") as handle:
        pickle.dump({"not": "a list"}, handle)

    try:
        load_instances(data_path)
    except TypeError as exc:
        assert "should contain a list" in str(exc)
    else:
        raise AssertionError("Expected TypeError")


def test_normalize_instance_rejects_bad_tuple_length():
    try:
        normalize_instance(0, ([[0, 0]], [[1, 0]], [3]))
    except ValueError as exc:
        assert "tuple length 4 or 6" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_normalize_instance_rejects_mismatched_loc_and_demand():
    try:
        normalize_instance(0, ([[0, 0]], [[1, 0], [0, 1]], [3], 40))
    except ValueError as exc:
        assert "loc and demand length mismatch" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_write_solutions_json_creates_parent_and_uses_cvrp_v1(tmp_path):
    output_path = tmp_path / "nested" / "predictions.json"
    solutions = [
        SolutionRecord(instance_id=0, routes=((1, 2), (3,))),
        SolutionRecord(instance_id=1, routes=((2,), (1,))),
    ]

    write_solutions_json(output_path, solutions)

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload == {
        "format_version": "cvrp_v1",
        "solutions": [
            {"instance_id": 0, "routes": [[1, 2], [3]]},
            {"instance_id": 1, "routes": [[2], [1]]},
        ],
    }


@pytest.mark.parametrize("route", [((0, 1),), ((-1, 2),)])
def test_write_solutions_json_rejects_non_positive_customer_ids(tmp_path, route):
    with pytest.raises(ValueError, match="positive 1-based integer"):
        write_solutions_json(
            tmp_path / "predictions.json",
            [SolutionRecord(instance_id=0, routes=route)],
        )


def test_write_solutions_json_rejects_non_integer_customer_ids(tmp_path):
    with pytest.raises(TypeError, match="positive 1-based integer"):
        write_solutions_json(
            tmp_path / "predictions.json",
            [SolutionRecord(instance_id=0, routes=((1.5,),))],
        )

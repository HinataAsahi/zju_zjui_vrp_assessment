import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.priority_data import (  # noqa: E402
    customer_feature_rows,
    normalized_rank_labels,
    reference_priority_order,
    routes_from_priority_scores,
    split_priority_order_by_capacity,
)
from src.vrp_io import CVRPInstance  # noqa: E402


def test_reference_priority_order_flattens_reference_routes():
    instance = CVRPInstance(
        instance_id=0,
        depot=(0.0, 0.0),
        loc=((0.0, 1.0), (0.0, 2.0), (1.0, 0.0)),
        demand=(1.0, 1.0, 1.0),
        capacity=3.0,
        reference_routes=((3, 1), (2,)),
        reference_cost=5.0,
    )

    assert reference_priority_order(instance) == (3, 1, 2)


def test_normalized_rank_labels_use_customer_id_order():
    instance = CVRPInstance(
        instance_id=0,
        depot=(0.0, 0.0),
        loc=((0.0, 1.0), (0.0, 2.0), (1.0, 0.0)),
        demand=(1.0, 1.0, 1.0),
        capacity=3.0,
        reference_routes=((3, 1), (2,)),
        reference_cost=5.0,
    )

    assert normalized_rank_labels(instance) == (0.5, 1.0, 0.0)


def test_reference_priority_order_rejects_missing_customer():
    instance = CVRPInstance(
        instance_id=7,
        depot=(0.0, 0.0),
        loc=((0.0, 1.0), (0.0, 2.0), (1.0, 0.0)),
        demand=(1.0, 1.0, 1.0),
        capacity=3.0,
        reference_routes=((3, 1),),
        reference_cost=5.0,
    )

    with pytest.raises(
        ValueError,
        match="reference routes must cover each customer exactly once",
    ):
        reference_priority_order(instance)


def test_reference_priority_order_rejects_duplicate_customer():
    instance = CVRPInstance(
        instance_id=8,
        depot=(0.0, 0.0),
        loc=((0.0, 1.0), (0.0, 2.0), (1.0, 0.0)),
        demand=(1.0, 1.0, 1.0),
        capacity=3.0,
        reference_routes=((3, 1), (1,)),
        reference_cost=5.0,
    )

    with pytest.raises(
        ValueError,
        match="reference routes must cover each customer exactly once",
    ):
        reference_priority_order(instance)


def test_customer_feature_rows_include_relative_and_normalized_features():
    instance = CVRPInstance(
        instance_id=0,
        depot=(1.0, 1.0),
        loc=((2.0, 1.0),),
        demand=(2.0,),
        capacity=4.0,
    )

    assert customer_feature_rows(instance) == ((2.0, 1.0, 1.0, 0.0, 2.0, 0.5, 1.0),)


def test_split_priority_order_by_capacity_starts_new_route_when_needed():
    instance = CVRPInstance(
        instance_id=0,
        depot=(0.0, 0.0),
        loc=((0.0, 1.0), (0.0, 2.0), (1.0, 0.0)),
        demand=(2.0, 2.0, 1.0),
        capacity=3.0,
    )

    assert split_priority_order_by_capacity(instance, (1, 3, 2)) == ((1, 3), (2,))


def test_split_priority_order_by_capacity_rejects_bad_order():
    instance = CVRPInstance(
        instance_id=0,
        depot=(0.0, 0.0),
        loc=((0.0, 1.0), (0.0, 2.0), (1.0, 0.0)),
        demand=(1.0, 1.0, 1.0),
        capacity=3.0,
    )

    with pytest.raises(ValueError, match="priority order must cover each customer"):
        split_priority_order_by_capacity(instance, (1, 2, 2))


def test_routes_from_priority_scores_sorts_lower_scores_first_with_customer_tiebreak():
    instance = CVRPInstance(
        instance_id=0,
        depot=(0.0, 0.0),
        loc=((0.0, 1.0), (0.0, 2.0), (1.0, 0.0)),
        demand=(1.0, 1.0, 1.0),
        capacity=2.0,
    )

    assert routes_from_priority_scores(instance, (0.2, 0.2, 0.1)) == ((3, 1), (2,))

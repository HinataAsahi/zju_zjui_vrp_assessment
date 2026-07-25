import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.preview_vrp_pkl import build_dataset_summary, render_markdown


def test_build_dataset_summary_handles_labeled_instances():
    data = [
        (
            [[0.0, 0.0]],
            [[1.0, 0.0], [0.0, 1.0]],
            [3.0, 4.0],
            40.0,
            [[1, 2]],
            2.0,
        ),
        (
            [[0.5, 0.5]],
            [[0.2, 0.3], [0.8, 0.9]],
            [1.0, 9.0],
            40.0,
            [[2], [1]],
            3.5,
        ),
    ]

    summary = build_dataset_summary("train_data.pkl", data, sample_count=1)

    assert summary["file"] == "train_data.pkl"
    assert summary["instance_count"] == 2
    assert summary["tuple_lengths"] == {"6": 2}
    assert summary["customer_counts"] == {"2": 2}
    assert summary["capacities"] == {"40.0": 2}
    assert summary["demand_range"] == [1.0, 9.0]
    assert summary["has_reference_labels"] is True
    assert summary["reference_cost_mean"] == 2.75
    assert summary["reference_route_count_mean"] == 1.5
    assert summary["samples"][0]["routes"] == [[1, 2]]
    assert summary["samples"][0]["cost"] == 2.0


def test_build_dataset_summary_handles_unlabeled_instances():
    data = [
        (
            [[0.0, 0.0]],
            [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
            [2.0, 5.0, 7.0],
            50.0,
        )
    ]

    summary = build_dataset_summary("check_data_to_students.pkl", data, sample_count=1)

    assert summary["instance_count"] == 1
    assert summary["tuple_lengths"] == {"4": 1}
    assert summary["customer_counts"] == {"3": 1}
    assert summary["capacities"] == {"50.0": 1}
    assert summary["has_reference_labels"] is False
    assert "reference_cost_mean" not in summary
    assert "routes" not in summary["samples"][0]
    assert "cost" not in summary["samples"][0]


def test_render_markdown_shows_dataset_structure_and_samples():
    summary = {
        "file": "train_data.pkl",
        "instance_count": 1,
        "tuple_lengths": {"6": 1},
        "customer_counts": {"2": 1},
        "capacities": {"40.0": 1},
        "demand_range": [3.0, 4.0],
        "has_reference_labels": True,
        "reference_cost_mean": 2.0,
        "reference_route_count_mean": 1.0,
        "samples": [
            {
                "instance_id": 0,
                "tuple_length": 6,
                "depot": [[0.0, 0.0]],
                "loc_preview": [[1.0, 0.0], [0.0, 1.0]],
                "demand_preview": [3.0, 4.0],
                "capacity": 40.0,
                "customer_count": 2,
                "routes": [[1, 2]],
                "cost": 2.0,
            }
        ],
    }

    markdown = render_markdown([summary])

    assert "# VRP `.pkl` 数据预览" in markdown
    assert "## `train_data.pkl`" in markdown
    assert "| 实例数量 | `1` |" in markdown
    assert "`(depot, loc, demand, capacity, routes, cost)`" in markdown
    assert "### 样本 0" in markdown
    assert "\"routes\": [[1, 2]]" in markdown

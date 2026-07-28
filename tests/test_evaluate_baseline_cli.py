import json
import pickle
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_baseline import evaluate_instances, parse_args  # noqa: E402
from src.vrp_io import CVRPInstance  # noqa: E402


def test_evaluate_instances_rejects_negative_limit():
    with pytest.raises(ValueError, match="limit must be non-negative"):
        evaluate_instances([], limit=-1)


def test_evaluate_parse_args_defaults_to_nearest_2opt():
    args = parse_args(["--input", "validation.pkl"])

    assert args.method == "nearest_2opt"


def test_evaluate_instances_accepts_nearest_2opt_method():
    instance = CVRPInstance(
        instance_id=0,
        depot=(0.0, 0.0),
        loc=((1.0, 0.0), (0.0, 1.0)),
        demand=(1.0, 1.0),
        capacity=2.0,
        reference_cost=4.0,
    )

    summary = evaluate_instances([instance], method="nearest_2opt")

    assert summary.instance_count == 1
    assert summary.feasible_count == 1
    assert summary.feasibility_rate == 1.0
    assert summary.average_cost > 0.0


def test_evaluate_baseline_cli_prints_summary_json(tmp_path):
    input_path = tmp_path / "validation.pkl"
    raw_instances = [
        ([[0, 0]], [[1, 0], [2, 0]], [1, 1], 2, [[1, 2]], 4.0),
        ([[0, 0]], [[0, 1]], [1], 2, [[1]], 2.0),
    ]
    with input_path.open("wb") as handle:
        pickle.dump(raw_instances, handle)

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "evaluate_baseline.py"),
            "--input",
            str(input_path),
            "--limit",
            "2",
            "--method",
            "nearest_2opt",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["instance_count"] == 2
    assert payload["feasible_count"] == 2
    assert payload["feasibility_rate"] == 1.0
    assert payload["average_cost"] > 0.0
    assert payload["average_gap"] is not None
    assert payload["average_inference_time"] >= 0.0


def test_evaluate_baseline_cli_rejects_negative_limit(tmp_path):
    input_path = tmp_path / "validation.pkl"
    with input_path.open("wb") as handle:
        pickle.dump([], handle)

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "evaluate_baseline.py"),
            "--input",
            str(input_path),
            "--limit",
            "-1",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "limit must be non-negative" in result.stderr


def test_evaluate_baseline_cli_rejects_unknown_method(tmp_path):
    input_path = tmp_path / "validation.pkl"
    with input_path.open("wb") as handle:
        pickle.dump([], handle)

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "evaluate_baseline.py"),
            "--input",
            str(input_path),
            "--method",
            "bad_method",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "invalid choice" in result.stderr


def test_evaluate_baseline_cli_handles_zero_limit(tmp_path):
    input_path = tmp_path / "validation.pkl"
    raw_instances = [
        ([[0, 0]], [[1, 0]], [1], 2, [[1]], 2.0),
    ]
    with input_path.open("wb") as handle:
        pickle.dump(raw_instances, handle)

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "evaluate_baseline.py"),
            "--input",
            str(input_path),
            "--limit",
            "0",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["instance_count"] == 0
    assert payload["feasibility_rate"] == 0.0
    assert payload["average_cost"] == 0.0
    assert payload["average_gap"] is None
    assert payload["average_inference_time"] is None

import json
import pickle
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_priority_model import parse_args  # noqa: E402
from src.priority_model import PriorityModelConfig, PriorityScoringModel  # noqa: E402
from src.vrp_eval import validate_solution  # noqa: E402
from src.vrp_io import load_instances  # noqa: E402


def _write_labeled_instances(path: Path) -> None:
    raw_instances = [
        ([[0, 0]], [[1, 0], [0, 1], [2, 0]], [1, 1, 1], 2, [[1, 2], [3]], 6.0),
        ([[0, 0]], [[1, 0], [2, 0]], [1, 1], 2, [[1, 2]], 4.0),
    ]
    with path.open("wb") as handle:
        pickle.dump(raw_instances, handle)


def _write_unlabeled_instances(path: Path) -> None:
    raw_instances = [
        ([[0, 0]], [[1, 0], [0, 1], [2, 0]], [1, 1, 1], 2),
        ([[0, 0]], [[1, 0], [2, 0]], [1, 1], 2),
    ]
    with path.open("wb") as handle:
        pickle.dump(raw_instances, handle)


def _write_checkpoint(path: Path) -> None:
    config = PriorityModelConfig(hidden_dim=16, num_heads=4, num_layers=1, dropout=0.0)
    model = PriorityScoringModel(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config": config.to_dict(),
            "model_state_dict": model.state_dict(),
            "epoch": 0,
            "best_validation": {},
        },
        path,
    )


def test_evaluate_priority_model_cli_prints_summary_and_writes_output(tmp_path):
    input_path = tmp_path / "validation.pkl"
    checkpoint_path = tmp_path / "checkpoints" / "priority.pt"
    output_path = tmp_path / "outputs" / "priority_predictions.json"
    _write_labeled_instances(input_path)
    _write_checkpoint(checkpoint_path)

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "evaluate_priority_model.py"),
            "--input",
            str(input_path),
            "--checkpoint",
            str(checkpoint_path),
            "--output",
            str(output_path),
            "--limit",
            "2",
            "--device",
            "cpu",
            "--no-postprocess",
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
    assert payload["average_gap"] is not None
    prediction_payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert prediction_payload["format_version"] == "cvrp_v1"
    assert len(prediction_payload["solutions"]) == 2
    instances = load_instances(input_path)
    for instance, solution in zip(instances, prediction_payload["solutions"]):
        routes = tuple(tuple(route) for route in solution["routes"])
        assert validate_solution(instance, routes).is_feasible is True


def test_evaluate_priority_model_cli_accepts_unlabeled_input(tmp_path):
    input_path = tmp_path / "check.pkl"
    checkpoint_path = tmp_path / "checkpoints" / "priority.pt"
    output_path = tmp_path / "outputs" / "priority_predictions.json"
    _write_unlabeled_instances(input_path)
    _write_checkpoint(checkpoint_path)

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "evaluate_priority_model.py"),
            "--input",
            str(input_path),
            "--checkpoint",
            str(checkpoint_path),
            "--output",
            str(output_path),
            "--limit",
            "2",
            "--device",
            "cpu",
            "--no-postprocess",
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
    assert payload["average_gap"] is None
    prediction_payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert prediction_payload["format_version"] == "cvrp_v1"
    instances = load_instances(input_path)
    for instance, solution in zip(instances, prediction_payload["solutions"]):
        routes = tuple(tuple(route) for route in solution["routes"])
        assert validate_solution(instance, routes).is_feasible is True


def test_evaluate_priority_model_rejects_negative_limit():
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--input",
                "validation.pkl",
                "--checkpoint",
                "priority.pt",
                "--limit",
                "-1",
            ]
        )

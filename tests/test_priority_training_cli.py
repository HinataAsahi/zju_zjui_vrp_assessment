import json
import pickle
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_priority_model import _format_progress_bar, parse_args  # noqa: E402


def _write_labeled_instances(path: Path) -> None:
    raw_instances = [
        ([[0, 0]], [[1, 0], [0, 1], [2, 0]], [1, 1, 1], 2, [[1, 2], [3]], 6.0),
        ([[0, 0]], [[1, 0], [2, 0]], [1, 1], 2, [[1, 2]], 4.0),
    ]
    with path.open("wb") as handle:
        pickle.dump(raw_instances, handle)


def test_format_progress_bar_shows_fraction_and_percentage():
    assert _format_progress_bar(1, 4, width=10) == "[##--------] 1/4 25.0%"
    assert _format_progress_bar(5, 4, width=10) == "[##########] 4/4 100.0%"


def test_train_priority_model_cli_writes_checkpoint_and_summary(tmp_path):
    train_path = tmp_path / "train.pkl"
    validation_path = tmp_path / "validation.pkl"
    checkpoint_path = tmp_path / "checkpoints" / "priority.pt"
    summary_path = tmp_path / "summaries" / "priority_summary.json"
    _write_labeled_instances(train_path)
    _write_labeled_instances(validation_path)

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "train_priority_model.py"),
            "--train-input",
            str(train_path),
            "--validation-input",
            str(validation_path),
            "--checkpoint-output",
            str(checkpoint_path),
            "--summary-output",
            str(summary_path),
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--hidden-dim",
            "16",
            "--num-heads",
            "4",
            "--num-layers",
            "1",
            "--dropout",
            "0",
            "--eval-limit",
            "2",
            "--device",
            "cpu",
            "--seed",
            "11",
            "--no-postprocess-eval",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert checkpoint_path.exists()
    assert "[train] device=cpu" in result.stderr
    assert "[epoch 1/1] train_start" in result.stderr
    assert "[epoch 1/1] batch 1/1 [########################] 1/1 100.0%" in result.stderr
    assert "[epoch 1/1] validation_start" in result.stderr
    assert "[epoch 1/1] validation_done" in result.stderr
    assert "[epoch 1/1] checkpoint_saved" in result.stderr
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    stdout_payload = json.loads(result.stdout)
    assert stdout_payload["best_validation"]["instance_count"] == 2
    assert payload["best_validation"]["feasible_count"] == 2
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    assert checkpoint["config"]["hidden_dim"] == 16
    assert "model_state_dict" in checkpoint


def test_train_priority_model_cli_accepts_mse_pairwise_loss(tmp_path):
    train_path = tmp_path / "train.pkl"
    validation_path = tmp_path / "validation.pkl"
    checkpoint_path = tmp_path / "checkpoints" / "priority_pairwise.pt"
    _write_labeled_instances(train_path)
    _write_labeled_instances(validation_path)

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "train_priority_model.py"),
            "--train-input",
            str(train_path),
            "--validation-input",
            str(validation_path),
            "--checkpoint-output",
            str(checkpoint_path),
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--hidden-dim",
            "16",
            "--num-heads",
            "4",
            "--num-layers",
            "1",
            "--dropout",
            "0",
            "--eval-limit",
            "2",
            "--device",
            "cpu",
            "--loss",
            "mse_pairwise",
            "--pairwise-weight",
            "0.5",
            "--pairwise-margin",
            "0.1",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "[train] device=cpu" in result.stderr
    assert "loss=mse_pairwise" in result.stderr
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    assert checkpoint["training_args"]["loss"] == "mse_pairwise"
    assert checkpoint["training_args"]["pairwise_weight"] == 0.5
    assert checkpoint["training_args"]["pairwise_margin"] == 0.1


def test_train_priority_model_rejects_unknown_loss():
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--train-input",
                "train.pkl",
                "--validation-input",
                "validation.pkl",
                "--checkpoint-output",
                "priority.pt",
                "--loss",
                "bad_loss",
            ]
        )


def test_train_priority_model_rejects_negative_eval_limit():
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--train-input",
                "train.pkl",
                "--validation-input",
                "validation.pkl",
                "--checkpoint-output",
                "priority.pt",
                "--eval-limit",
                "-1",
            ]
        )


def test_train_priority_model_rejects_zero_eval_limit():
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--train-input",
                "train.pkl",
                "--validation-input",
                "validation.pkl",
                "--checkpoint-output",
                "priority.pt",
                "--eval-limit",
                "0",
            ]
        )


@pytest.mark.parametrize("dropout", ["nan", "1", "1.5"])
def test_train_priority_model_rejects_invalid_dropout(dropout):
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--train-input",
                "train.pkl",
                "--validation-input",
                "validation.pkl",
                "--checkpoint-output",
                "priority.pt",
                "--dropout",
                dropout,
            ]
        )


def test_train_priority_model_rejects_nan_learning_rate():
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--train-input",
                "train.pkl",
                "--validation-input",
                "validation.pkl",
                "--checkpoint-output",
                "priority.pt",
                "--learning-rate",
                "nan",
            ]
        )

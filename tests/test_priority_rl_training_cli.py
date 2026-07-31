import json
import pickle
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_priority_rl import parse_args  # noqa: E402
from src.priority_model import PriorityModelConfig, PriorityScoringModel  # noqa: E402


def _write_labeled_instances(path: Path) -> None:
    raw_instances = [
        ([[0, 0]], [[1, 0], [0, 1], [2, 0]], [1, 1, 1], 2, [[1, 2], [3]], 6.0),
        ([[0, 0]], [[1, 0], [2, 0]], [1, 1], 2, [[1, 2]], 4.0),
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


def test_train_priority_rl_cli_writes_checkpoint_summary_and_logs_progress(tmp_path):
    train_path = tmp_path / "train.pkl"
    validation_path = tmp_path / "validation.pkl"
    checkpoint_path = tmp_path / "checkpoints" / "priority_rl.pt"
    summary_path = tmp_path / "outputs" / "priority_rl_summary.json"
    _write_labeled_instances(train_path)
    _write_labeled_instances(validation_path)

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "train_priority_rl.py"),
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
            "--samples-per-instance",
            "2",
            "--temperature",
            "1.0",
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
            "--no-postprocess-reward",
            "--no-postprocess-eval",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert checkpoint_path.exists()
    assert summary_path.exists()
    assert "[rl-train] device=cpu" in result.stderr
    assert "[rl-epoch 1/1] train_start" in result.stderr
    assert "[rl-epoch 1/1] batch 1/1 [########################] 1/1 100.0%" in result.stderr
    assert "[rl-epoch 1/1] validation_start" in result.stderr
    assert "[rl-epoch 1/1] validation_done" in result.stderr
    assert "[rl-epoch 1/1] checkpoint_saved" in result.stderr
    assert "[rl-train] done" in result.stderr
    stdout_payload = json.loads(result.stdout)
    summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert stdout_payload["best_validation"]["instance_count"] == 2
    assert summary_payload["history"][0]["average_sampled_cost"] > 0
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    assert checkpoint["config"]["hidden_dim"] == 16
    assert checkpoint["training_mode"] == "priority_rl_reinforce"


def test_train_priority_rl_cli_can_initialize_from_priority_checkpoint(tmp_path):
    train_path = tmp_path / "train.pkl"
    validation_path = tmp_path / "validation.pkl"
    init_checkpoint_path = tmp_path / "checkpoints" / "init_priority.pt"
    checkpoint_path = tmp_path / "checkpoints" / "priority_rl.pt"
    _write_labeled_instances(train_path)
    _write_labeled_instances(validation_path)
    _write_checkpoint(init_checkpoint_path)

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "train_priority_rl.py"),
            "--train-input",
            str(train_path),
            "--validation-input",
            str(validation_path),
            "--init-checkpoint",
            str(init_checkpoint_path),
            "--checkpoint-output",
            str(checkpoint_path),
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--samples-per-instance",
            "2",
            "--temperature",
            "1.0",
            "--eval-limit",
            "2",
            "--device",
            "cpu",
            "--seed",
            "11",
            "--no-postprocess-reward",
            "--no-postprocess-eval",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    assert checkpoint["init_checkpoint"] == str(init_checkpoint_path)
    assert checkpoint["config"]["hidden_dim"] == 16


def test_train_priority_rl_cli_resumes_from_last_checkpoint_epoch(tmp_path):
    train_path = tmp_path / "train.pkl"
    validation_path = tmp_path / "validation.pkl"
    best_checkpoint_path = tmp_path / "checkpoints" / "priority_rl_best.pt"
    last_checkpoint_path = tmp_path / "checkpoints" / "priority_rl_last.pt"
    summary_path = tmp_path / "outputs" / "priority_rl_summary.json"
    _write_labeled_instances(train_path)
    _write_labeled_instances(validation_path)

    first_result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "train_priority_rl.py"),
            "--train-input",
            str(train_path),
            "--validation-input",
            str(validation_path),
            "--checkpoint-output",
            str(best_checkpoint_path),
            "--last-checkpoint-output",
            str(last_checkpoint_path),
            "--summary-output",
            str(summary_path),
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--samples-per-instance",
            "2",
            "--temperature",
            "1.0",
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
            "--no-postprocess-reward",
            "--no-postprocess-eval",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert first_result.returncode == 0, first_result.stderr
    assert last_checkpoint_path.exists()

    resumed_result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "train_priority_rl.py"),
            "--train-input",
            str(train_path),
            "--validation-input",
            str(validation_path),
            "--resume-checkpoint",
            str(last_checkpoint_path),
            "--checkpoint-output",
            str(best_checkpoint_path),
            "--last-checkpoint-output",
            str(last_checkpoint_path),
            "--summary-output",
            str(summary_path),
            "--epochs",
            "2",
            "--batch-size",
            "2",
            "--samples-per-instance",
            "2",
            "--temperature",
            "1.0",
            "--eval-limit",
            "2",
            "--device",
            "cpu",
            "--seed",
            "11",
            "--no-postprocess-reward",
            "--no-postprocess-eval",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert resumed_result.returncode == 0, resumed_result.stderr
    assert "[rl-train] resume_checkpoint=" in resumed_result.stderr
    assert "[rl-epoch 1/2] train_start" not in resumed_result.stderr
    assert "[rl-epoch 2/2] train_start" in resumed_result.stderr
    summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert [record["epoch"] for record in summary_payload["history"]] == [1, 2]
    last_checkpoint = torch.load(last_checkpoint_path, map_location="cpu")
    assert last_checkpoint["epoch"] == 2
    assert [record["epoch"] for record in last_checkpoint["history"]] == [1, 2]
    assert last_checkpoint["resume_checkpoint"] == str(last_checkpoint_path)


def test_train_priority_rl_cli_rejects_singleton_runtime_batch(tmp_path):
    train_path = tmp_path / "train.pkl"
    validation_path = tmp_path / "validation.pkl"
    checkpoint_path = tmp_path / "checkpoints" / "priority_rl.pt"
    _write_labeled_instances(train_path)
    _write_labeled_instances(validation_path)

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "train_priority_rl.py"),
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
            "--samples-per-instance",
            "1",
            "--train-limit",
            "1",
            "--temperature",
            "1.0",
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
            "--no-postprocess-reward",
            "--no-postprocess-eval",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    assert (
        "[rl-epoch 1/1] batch 1/1 skipped singleton trajectory batch"
        in result.stderr
    )
    assert (
        "training set must produce at least one non-singleton trajectory batch"
        in result.stderr
    )


def test_train_priority_rl_cli_accepts_valid_partial_runtime_batch(tmp_path):
    train_path = tmp_path / "train.pkl"
    validation_path = tmp_path / "validation.pkl"
    checkpoint_path = tmp_path / "checkpoints" / "priority_rl.pt"
    _write_labeled_instances(train_path)
    _write_labeled_instances(validation_path)

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "train_priority_rl.py"),
            "--train-input",
            str(train_path),
            "--validation-input",
            str(validation_path),
            "--checkpoint-output",
            str(checkpoint_path),
            "--epochs",
            "1",
            "--batch-size",
            "3",
            "--samples-per-instance",
            "1",
            "--train-limit",
            "2",
            "--temperature",
            "1.0",
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
            "--no-postprocess-reward",
            "--no-postprocess-eval",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert checkpoint_path.exists()
    assert "skipped singleton trajectory batch" not in result.stderr
    stdout_payload = json.loads(result.stdout)
    assert stdout_payload["train_instances"] == 2


def test_train_priority_rl_rejects_invalid_temperature():
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--train-input",
                "train.pkl",
                "--validation-input",
                "validation.pkl",
                "--checkpoint-output",
                "priority_rl.pt",
                "--temperature",
                "0",
            ]
        )


def test_train_priority_rl_rejects_zero_samples_per_instance():
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--train-input",
                "train.pkl",
                "--validation-input",
                "validation.pkl",
                "--checkpoint-output",
                "priority_rl.pt",
                "--samples-per-instance",
                "0",
            ]
        )


def test_train_priority_rl_rejects_single_trajectory_batch():
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--train-input",
                "train.pkl",
                "--validation-input",
                "validation.pkl",
                "--checkpoint-output",
                "priority_rl.pt",
                "--batch-size",
                "1",
                "--samples-per-instance",
                "1",
            ]
        )

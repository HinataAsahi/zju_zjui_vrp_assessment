import json
import pickle
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _write_sample_inputs(tmp_path):
    input_path = tmp_path / "instances.pkl"
    solutions_path = tmp_path / "solutions.json"
    raw_instances = [
        (
            [[0.0, 0.0]],
            [[1.0, 0.0], [0.0, 1.0]],
            [1.0, 1.0],
            2.0,
            [[1, 2]],
            3.41421356237,
        )
    ]
    with input_path.open("wb") as handle:
        pickle.dump(raw_instances, handle)
    solutions_path.write_text(
        json.dumps(
            {
                "format_version": "cvrp_v1",
                "solutions": [{"instance_id": 0, "routes": [[1, 2]]}],
            }
        ),
        encoding="utf-8",
    )
    return input_path, solutions_path


def test_visualize_cli_writes_png_for_solution(tmp_path):
    input_path, solutions_path = _write_sample_inputs(tmp_path)
    image_path = tmp_path / "figure.png"

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "visualize_cvrp_solution.py"),
            "--input",
            str(input_path),
            "--solutions",
            str(solutions_path),
            "--instance-id",
            "0",
            "--output",
            str(image_path),
            "--title",
            "Example",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert image_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_visualize_cli_rejects_missing_instance_id(tmp_path):
    input_path, solutions_path = _write_sample_inputs(tmp_path)
    image_path = tmp_path / "missing.png"

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "visualize_cvrp_solution.py"),
            "--input",
            str(input_path),
            "--solutions",
            str(solutions_path),
            "--instance-id",
            "7",
            "--output",
            str(image_path),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    assert "instance_id 7" in result.stderr
    assert not image_path.exists()

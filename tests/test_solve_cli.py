import json
import pickle
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_solve_cli_writes_valid_cvrp_v1_json(tmp_path):
    input_path = tmp_path / "check.pkl"
    output_path = tmp_path / "nested" / "predictions.json"
    raw_instances = [
        ([[0, 0]], [[1, 0], [2, 0], [10, 0]], [4, 4, 4], 8),
        ([[0, 0]], [[0, 1], [0, 2]], [1, 1], 2),
    ]
    with input_path.open("wb") as handle:
        pickle.dump(raw_instances, handle)

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "solve.py"),
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--device",
            "cuda:0",
            "--seed",
            "2026",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["format_version"] == "cvrp_v1"
    assert payload["solutions"] == [
        {"instance_id": 0, "routes": [[1, 2], [3]]},
        {"instance_id": 1, "routes": [[1, 2]]},
    ]

import json
import pickle
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


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

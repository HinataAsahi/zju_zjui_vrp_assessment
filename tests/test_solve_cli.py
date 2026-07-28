import json
import pickle
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from solve import parse_args, solve_instances  # noqa: E402
from src.vrp_eval import validate_solution  # noqa: E402
from src.vrp_io import CVRPInstance, load_instances  # noqa: E402


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
            "--method",
            "nearest",
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


def test_parse_args_defaults_to_nearest_2opt():
    args = parse_args(
        [
            "--input",
            "input.pkl",
            "--output",
            "predictions.json",
        ]
    )

    assert args.method == "nearest_2opt"


def test_solve_instances_accepts_nearest_2opt_method():
    instance = CVRPInstance(
        instance_id=3,
        depot=(0.0, 0.0),
        loc=((1.0, 0.0), (0.0, 1.0), (1.0, 1.0)),
        demand=(1.0, 1.0, 1.0),
        capacity=3.0,
    )

    solutions = solve_instances([instance], method="nearest_2opt")

    assert len(solutions) == 1
    assert solutions[0].instance_id == 3
    assert validate_solution(instance, solutions[0].routes).is_feasible is True


def test_solve_cli_rejects_unknown_method(tmp_path):
    input_path = tmp_path / "check.pkl"
    output_path = tmp_path / "predictions.json"
    with input_path.open("wb") as handle:
        pickle.dump([], handle)

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "solve.py"),
            "--input",
            str(input_path),
            "--output",
            str(output_path),
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


def test_solve_cli_accepts_nearest_2opt_relocate_best_method(tmp_path):
    input_path = tmp_path / "check.pkl"
    output_path = tmp_path / "predictions_relocate.json"
    raw_instances = [
        ([[0, 0]], [[1, 0], [10, 0], [11, 0]], [1, 1, 1], 3),
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
            "--method",
            "nearest_2opt_relocate_best",
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
    instance = load_instances(input_path)[0]
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    routes = tuple(tuple(route) for route in payload["solutions"][0]["routes"])
    assert payload["format_version"] == "cvrp_v1"
    assert payload["solutions"][0]["instance_id"] == 0
    assert validate_solution(instance, routes).is_feasible is True


def test_solve_cli_help_describes_relocate_method():
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "solve.py"),
            "--help",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert "inter-route best relocate" in result.stdout


def test_solve_cli_accepts_nearest_2opt_relocate_limited_method(tmp_path):
    input_path = tmp_path / "check.pkl"
    output_path = tmp_path / "predictions_limited.json"
    raw_instances = [
        ([[0, 0]], [[1, 0], [10, 0], [11, 0]], [1, 1, 1], 3),
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
            "--method",
            "nearest_2opt_relocate_limited",
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
    instance = load_instances(input_path)[0]
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    routes = tuple(tuple(route) for route in payload["solutions"][0]["routes"])
    assert payload["format_version"] == "cvrp_v1"
    assert payload["solutions"][0]["instance_id"] == 0
    assert validate_solution(instance, routes).is_feasible is True


def test_solve_cli_help_describes_limited_relocate_method():
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "solve.py"),
            "--help",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert "limited relocate" in result.stdout

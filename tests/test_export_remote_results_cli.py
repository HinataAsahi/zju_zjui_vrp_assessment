import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


RESULT_FILES = [
    "checkpoints/priority_rl/priority_rl_finetune.pt",
    "checkpoints/priority_rl/priority_rl_finetune_last.pt",
    "outputs/priority_rl/rl_finetune_summary.json",
    "outputs/priority_rl/predictions_priority_rl.json",
]


def _write_result_files(source_root: Path) -> None:
    for relative_path in RESULT_FILES:
        path = source_root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"content for {relative_path}\n", encoding="utf-8")


def test_export_remote_results_cli_copies_required_files_with_relative_paths(tmp_path):
    source_root = tmp_path / "remote_project"
    export_dir = tmp_path / "exports" / "priority_rl_results"
    _write_result_files(source_root)

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "export_remote_results.py"),
            "--source-root",
            str(source_root),
            "--export-dir",
            str(export_dir),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["export_dir"] == str(export_dir)
    assert payload["copied_files"] == RESULT_FILES
    for relative_path in RESULT_FILES:
        assert (export_dir / relative_path).read_text(encoding="utf-8") == (
            f"content for {relative_path}\n"
        )


def test_export_remote_results_cli_rejects_missing_required_file(tmp_path):
    source_root = tmp_path / "remote_project"
    export_dir = tmp_path / "exports" / "priority_rl_results"
    _write_result_files(source_root)
    (source_root / "outputs/priority_rl/predictions_priority_rl.json").unlink()

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "export_remote_results.py"),
            "--source-root",
            str(source_root),
            "--export-dir",
            str(export_dir),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    assert "missing required result files" in result.stderr
    assert "outputs/priority_rl/predictions_priority_rl.json" in result.stderr
    assert not export_dir.exists()

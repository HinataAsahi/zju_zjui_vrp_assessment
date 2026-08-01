"""Collect final result files for future remote-server downloads.

This utility is for larger remote-server workflows where dragging one prepared
export directory is simpler than hunting through checkpoints and outputs. It
was not used for the current VRP assessment training on the RTX 4060 laptop.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


REQUIRED_RESULT_FILES = [
    Path("checkpoints/priority_rl/priority_rl_finetune.pt"),
    Path("checkpoints/priority_rl/priority_rl_finetune_last.pt"),
    Path("outputs/priority_rl/rl_finetune_summary.json"),
    Path("outputs/priority_rl/predictions_priority_rl.json"),
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect priority-RL result files into one export directory for "
            "future remote-server downloads."
        ),
        epilog=(
            "Note: this helper serves future remote-server workflows and was "
            "not used for the current VRP assessment training on the RTX 4060 laptop."
        ),
    )
    parser.add_argument(
        "--source-root",
        default=".",
        help="Project root containing checkpoints/ and outputs/. Defaults to current directory.",
    )
    parser.add_argument(
        "--export-dir",
        default="exports/priority_rl_results",
        help="Directory where files are copied with their relative paths preserved.",
    )
    return parser.parse_args(argv)


def collect_remote_results(source_root: Path, export_dir: Path) -> list[Path]:
    missing = [
        relative_path
        for relative_path in REQUIRED_RESULT_FILES
        if not (source_root / relative_path).is_file()
    ]
    if missing:
        missing_text = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(f"missing required result files:\n{missing_text}")

    copied: list[Path] = []
    for relative_path in REQUIRED_RESULT_FILES:
        source_path = source_root / relative_path
        target_path = export_dir / relative_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target_path)
        copied.append(relative_path)
    return copied


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    source_root = Path(args.source_root)
    export_dir = Path(args.export_dir)
    try:
        copied = collect_remote_results(source_root, export_dir)
    except FileNotFoundError as error:
        print(str(error), file=sys.stderr)
        return 1

    print(
        json.dumps(
            {
                "source_root": str(source_root),
                "export_dir": str(export_dir),
                "copied_files": [str(path) for path in copied],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

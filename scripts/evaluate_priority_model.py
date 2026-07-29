"""Evaluate a trained customer-priority model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.priority_inference import (  # noqa: E402
    evaluate_priority_model_instances,
    evaluation_summary_to_dict,
)
from src.priority_model import (  # noqa: E402
    load_priority_model_checkpoint,
    resolve_torch_device,
)
from src.vrp_io import load_instances, write_solutions_json  # noqa: E402


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a supervised-imitation priority model for CVRP.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--input", required=True, help="Input .pkl file.")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint path.")
    parser.add_argument("--output", help="Optional cvrp_v1 prediction JSON path.")
    parser.add_argument("--limit", type=_non_negative_int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--postprocess",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use heuristic postprocessing after priority-based capacity split.",
    )
    return parser.parse_args(argv)


def evaluate_priority_model(args: argparse.Namespace) -> dict:
    device = resolve_torch_device(args.device)
    model, checkpoint = load_priority_model_checkpoint(args.checkpoint, device=device)
    summary, solutions = evaluate_priority_model_instances(
        model,
        load_instances(args.input),
        device=device,
        limit=args.limit,
        postprocess=args.postprocess,
    )
    if args.output:
        write_solutions_json(args.output, solutions)

    return {
        **evaluation_summary_to_dict(summary),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "postprocess": args.postprocess,
        "output": args.output,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = evaluate_priority_model(args)
    except Exception as exc:
        print(f"evaluate_priority_model.py failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Train a customer-priority model by supervised imitation."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from statistics import mean

import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.priority_inference import (  # noqa: E402
    evaluate_priority_model_instances,
    evaluation_summary_to_dict,
)
from src.priority_model import (  # noqa: E402
    PriorityDataset,
    PriorityModelConfig,
    PriorityScoringModel,
    collate_priority_samples,
    masked_mse_loss,
    resolve_torch_device,
)
from src.vrp_io import load_instances  # noqa: E402


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _non_negative_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def _dropout_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0 or parsed >= 1:
        raise argparse.ArgumentTypeError("dropout must be in [0, 1)")
    return parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a supervised-imitation priority model for CVRP.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--train-input", required=True, help="Labeled training .pkl.")
    parser.add_argument("--validation-input", required=True, help="Labeled validation .pkl.")
    parser.add_argument("--checkpoint-output", required=True, help="Best checkpoint path.")
    parser.add_argument("--summary-output", help="Optional JSON training summary path.")
    parser.add_argument("--train-limit", type=_non_negative_int, default=None)
    parser.add_argument("--eval-limit", type=_positive_int, default=100)
    parser.add_argument("--epochs", type=_positive_int, default=50)
    parser.add_argument("--batch-size", type=_positive_int, default=64)
    parser.add_argument("--learning-rate", type=_positive_float, default=1e-3)
    parser.add_argument("--weight-decay", type=_non_negative_float, default=1e-4)
    parser.add_argument("--grad-clip", type=_non_negative_float, default=1.0)
    parser.add_argument("--hidden-dim", type=_positive_int, default=128)
    parser.add_argument("--num-heads", type=_positive_int, default=4)
    parser.add_argument("--num-layers", type=_positive_int, default=2)
    parser.add_argument("--dropout", type=_dropout_float, default=0.1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--postprocess-eval",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Evaluate validation routes with the heuristic postprocessor.",
    )
    return parser.parse_args(argv)


def _load_limited_instances(path: str, limit: int | None):
    instances = load_instances(path)
    if limit is not None:
        return instances[:limit]
    return instances


def _save_checkpoint(
    path: str,
    model: PriorityScoringModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    config: PriorityModelConfig,
    args: argparse.Namespace,
    best_validation: dict,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config": config.to_dict(),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "best_validation": best_validation,
            "training_args": vars(args),
        },
        output_path,
    )


def train_priority_model(args: argparse.Namespace) -> dict:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = resolve_torch_device(args.device)
    train_instances = _load_limited_instances(args.train_input, args.train_limit)
    validation_instances = load_instances(args.validation_input)
    if not train_instances:
        raise ValueError("training set is empty")
    if args.num_heads > args.hidden_dim or args.hidden_dim % args.num_heads != 0:
        raise ValueError("hidden-dim must be divisible by num-heads")

    train_dataset = PriorityDataset(train_instances, require_labels=True)
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_priority_samples,
        generator=generator,
    )

    config = PriorityModelConfig(
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )
    model = PriorityScoringModel(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    history = []
    best_validation: dict | None = None
    best_cost = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            features = batch["features"].to(device)
            labels = batch["labels"].to(device)
            mask = batch["mask"].to(device)

            optimizer.zero_grad(set_to_none=True)
            scores = model(features, mask)
            loss = masked_mse_loss(scores, labels, mask)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        summary, _solutions = evaluate_priority_model_instances(
            model,
            validation_instances,
            device=device,
            limit=args.eval_limit,
            postprocess=args.postprocess_eval,
        )
        validation_payload = evaluation_summary_to_dict(summary)
        record = {
            "epoch": epoch,
            "train_loss": mean(losses) if losses else 0.0,
            "validation": validation_payload,
        }
        history.append(record)

        if summary.average_cost < best_cost:
            best_cost = summary.average_cost
            best_validation = {
                "epoch": epoch,
                **validation_payload,
            }
            _save_checkpoint(
                args.checkpoint_output,
                model,
                optimizer,
                epoch,
                config,
                args,
                best_validation,
            )

    if best_validation is None:
        raise RuntimeError("training did not produce a checkpoint")

    result = {
        "checkpoint_output": args.checkpoint_output,
        "train_instances": len(train_instances),
        "validation_instances": len(validation_instances),
        "best_validation": best_validation,
        "history": history,
    }
    if args.summary_output:
        summary_path = Path(args.summary_output)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return result


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = train_priority_model(args)
    except Exception as exc:
        print(f"train_priority_model.py failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

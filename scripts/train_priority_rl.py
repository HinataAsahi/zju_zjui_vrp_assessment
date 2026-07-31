"""Finetune a customer-priority model with REINFORCE."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
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
    load_priority_model_checkpoint,
    resolve_torch_device,
)
from src.priority_rl import (  # noqa: E402
    compute_order_rewards,
    reinforce_policy_loss,
    sample_priority_orders,
)
from src.vrp_io import CVRPInstance, load_instances  # noqa: E402


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


def _validate_args(args: argparse.Namespace) -> argparse.Namespace:
    if args.batch_size * args.samples_per_instance <= 1:
        raise argparse.ArgumentTypeError(
            "batch-size * samples-per-instance must be greater than 1 "
            "for REINFORCE baseline"
        )
    return args


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _format_progress_bar(current: int, total: int, width: int = 24) -> str:
    if total <= 0:
        raise ValueError("total must be positive")
    bounded_current = min(max(current, 0), total)
    ratio = bounded_current / total
    filled = int(width * ratio)
    return (
        f"[{'#' * filled}{'-' * (width - filled)}] "
        f"{bounded_current}/{total} {ratio * 100:.1f}%"
    )


def _progress(message: str, final: bool = False) -> None:
    end = "\n" if final else ""
    print(f"\r{message}", file=sys.stderr, end=end, flush=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Finetune a priority model with REINFORCE for CVRP.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--train-input", required=True, help="Training .pkl.")
    parser.add_argument("--validation-input", required=True, help="Validation .pkl.")
    parser.add_argument("--checkpoint-output", required=True, help="Best checkpoint path.")
    parser.add_argument("--summary-output", help="Optional JSON training summary path.")
    parser.add_argument("--init-checkpoint", help="Optional priority checkpoint to finetune.")
    parser.add_argument("--train-limit", type=_non_negative_int, default=None)
    parser.add_argument("--eval-limit", type=_positive_int, default=100)
    parser.add_argument("--epochs", type=_positive_int, default=20)
    parser.add_argument("--batch-size", type=_positive_int, default=32)
    parser.add_argument("--samples-per-instance", type=_positive_int, default=2)
    parser.add_argument("--temperature", type=_positive_float, default=1.0)
    parser.add_argument("--learning-rate", type=_positive_float, default=1e-5)
    parser.add_argument("--weight-decay", type=_non_negative_float, default=1e-4)
    parser.add_argument("--grad-clip", type=_non_negative_float, default=1.0)
    parser.add_argument("--hidden-dim", type=_positive_int, default=128)
    parser.add_argument("--num-heads", type=_positive_int, default=4)
    parser.add_argument("--num-layers", type=_positive_int, default=2)
    parser.add_argument("--dropout", type=_dropout_float, default=0.1)
    parser.add_argument("--log-every", type=_positive_int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--postprocess-reward",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use heuristic postprocessing before computing training reward.",
    )
    parser.add_argument(
        "--postprocess-eval",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use heuristic postprocessing during validation.",
    )
    try:
        return _validate_args(parser.parse_args(argv))
    except argparse.ArgumentTypeError as error:
        parser.error(str(error))


def _load_limited_instances(path: str, limit: int | None) -> list[CVRPInstance]:
    instances = load_instances(path)
    if limit is not None:
        return instances[:limit]
    return instances


def _build_model(
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[PriorityScoringModel, PriorityModelConfig]:
    if args.init_checkpoint:
        model, checkpoint = load_priority_model_checkpoint(args.init_checkpoint, device=device)
        config = PriorityModelConfig(**checkpoint["config"])
        model.train()
        return model, config

    if args.num_heads > args.hidden_dim or args.hidden_dim % args.num_heads != 0:
        raise ValueError("hidden-dim must be divisible by num-heads")
    config = PriorityModelConfig(
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )
    return PriorityScoringModel(config).to(device), config


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
            "training_mode": "priority_rl_reinforce",
            "init_checkpoint": args.init_checkpoint,
        },
        output_path,
    )


def _batch_instances(
    instances_by_id: dict[int, CVRPInstance],
    instance_ids: torch.Tensor,
) -> list[CVRPInstance]:
    return [instances_by_id[int(instance_id)] for instance_id in instance_ids.tolist()]


def train_priority_rl(args: argparse.Namespace) -> dict:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = resolve_torch_device(args.device)
    train_instances = _load_limited_instances(args.train_input, args.train_limit)
    validation_instances = load_instances(args.validation_input)
    if not train_instances:
        raise ValueError("training set is empty")

    train_dataset = PriorityDataset(train_instances, require_labels=False)
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_priority_samples,
        generator=generator,
    )
    instances_by_id = {instance.instance_id: instance for instance in train_instances}

    model, config = _build_model(args, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    history = []
    best_validation: dict | None = None
    best_cost = float("inf")
    total_batches = len(train_loader)
    _log(
        "[rl-train] "
        f"device={device} train_instances={len(train_instances)} "
        f"validation_instances={len(validation_instances)} epochs={args.epochs} "
        f"batch_size={args.batch_size} samples_per_instance={args.samples_per_instance} "
        f"temperature={args.temperature} eval_limit={args.eval_limit} "
        f"postprocess_reward={args.postprocess_reward} "
        f"postprocess_eval={args.postprocess_eval} init_checkpoint={args.init_checkpoint}"
    )

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        losses: list[float] = []
        sampled_costs: list[float] = []
        trainable_batches = 0
        _log(f"[rl-epoch {epoch}/{args.epochs}] train_start batches={total_batches}")

        for batch_index, batch in enumerate(train_loader, start=1):
            features = batch["features"].to(device)
            mask = batch["mask"].to(device)
            actual_trajectory_count = features.shape[0] * args.samples_per_instance
            if actual_trajectory_count <= 1:
                _progress(
                    f"[rl-epoch {epoch}/{args.epochs}] "
                    f"batch {batch_index}/{total_batches} "
                    "skipped singleton trajectory batch",
                    final=True,
                )
                continue
            trainable_batches += 1
            selected_instances = _batch_instances(instances_by_id, batch["instance_ids"])

            optimizer.zero_grad(set_to_none=True)
            scores = model(features, mask)
            orders, log_probs = sample_priority_orders(
                scores,
                mask,
                samples_per_instance=args.samples_per_instance,
                temperature=args.temperature,
            )
            rewards = compute_order_rewards(
                selected_instances,
                orders,
                postprocess_reward=args.postprocess_reward,
            )
            loss, _advantages = reinforce_policy_loss(log_probs, rewards)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            loss_value = float(loss.detach().cpu())
            average_sampled_cost = float((-rewards).mean().detach().cpu())
            losses.append(loss_value)
            sampled_costs.append(average_sampled_cost)
            if (
                batch_index == 1
                or batch_index % args.log_every == 0
                or batch_index == total_batches
            ):
                _progress(
                    f"[rl-epoch {epoch}/{args.epochs}] "
                    f"batch {batch_index}/{total_batches} "
                    f"{_format_progress_bar(batch_index, total_batches)} "
                    f"policy_loss={loss_value:.6f} "
                    f"sampled_cost={average_sampled_cost:.6f}",
                    final=batch_index == total_batches,
                )

        if trainable_batches == 0:
            raise ValueError(
                "training set must produce at least one non-singleton trajectory batch"
            )
        train_loss = mean(losses) if losses else 0.0
        average_sampled_cost = mean(sampled_costs) if sampled_costs else 0.0
        _log(
            f"[rl-epoch {epoch}/{args.epochs}] "
            f"validation_start limit={args.eval_limit} "
            f"postprocess={args.postprocess_eval} "
            f"policy_loss={train_loss:.6f} "
            f"average_sampled_cost={average_sampled_cost:.6f}"
        )
        summary, _solutions = evaluate_priority_model_instances(
            model,
            validation_instances,
            device=device,
            limit=args.eval_limit,
            postprocess=args.postprocess_eval,
        )
        gap_text = (
            f"{summary.average_gap:.6f}"
            if summary.average_gap is not None
            else "null"
        )
        _log(
            f"[rl-epoch {epoch}/{args.epochs}] validation_done "
            f"feasible={summary.feasible_count}/{summary.instance_count} "
            f"average_cost={summary.average_cost:.6f} "
            f"average_gap={gap_text} "
            f"elapsed_seconds={time.perf_counter() - epoch_start:.2f}"
        )
        validation_payload = evaluation_summary_to_dict(summary)
        record = {
            "epoch": epoch,
            "policy_loss": train_loss,
            "average_sampled_cost": average_sampled_cost,
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
            _log(
                f"[rl-epoch {epoch}/{args.epochs}] "
                f"checkpoint_saved path={args.checkpoint_output}"
            )

    if best_validation is None:
        raise RuntimeError("training did not produce a checkpoint")

    result = {
        "checkpoint_output": args.checkpoint_output,
        "init_checkpoint": args.init_checkpoint,
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
    _log(
        "[rl-train] done "
        f"best_epoch={best_validation['epoch']} "
        f"best_average_cost={best_validation['average_cost']:.6f}"
    )
    return result


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = train_priority_rl(args)
    except Exception as exc:
        print(f"train_priority_rl.py failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

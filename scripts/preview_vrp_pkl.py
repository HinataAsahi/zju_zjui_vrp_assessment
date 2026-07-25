"""Generate readable Markdown previews for VRP pickle datasets."""

from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any, Iterable


DEFAULT_INPUTS = [
    "VRP_project/VRPData/train_data.pkl",
    "VRP_project/VRPData/validation_data.pkl",
    "VRP_project/VRPData/check_data_to_students.pkl",
]


def _float(value: Any) -> float:
    return float(value)


def _to_preview(values: Iterable[Any], limit: int) -> list[Any]:
    preview = list(values)[:limit]
    return _jsonable(preview)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "item"):
        return _jsonable(value.item())
    if isinstance(value, float):
        return round(value, 6)
    return value


def _counter_as_strings(counter: Counter[Any]) -> dict[str, int]:
    return {str(key): counter[key] for key in sorted(counter)}


def build_dataset_summary(
    file_label: str,
    data: list[tuple[Any, ...]],
    sample_count: int = 2,
    preview_items: int = 5,
) -> dict[str, Any]:
    """Summarize one VRP pickle dataset without mutating the input data."""
    tuple_lengths = Counter(len(instance) for instance in data)
    customer_counts = Counter(len(instance[1]) for instance in data)
    capacities = Counter(_float(instance[3]) for instance in data)
    demands = [_float(demand) for instance in data for demand in instance[2]]
    has_reference_labels = all(len(instance) >= 6 for instance in data)

    summary: dict[str, Any] = {
        "file": file_label,
        "instance_count": len(data),
        "tuple_lengths": _counter_as_strings(tuple_lengths),
        "customer_counts": _counter_as_strings(customer_counts),
        "capacities": _counter_as_strings(capacities),
        "demand_range": [min(demands), max(demands)] if demands else [],
        "has_reference_labels": has_reference_labels,
        "samples": [],
    }

    if has_reference_labels:
        costs = [_float(instance[5]) for instance in data]
        route_counts = [len(instance[4]) for instance in data]
        summary["reference_cost_mean"] = round(mean(costs), 6)
        summary["reference_route_count_mean"] = round(mean(route_counts), 6)

    for instance_id, instance in enumerate(data[:sample_count]):
        depot, loc, demand, capacity = instance[:4]
        sample = {
            "instance_id": instance_id,
            "tuple_length": len(instance),
            "depot": _jsonable(depot),
            "loc_preview": _to_preview(loc, preview_items),
            "demand_preview": _to_preview(demand, preview_items),
            "capacity": _float(capacity),
            "customer_count": len(loc),
        }
        if len(instance) >= 6:
            sample["routes"] = _jsonable(instance[4])
            sample["cost"] = round(_float(instance[5]), 6)
        summary["samples"].append(sample)

    return summary


def _tuple_signature(summary: dict[str, Any]) -> str:
    if summary["has_reference_labels"]:
        return "`(depot, loc, demand, capacity, routes, cost)`"
    return "`(depot, loc, demand, capacity)`"


def _json_line(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(", ", ": "))


def render_markdown(summaries: list[dict[str, Any]]) -> str:
    """Render dataset summaries as a readable Markdown document."""
    lines = [
        "# VRP `.pkl` 数据预览",
        "",
        "`.pkl` 是 Python pickle 二进制序列化文件，VSCode 的文本编辑器不能直接阅读。",
        "本文件是通过 Python 加载后生成的可读预览，只展示结构、统计信息和少量样本。",
        "",
        "## 快速理解",
        "",
        "- 训练集和验证集通常包含参考 `routes` 与 `cost`，tuple 长度为 6。",
        "- 公开测试集不包含参考答案，tuple 长度为 4。",
        "- `loc[0]` 对应输出中的客户 `1`，即内部数组下标是 0-based，提交 routes 是 1-based。",
        "- depot 是隐含起终点，不应写进输出 routes。",
        "",
    ]

    for summary in summaries:
        lines.extend(
            [
                f"## `{summary['file']}`",
                "",
                "| 项目 | 值 |",
                "| --- | --- |",
                f"| 实例数量 | `{summary['instance_count']}` |",
                f"| tuple 结构 | {_tuple_signature(summary)} |",
                f"| tuple 长度分布 | `{_json_line(summary['tuple_lengths'])}` |",
                f"| 客户数量分布 | `{_json_line(summary['customer_counts'])}` |",
                f"| capacity 分布 | `{_json_line(summary['capacities'])}` |",
                f"| demand 范围 | `{_json_line(summary['demand_range'])}` |",
                f"| 是否有参考 routes/cost | `{'是' if summary['has_reference_labels'] else '否'}` |",
            ]
        )

        if summary["has_reference_labels"]:
            lines.extend(
                [
                    f"| 参考 cost 平均值 | `{summary['reference_cost_mean']}` |",
                    f"| 参考 route 数平均值 | `{summary['reference_route_count_mean']}` |",
                ]
            )

        lines.append("")
        lines.append("### 样本预览")
        lines.append("")
        for sample in summary["samples"]:
            lines.extend(
                [
                    f"### 样本 {sample['instance_id']}",
                    "",
                    "```json",
                    _json_line(sample),
                    "```",
                    "",
                ]
            )

    return "\n".join(lines).rstrip() + "\n"


def load_pickle(path: Path) -> list[tuple[Any, ...]]:
    with path.open("rb") as handle:
        data = pickle.load(handle)
    if not isinstance(data, list):
        raise TypeError(f"{path} should contain a list, got {type(data).__name__}")
    return data


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a readable Markdown preview for VRP pickle datasets."
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        default=DEFAULT_INPUTS,
        help="Pickle dataset paths. Defaults to the three provided VRP datasets.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="docs/data_preview/vrp_pkl_preview.md",
        help="Markdown output path.",
    )
    parser.add_argument("--samples", type=int, default=2, help="Samples per dataset.")
    parser.add_argument(
        "--preview-items",
        type=int,
        default=5,
        help="Number of loc/demand values shown per sample.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summaries = []
    for input_path in args.inputs:
        path = Path(input_path)
        data = load_pickle(path)
        summaries.append(
            build_dataset_summary(
                path.name,
                data,
                sample_count=args.samples,
                preview_items=args.preview_items,
            )
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_markdown(summaries), encoding="utf-8")
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

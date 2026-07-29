"""Inference helpers for priority-score CVRP models."""

from __future__ import annotations

import time
from typing import Sequence

import torch

from src.priority_model import PriorityScoringModel, predict_priority_scores
from src.priority_pipeline import solve_from_priority_scores
from src.vrp_eval import EvaluationSummary, summarize_results
from src.vrp_io import CVRPInstance, SolutionRecord


def evaluation_summary_to_dict(summary: EvaluationSummary) -> dict[str, float | int | None]:
    return {
        "instance_count": summary.instance_count,
        "feasible_count": summary.feasible_count,
        "feasibility_rate": summary.feasibility_rate,
        "average_cost": summary.average_cost,
        "average_gap": summary.average_gap,
        "average_inference_time": summary.average_inference_time,
    }


def solve_instances_with_priority_model(
    model: PriorityScoringModel,
    instances: Sequence[CVRPInstance],
    device: torch.device | str,
    postprocess: bool = True,
) -> tuple[list[SolutionRecord], list[float]]:
    solutions: list[SolutionRecord] = []
    inference_times: list[float] = []
    for instance in instances:
        start = time.perf_counter()
        scores = predict_priority_scores(model, instance, device=device)
        routes = solve_from_priority_scores(instance, scores, postprocess=postprocess)
        inference_times.append(time.perf_counter() - start)
        solutions.append(SolutionRecord(instance_id=instance.instance_id, routes=routes))
    return solutions, inference_times


def evaluate_priority_model_instances(
    model: PriorityScoringModel,
    instances: Sequence[CVRPInstance],
    device: torch.device | str,
    limit: int | None = None,
    postprocess: bool = True,
) -> tuple[EvaluationSummary, list[SolutionRecord]]:
    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative")

    selected = list(instances[:limit]) if limit is not None else list(instances)
    solutions, inference_times = solve_instances_with_priority_model(
        model,
        selected,
        device=device,
        postprocess=postprocess,
    )
    summary = summarize_results(
        selected,
        [solution.routes for solution in solutions],
        inference_times,
    )
    return summary, solutions

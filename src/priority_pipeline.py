"""Route construction and post-processing for priority model scores."""

from __future__ import annotations

from typing import Sequence

from src.heuristics import (
    improve_routes_2opt,
    improve_routes_relocate_best,
    improve_routes_swap_best,
    relocate_limited_passes,
    swap_limited_passes,
)
from src.priority_data import routes_from_priority_scores
from src.vrp_io import CVRPInstance


def postprocess_priority_routes(
    instance: CVRPInstance,
    routes: Sequence[Sequence[int]],
    capacity_tol: float = 1e-9,
    improvement_tol: float = 1e-12,
) -> tuple[tuple[int, ...], ...]:
    improved = improve_routes_2opt(
        instance,
        routes,
        improvement_tol=improvement_tol,
    )
    improved = improve_routes_relocate_best(
        instance,
        improved,
        capacity_tol=capacity_tol,
        improvement_tol=improvement_tol,
        max_passes=relocate_limited_passes(instance.customer_count),
    )
    return improve_routes_swap_best(
        instance,
        improved,
        capacity_tol=capacity_tol,
        improvement_tol=improvement_tol,
        max_passes=swap_limited_passes(instance.customer_count),
    )


def solve_from_priority_scores(
    instance: CVRPInstance,
    scores: Sequence[float],
    postprocess: bool = True,
) -> tuple[tuple[int, ...], ...]:
    routes = routes_from_priority_scores(instance, scores)
    if not postprocess:
        return routes
    return postprocess_priority_routes(instance, routes)

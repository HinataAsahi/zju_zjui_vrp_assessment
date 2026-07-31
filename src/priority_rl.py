"""Reinforcement-learning helpers for priority-score CVRP models."""

from __future__ import annotations

from typing import Sequence

import torch

from src.priority_data import split_priority_order_by_capacity
from src.priority_pipeline import postprocess_priority_routes
from src.vrp_eval import compute_total_cost
from src.vrp_io import CVRPInstance


def sample_priority_orders(
    scores: torch.Tensor,
    mask: torch.Tensor,
    samples_per_instance: int,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample customer permutations from priority scores.

    Lower scores mean earlier service, so logits are `-scores / temperature`.
    Returned orders contain zero-based customer indexes. Padded positions are
    `-1`, matching invalid customer slots from the input mask.
    """

    if scores.ndim != 2:
        raise ValueError("scores must have shape [batch, customers]")
    if mask.shape != scores.shape:
        raise ValueError("mask must have the same shape as scores")
    if samples_per_instance <= 0:
        raise ValueError("samples_per_instance must be positive")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if not mask.any(dim=1).all():
        raise ValueError("each instance must contain at least one valid customer")

    batch_size, max_customers = scores.shape
    orders = torch.full(
        (batch_size, samples_per_instance, max_customers),
        -1,
        dtype=torch.long,
        device=scores.device,
    )
    log_probs = torch.zeros(
        batch_size,
        samples_per_instance,
        dtype=scores.dtype,
        device=scores.device,
    )

    for batch_index in range(batch_size):
        valid_indexes = torch.nonzero(mask[batch_index], as_tuple=False).flatten()
        valid_scores = scores[batch_index, valid_indexes]
        for sample_index in range(samples_per_instance):
            remaining_indexes = valid_indexes
            remaining_scores = valid_scores
            selected: list[torch.Tensor] = []
            sample_log_prob = torch.zeros((), dtype=scores.dtype, device=scores.device)
            while remaining_indexes.numel() > 0:
                logits = -remaining_scores / temperature
                distribution = torch.distributions.Categorical(logits=logits)
                choice = distribution.sample()
                sample_log_prob = sample_log_prob + distribution.log_prob(choice)
                selected.append(remaining_indexes[choice])
                keep = torch.ones(
                    remaining_indexes.numel(),
                    dtype=torch.bool,
                    device=scores.device,
                )
                keep[choice] = False
                remaining_indexes = remaining_indexes[keep]
                remaining_scores = remaining_scores[keep]

            selected_order = torch.stack(selected)
            orders[batch_index, sample_index, : selected_order.numel()] = selected_order
            log_probs[batch_index, sample_index] = sample_log_prob

    return orders, log_probs


def compute_order_rewards(
    instances: Sequence[CVRPInstance],
    orders: torch.Tensor,
    postprocess_reward: bool = True,
) -> torch.Tensor:
    """Return negative route costs for sampled orders.

    `orders` has shape [batch, samples, customers] and contains zero-based
    customer indexes, with `-1` for padded positions.
    """

    if orders.ndim != 3:
        raise ValueError("orders must have shape [batch, samples, customers]")
    if len(instances) != orders.shape[0]:
        raise ValueError("instances length must match orders batch size")

    rewards: list[list[float]] = []
    cpu_orders = orders.detach().cpu()
    for instance, instance_orders in zip(instances, cpu_orders):
        instance_rewards: list[float] = []
        for order_tensor in instance_orders:
            order = tuple(int(index) + 1 for index in order_tensor.tolist() if index >= 0)
            routes = split_priority_order_by_capacity(instance, order)
            if postprocess_reward:
                routes = postprocess_priority_routes(instance, routes)
            instance_rewards.append(-compute_total_cost(instance, routes))
        rewards.append(instance_rewards)

    return torch.tensor(
        rewards,
        dtype=torch.float32,
        device=orders.device,
    )


def reinforce_policy_loss(
    log_probs: torch.Tensor,
    rewards: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute REINFORCE loss with a batch-mean reward baseline."""

    if log_probs.shape != rewards.shape:
        raise ValueError("log_probs and rewards must have the same shape")
    baseline = rewards.mean()
    advantages = rewards - baseline
    loss = -(log_probs * advantages.detach()).mean()
    return loss, advantages

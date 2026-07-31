import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.priority_rl import sample_priority_orders  # noqa: E402
from src.priority_data import split_priority_order_by_capacity  # noqa: E402
from src.priority_rl import compute_order_rewards, reinforce_policy_loss  # noqa: E402
from src.vrp_eval import compute_total_cost  # noqa: E402
from src.vrp_io import CVRPInstance  # noqa: E402


def test_sample_priority_orders_returns_valid_permutations_with_padding():
    torch.manual_seed(3)
    scores = torch.tensor(
        [
            [0.0, 1.0, 2.0],
            [0.2, 0.1, 0.0],
        ],
        dtype=torch.float32,
    )
    mask = torch.tensor(
        [
            [True, True, True],
            [True, True, False],
        ]
    )

    orders, log_probs = sample_priority_orders(
        scores,
        mask,
        samples_per_instance=3,
        temperature=1.0,
    )

    assert orders.shape == (2, 3, 3)
    assert log_probs.shape == (2, 3)
    assert torch.isfinite(log_probs).all()
    for order in orders[0]:
        assert sorted(order.tolist()) == [0, 1, 2]
    for order in orders[1]:
        assert sorted(order[:2].tolist()) == [0, 1]
        assert order[2].item() == -1


@pytest.mark.parametrize("temperature", [0.0, -1.0])
def test_sample_priority_orders_rejects_non_positive_temperature(temperature):
    scores = torch.tensor([[0.0, 1.0]], dtype=torch.float32)
    mask = torch.tensor([[True, True]])

    with pytest.raises(ValueError, match="temperature must be positive"):
        sample_priority_orders(
            scores,
            mask,
            samples_per_instance=1,
            temperature=temperature,
        )


def test_sample_priority_orders_rejects_zero_samples_per_instance():
    scores = torch.tensor([[0.0, 1.0]], dtype=torch.float32)
    mask = torch.tensor([[True, True]])

    with pytest.raises(ValueError, match="samples_per_instance must be positive"):
        sample_priority_orders(
            scores,
            mask,
            samples_per_instance=0,
            temperature=1.0,
        )


def _instance() -> CVRPInstance:
    return CVRPInstance(
        instance_id=0,
        depot=(0.0, 0.0),
        loc=((1.0, 0.0), (2.0, 0.0), (0.0, 2.0)),
        demand=(1.0, 1.0, 1.0),
        capacity=2.0,
        reference_routes=((1, 2), (3,)),
        reference_cost=6.0,
    )


def test_compute_order_rewards_returns_negative_costs_without_postprocess():
    instance = _instance()
    orders = torch.tensor([[[0, 1, 2]]], dtype=torch.long)

    rewards = compute_order_rewards(
        [instance],
        orders,
        postprocess_reward=False,
    )

    routes = split_priority_order_by_capacity(instance, (1, 2, 3))
    expected_cost = compute_total_cost(instance, routes)
    assert rewards.shape == (1, 1)
    assert rewards.item() == pytest.approx(-expected_cost)


def test_compute_order_rewards_ignores_padded_indexes():
    instance = CVRPInstance(
        instance_id=1,
        depot=(0.0, 0.0),
        loc=((1.0, 0.0), (2.0, 0.0)),
        demand=(1.0, 1.0),
        capacity=2.0,
    )
    orders = torch.tensor([[[0, 1, -1]]], dtype=torch.long)

    rewards = compute_order_rewards(
        [instance],
        orders,
        postprocess_reward=False,
    )

    assert rewards.shape == (1, 1)
    assert torch.isfinite(rewards).all()


def test_reinforce_policy_loss_uses_batch_average_baseline_and_backpropagates():
    log_probs = torch.tensor([[-0.2, -0.4]], dtype=torch.float32, requires_grad=True)
    rewards = torch.tensor([[-10.0, -8.0]], dtype=torch.float32)

    loss, advantages = reinforce_policy_loss(log_probs, rewards)
    loss.backward()

    assert advantages.tolist() == pytest.approx([[-1.0, 1.0]])
    assert loss.ndim == 0
    assert log_probs.grad is not None
    assert torch.isfinite(log_probs.grad).all()

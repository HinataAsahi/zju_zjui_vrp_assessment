import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.priority_model import (  # noqa: E402
    PriorityDataset,
    PriorityModelConfig,
    PriorityScoringModel,
    collate_priority_samples,
    masked_mse_loss,
    masked_pairwise_ranking_loss,
    predict_priority_scores,
)
from src.vrp_io import CVRPInstance  # noqa: E402


def _instance(
    instance_id: int = 0,
    loc=((1.0, 0.0), (0.0, 1.0), (2.0, 0.0)),
    demand=(1.0, 2.0, 1.0),
    capacity: float = 3.0,
    reference_routes=((2, 1), (3,)),
) -> CVRPInstance:
    return CVRPInstance(
        instance_id=instance_id,
        depot=(0.0, 0.0),
        loc=loc,
        demand=demand,
        capacity=capacity,
        reference_routes=reference_routes,
        reference_cost=5.0,
    )


def test_priority_dataset_builds_feature_and_rank_tensors():
    dataset = PriorityDataset([_instance()])

    sample = dataset[0]

    assert sample.instance_id == 0
    assert sample.features.shape == (3, 7)
    assert sample.labels.tolist() == [0.5, 0.0, 1.0]
    assert sample.features.dtype == torch.float32
    assert sample.labels.dtype == torch.float32


def test_collate_priority_samples_pads_variable_customer_counts():
    samples = [
        PriorityDataset([_instance(instance_id=0)])[0],
        PriorityDataset(
            [
                _instance(
                    instance_id=1,
                    loc=((1.0, 0.0),),
                    demand=(1.0,),
                    capacity=2.0,
                    reference_routes=((1,),),
                )
            ]
        )[0],
    ]

    batch = collate_priority_samples(samples)

    assert batch["features"].shape == (2, 3, 7)
    assert batch["labels"].shape == (2, 3)
    assert batch["mask"].tolist() == [
        [True, True, True],
        [True, False, False],
    ]
    assert batch["instance_ids"].tolist() == [0, 1]


def test_priority_model_forward_and_masked_mse_loss():
    torch.manual_seed(7)
    model = PriorityScoringModel(
        PriorityModelConfig(feature_dim=7, hidden_dim=16, num_heads=4, num_layers=1)
    )
    batch = collate_priority_samples([PriorityDataset([_instance()])[0]])

    scores = model(batch["features"], batch["mask"])
    loss = masked_mse_loss(scores, batch["labels"], batch["mask"])

    assert scores.shape == (1, 3)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_masked_pairwise_ranking_loss_penalizes_wrong_customer_order():
    labels = torch.tensor([[0.0, 0.5, 1.0]], dtype=torch.float32)
    mask = torch.tensor([[True, True, True]])
    correctly_ordered_scores = torch.tensor([[0.0, 0.3, 0.6]], dtype=torch.float32)
    reversed_scores = torch.tensor([[0.6, 0.3, 0.0]], dtype=torch.float32)

    correct_loss = masked_pairwise_ranking_loss(
        correctly_ordered_scores,
        labels,
        mask,
        margin=0.1,
    )
    reversed_loss = masked_pairwise_ranking_loss(
        reversed_scores,
        labels,
        mask,
        margin=0.1,
    )

    assert correct_loss.item() == pytest.approx(0.0)
    assert reversed_loss.item() == pytest.approx(0.5)


def test_predict_priority_scores_returns_one_score_per_customer():
    torch.manual_seed(7)
    model = PriorityScoringModel(
        PriorityModelConfig(feature_dim=7, hidden_dim=16, num_heads=4, num_layers=1)
    )

    scores = predict_priority_scores(model, _instance(), device=torch.device("cpu"))

    assert len(scores) == 3
    assert all(isinstance(score, float) for score in scores)

"""PyTorch model utilities for customer-priority supervised imitation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import torch
from torch import nn
from torch.utils.data import Dataset

from src.priority_data import customer_feature_rows, normalized_rank_labels
from src.vrp_io import CVRPInstance


@dataclass(frozen=True)
class PriorityModelConfig:
    feature_dim: int = 7
    hidden_dim: int = 128
    num_heads: int = 4
    num_layers: int = 2
    dropout: float = 0.1

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


@dataclass(frozen=True)
class PrioritySample:
    instance_id: int
    features: torch.Tensor
    labels: torch.Tensor


class PriorityDataset(Dataset[PrioritySample]):
    """One item per CVRP instance, with one row per customer."""

    def __init__(
        self,
        instances: Sequence[CVRPInstance],
        require_labels: bool = True,
    ) -> None:
        self.samples: list[PrioritySample] = []
        for instance in instances:
            features = torch.tensor(customer_feature_rows(instance), dtype=torch.float32)
            if require_labels:
                labels = torch.tensor(
                    normalized_rank_labels(instance),
                    dtype=torch.float32,
                )
            else:
                labels = torch.zeros(instance.customer_count, dtype=torch.float32)
            self.samples.append(
                PrioritySample(
                    instance_id=instance.instance_id,
                    features=features,
                    labels=labels,
                )
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> PrioritySample:
        return self.samples[index]


def collate_priority_samples(samples: Sequence[PrioritySample]) -> dict[str, torch.Tensor]:
    if not samples:
        raise ValueError("samples must be non-empty")

    batch_size = len(samples)
    max_customers = max(sample.features.shape[0] for sample in samples)
    feature_dim = samples[0].features.shape[1]
    features = torch.zeros(batch_size, max_customers, feature_dim, dtype=torch.float32)
    labels = torch.zeros(batch_size, max_customers, dtype=torch.float32)
    mask = torch.zeros(batch_size, max_customers, dtype=torch.bool)
    instance_ids = torch.tensor(
        [sample.instance_id for sample in samples],
        dtype=torch.long,
    )

    for row, sample in enumerate(samples):
        customer_count = sample.features.shape[0]
        features[row, :customer_count] = sample.features
        labels[row, :customer_count] = sample.labels
        mask[row, :customer_count] = True

    return {
        "features": features,
        "labels": labels,
        "mask": mask,
        "instance_ids": instance_ids,
    }


class PriorityScoringModel(nn.Module):
    """Scores each customer; lower scores are served earlier."""

    def __init__(self, config: PriorityModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or PriorityModelConfig()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.config.hidden_dim,
            nhead=self.config.num_heads,
            dim_feedforward=self.config.hidden_dim * 4,
            dropout=self.config.dropout,
            activation="gelu",
            batch_first=True,
        )
        self.input_projection = nn.Linear(
            self.config.feature_dim,
            self.config.hidden_dim,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.config.num_layers,
            enable_nested_tensor=False,
        )
        self.score_head = nn.Sequential(
            nn.LayerNorm(self.config.hidden_dim),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3:
            raise ValueError("features must have shape [batch, customers, features]")
        if mask.shape != features.shape[:2]:
            raise ValueError("mask must have shape [batch, customers]")

        hidden = self.input_projection(features)
        hidden = self.encoder(hidden, src_key_padding_mask=~mask)
        scores = self.score_head(hidden).squeeze(-1)
        return scores.masked_fill(~mask, 0.0)


def masked_mse_loss(
    scores: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    if scores.shape != labels.shape or scores.shape != mask.shape:
        raise ValueError("scores, labels, and mask must have the same shape")
    if not mask.any():
        raise ValueError("mask must contain at least one valid customer")
    return ((scores - labels) ** 2)[mask].mean()


def resolve_torch_device(requested_device: str) -> torch.device:
    device = torch.device(requested_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA device {requested_device!r} was requested, but CUDA is not available"
        )
    return device


def load_priority_model_checkpoint(
    checkpoint_path: str,
    device: torch.device | str,
) -> tuple[PriorityScoringModel, dict]:
    resolved_device = torch.device(device)
    checkpoint = torch.load(
        checkpoint_path,
        map_location=resolved_device,
        weights_only=True,
    )
    config = PriorityModelConfig(**checkpoint["config"])
    model = PriorityScoringModel(config).to(resolved_device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def predict_priority_scores(
    model: PriorityScoringModel,
    instance: CVRPInstance,
    device: torch.device | str = torch.device("cpu"),
) -> tuple[float, ...]:
    model.eval()
    resolved_device = torch.device(device)
    features = torch.tensor(customer_feature_rows(instance), dtype=torch.float32)
    features = features.unsqueeze(0).to(resolved_device)
    mask = torch.ones(
        1,
        instance.customer_count,
        dtype=torch.bool,
        device=resolved_device,
    )
    with torch.no_grad():
        scores = model(features, mask).squeeze(0).detach().cpu()
    return tuple(float(score) for score in scores.tolist())

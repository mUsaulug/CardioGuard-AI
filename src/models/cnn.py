"""1D CNN model components for ECG classification."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class ECGCNNConfig:
    """Configuration for ECGCNN."""

    in_channels: int = 12
    num_filters: int = 64
    kernel_size: int = 7
    dropout: float = 0.3


class ECGBackbone(nn.Module):
    """Conv1D CNN backbone returning pooled embeddings."""

    def __init__(self, config: ECGCNNConfig) -> None:
        super().__init__()
        padding = config.kernel_size // 2
        self.features = nn.Sequential(
            nn.Conv1d(config.in_channels, config.num_filters, config.kernel_size, padding=padding),
            nn.BatchNorm1d(config.num_filters),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout),
            nn.Conv1d(config.num_filters, config.num_filters, config.kernel_size, padding=padding),
            nn.BatchNorm1d(config.num_filters),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout),
            nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass expecting shape (batch, channels, timesteps)."""

        features = self.features(x)
        return features.squeeze(-1)


class BinaryHead(nn.Module):
    """Binary classification head returning a single logit per sample."""

    def __init__(self, in_features: int) -> None:
        super().__init__()
        self.classifier = nn.Linear(in_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x).squeeze(-1)


class FiveClassHead(nn.Module):
    """Five-class classification head returning logits for 5 classes."""

    def __init__(self, in_features: int) -> None:
        super().__init__()
        self.classifier = nn.Linear(in_features, 5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


class LocalizationHead(nn.Module):
    """Localization head returning regression logits."""

    def __init__(self, in_features: int, output_dim: int = 2) -> None:
        super().__init__()
        self.regressor = nn.Linear(in_features, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.regressor(x)


class ECGCNN(nn.Module):
    """Conv1D-based ECG classifier returning a single logit."""

    def __init__(self, config: ECGCNNConfig) -> None:
        super().__init__()
        self.backbone = ECGBackbone(config)
        self.head = BinaryHead(config.num_filters)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass expecting shape (batch, channels, timesteps)."""

        embeddings = self.backbone(x)
        return self.head(embeddings)


class CNNEncoder(ECGBackbone):
    """Backward-compatible alias for ECGBackbone."""

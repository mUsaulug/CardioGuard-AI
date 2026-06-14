"""Unit tests for XAI sanity checker edge cases (WP-15)."""

import numpy as np
import pytest
import torch
from torch import nn

from src.xai.sanity import XAISanityChecker
from src.xai.pipeline import _resolve_sanity_gradcam_target


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv1d(1, 2, kernel_size=3)
        self.head = nn.Linear(2, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv(x))
        x = x.mean(dim=2)
        return self.head(x)


def test_compute_similarity_handles_none():
    model = _TinyModel()
    checker = XAISanityChecker(model)
    other = np.linspace(0, 1, 10)
    assert checker._compute_similarity(None, other) == 0.0
    assert checker._compute_similarity(other, None) == 0.0


def test_resolve_sanity_gradcam_target_norm_fallback():
    label, heatmap = _resolve_sanity_gradcam_target(
        "NORM",
        {},
        {"MI": 0.1, "STTC": 0.05, "CD": 0.02, "HYP": 0.01},
    )
    assert label == "MI"
    assert heatmap is None


def test_resolve_sanity_gradcam_target_uses_primary_when_available():
    cam = np.ones((1, 12, 100))
    label, heatmap = _resolve_sanity_gradcam_target(
        "STTC",
        {"STTC": cam},
        {"STTC": 0.8},
    )
    assert label == "STTC"
    assert heatmap is cam

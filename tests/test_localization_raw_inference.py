"""Localization inference uses raw amplitudes (R3-02), matching training."""

import numpy as np
import torch
from torch import nn

from src.utils.signal import apply_superclass_normalization
from src.pipeline.inference.run_inference_localization import predict


class _CaptureLocalizationModel(nn.Module):
    def __init__(self, timesteps: int):
        super().__init__()
        self.timesteps = timesteps
        self.captured: np.ndarray | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.captured = x.detach().cpu().numpy()
        return torch.zeros(x.shape[0], 5)


def test_localization_standalone_uses_raw_not_zscore():
    timesteps = 5000
    model = _CaptureLocalizationModel(timesteps)
    model.eval()

    rng = np.random.default_rng(0)
    raw = rng.standard_normal((12, timesteps)).astype(np.float32) * 25.0 + 10.0
    predict(raw, model, torch.device("cpu"), explain=False)

    assert model.captured is not None
    fed = model.captured[0]
    normed = apply_superclass_normalization(raw)
    assert np.allclose(fed, raw, atol=1e-4)
    assert not np.allclose(fed, normed, atol=0.1)

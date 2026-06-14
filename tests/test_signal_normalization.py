"""Tests for superclass ECG normalization at inference (WP-01)."""
from pathlib import Path

import numpy as np
import pytest

from src.utils.signal import (
    apply_superclass_normalization,
    ensure_channel_first,
    load_superclass_norm_stats,
    normalize_signal,
)

PROJECT_ROOT = Path(__file__).parent.parent


def test_load_superclass_norm_stats_from_json_fallback():
    """Repo ships feature_config JSON when npz is absent."""
    mean, std = load_superclass_norm_stats()
    assert mean.shape == (12,)
    assert std.shape == (12,)
    assert np.all(std > 0)


def test_normalization_changes_signal():
    raw = np.random.randn(12, 1000).astype(np.float32)
    normed = apply_superclass_normalization(raw)
    assert normed.shape == (12, 1000)
    assert not np.allclose(raw, normed)


def test_normalized_leads_near_zero_mean_unit_scale():
    """After norm, channel means should be ~0 relative to training stats."""
    mean, std = load_superclass_norm_stats()
    rng = np.random.default_rng(42)
    raw = mean[:, None] + std[:, None] * rng.standard_normal((12, 5000))
    normed = normalize_signal(raw, mean, std)
    assert np.allclose(normed.mean(axis=1), 0, atol=0.06)
    assert np.allclose(normed.std(axis=1), 1, atol=0.05)


def test_missing_stats_raises(tmp_path):
    load_superclass_norm_stats.cache_clear()
    with pytest.raises(FileNotFoundError):
        load_superclass_norm_stats(
            stats_npz=str(tmp_path / "missing.npz"),
            stats_json=str(tmp_path / "missing.json"),
        )
    load_superclass_norm_stats.cache_clear()


def test_predict_superclass_still_valid(client):
    """API happy path after normalization wired in core_predict."""
    sample_path = PROJECT_ROOT / "sample.npy"
    if not sample_path.exists():
        pytest.skip("sample.npy not found")
    with open(sample_path, "rb") as f:
        response = client.post(
            "/predict/superclass?ensemble_weight=0.15&explain=false",
            files={"file": ("sample.npy", f, "application/octet-stream")},
        )
    assert response.status_code == 200
    probs = response.json()["probabilities"]
    for cls in ["MI", "STTC", "CD", "HYP", "NORM"]:
        assert cls in probs
        assert 0.0 <= probs[cls] <= 1.0

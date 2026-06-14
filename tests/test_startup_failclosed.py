"""Fail-closed startup tests (WP-08)."""

from pathlib import Path

import pytest

from src.backend.main import AppState


def test_load_models_raises_without_superclass_checkpoint(tmp_path):
    """Superclass checkpoint is required — missing file must abort load."""
    state = AppState()
    missing = tmp_path / "missing_superclass.pt"
    with pytest.raises(RuntimeError, match="superclass checkpoint"):
        state.load_models(superclass_checkpoint=missing)


def test_load_models_raises_without_xgb(tmp_path):
    """XGB OVR bundle is required for ensemble inference."""
    state = AppState()
    with pytest.raises(RuntimeError, match="Required XGB"):
        state.load_models(
            superclass_checkpoint=Path("checkpoints/ecgcnn_superclass.pt"),
            xgb_dir=tmp_path / "empty_xgb",
        )

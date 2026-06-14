"""Tests for shared ECG I/O (R3-06)."""

import io

import numpy as np
import pytest

from src.utils.signal_io import load_ecg_array_from_path, load_ecg_from_bytes


def test_load_ecg_from_bytes_npy():
    raw = np.random.randn(12, 5000).astype(np.float32)
    bio = io.BytesIO()
    np.save(bio, raw)
    buf = bio.getvalue()

    signal, meta = load_ecg_from_bytes(buf, "sample.npy", validate=True)
    assert signal.shape == (12, 5000)
    assert signal.ndim == 2


def test_load_ecg_array_from_path_npy(tmp_path):
    raw = np.random.randn(12, 32).astype(np.float32)
    path = tmp_path / "ecg.npy"
    np.save(path, raw)

    loaded = load_ecg_array_from_path(path)
    assert np.allclose(loaded, raw)


def test_load_ecg_from_bytes_unsupported():
    with pytest.raises(ValueError, match="Unsupported"):
        load_ecg_from_bytes(b"data", "ecg.csv", validate=False)

"""ECG input validation tests (WP-03)."""

import io

import numpy as np
import pytest

from src.utils.signal import (
    PTBXL_EXPECTED_TIMESTEPS,
    validate_ecg_signal,
)


def _valid_signal() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.normal(0, 0.5, (12, PTBXL_EXPECTED_TIMESTEPS)).astype(np.float32)


class TestValidateEcgSignal:
    def test_valid_sample_shape(self):
        arr = _valid_signal()
        out, meta = validate_ecg_signal(arr)
        assert out.shape == (12, PTBXL_EXPECTED_TIMESTEPS)
        assert meta["shape"] == [12, PTBXL_EXPECTED_TIMESTEPS]
        assert meta["validation"]["valid"] is True
        assert meta["validation"]["ptbxl_standard"] is True

    def test_transposes_timesteps_first(self):
        arr = _valid_signal().T  # (1000, 12)
        out, meta = validate_ecg_signal(arr)
        assert out.shape == (12, PTBXL_EXPECTED_TIMESTEPS)
        assert meta["shape"] == [12, PTBXL_EXPECTED_TIMESTEPS]

    def test_rejects_wrong_lead_count(self):
        arr = np.zeros((8, 1000), dtype=np.float32)
        with pytest.raises(ValueError, match="12 leads"):
            validate_ecg_signal(arr)

    def test_rejects_nan(self):
        arr = _valid_signal()
        arr[0, 0] = np.nan
        with pytest.raises(ValueError, match="non-finite"):
            validate_ecg_signal(arr)

    def test_rejects_flat_signal(self):
        arr = np.zeros((12, 1000), dtype=np.float32)
        with pytest.raises(ValueError, match="flat"):
            validate_ecg_signal(arr)

    def test_flags_non_standard_timesteps(self):
        arr = np.random.randn(12, 500).astype(np.float32)
        _, meta = validate_ecg_signal(arr)
        assert meta["validation"]["ptbxl_standard"] is False
        assert meta["quality_flags"] is not None


def test_api_rejects_invalid_shape(client):
    bad = np.zeros((8, 1000), dtype=np.float32)
    buf = io.BytesIO()
    np.save(buf, bad)
    buf.seek(0)
    response = client.post(
        "/predict/superclass?explain=false",
        files={"file": ("bad.npy", buf.read(), "application/octet-stream")},
    )
    assert response.status_code == 400
    assert "12 leads" in response.json()["detail"]


def test_api_rejects_nan_signal(client):
    bad = _valid_signal()
    bad[1, 10] = np.nan
    buf = io.BytesIO()
    np.save(buf, bad)
    buf.seek(0)
    response = client.post(
        "/predict/superclass?explain=false",
        files={"file": ("nan.npy", buf.read(), "application/octet-stream")},
    )
    assert response.status_code == 400
    assert "non-finite" in response.json()["detail"]


def test_api_sample_npy_reports_real_shape(client):
    from pathlib import Path

    sample_path = Path(__file__).resolve().parents[1] / "sample.npy"
    if not sample_path.exists():
        pytest.skip("sample.npy not found")

    with open(sample_path, "rb") as f:
        response = client.post(
            "/predict/superclass?explain=false&full=true",
            files={"file": ("sample.npy", f, "application/octet-stream")},
        )
    assert response.status_code == 200
    airesult = response.json()["airesult"]
    assert airesult["input"]["shape"] == [12, 1000]
    assert airesult["input"]["validation"]["ptbxl_standard"] is True

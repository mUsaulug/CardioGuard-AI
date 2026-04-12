"""
API Endpoint Tests for CardioGuard-AI.

Tests all REST endpoints using FastAPI TestClient.
"""
import pytest
from pathlib import Path
from fastapi.testclient import TestClient

# Import app - this triggers model loading on first use
from src.backend.main import app

client = TestClient(app)

PROJECT_ROOT = Path(__file__).parent.parent


# --- Health & Ready ---

def test_health_returns_200():
    """GET /health should return 200 with status=healthy."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "timestamp" in data


def test_ready_returns_model_status():
    """GET /ready should return model loading status."""
    response = client.get("/ready")
    assert response.status_code == 200
    data = response.json()
    assert "ready" in data
    assert "models_loaded" in data
    assert isinstance(data["models_loaded"], dict)
    # Check expected model keys
    for key in ["superclass", "localization", "xgb", "thresholds"]:
        assert key in data["models_loaded"]


# --- Superclass Prediction ---

def test_predict_superclass_with_npy():
    """POST /predict/superclass with .npy file should return valid prediction."""
    sample_path = PROJECT_ROOT / "sample.npy"
    if not sample_path.exists():
        pytest.skip("sample.npy not found")

    with open(sample_path, "rb") as f:
        response = client.post(
            "/predict/superclass?ensemble_weight=0.5&explain=false",
            files={"file": ("sample.npy", f, "application/octet-stream")},
        )

    assert response.status_code == 200
    data = response.json()

    # Check required fields
    assert data["mode"] == "multilabel-superclass"
    assert "probabilities" in data
    assert "predicted_labels" in data
    assert "primary" in data
    assert "sources" in data
    assert "versions" in data

    # Check probabilities have all classes
    probs = data["probabilities"]
    for cls in ["MI", "STTC", "CD", "HYP", "NORM"]:
        assert cls in probs
        assert 0.0 <= probs[cls] <= 1.0

    # Check primary
    assert "label" in data["primary"]
    assert "confidence" in data["primary"]
    assert "rule" in data["primary"]


def test_predict_superclass_with_npz():
    """POST /predict/superclass with .npz file should return valid prediction."""
    sample_path = PROJECT_ROOT / "test_mi_sample.npz"
    if not sample_path.exists():
        pytest.skip("test_mi_sample.npz not found")

    with open(sample_path, "rb") as f:
        response = client.post(
            "/predict/superclass?ensemble_weight=0.5&explain=false",
            files={"file": ("test_mi_sample.npz", f, "application/octet-stream")},
        )

    assert response.status_code == 200
    data = response.json()
    assert "probabilities" in data
    assert "predicted_labels" in data


def test_predict_superclass_consistency_present():
    """Response should include consistency info when binary model loaded."""
    sample_path = PROJECT_ROOT / "sample.npy"
    if not sample_path.exists():
        pytest.skip("sample.npy not found")

    with open(sample_path, "rb") as f:
        response = client.post(
            "/predict/superclass?explain=false",
            files={"file": ("sample.npy", f, "application/octet-stream")},
        )

    assert response.status_code == 200
    data = response.json()
    # Consistency should be present if binary model is loaded
    if data.get("consistency") is not None:
        c = data["consistency"]
        assert "agreement" in c
        assert "triage_level" in c
        assert "superclass_mi_prob" in c
        assert "binary_mi_prob" in c


def test_predict_superclass_invalid_file():
    """POST with non-ECG file should return 400."""
    response = client.post(
        "/predict/superclass",
        files={"file": ("test.txt", b"not an ecg file", "text/plain")},
    )
    assert response.status_code == 400


def test_predict_superclass_ensemble_weight_bounds():
    """Ensemble weight outside 0-1 should return 422."""
    sample_path = PROJECT_ROOT / "sample.npy"
    if not sample_path.exists():
        pytest.skip("sample.npy not found")

    with open(sample_path, "rb") as f:
        response = client.post(
            "/predict/superclass?ensemble_weight=1.5",
            files={"file": ("sample.npy", f, "application/octet-stream")},
        )
    assert response.status_code == 422


# --- MI Localization ---

def test_predict_localization():
    """POST /predict/mi-localization should return valid response."""
    sample_path = PROJECT_ROOT / "sample.npy"
    if not sample_path.exists():
        pytest.skip("sample.npy not found")

    with open(sample_path, "rb") as f:
        response = client.post(
            "/predict/mi-localization?threshold=0.5&explain=false",
            files={"file": ("sample.npy", f, "application/octet-stream")},
        )

    assert response.status_code == 200
    data = response.json()
    assert "mi_detected" in data
    assert "regions" in data
    assert "probabilities" in data
    assert isinstance(data["regions"], list)


# --- Artifact Serving ---

def test_path_traversal_blocked():
    """Path traversal attempts should return 400."""
    response = client.get("/runs/test_run/../../etc/passwd")
    assert response.status_code == 400


def test_invalid_run_id_rejected():
    """Invalid run_id format should return 400."""
    response = client.get("/runs/../../bad/file.txt")
    assert response.status_code == 400


def test_nonexistent_artifact_returns_404():
    """Valid format but nonexistent artifact should return 404."""
    response = client.get("/runs/nonexistent_run_id/visuals/test.png")
    assert response.status_code == 404

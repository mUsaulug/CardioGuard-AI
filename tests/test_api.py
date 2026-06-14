"""
API Endpoint Tests for CardioGuard-AI.

Tests all REST endpoints using FastAPI TestClient.
"""
import pytest
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent


# --- Health & Ready ---

def test_health_returns_200(client):
    """GET /health should return 200 with status=healthy."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "timestamp" in data


def test_ready_returns_model_status(client):
    """GET /ready should return model loading status."""
    response = client.get("/ready")
    assert response.status_code == 200
    data = response.json()
    assert "ready" in data
    assert "models_loaded" in data
    assert "degraded" in data
    assert "degraded_models" in data
    assert isinstance(data["models_loaded"], dict)
    # Check expected model keys
    for key in ["superclass", "localization", "xgb", "thresholds"]:
        assert key in data["models_loaded"]


def test_ready_reports_degraded(client):
    """GET /ready should expose degraded mode when optional models are missing."""
    import src.backend.main as main_mod

    main_mod.state.degraded = True
    main_mod.state.degraded_models = ["binary"]
    response = client.get("/ready")
    assert response.status_code == 200
    data = response.json()
    assert data["degraded"] is True
    assert "binary" in data["degraded_models"]


# --- Superclass Prediction ---

def test_predict_superclass_with_npy(client):
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


def test_predict_superclass_default_ensemble_weight(client):
    """Default ensemble uses CNN weight from thresholds artifact (0.15)."""
    from src.config import get_ensemble_cnn_weight

    sample_path = PROJECT_ROOT / "sample.npy"
    if not sample_path.exists():
        pytest.skip("sample.npy not found")

    w = get_ensemble_cnn_weight()
    with open(sample_path, "rb") as f:
        response = client.post(
            "/predict/superclass?explain=false",
            files={"file": ("sample.npy", f, "application/octet-stream")},
        )

    assert response.status_code == 200
    data = response.json()
    sources = data["sources"]
    assert sources.get("xgb") is not None
    for cls in ["MI", "STTC", "CD", "HYP"]:
        cnn_p = sources["cnn"][cls]
        xgb_p = sources["xgb"][cls]
        ens_p = sources["ensemble"][cls]
        expected = w * cnn_p + (1.0 - w) * xgb_p
        assert abs(ens_p - expected) < 1e-4, cls


def test_predict_superclass_full_airesult_reports_ensemble_weight(client):
    """full=true AIResult must echo the requested ensemble_weight."""
    sample_path = PROJECT_ROOT / "sample.npy"
    if not sample_path.exists():
        pytest.skip("sample.npy not found")

    custom_w = 0.5
    with open(sample_path, "rb") as f:
        response = client.post(
            f"/predict/superclass?ensemble_weight={custom_w}&explain=false&full=true",
            files={"file": ("sample.npy", f, "application/octet-stream")},
        )

    assert response.status_code == 200
    data = response.json()
    airesult = data.get("airesult")
    assert airesult is not None
    assert airesult["versions"]["ensemble_best_alpha"] == custom_w


def test_predict_superclass_with_npz(client):
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


def test_predict_superclass_consistency_present(client):
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


def test_predict_superclass_v12_fields_without_explain(client):
    """v1.2 additive fields: glossary always present; explanation null without explain."""
    sample_path = PROJECT_ROOT / "sample.npy"
    if not sample_path.exists():
        pytest.skip("sample.npy not found")

    with open(sample_path, "rb") as f:
        response = client.post(
            "/predict/superclass?explain=false&full=false",
            files={"file": ("sample.npy", f, "application/octet-stream")},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["versions"]["api_version"] == "1.2.0"
    assert "glossary" in data
    assert isinstance(data["glossary"], dict)
    assert "MI" in data["glossary"]
    assert data.get("explanation") is None
    assert "localization" in data
    assert data.get("airesult") is None


def test_predict_superclass_v12_with_explain_and_full(client):
    """v1.2: explanation + localization + airesult when explain=true and full=true."""
    sample_path = PROJECT_ROOT / "sample.npy"
    if not sample_path.exists():
        pytest.skip("sample.npy not found")

    with open(sample_path, "rb") as f:
        response = client.post(
            "/predict/superclass?explain=true&sanity_check=false&full=true",
            files={"file": ("sample.npy", f, "application/octet-stream")},
        )

    assert response.status_code == 200
    data = response.json()

    assert data["versions"]["api_version"] == "1.2.0"
    assert data.get("explanation") is not None
    exp = data["explanation"]
    for key in [
        "narrative",
        "coherence_score",
        "sanity_passed",
        "gradcam_summary",
        "shap_summary",
        "dominant_source",
        "conflicts",
    ]:
        assert key in exp

    assert data.get("airesult") is not None
    airesult = data["airesult"]
    assert "predictions" in airesult
    assert "triage" in airesult
    assert "identity" in airesult

    if data.get("localization") is not None:
        loc = data["localization"]
        assert "labels_tr" in loc
        assert "probabilities" in loc


def test_explain_produces_servable_png_and_calibrated_coherence(client):
    """Regression: explain path must emit a real report PNG (matplotlib Agg under
    the server threadpool) and a calibrated coherence score with readable SHAP names."""
    sample_path = PROJECT_ROOT / "sample.npy"
    if not sample_path.exists():
        pytest.skip("sample.npy not found")

    with open(sample_path, "rb") as f:
        response = client.post(
            "/predict/superclass?explain=true&sanity_check=true&full=false",
            files={"file": ("sample.npy", f, "application/octet-stream")},
        )
    assert response.status_code == 200
    data = response.json()

    # 1. A report PNG artifact must exist (this silently broke before the Agg fix).
    xai = data["xai"]
    assert xai and xai["enabled"]
    pngs = [a for a in xai["artifacts"] if a["mime"] == "image/png"]
    assert pngs, "explain path produced no PNG artifact"

    # 2. The PNG URL must actually be servable as an image.
    served = client.get(pngs[0]["url"])
    assert served.status_code == 200
    assert served.headers["content-type"] == "image/png"
    assert len(served.content) > 1000

    # 3. Coherence must be calibrated, never a synthetic perfect 1.0.
    coherence = data["explanation"]["coherence_score"]
    assert 0.0 < coherence < 1.0

    # 4. SHAP summary must use readable labels, not raw "feature_13".
    import re

    assert not re.search(r"\bfeature_\d+\b", data["explanation"]["shap_summary"])


def test_predict_superclass_invalid_file(client):
    """POST with non-ECG file should return 400."""
    response = client.post(
        "/predict/superclass",
        files={"file": ("test.txt", b"not an ecg file", "text/plain")},
    )
    assert response.status_code == 400


def test_predict_superclass_ensemble_weight_bounds(client):
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

def test_predict_localization(client):
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

def test_path_traversal_blocked(client):
    """Path traversal in file_path should return 400."""
    response = client.get("/runs/test_run/visuals/..%2F..%2F..%2Fetc%2Fpasswd")
    assert response.status_code == 400


def test_invalid_run_id_rejected(client):
    """Invalid run_id format should return 400."""
    response = client.get("/runs/not valid!/file.txt")
    assert response.status_code == 400


def test_nonexistent_artifact_returns_404(client):
    """Valid format but nonexistent artifact should return 404."""
    response = client.get("/runs/nonexistent_run_id/visuals/test.png")
    assert response.status_code == 404


# --- Client debug log ---

def test_client_debug_log_disabled_returns_404(client, monkeypatch):
    """Debug endpoints must be env-gated (off by default in production)."""
    monkeypatch.setenv("ENABLE_DEBUG_ENDPOINTS", "0")
    post = client.post(
        "/debug/client-log",
        json={"ts": "2026-06-09T12:00:00Z", "message": "nope"},
    )
    assert post.status_code == 404
    get = client.get("/debug/client-log")
    assert get.status_code == 404


def test_client_debug_log_roundtrip(client, tmp_path, monkeypatch):
    """POST /debug/client-log append + GET tail for browser QA."""
    import src.backend.main as main_mod

    log_file = tmp_path / "client-events.jsonl"
    monkeypatch.setattr(main_mod, "CLIENT_LOG_FILE", log_file)

    payload = {
        "ts": "2026-06-09T12:00:00Z",
        "level": "info",
        "category": "llm",
        "message": "test event",
        "meta": {"model": "openrouter/free"},
    }
    post = client.post("/debug/client-log", json=payload)
    assert post.status_code == 200
    assert post.json()["ok"] is True

    get = client.get("/debug/client-log?tail=10")
    assert get.status_code == 200
    data = get.json()
    assert data["count"] == 1
    assert data["events"][0]["message"] == "test event"


def test_llm_status_endpoint(client):
    r = client.get("/api/llm/status")
    assert r.status_code == 200
    data = r.json()
    assert data["proxy_enabled"] is True
    assert isinstance(data["default_models"], list)
    assert len(data["default_models"]) >= 1


def test_llm_chat_requires_key_when_locked_down(client, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("ALLOW_CLIENT_LLM_KEY", "0")
    r = client.post(
        "/api/llm/chat",
        json={
            "model": "openrouter/free",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
        },
    )
    assert r.status_code == 401


def test_llm_chat_proxies_with_client_key_header(client, monkeypatch):
    import httpx

    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("ALLOW_CLIENT_LLM_KEY", "1")

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def post(self, *args, **kwargs):
            assert kwargs.get("headers", {}).get("Authorization") == "Bearer sk-or-test"
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": "Merhaba"}}]},
            )

    monkeypatch.setattr("src.backend.llm_proxy.httpx.AsyncClient", FakeAsyncClient)

    r = client.post(
        "/api/llm/chat",
        headers={"X-OpenRouter-Key": "sk-or-test"},
        json={
            "model": "openrouter/free",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
        },
    )
    assert r.status_code == 200
    assert r.json()["choices"][0]["message"]["content"] == "Merhaba"

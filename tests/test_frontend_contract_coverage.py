"""Contract coverage: backend API can supply full AnalysisContext shape."""

from pathlib import Path

import pytest
from src.contracts.frontend_context import (
    build_analysis_context_from_api,
    validate_analysis_context,
)

PROJECT_ROOT = Path(__file__).parent.parent


def test_analysis_context_buildable_from_api_with_explain(client):
    sample_path = PROJECT_ROOT / "sample.npy"
    if not sample_path.exists():
        pytest.skip("sample.npy not found")

    with open(sample_path, "rb") as f:
        response = client.post(
            "/predict/superclass?explain=true&full=true",
            files={"file": ("sample.npy", f, "application/octet-stream")},
        )

    assert response.status_code == 200
    api_data = response.json()

    ctx = build_analysis_context_from_api(api_data, file_name="sample.npy")
    errors = validate_analysis_context(ctx)
    assert errors == [], f"AnalysisContext validation failed: {errors}"

    assert ctx["primary"]["label"] in ["MI", "STTC", "CD", "HYP", "NORM"]
    assert isinstance(ctx["predictedLabels"], list)
    assert ctx["glossary"].get("MI")
    assert ctx["xai"] is not None
    assert ctx["xai"]["gradcam_summary"]
    assert ctx["xai"]["shap_summary"]


def test_analysis_context_buildable_without_explain(client):
    sample_path = PROJECT_ROOT / "sample.npy"
    if not sample_path.exists():
        pytest.skip("sample.npy not found")

    with open(sample_path, "rb") as f:
        response = client.post(
            "/predict/superclass?explain=false",
            files={"file": ("sample.npy", f, "application/octet-stream")},
        )

    assert response.status_code == 200
    api_data = response.json()
    ctx = build_analysis_context_from_api(api_data, file_name="sample.npy")

    # Without explain, xai may be null — still valid for partial context
    assert ctx["glossary"]
    assert "probabilities" in ctx
    assert "NORM" in ctx["probabilities"]

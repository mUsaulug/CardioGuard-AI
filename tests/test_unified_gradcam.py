"""Grad-CAM temporal summary regression tests."""

import numpy as np

from src.xai.unified import UnifiedExplainer


def test_visual_evidence_2d_cam_no_absurd_seconds():
    explainer = UnifiedExplainer()
    # Simulates GradCAM output (1, 1000) — old code produced ~3690s bug
    gradcam = {
        "MI": np.random.rand(1, 1000).astype(np.float32),
        "STTC": np.random.rand(1, 1000).astype(np.float32),
    }
    visual = explainer._extract_visual_evidence(gradcam)
    assert len(visual) == 2
    for line in visual:
        assert "3690" not in line
        assert "s civarında" in line or "civarında" in line


def test_narrative_uses_primary_label_not_max_prob():
    explainer = UnifiedExplainer()
    probs = {"MI": 0.4, "STTC": 0.62, "CD": 0.1, "HYP": 0.05}
    text = explainer._generate_narrative(
        probs,
        visual=["MI: test"],
        feature=["MI: feat"],
        source="XGBoost (Feature)",
        conflicts=[],
        primary_label="MI",
    )
    assert "MI" in text
    assert "%40.0" in text


def _make_shap(top_importance):
    return {
        "MI": {
            "top_features": [
                {"feature": "CNN gömme boyutu 3", "importance": top_importance},
                {"feature": "CNN gömme boyutu 7", "importance": 0.05},
                {"feature": "CNN gömme boyutu 9", "importance": 0.02},
            ]
        }
    }


def test_coherence_never_perfect_one():
    """Calibrated coherence must never hit a synthetic 1.0."""
    explainer = UnifiedExplainer()
    # Strongly peaked Grad-CAM + dominant SHAP would max the old metric to 1.0.
    cam = np.zeros((1, 1000), dtype=np.float32)
    cam[0, 500] = 1.0
    gradcam = {"MI": cam}
    score, _ = explainer._analyze_coherence(gradcam, _make_shap(0.9))
    assert 0.0 < score <= 0.97
    assert score != 1.0


def test_coherence_lower_when_classes_disagree():
    explainer = UnifiedExplainer()
    cam = np.random.rand(1, 1000).astype(np.float32)
    # Grad-CAM highlights STTC but SHAP only has MI -> zero class agreement.
    agree, _ = explainer._analyze_coherence({"MI": cam}, _make_shap(0.9))
    disagree, conflicts = explainer._analyze_coherence({"STTC": cam}, _make_shap(0.9))
    assert disagree < agree
    assert any("farklı" in c for c in conflicts)


def test_coherence_insufficient_data_is_modest():
    explainer = UnifiedExplainer()
    score, _ = explainer._analyze_coherence({}, {})
    assert score <= 0.5

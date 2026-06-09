"""Unit tests for explanation summary contract."""

from src.contracts.explanation_summary import summarize_explanation


def test_summarize_explanation_none():
    assert summarize_explanation(None) is None
    assert summarize_explanation({}) is None


def test_summarize_explanation_full():
    explanation = {
        "narrative": "MI odaklı açıklama.",
        "coherence_score": 0.82,
        "dominant_source": "Grad-CAM",
        "visual_summary": ["V2-V4 bölgesinde yüksek aktivasyon."],
        "feature_summary": ["embedding_12: +0.41", "embedding_7: +0.22"],
        "conflicts": ["CNN ve XGB farklı STTC eşiği öneriyor."],
        "sanity_check": {
            "overall": {
                "status": "RELIABLE",
                "passed_checks": 4,
                "total_checks": 4,
            }
        },
    }
    result = summarize_explanation(explanation)
    assert result is not None
    assert result["narrative"] == "MI odaklı açıklama."
    assert result["coherence_score"] == 0.82
    assert result["sanity_passed"] is True
    assert "Grad-CAM" in result["gradcam_summary"]
    assert "embedding_12" in result["shap_summary"]
    assert len(result["conflicts"]) == 1


def test_summarize_explanation_without_sanity():
    explanation = {
        "narrative": "Özet",
        "coherence_score": 0.5,
        "visual_summary": [],
        "feature_summary": [],
    }
    result = summarize_explanation(explanation)
    assert result is not None
    assert result["sanity_passed"] is None
    assert "mevcut değil" in result["gradcam_summary"].lower() or "Grad-CAM" in result["gradcam_summary"]


def test_summarize_explanation_unreliable_sanity():
    explanation = {
        "narrative": "Özet",
        "coherence_score": 0.3,
        "visual_summary": ["x"],
        "feature_summary": ["y"],
        "sanity_check": {
            "overall": {
                "status": "UNRELIABLE",
                "passed_checks": 1,
                "total_checks": 4,
            }
        },
    }
    result = summarize_explanation(explanation)
    assert result is not None
    assert result["sanity_passed"] is False

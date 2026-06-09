"""
Explanation summary helpers for API v1.2.

Maps pipeline explanation dict to frontend-friendly string summaries.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def _extract_sanity_passed(sanity_check: Optional[Dict[str, Any]]) -> Optional[bool]:
    """Map sanity_check.overall to a tri-state bool (None = not run)."""
    if not sanity_check:
        return None

    overall = sanity_check.get("overall")
    if not isinstance(overall, dict):
        return None

    status = overall.get("status")
    if status in ("RELIABLE", "ACCEPTABLE"):
        return True
    if status == "UNRELIABLE":
        return False

    passed_checks = overall.get("passed_checks")
    total_checks = overall.get("total_checks", 4)
    if isinstance(passed_checks, int) and isinstance(total_checks, int) and total_checks > 0:
        return passed_checks >= (total_checks // 2)

    return None


def _join_summary(parts: List[str], fallback: str) -> str:
    cleaned = [p.strip() for p in parts if p and str(p).strip()]
    return " ".join(cleaned) if cleaned else fallback


def summarize_explanation(explanation_dict: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Convert pipeline explanation dict to API ExplanationInfo fields.

    Returns None when explanation_dict is None/empty.
    """
    if not explanation_dict:
        return None

    visual_summary = explanation_dict.get("visual_summary") or []
    if not isinstance(visual_summary, list):
        visual_summary = [str(visual_summary)]

    feature_summary = explanation_dict.get("feature_summary") or []
    if not isinstance(feature_summary, list):
        feature_summary = [str(feature_summary)]

    dominant_source = str(explanation_dict.get("dominant_source") or "").strip()

    gradcam_parts: List[str] = []
    if dominant_source:
        gradcam_parts.append(f"Kaynak: {dominant_source}.")
    gradcam_parts.extend(str(v) for v in visual_summary)

    gradcam_summary = _join_summary(
        gradcam_parts,
        "Grad-CAM analizi mevcut değil.",
    )
    shap_summary = _join_summary(
        [str(f) for f in feature_summary],
        "SHAP analizi mevcut değil.",
    )

    conflicts = explanation_dict.get("conflicts") or []
    if not isinstance(conflicts, list):
        conflicts = [str(conflicts)]

    coherence = explanation_dict.get("coherence_score", 0.5)
    try:
        coherence_score = float(coherence)
    except (TypeError, ValueError):
        coherence_score = 0.5

    return {
        "narrative": str(explanation_dict.get("narrative") or ""),
        "coherence_score": coherence_score,
        "sanity_passed": _extract_sanity_passed(explanation_dict.get("sanity_check")),
        "gradcam_summary": gradcam_summary,
        "shap_summary": shap_summary,
        "dominant_source": dominant_source,
        "conflicts": [str(c) for c in conflicts],
    }

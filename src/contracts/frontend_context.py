"""
Build frontend AnalysisContext-shaped dict from API v1.2 response.

Used for contract coverage tests and optional frontend mapper reference.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


REQUIRED_CONTEXT_KEYS = {
    "sessionId",
    "fileName",
    "timestamp",
    "primary",
    "predictedLabels",
    "probabilities",
    "thresholds",
    "sources",
    "consistency",
    "localization",
    "xai",
    "glossary",
}

REQUIRED_PRIMARY_KEYS = {"label", "confidence", "rule"}
REQUIRED_XAI_KEYS = {
    "narrative",
    "coherence_score",
    "sanity_passed",
    "gradcam_summary",
    "shap_summary",
}


def build_analysis_context_from_api(
    api_response: Dict[str, Any],
    file_name: str = "sample.npy",
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Map SuperclassPredictionResponse v1.2 JSON to AnalysisContext shape."""
    explanation = api_response.get("explanation")
    localization = api_response.get("localization")

    xai = None
    if explanation:
        xai = {
            "narrative": explanation.get("narrative", ""),
            "coherence_score": explanation.get("coherence_score", 0.5),
            "sanity_passed": explanation.get("sanity_passed"),
            "gradcam_summary": explanation.get("gradcam_summary", ""),
            "shap_summary": explanation.get("shap_summary", ""),
        }

    loc = None
    if localization:
        loc = {
            "regions": localization.get("regions", []),
            "probabilities": localization.get("probabilities", {}),
            "labels_tr": localization.get("labels_tr", {}),
        }

    return {
        "sessionId": session_id or str(uuid.uuid4()),
        "fileName": file_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "primary": api_response.get("primary", {}),
        "predictedLabels": api_response.get("predicted_labels", []),
        "probabilities": api_response.get("probabilities", {}),
        "thresholds": api_response.get("thresholds", {}),
        "sources": api_response.get("sources", {}),
        "consistency": api_response.get("consistency"),
        "localization": loc,
        "xai": xai,
        "glossary": api_response.get("glossary", {}),
    }


def validate_analysis_context(ctx: Dict[str, Any]) -> List[str]:
    """Return list of validation errors (empty = valid)."""
    errors: List[str] = []

    missing = REQUIRED_CONTEXT_KEYS - set(ctx.keys())
    if missing:
        errors.append(f"Missing top-level keys: {sorted(missing)}")

    primary = ctx.get("primary")
    if not isinstance(primary, dict):
        errors.append("primary must be a dict")
    else:
        missing_primary = REQUIRED_PRIMARY_KEYS - set(primary.keys())
        if missing_primary:
            errors.append(f"primary missing keys: {sorted(missing_primary)}")

    probs = ctx.get("probabilities")
    if isinstance(probs, dict):
        for cls in ["MI", "STTC", "CD", "HYP", "NORM"]:
            if cls not in probs:
                errors.append(f"probabilities missing {cls}")

    xai = ctx.get("xai")
    if xai is not None:
        if not isinstance(xai, dict):
            errors.append("xai must be dict or null")
        else:
            missing_xai = REQUIRED_XAI_KEYS - set(xai.keys())
            if missing_xai:
                errors.append(f"xai missing keys: {sorted(missing_xai)}")

    glossary = ctx.get("glossary")
    if not isinstance(glossary, dict) or not glossary:
        errors.append("glossary must be non-empty dict")

    return errors

"""
API response mapping helpers — keeps main.py thin.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.config import (
    GLOSSARY,
    MI_LOCALIZATION_LABELS,
    MI_LOCALIZATION_LABELS_TR,
)
from src.contracts.explanation_summary import summarize_explanation


def map_localization_inline(
    mi_localization: Optional[Dict[str, Any]],
    mi_detected: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    Map pipeline mi_localization dict to LocalizationInline shape.

    Pipeline format:
        {AMI: 0.1, ASMI: 0.2, ..., predicted_regions: [...]}
    """
    if not mi_localization:
        return None

    probabilities = {
        label: float(mi_localization[label])
        for label in MI_LOCALIZATION_LABELS
        if label in mi_localization
    }
    regions = mi_localization.get("predicted_regions") or []
    if not isinstance(regions, list):
        regions = []

    detected = mi_detected or len(regions) > 0
    if not detected and not probabilities:
        return None

    return {
        "mi_detected": detected,
        "regions": [str(r) for r in regions],
        "probabilities": probabilities,
        "labels": list(MI_LOCALIZATION_LABELS),
        "labels_tr": dict(MI_LOCALIZATION_LABELS_TR),
    }


def map_explanation_info(
    explanation_dict: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Map pipeline explanation to API ExplanationInfo fields."""
    return summarize_explanation(explanation_dict)


def build_glossary_subset() -> Dict[str, str]:
    """Return canonical glossary for frontend AnalysisContext."""
    return dict(GLOSSARY)

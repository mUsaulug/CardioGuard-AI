"""
Summary utilities for visual explainability outputs.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from src.xai.visualize import LEAD_NAMES


def compute_lead_attention(
    cam: np.ndarray,
    signal: Optional[np.ndarray] = None,
    lead_names: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Compute normalized lead attention and return the top lead summary.
    """
    if lead_names is None:
        lead_names = LEAD_NAMES

    n_leads = len(lead_names)
    if signal is not None:
        if signal.shape[0] != 12 and signal.shape[1] == 12:
            signal = signal.T
        signal_magnitude = np.abs(signal)
        weighted_attention = signal_magnitude * cam[np.newaxis, :]
        lead_attention = weighted_attention.mean(axis=1)
    else:
        lead_attention = np.full(n_leads, cam.mean())

    lead_attention = lead_attention / (lead_attention.sum() + 1e-8)
    top_idx = int(np.argmax(lead_attention))
    return {
        "lead": lead_names[top_idx],
        "score": float(lead_attention[top_idx]),
    }


def compute_top_shap_feature(
    shap_values: np.ndarray,
    feature_names: Optional[List[str]] = None,
    sample_idx: int = 0,
) -> Dict[str, float]:
    """
    Get the strongest SHAP feature for a single sample.
    """
    if shap_values.ndim == 1:
        sample_shap = shap_values
    else:
        sample_shap = shap_values[sample_idx]

    if feature_names is None:
        feature_names = [f"Feature_{i}" for i in range(len(sample_shap))]

    top_idx = int(np.argmax(np.abs(sample_shap)))
    return {
        "feature": feature_names[top_idx],
        "importance": float(np.abs(sample_shap[top_idx])),
    }


def summarize_visual_explanations(
    cam: np.ndarray,
    signal: Optional[np.ndarray],
    shap_values: np.ndarray,
    feature_names: Optional[List[str]] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Produce a simple summary: top lead attention + top SHAP feature.
    """
    lead_summary = compute_lead_attention(cam, signal=signal)
    shap_summary = compute_top_shap_feature(shap_values, feature_names=feature_names)
    return {
        "lead_attention": lead_summary,
        "shap_summary": shap_summary,
    }


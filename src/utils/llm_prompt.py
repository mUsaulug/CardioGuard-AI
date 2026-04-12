"""
LLM prompt templates for clinical explanation generation.
"""

from __future__ import annotations

from typing import Dict, Iterable, List


def build_clinical_prompt(
    model_prediction: str,
    probability: float,
    lead_attention: Dict[str, float],
    shap_summary: Dict[str, float],
    gradcam_images: Iterable[str],
) -> str:
    """
    Build a structured prompt for an LLM to generate clinical explanations.

    Inputs:
        model_prediction: Model class prediction (e.g., "MI" or "NORM")
        probability: Ensemble probability for the predicted class
        lead_attention: {"lead": name, "score": value}
        shap_summary: {"feature": name, "importance": value}
        gradcam_images: List of Grad-CAM image paths/URIs

    Output:
        A prompt that instructs the LLM to return a clinical explanation and
        risk factor in Turkish.
    """
    gradcam_list = ", ".join(gradcam_images) if gradcam_images else "N/A"
    lead_name = lead_attention.get("lead", "Unknown")
    lead_score = lead_attention.get("score", 0.0)
    shap_feature = shap_summary.get("feature", "Unknown")
    shap_score = shap_summary.get("importance", 0.0)

    return (
        "Sen bir kardiyoloji asistanısın. Aşağıdaki model çıktılarından "
        "kısa ve klinik-dil bir açıklama üret.\n\n"
        f"Model tahmini: {model_prediction}\n"
        f"Olasılık (ensemble): {probability:.3f}\n"
        f"En yüksek lead dikkat skoru: {lead_name} ({lead_score:.3f})\n"
        f"En güçlü SHAP özelliği: {shap_feature} ({shap_score:.3f})\n"
        f"Grad-CAM görselleri: {gradcam_list}\n\n"
        "Çıktı formatı:\n"
        "Klinik Açıklama: <tek paragraf, klinik dil>\n"
        "Risk Faktörü: <tek cümle, en olası risk faktörü>\n"
    )


def format_explanation_text(
    model_prediction: str,
    probability: float,
    lead_attention: Dict[str, float],
    shap_summary: Dict[str, float],
) -> str:
    """
    Build a deterministic explanation text without an LLM call.
    """
    lead_name = lead_attention.get("lead", "Unknown")
    lead_score = lead_attention.get("score", 0.0)
    shap_feature = shap_summary.get("feature", "Unknown")

    return (
        f"Klinik özet: Model {model_prediction} olasılığını {probability:.3f} olarak "
        f"değerlendirdi. En belirgin dikkat {lead_name} lead'inde "
        f"(skor={lead_score:.3f}). En güçlü SHAP katkısı {shap_feature} "
        "özelliğinden geldi."
    )


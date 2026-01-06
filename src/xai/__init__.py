"""CardioGuard-AI XAI (Explainable AI) Module."""

from src.xai.gradcam import GradCAM, generate_relevant_gradcam
from src.xai.shap_xgb import (
    explain_xgb,
    get_top_features,
    plot_shap_summary,
    plot_shap_waterfall,
)
from src.xai.visualize import (
    LEAD_NAMES,
    plot_ecg_with_localization,
    plot_ecg_with_prediction,
    plot_gradcam_heatmap,
    plot_lead_attention,
    generate_xai_report_png,
)
from src.xai.summary import (
    compute_lead_attention,
    compute_top_shap_feature,
    summarize_visual_explanations,
)
from src.xai.sanity import XAISanityChecker, run_sanity_check_standalone
from src.xai.combined import CombinedExplainer, ExplanationCard, create_explanation_card
from src.xai.reporting import XAIReporter, generate_run_id, quick_report

__all__ = [
    # Grad-CAM
    "GradCAM",
    "generate_relevant_gradcam",
    # SHAP
    "explain_xgb",
    "get_top_features",
    "plot_shap_summary",
    "plot_shap_waterfall",
    # Visualization
    "LEAD_NAMES",
    "plot_ecg_with_localization",
    "plot_ecg_with_prediction",
    "plot_gradcam_heatmap",
    "plot_lead_attention",
    "generate_xai_report_png",
    # Summary
    "compute_lead_attention",
    "compute_top_shap_feature",
    "summarize_visual_explanations",
    # Sanity
    "XAISanityChecker",
    "run_sanity_check_standalone",
    # Combined
    "CombinedExplainer",
    "ExplanationCard",
    "create_explanation_card",
    # Reporting
    "XAIReporter",
    "generate_run_id",
    "quick_report",
]

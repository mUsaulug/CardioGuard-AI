"""
XAI pipeline — single entry point coordinating Grad-CAM, SHAP, synthesis,
sanity checks, visualization, and manifest writing.

Callers use explain(); they do not coordinate individual XAI modules.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch import nn

from src.config import SUPERCLASS_LABELS
from src.xai.gradcam import GradCAM, generate_relevant_gradcam
from src.xai.shap_ovr import explain_single_sample
from src.xai.unified import UnifiedExplainer
from src.xai.sanity import XAISanityChecker
from src.xai.visualize import generate_xai_report_png


@dataclass
class PredictResult:
    """Pure inference output consumed by the XAI pipeline."""

    signal: np.ndarray
    signal_tensor: torch.Tensor
    cnn_probs: Dict[str, float]
    xgb_probs: Dict[str, float]
    ensemble_probs: Dict[str, float]
    embeddings: Optional[np.ndarray]
    predicted_labels: List[str]
    primary_label: str
    primary_confidence: float
    norm_prob: float
    thresholds: Dict[str, float]
    ensemble_weight: float
    localization: Optional[Dict[str, Any]] = None
    consistency: Optional[Any] = None


@dataclass
class ExplanationResult:
    """Structured XAI output from the pipeline."""

    gradcam: Dict[str, Any]
    shap: Dict[str, Any]
    narrative: str
    sanity: Optional[Dict[str, Any]] = None
    artifacts: List[Dict[str, str]] = field(default_factory=list)
    coherence_score: float = 0.5
    combined_heatmap: Optional[np.ndarray] = None
    contrastive: Optional[Dict[str, Any]] = None
    dominant_source: str = ""
    visual_summary: List[str] = field(default_factory=list)
    feature_summary: List[str] = field(default_factory=list)
    conflicts: List[str] = field(default_factory=list)

    def to_explanation_dict(self) -> Dict[str, Any]:
        """Return legacy dict shape expected by API and tests."""
        result: Dict[str, Any] = {
            "narrative": self.narrative,
            "coherence_score": self.coherence_score,
            "dominant_source": self.dominant_source,
            "visual_summary": self.visual_summary,
            "feature_summary": self.feature_summary,
            "conflicts": self.conflicts,
            "combined_heatmap": self.combined_heatmap,
            "contrastive": self.contrastive,
            "raw_gradcam": self.gradcam,
            "raw_shap": self.shap,
        }
        if self.sanity is not None:
            result["sanity_check"] = self.sanity
        return result


def explain(
    predict_result: PredictResult,
    cnn_model: nn.Module,
    xgb_data: Dict[str, Any],
    run_dir: Optional[Path],
    sample_id: str,
    sanity_check: bool = False,
    save_plot: Optional[Path] = None,
) -> ExplanationResult:
    """
    Coordinate Grad-CAM, SHAP, unified synthesis, sanity, visualization, manifest.

    Caller passes PredictResult from core_predict(); no XAI module knowledge required.
    """
    target_layer = cnn_model.get_cam_layer()
    gradcam_res = generate_relevant_gradcam(
        cnn_model,
        target_layer,
        predict_result.signal_tensor,
        predict_result.cnn_probs,
        predict_result.thresholds,
        top_k=2,
    )

    relevant_for_shap = list(gradcam_res.keys())
    primary_label = predict_result.primary_label
    if primary_label != "NORM" and primary_label not in relevant_for_shap:
        relevant_for_shap.append(primary_label)

    embeddings = predict_result.embeddings
    if embeddings is None and xgb_data.get("models") and hasattr(cnn_model, "backbone"):
        with torch.no_grad():
            embeddings = cnn_model.backbone(predict_result.signal_tensor).cpu().numpy()

    shap_res: Dict[str, Any] = {}
    if xgb_data.get("models") and relevant_for_shap and embeddings is not None:
        # SHAP features are latent CNN embedding dimensions (no raw-ECG meaning).
        # Label them transparently in Turkish instead of leaking "feature_13".
        feature_names = xgb_data.get("feature_names")
        if not feature_names:
            n_feat = int(np.asarray(embeddings).reshape(1, -1).shape[1])
            feature_names = [f"CNN gömme boyutu {i}" for i in range(n_feat)]
        shap_res = explain_single_sample(
            xgb_data["models"],
            embeddings,
            relevant_classes=relevant_for_shap,
            feature_names=feature_names,
        )

    unifier = UnifiedExplainer()
    sorted_classes = sorted(
        predict_result.ensemble_probs.items(), key=lambda x: x[1], reverse=True
    )
    runnerup_cls = sorted_classes[1][0] if len(sorted_classes) > 1 else None

    synthesized = unifier.synthesize(
        gradcam_res,
        shap_res,
        predict_result.ensemble_probs,
        predict_result.ensemble_weight,
        primary_label=primary_label,
        runnerup_label=runnerup_cls,
    )

    sanity_result: Optional[Dict[str, Any]] = None
    if sanity_check:
        class_idx = (
            SUPERCLASS_LABELS.index(primary_label) if primary_label != "NORM" else 0
        )

        def explanation_func(m: nn.Module, inp: torch.Tensor) -> np.ndarray:
            gcam = GradCAM(m, m.get_cam_layer())
            return gcam.generate(inp, class_index=class_idx)

        checker = XAISanityChecker(cnn_model)
        sanity_result = checker.run_checks(
            predict_result.signal_tensor,
            gradcam_res.get(primary_label) if gradcam_res else None,
            explanation_func,
        )

    # If sanity flags the explanation as unreliable, the coherence claim cannot
    # be trusted either — cap it so the UI never shows high confidence on junk.
    if sanity_result is not None:
        status = str(sanity_result.get("overall", {}).get("status", "")).upper()
        if status == "UNRELIABLE":
            synthesized["coherence_score"] = min(
                float(synthesized.get("coherence_score", 0.5)), 0.35
            )
            synthesized.setdefault("conflicts", []).append(
                "Sanity kontrolü açıklamayı güvenilmez buldu; tutarlılık düşürüldü."
            )

    plot_path = save_plot
    if plot_path is None and run_dir is not None:
        (run_dir / "visuals").mkdir(parents=True, exist_ok=True)
        plot_path = run_dir / "visuals" / f"{sample_id}_report.png"

    if plot_path is not None:
        try:
            shap_features = []
            primary_shap = shap_res.get(primary_label, {})
            if isinstance(primary_shap, dict):
                for feat in primary_shap.get("top_features", []):
                    shap_features.append({
                        "feature_idx": feat.get("feature", "Unknown"),
                        "shap_value": feat.get("importance", 0),
                    })

            primary_gradcam = gradcam_res.get(primary_label) if gradcam_res else None
            generate_xai_report_png(
                signal=predict_result.signal,
                combined_heatmap=primary_gradcam,
                shap_features=shap_features,
                sanity_metrics=sanity_result or {},
                prediction={
                    "pred_class": primary_label,
                    "pred_proba": predict_result.primary_confidence,
                },
                output_path=plot_path,
                sampling_rate=100,
            )
            print(f"Explanation plot saved to {plot_path}")
        except Exception as e:
            print(f"Warning: Could not save plot: {e}")
            import traceback
            traceback.print_exc()

    explanation_dict = {
        **synthesized,
        "raw_gradcam": gradcam_res,
        "raw_shap": shap_res,
    }
    if sanity_result is not None:
        explanation_dict["sanity_check"] = sanity_result

    artifacts: List[Dict[str, str]] = []
    if run_dir is not None:
        artifacts = _write_manifest(
            run_dir=run_dir,
            sample_id=sample_id,
            explanation_result=explanation_dict,
            sanity_result=sanity_result,
        )

    return ExplanationResult(
        gradcam=gradcam_res,
        shap=shap_res,
        narrative=synthesized.get("narrative", ""),
        sanity=sanity_result,
        artifacts=artifacts,
        coherence_score=synthesized.get("coherence_score", 0.5),
        combined_heatmap=synthesized.get("combined_heatmap"),
        contrastive=synthesized.get("contrastive"),
        dominant_source=synthesized.get("dominant_source", ""),
        visual_summary=synthesized.get("visual_summary", []),
        feature_summary=synthesized.get("feature_summary", []),
        conflicts=synthesized.get("conflicts", []),
    )


def _write_manifest(
    run_dir: Path,
    sample_id: str,
    explanation_result: Optional[Dict[str, Any]],
    sanity_result: Optional[Dict[str, Any]],
) -> List[Dict[str, str]]:
    """
    Write manifest.json for XAI artifacts.

    This is the ONLY place where manifest is written.
    Backend reads this file and serves artifacts.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "visuals").mkdir(exist_ok=True)
    (run_dir / "text").mkdir(exist_ok=True)
    (run_dir / "tensors").mkdir(exist_ok=True)

    artifacts: List[Dict[str, str]] = []

    visuals_dir = run_dir / "visuals"
    for png in visuals_dir.glob("*.png"):
        artifacts.append({
            "type": "report_png",
            "path": f"visuals/{png.name}",
            "mime": "image/png",
        })

    if explanation_result:
        narrative = _generate_narrative(explanation_result, sample_id)
        narrative_path = run_dir / "text" / f"{sample_id}__narrative.md"
        with open(narrative_path, "w", encoding="utf-8") as f:
            f.write(narrative)
        artifacts.append({
            "type": "narrative_md",
            "path": f"text/{sample_id}__narrative.md",
            "mime": "text/markdown",
        })

    manifest = {
        "run_id": run_dir.name,
        "created_at": datetime.now(timezone.utc).isoformat() + "Z",
        "task": "multiclass",
        "sample_id": sample_id,
        "artifacts": artifacts,
        "sanity": sanity_result.get("overall") if sanity_result else None,
        "highlights": explanation_result.get("top_windows") if explanation_result else None,
    }

    manifest_path = run_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return artifacts


def _generate_narrative(explanation: Dict[str, Any], sample_id: str) -> str:
    """Generate narrative markdown from explanation result."""
    narrative = f"# XAI Explanation: {sample_id}\n\n"

    shap_res = explanation.get("raw_shap", {})
    if shap_res:
        narrative += "## Top Features (SHAP)\n\n"
        for cls, data in shap_res.items():
            if isinstance(data, dict) and "top_features" in data:
                narrative += f"### {cls}\n"
                for feat in data.get("top_features", [])[:5]:
                    narrative += f"- {feat.get('feature', '?')}: {feat.get('importance', 0):.4f}\n"
                narrative += "\n"

    gradcam_res = explanation.get("raw_gradcam", {})
    if gradcam_res:
        narrative += "## Temporal Attention (Grad-CAM)\n\n"
        narrative += f"Generated for classes: {list(gradcam_res.keys())}\n\n"

    sanity = explanation.get("sanity_check", {})
    if sanity:
        overall = sanity.get("overall", {})
        status = overall.get("status", "UNKNOWN")
        narrative += f"## Sanity Check: {status}\n\n"
        narrative += f"- Passed: {overall.get('passed_checks', 0)}/{overall.get('total_checks', 0)}\n"

    return narrative

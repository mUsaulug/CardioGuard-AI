"""
Combined SHAP + Grad-CAM Explainer Module.

Creates unified explanations that combine the strengths of both:
- SHAP: Which embedding dimensions drive the XGBoost decision
- Grad-CAM: Which time regions in ECG signal are important for CNN

Combined output: SHAP-weighted Grad-CAM that shows "which time regions
contribute to the most important embedding dimensions"

Usage:
    explainer = CombinedExplainer(cnn_model, xgb_models)
    result = explainer.explain(signal_tensor, embeddings)
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
import json
import numpy as np
import torch
from torch import nn

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


@dataclass
class ExplanationCard:
    """Structured explanation card for RAG/LLM consumption."""
    
    # Metadata
    sample_id: str
    task: str  # binary, multiclass, localization
    model_id: str
    timestamp: str
    
    # Prediction
    pred_class: str
    pred_proba: float
    runnerup_class: Optional[str]
    runnerup_proba: Optional[float]
    margin: Optional[float]
    true_label: Optional[str]
    
    # SHAP
    shap_top_features: List[Dict[str, Any]]
    shap_expected_value: float
    
    # Grad-CAM
    gradcam_top_windows: List[Dict[str, Any]]
    
    # Combined
    combined_top_windows: List[Dict[str, Any]]
    contrastive_mode: str  # "pred_only" or "pred_minus_runnerup"
    
    # Sanity
    sanity_summary: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, default=str)


class CombinedExplainer:
    """
    Combines SHAP (XGBoost) and Grad-CAM (CNN) explanations.
    
    Strategy:
    1. Get SHAP values for each embedding dimension from XGBoost
    2. Get Grad-CAM for CNN showing temporal importance
    3. Weight Grad-CAM by SHAP importance of embedding dimensions
    4. Optional: Contrastive mode (pred class vs runnerup)
    """
    
    def __init__(
        self,
        cnn_model: nn.Module,
        xgb_models: Optional[Dict[str, Any]] = None,
        class_order: List[str] = None,
        top_k: int = 8,
        window_ms: int = 80,
        sampling_rate: int = 100
    ):
        """
        Initialize combined explainer.
        
        Args:
            cnn_model: CNN model (with backbone)
            xgb_models: Dict with 'models' and optional 'calibrators'
            class_order: Order of classes
            top_k: Number of top features/windows to report
            window_ms: Window size for temporal segmentation
            sampling_rate: ECG sampling rate
        """
        self.cnn_model = cnn_model
        self.xgb_models = xgb_models
        self.class_order = class_order or ["MI", "STTC", "CD", "HYP"]
        self.top_k = top_k
        self.window_ms = window_ms
        self.sampling_rate = sampling_rate
        self.window_samples = int(window_ms * sampling_rate / 1000)
    
    def explain(
        self,
        signal_tensor: torch.Tensor,
        embeddings: np.ndarray,
        target_class: str,
        probs: Dict[str, float],
        contrastive: bool = True,
        gradcam_func: Optional[callable] = None
    ) -> Dict[str, Any]:
        """
        Generate combined explanation.
        
        Args:
            signal_tensor: ECG signal (1, C, T)
            embeddings: CNN embeddings (1, D)
            target_class: Class to explain
            probs: Probability dict for all classes
            contrastive: If True, also compute pred vs runnerup
            gradcam_func: Optional custom Grad-CAM function
            
        Returns:
            Combined explanation dict with all components
        """
        result = {
            "target_class": target_class,
            "contrastive_mode": "pred_minus_runnerup" if contrastive else "pred_only"
        }
        
        # 1. SHAP explanation for target class
        shap_result = self._compute_shap(embeddings, target_class)
        result["shap"] = shap_result
        
        # 2. Grad-CAM (if function provided)
        gradcam_result = None
        if gradcam_func is not None:
            gradcam_result = gradcam_func(self.cnn_model, signal_tensor, target_class)
            result["gradcam"] = self._format_gradcam(gradcam_result)
        
        # 3. Combined: SHAP-weighted activation map
        # Check if we have valid shap values and gradcam result
        has_shap = shap_result.get("shap_values") is not None
        has_gradcam = gradcam_result is not None and (
            isinstance(gradcam_result, np.ndarray) or 
            (isinstance(gradcam_result, dict) and gradcam_result)
        )
        
        if has_shap and has_gradcam:
            result["combined"] = self._compute_combined(
                shap_result, gradcam_result, signal_tensor
            )
        else:
            result["combined"] = {"top_windows": [], "heatmap": None}
        
        # 4. Contrastive (pred vs runnerup)
        if contrastive:
            # Find runnerup
            sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
            if len(sorted_probs) >= 2:
                runnerup = sorted_probs[1][0] if sorted_probs[0][0] == target_class else sorted_probs[0][0]
                result["contrastive"] = self._compute_contrastive(
                    embeddings, target_class, runnerup
                )
        
        return result
    
    def _compute_shap(
        self, 
        embeddings: np.ndarray, 
        target_class: str
    ) -> Dict[str, Any]:
        """Compute SHAP values for embeddings using XGBoost."""
        if not SHAP_AVAILABLE:
            return {"error": "SHAP not installed", "shap_values": None}
        
        if self.xgb_models is None or target_class not in self.xgb_models.get("models", {}):
            return {"error": f"No XGBoost model for {target_class}", "shap_values": None}
        
        model = self.xgb_models["models"][target_class]
        
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        
        try:
            # Unwrap ManualCalibratedModel if present
            if hasattr(model, "base_model"):
                model = model.base_model
                
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(embeddings)
            
            if isinstance(shap_values, list):
                shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
            
            shap_values = shap_values[0]  # Single sample
            expected_value = float(explainer.expected_value) if np.isscalar(explainer.expected_value) else float(explainer.expected_value[1])
            
            # Get top-K features
            abs_shap = np.abs(shap_values)
            top_indices = np.argsort(abs_shap)[::-1][:self.top_k]
            
            top_features = []
            for rank, idx in enumerate(top_indices):
                top_features.append({
                    "rank": rank + 1,
                    "feature_idx": int(idx),
                    "shap_value": float(shap_values[idx]),
                    "direction": "positive" if shap_values[idx] > 0 else "negative",
                    "abs_importance": float(abs_shap[idx])
                })
            
            return {
                "shap_values": shap_values,
                "expected_value": expected_value,
                "top_features": top_features,
                "contribution_sum": float(shap_values.sum())
            }
            
        except Exception as e:
            return {"error": str(e), "shap_values": None}
    
    def _format_gradcam(self, gradcam: Any) -> Dict[str, Any]:
        """Format Grad-CAM output to standard structure."""
        if isinstance(gradcam, np.ndarray):
            heatmap = gradcam.flatten()
        elif isinstance(gradcam, dict):
            # Assume it has a heatmap key
            heatmap = gradcam.get("heatmap", gradcam.get("cam", np.array([])))
            if isinstance(heatmap, np.ndarray):
                heatmap = heatmap.flatten()
            else:
                return {"heatmap": None, "top_windows": []}
        else:
            return {"heatmap": None, "top_windows": []}
        
        # Find top windows
        top_windows = self._find_top_windows(heatmap)
        
        return {
            "heatmap": heatmap,
            "top_windows": top_windows,
            "shape": heatmap.shape
        }
    
    def _find_top_windows(self, heatmap: np.ndarray) -> List[Dict[str, Any]]:
        """Find top-K salient time windows in heatmap."""
        if len(heatmap) == 0:
            return []
        
        n_windows = max(1, len(heatmap) // self.window_samples)
        window_importance = []
        
        for i in range(n_windows):
            start = i * self.window_samples
            end = min(start + self.window_samples, len(heatmap))
            mean_activation = heatmap[start:end].mean()
            max_activation = heatmap[start:end].max()
            
            start_ms = int(start * 1000 / self.sampling_rate)
            end_ms = int(end * 1000 / self.sampling_rate)
            
            window_importance.append({
                "window_idx": i,
                "start_ms": start_ms,
                "end_ms": end_ms,
                "mean_activation": float(mean_activation),
                "max_activation": float(max_activation)
            })
        
        # Sort by mean activation
        window_importance.sort(key=lambda x: x["mean_activation"], reverse=True)
        
        return window_importance[:self.top_k]
    
    def _compute_combined(
        self,
        shap_result: Dict[str, Any],
        gradcam_result: Any,
        signal_tensor: torch.Tensor
    ) -> Dict[str, Any]:
        """
        Compute SHAP-weighted Grad-CAM.
        
        Concept: Weight the temporal importance by how important
        each embedding dimension is according to SHAP.
        
        For a true implementation, we would need per-dimension Grad-CAM,
        but for efficiency we use a simplified weighting approach.
        """
        shap_values = shap_result.get("shap_values")
        if shap_values is None:
            return {"heatmap": None, "top_windows": []}
        
        # Get the gradcam heatmap
        if isinstance(gradcam_result, dict):
            heatmap = gradcam_result.get("heatmap")
            if heatmap is None:
                heatmap = gradcam_result.get("cam")
        else:
            heatmap = gradcam_result
        
        if heatmap is None or not isinstance(heatmap, np.ndarray):
            return {"heatmap": None, "top_windows": []}
        
        heatmap = np.asarray(heatmap).flatten()
        
        # Weighted combination: scale by total SHAP contribution
        # (simplified - full version would compute per-dim CAM)
        total_contribution = abs(shap_values.sum())
        scaling_factor = 1.0 + 0.5 * np.tanh(total_contribution)  # Soft scaling
        
        combined_heatmap = heatmap * scaling_factor
        
        # Normalize
        if combined_heatmap.max() > 0:
            combined_heatmap = combined_heatmap / combined_heatmap.max()
        
        # Find windows
        top_windows = self._find_top_windows(combined_heatmap)
        
        # Add SHAP context to windows
        shap_top = [f["feature_idx"] for f in shap_result.get("top_features", [])[:3]]
        for w in top_windows:
            w["contributing_dims"] = shap_top
        
        return {
            "heatmap": combined_heatmap,
            "top_windows": top_windows,
            "scaling_factor": float(scaling_factor),
            "shap_contribution": float(total_contribution)
        }
    
    def _compute_contrastive(
        self,
        embeddings: np.ndarray,
        pred_class: str,
        runnerup_class: str
    ) -> Dict[str, Any]:
        """
        Compute contrastive explanation: pred vs runnerup.
        
        Shows which features push toward pred and away from runnerup.
        """
        pred_shap = self._compute_shap(embeddings, pred_class)
        runnerup_shap = self._compute_shap(embeddings, runnerup_class)
        
        if pred_shap.get("shap_values") is None or runnerup_shap.get("shap_values") is None:
            return {"error": "Could not compute contrastive SHAP"}
        
        # Difference: positive = pushes toward pred, negative = toward runnerup
        diff = pred_shap["shap_values"] - runnerup_shap["shap_values"]
        
        # Top features that distinguish
        abs_diff = np.abs(diff)
        top_indices = np.argsort(abs_diff)[::-1][:self.top_k]
        
        distinguishing_features = []
        for rank, idx in enumerate(top_indices):
            direction = "toward_pred" if diff[idx] > 0 else "toward_runnerup"
            distinguishing_features.append({
                "rank": rank + 1,
                "feature_idx": int(idx),
                "diff_value": float(diff[idx]),
                "direction": direction,
                "pred_shap": float(pred_shap["shap_values"][idx]),
                "runnerup_shap": float(runnerup_shap["shap_values"][idx])
            })
        
        return {
            "pred_class": pred_class,
            "runnerup_class": runnerup_class,
            "distinguishing_features": distinguishing_features,
            "total_margin": float(diff.sum())
        }
    
    def generate_narrative(
        self,
        explanation: Dict[str, Any],
        pred_class: str,
        pred_proba: float
    ) -> str:
        """
        Generate human-readable narrative from explanation.
        
        For RAG/LLM consumption.
        """
        lines = []
        lines.append(f"**Prediction: {pred_class}** (confidence: {pred_proba:.1%})")
        lines.append("")
        
        # SHAP summary
        shap_data = explanation.get("shap", {})
        if shap_data.get("top_features"):
            lines.append("**Key Evidence (SHAP):**")
            for f in shap_data["top_features"][:3]:
                direction = "↑" if f["direction"] == "positive" else "↓"
                lines.append(f"- Embedding-{f['feature_idx']}: {direction} {abs(f['shap_value']):.3f}")
            lines.append("")
        
        # Combined temporal windows
        combined = explanation.get("combined", {})
        if combined.get("top_windows"):
            lines.append("**Salient Time Regions:**")
            for w in combined["top_windows"][:3]:
                lines.append(f"- {w['start_ms']}–{w['end_ms']} ms (activation: {w['mean_activation']:.2f})")
            lines.append("")
        
        # Contrastive
        contrastive = explanation.get("contrastive", {})
        if contrastive.get("distinguishing_features"):
            runnerup = contrastive.get("runnerup_class", "other")
            lines.append(f"**Why {pred_class} and not {runnerup}?**")
            for f in contrastive["distinguishing_features"][:2]:
                lines.append(f"- Embedding-{f['feature_idx']}: {f['direction']} ({f['diff_value']:.3f})")
        
        return "\n".join(lines)


def create_explanation_card(
    sample_id: str,
    task: str,
    model_id: str,
    prediction: Dict[str, Any],
    explanation: Dict[str, Any],
    sanity: Dict[str, Any],
    true_label: Optional[str] = None
) -> ExplanationCard:
    """
    Factory function to create a structured ExplanationCard.
    """
    from datetime import datetime
    
    # Extract prediction info
    pred_class = prediction.get("pred_class", "UNKNOWN")
    pred_proba = prediction.get("pred_proba", 0.0)
    runnerup = prediction.get("runnerup")
    runnerup_proba = prediction.get("runnerup_proba")
    margin = pred_proba - runnerup_proba if runnerup_proba else None
    
    # SHAP
    shap_data = explanation.get("shap", {})
    shap_top = shap_data.get("top_features", [])
    shap_ev = shap_data.get("expected_value", 0.0)
    
    # Grad-CAM
    gradcam_data = explanation.get("gradcam", {})
    gradcam_windows = gradcam_data.get("top_windows", [])
    
    # Combined
    combined_data = explanation.get("combined", {})
    combined_windows = combined_data.get("top_windows", [])
    contrastive_mode = explanation.get("contrastive_mode", "pred_only")
    
    # Sanity summary
    sanity_summary = {
        "overall_status": sanity.get("overall", {}).get("status", "UNKNOWN"),
        "passed_checks": sanity.get("overall", {}).get("passed_checks", 0),
        "total_checks": sanity.get("overall", {}).get("total_checks", 4)
    }
    
    return ExplanationCard(
        sample_id=sample_id,
        task=task,
        model_id=model_id,
        timestamp=datetime.utcnow().isoformat() + "Z",
        pred_class=pred_class,
        pred_proba=pred_proba,
        runnerup_class=runnerup,
        runnerup_proba=runnerup_proba,
        margin=margin,
        true_label=true_label,
        shap_top_features=shap_top,
        shap_expected_value=shap_ev,
        gradcam_top_windows=gradcam_windows,
        combined_top_windows=combined_windows,
        contrastive_mode=contrastive_mode,
        sanity_summary=sanity_summary
    )

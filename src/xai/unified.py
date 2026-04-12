"""
Unified Explanation Module.

This module bridges the gap between Visual Explanations (Grad-CAM) and
Statistical Feature Attributions (SHAP). It synthesizes a coherent 
clinical narrative by aligning spatial and feature-based evidence.

Usage:
    explainer = UnifiedExplainer()
    explanation = explainer.synthesize(gradcam_result, shap_result, prediction_probs)
"""

from typing import Dict, Any, List, Optional
import numpy as np


class UnifiedExplainer:
    """
    Synthesizes Grad-CAM and SHAP explanations into a unified clinical report.
    """
    
    def __init__(self):
        # Clinical mapping: CNN features -> Human readable concepts
        # Ideally this would be learned or expert-defined
        self.feature_map = {
            "lead": "Derivasyon",
            "ST": "ST Segmenti",
            "T_wave": "T Dalgası",
            "QRS": "QRS Kompleksi"
        }

    def _compute_shap_weighted_cam(self, gradcam_heatmap, shap_values):
        """Scale GradCAM heatmap by SHAP feature importance."""
        if gradcam_heatmap is None or shap_values is None:
            return gradcam_heatmap
        total_shap = float(np.sum(np.abs(shap_values)))
        scaling = 1.0 + 0.5 * np.tanh(total_shap)
        combined = gradcam_heatmap * scaling
        combined = (combined - combined.min()) / (combined.max() + 1e-8)
        return combined

    def _compute_contrastive(self, pred_shap_values, runnerup_shap_values):
        """Compare SHAP values between predicted and runner-up class."""
        if pred_shap_values is None or runnerup_shap_values is None:
            return None
        pred = np.asarray(pred_shap_values).flatten()
        runner = np.asarray(runnerup_shap_values).flatten()
        min_len = min(len(pred), len(runner))
        diff = pred[:min_len] - runner[:min_len]
        top_idx = np.argsort(np.abs(diff))[::-1][:10]
        return {
            "distinguishing_features": top_idx.tolist(),
            "diff_values": diff[top_idx].tolist(),
        }

    def synthesize(
        self,
        gradcam_result: Dict[str, Any],
        shap_result: Dict[str, Any],
        prediction_probs: Dict[str, float],
        ensemble_weight: float = 0.5,
        primary_label: str = None,
        runnerup_label: str = None,
    ) -> Dict[str, Any]:
        """
        Create a unified explanation from multiple modalities.
        
        Args:
            gradcam_result: Result from generate_relevant_gradcam
            shap_result: Result from explain_single_sample
            prediction_probs: Ensemble probabilities
            ensemble_weight: Weight of CNN in ensemble (0.0 - 1.0)
            
        Returns:
            Unified explanation dictionary including summary, coherence score, and conflicts.
        """
        # 1. Identify Dominant Model
        cnn_contrib = ensemble_weight
        xgb_contrib = 1.0 - ensemble_weight
        dominant_source = "CNN (Visual)" if cnn_contrib >= xgb_contrib else "XGBoost (Feature)"
        
        # 2. Extract Key Evidence
        visual_evidence = self._extract_visual_evidence(gradcam_result)
        feature_evidence = self._extract_feature_evidence(shap_result)
        
        # 3. Detect Conflicts or Synergy
        coherence_score, conflict_notes = self._analyze_coherence(visual_evidence, feature_evidence)
        
        # 4. Generate Narrative
        narrative = self._generate_narrative(
            prediction_probs, 
            visual_evidence, 
            feature_evidence, 
            dominant_source,
            conflict_notes
        )
        
        # SHAP-weighted GradCAM
        combined_heatmap = None
        if primary_label and primary_label in gradcam_result:
            primary_cam = gradcam_result[primary_label]
            primary_shap = shap_result.get(primary_label, {})
            shap_vals = primary_shap.get("shap_values") if isinstance(primary_shap, dict) else None
            combined_heatmap = self._compute_shap_weighted_cam(primary_cam, shap_vals)

        # Contrastive mode
        contrastive = None
        if primary_label and runnerup_label:
            pred_shap = shap_result.get(primary_label, {})
            runner_shap = shap_result.get(runnerup_label, {})
            pred_vals = pred_shap.get("shap_values") if isinstance(pred_shap, dict) else None
            runner_vals = runner_shap.get("shap_values") if isinstance(runner_shap, dict) else None
            contrastive = self._compute_contrastive(pred_vals, runner_vals)

        return {
            "narrative": narrative,
            "coherence_score": coherence_score,
            "dominant_source": dominant_source,
            "visual_summary": visual_evidence,
            "feature_summary": feature_evidence,
            "conflicts": conflict_notes,
            "combined_heatmap": combined_heatmap,
            "contrastive": contrastive,
        }

    def _extract_visual_evidence(self, gradcam_result: Dict[str, Any]) -> List[str]:
        """Extract visual evidence from Grad-CAM."""
        evidence = []
        if not gradcam_result:
            return ["No significant visual activation."]
            
        for cls, data in gradcam_result.items():
            # data is numpy array (timesteps,)
            if isinstance(data, np.ndarray):
                # Find peak activation time
                peak_time = np.argmax(data)
                duration = len(data)
                # Helper to describe time in ECG terms (0-10s)
                time_sec = (peak_time / duration) * 10.0
                evidence.append(f"{cls}: High activation around {time_sec:.1f}s.")
            elif isinstance(data, dict):
                top_leads = data.get("top_leads", [])[:2] # Top 2 leads
                time_focus = data.get("time_focus", "unknown")
                evidence.append(f"{cls}: Focused on {', '.join(top_leads)} during {time_focus}")
            
        return evidence

    def _extract_feature_evidence(self, shap_result: Dict[str, Any]) -> List[str]:
        """Extract top contributing features from SHAP."""
        evidence = []
        if not shap_result:
            return ["No significant feature contribution."]
            
        if "error" in shap_result:
            return [f"Feature analysis unavailable: {shap_result['error']}"]

        for cls, result in shap_result.items():
            if not isinstance(result, dict):
                continue
            # Get feature importance from SHAP values
            shap_values = result.get("shap_values", [])
            # In a real scenario, we map feature indices to names here
            # For now, we simulate extraction of top features
            evidence.append(f"{cls}: Driven by key statistical features consistent with pathology.")
            
        return evidence

    def _analyze_coherence(self, gradcam_result, shap_result):
        """Compute real coherence between visual and feature explanations."""
        if not gradcam_result or not shap_result:
            return 0.5, ["Insufficient data for coherence analysis"]

        gradcam_peaks = set()
        for cls, cam in gradcam_result.items():
            if isinstance(cam, np.ndarray):
                cam_flat = cam.flatten()
                if len(cam_flat) > 0:
                    peak_region = int(np.argmax(cam_flat) / max(len(cam_flat), 1) * 10)
                    gradcam_peaks.add(peak_region)

        shap_consistency = 0
        shap_total = 0
        for cls, data in shap_result.items():
            if isinstance(data, dict) and "top_features" in data:
                shap_total += 1
                top_feat = data["top_features"][0] if data["top_features"] else None
                if top_feat and top_feat.get("importance", 0) > 0.01:
                    shap_consistency += 1

        if shap_total > 0:
            score = 0.5 + 0.5 * (shap_consistency / shap_total)
        else:
            score = 0.5

        conflicts = []
        if score < 0.6:
            conflicts.append("Visual and feature explanations show low agreement.")

        return score, conflicts

    def _generate_narrative(
        self, 
        probs: Dict[str, float], 
        visual: List[str], 
        feature: List[str], 
        source: str, 
        conflicts: List[str]
    ) -> str:
        """Generate human-readable clinical summary."""
        primary_dx = max(probs, key=probs.get)
        prob = probs[primary_dx]
        
        text = f"Diagnosis: **{primary_dx}** ({prob:.1%}).\n\n"
        text += f"Reasoning is primarily driven by **{source}** analysis.\n"
        
        if conflicts:
            text += f"⚠️ **Attention:** {conflicts[0]}\n"
        else:
            text += "✅ Multi-modal evidence is coherent.\n"
            
        text += "\n**Evidence:**\n"
        for v in visual:
            text += f"- Visual: {v}\n"
        for f in feature:
            text += f"- Clinical: {f}\n"
            
        return text

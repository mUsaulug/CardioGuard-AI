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
        coherence_score, conflict_notes = self._analyze_coherence(gradcam_result, shap_result)
        
        # 4. Generate Narrative
        narrative = self._generate_narrative(
            prediction_probs,
            visual_evidence,
            feature_evidence,
            dominant_source,
            conflict_notes,
            primary_label=primary_label,
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

    @staticmethod
    def _temporal_cam_series(cam: np.ndarray) -> np.ndarray:
        """Reduce Grad-CAM map to 1D temporal attention (timesteps,)."""
        arr = np.asarray(cam)
        if arr.ndim == 1:
            return arr
        if arr.ndim == 2:
            # Common shapes: (batch, time) or (leads, time)
            if arr.shape[0] == 1:
                return arr[0]
            if arr.shape[-1] >= arr.shape[0]:
                return np.mean(arr, axis=0)
            return np.mean(arr, axis=1)
        return arr.reshape(-1)

    def _extract_visual_evidence(self, gradcam_result: Dict[str, Any]) -> List[str]:
        """Extract visual evidence from Grad-CAM."""
        evidence = []
        if not gradcam_result:
            return ["Anlamlı görsel aktivasyon bulunamadı."]

        for cls, data in gradcam_result.items():
            if isinstance(data, np.ndarray):
                series = self._temporal_cam_series(data)
                if series.size == 0:
                    continue
                peak_idx = int(np.argmax(series))
                duration = int(series.size)
                time_sec = (peak_idx / max(duration - 1, 1)) * 10.0
                evidence.append(
                    f"{cls}: Zaman ekseninde ~{time_sec:.1f}s civarında yüksek aktivasyon."
                )
            elif isinstance(data, dict):
                top_leads = data.get("top_leads", [])[:2]
                time_focus = data.get("time_focus", "bilinmiyor")
                evidence.append(
                    f"{cls}: {', '.join(top_leads) or 'seçili derivasyonlarda'} odak ({time_focus})."
                )

        return evidence or ["Anlamlı görsel aktivasyon bulunamadı."]

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
            top_feats = result.get("top_features") or []
            if top_feats:
                names = []
                for feat in top_feats[:3]:
                    label = feat.get("feature", feat.get("feature_idx", "?"))
                    imp = feat.get("importance", 0.0)
                    names.append(f"{label} ({imp:+.3f})")
                evidence.append(f"{cls}: Öne çıkan özellikler — {', '.join(names)}.")
            else:
                evidence.append(
                    f"{cls}: İstatistiksel özellikler patoloji yönünde katkı sağlıyor."
                )

        return evidence

    def _analyze_coherence(self, gradcam_result, shap_result):
        """Compute a calibrated coherence score between Grad-CAM and SHAP.

        Combines three real signals (never a synthetic perfect 1.0):
        1. Class agreement (Jaccard) between modalities that produced evidence.
        2. Grad-CAM focus: how peaked the temporal attention is (diffuse = low).
        3. SHAP dominance: how clearly one feature leads the attribution.
        """
        if not gradcam_result or not shap_result:
            return 0.4, ["Yetersiz veri: tutarlılık güvenilir hesaplanamadı."]

        gradcam_classes = {
            cls
            for cls, cam in gradcam_result.items()
            if isinstance(cam, np.ndarray) and np.asarray(cam).size > 0
        }
        shap_classes = {
            cls
            for cls, data in shap_result.items()
            if isinstance(data, dict) and data.get("top_features")
        }
        if not gradcam_classes or not shap_classes:
            return 0.4, ["Modalitelerden biri anlamlı kanıt üretmedi."]

        shared = gradcam_classes & shap_classes
        union = gradcam_classes | shap_classes
        agreement = len(shared) / len(union) if union else 0.0

        # Grad-CAM focus: 1 - normalized-mean (a single sharp peak -> high focus).
        focus_scores: List[float] = []
        for cls in (shared or gradcam_classes):
            cam = gradcam_result.get(cls)
            if isinstance(cam, np.ndarray):
                series = np.abs(self._temporal_cam_series(cam))
                if series.size and series.max() > 0:
                    norm = series / series.max()
                    focus_scores.append(float(min(max(1.0 - norm.mean(), 0.0), 1.0)))
        focus = float(np.mean(focus_scores)) if focus_scores else 0.5

        # SHAP dominance: top-1 share of the top-5 absolute importances.
        dom_scores: List[float] = []
        for cls in (shared or shap_classes):
            data = shap_result.get(cls)
            if isinstance(data, dict):
                imps = [abs(f.get("importance", 0.0)) for f in (data.get("top_features") or [])[:5]]
                total = sum(imps)
                if total > 0:
                    dom_scores.append(imps[0] / total)
        dominance = float(np.mean(dom_scores)) if dom_scores else 0.5

        score = 0.5 * agreement + 0.25 * focus + 0.25 * dominance
        # Cap: never claim a perfect 1.0; floor avoids absolute zero.
        score = float(min(max(score, 0.05), 0.97))

        conflicts: List[str] = []
        if agreement < 0.5:
            conflicts.append(
                "Görsel (Grad-CAM) ve istatistiksel (SHAP) kanıtlar farklı sınıfları öne çıkarıyor."
            )
        if score < 0.5:
            conflicts.append("Açıklama modaliteleri arasında düşük uyum.")

        return score, conflicts

    def _generate_narrative(
        self,
        probs: Dict[str, float],
        visual: List[str],
        feature: List[str],
        source: str,
        conflicts: List[str],
        primary_label: Optional[str] = None,
    ) -> str:
        """Generate human-readable clinical summary (Turkish)."""
        primary_dx = primary_label or max(probs, key=probs.get)
        prob = probs.get(primary_dx, 0.0)

        source_tr = source
        if "XGBoost" in source:
            source_tr = "XGBoost (özellik)"
        elif "CNN" in source:
            source_tr = "CNN (görsel)"

        text = f"**Birincil bulgu:** {primary_dx} — ensemble olasılığı **%{prob * 100:.1f}**.\n\n"
        text += f"Açıklama ağırlığı: **{source_tr}**.\n"

        if conflicts:
            text += f"⚠️ **Dikkat:** {conflicts[0]}\n"
        else:
            text += "✅ Görsel (Grad-CAM) ve istatistiksel (SHAP) kanıtlar uyumlu.\n"

        text += "\n**Kanıtlar:**\n"
        for v in visual:
            text += f"- Grad-CAM: {v}\n"
        for f in feature:
            text += f"- SHAP: {f}\n"

        return text

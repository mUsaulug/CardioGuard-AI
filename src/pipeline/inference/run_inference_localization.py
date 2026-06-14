"""
MI Localization Inference Entrypoint.

Runs inference with localization CNN for 5 anatomical regions.
Generates XAI artifacts when explain=True.

Usage:
    python -m src.pipeline.inference.run_inference_localization --input sample.npz
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, Optional, List

import numpy as np
import torch
from torch import nn

from src.models.cnn import ECGCNNConfig, ECGCNN
from src.data.mi_localization import MI_LOCALIZATION_REGIONS
from src.utils.signal import validate_ecg_signal


def predict(
    signal: np.ndarray,
    model: nn.Module,
    device: torch.device,
    threshold: float = 0.5,
    explain: bool = False,
    run_dir: Optional[Path] = None,
    sample_id: str = "sample",
) -> Dict[str, Any]:
    """
    Run MI localization prediction.
    
    Args:
        signal: ECG signal (channels, timesteps)
        model: Trained localization model
        device: Torch device
        threshold: Detection threshold
        explain: Generate XAI artifacts
        run_dir: Directory for XAI artifacts
        sample_id: Sample identifier for artifact naming
        
    Returns:
        Prediction result dict
    """
    # Match training: validated channel-first raw amplitudes (no superclass z-score).
    signal, _ = validate_ecg_signal(signal)

    with torch.no_grad():
        signal_tensor = torch.as_tensor(signal, dtype=torch.float32).unsqueeze(0).to(device)
        logits = model(signal_tensor)
        probs = torch.sigmoid(logits).cpu().numpy()[0]
    
    probs_dict = {
        region: float(probs[i])
        for i, region in enumerate(MI_LOCALIZATION_REGIONS)
    }
    
    detected_regions = [r for r, p in probs_dict.items() if p >= threshold]
    mi_detected = len(detected_regions) > 0
    
    # XAI generation
    explanation_result = None
    if explain and run_dir:
        explanation_result = _generate_xai(
            model=model,
            signal=signal,
            signal_tensor=signal_tensor,
            probs_dict=probs_dict,
            detected_regions=detected_regions,
            run_dir=run_dir,
            sample_id=sample_id,
        )
        
        # Write manifest
        _write_manifest(
            run_dir=run_dir,
            sample_id=sample_id,
            probs_dict=probs_dict,
            detected_regions=detected_regions,
        )
    
    return {
        "mi_detected": mi_detected,
        "regions": detected_regions,
        "probabilities": probs_dict,
        "threshold": threshold,
        "explanation": explanation_result,
        "run_id": run_dir.name if run_dir else None,
        "run_dir": str(run_dir) if run_dir else None,
    }


def _ensure_channel_first(signal: np.ndarray) -> np.ndarray:
    """Ensure signal is (channels, timesteps) format."""
    if signal.ndim == 1:
        signal = signal.reshape(1, -1)
    if signal.shape[0] == 12:
        return signal
    if signal.shape[1] == 12:
        return signal.T
    if signal.shape[0] > signal.shape[1]:
        return signal.T
    return signal


def _generate_xai(
    model: nn.Module,
    signal: np.ndarray,
    signal_tensor: torch.Tensor,
    probs_dict: Dict[str, float],
    detected_regions: List[str],
    run_dir: Path,
    sample_id: str,
) -> Dict[str, Any]:
    """Generate XAI artifacts for localization."""
    explanation = {"gradcam": None}
    
    try:
        from src.xai.gradcam import GradCAM
        from src.xai.visualize import generate_xai_report_png
        
        # Find target layer
        target_layer = None
        if hasattr(model, 'backbone') and hasattr(model.backbone, 'features'):
            features = model.backbone.features
            target_layer = features[4] if len(features) > 4 else features[0]
        
        if target_layer is not None:
            # Get dominant class
            dominant_region = max(probs_dict, key=probs_dict.get)
            dominant_idx = MI_LOCALIZATION_REGIONS.index(dominant_region)
            
            # Generate Grad-CAM
            signal_grad = signal_tensor.clone().requires_grad_(True)
            gradcam = GradCAM(model, target_layer)
            heatmap = gradcam.generate(signal_grad, class_index=dominant_idx)
            gradcam.cleanup()
            
            explanation["gradcam"] = {
                "class": dominant_region,
                "heatmap_shape": list(heatmap.shape) if hasattr(heatmap, 'shape') else None,
            }
            
            # Generate visual report
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "visuals").mkdir(exist_ok=True)
            
            prediction = {
                "pred_class": dominant_region,
                "pred_proba": probs_dict[dominant_region],
                "affected_leads": detected_regions,
            }
            
            visual_path = run_dir / "visuals" / f"{sample_id}__report.png"
            try:
                generate_xai_report_png(
                    signal=signal,
                    combined_heatmap=heatmap.squeeze() if hasattr(heatmap, 'squeeze') else heatmap,
                    shap_features=[],
                    sanity_metrics={"overall": {"status": "SKIPPED"}},
                    prediction=prediction,
                    output_path=visual_path
                )
            except Exception as e:
                print(f"Warning: Visual report generation failed: {e}")
    
    except ImportError as e:
        print(f"Warning: XAI modules not available: {e}")
    except Exception as e:
        print(f"Warning: XAI generation failed: {e}")
    
    return explanation


def _write_manifest(
    run_dir: Path,
    sample_id: str,
    probs_dict: Dict[str, float],
    detected_regions: List[str],
) -> None:
    """Write manifest.json for localization XAI artifacts."""
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "visuals").mkdir(exist_ok=True)
    (run_dir / "text").mkdir(exist_ok=True)
    
    from src.xai.manifest_io import discover_visual_artifacts, write_run_manifest

    artifacts = discover_visual_artifacts(run_dir)
    narrative = f"""# MI Localization XAI

## Detected Regions
{', '.join(detected_regions) if detected_regions else 'None detected'}

## Probabilities
"""
    for region, prob in sorted(probs_dict.items(), key=lambda x: x[1], reverse=True):
        narrative += f"- {region}: {prob:.1%}\n"
    
    narrative_path = run_dir / "text" / f"{sample_id}__narrative.md"
    with open(narrative_path, "w", encoding="utf-8") as f:
        f.write(narrative)
    
    artifacts.append({
        "type": "narrative_md",
        "path": f"text/{sample_id}__narrative.md",
        "mime": "text/markdown"
    })

    write_run_manifest(
        run_dir=run_dir,
        sample_id=sample_id,
        task="localization",
        artifacts=artifacts,
        sanity=None,
        highlights=None,
    )

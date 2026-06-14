"""
Multi-label Superclass Inference Entrypoint.

Runs inference with CNN + XGBoost OVR ensemble.
Outputs multi-label predictions + primary label based on priority rule.

Usage:
    python -m src.pipeline.run_inference_superclass --input sample.npz --output result.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import numpy as np
import torch
from torch import nn
import joblib
from sklearn.isotonic import IsotonicRegression  # Import needed for instance check

from src.models.cnn import ECGCNNConfig, ECGBackbone, ECGCNN
from src.data.mi_localization import MI_LOCALIZATION_REGIONS
from src.config import SUPERCLASS_LABELS, MI_LOCALIZATION_LABELS, get_ensemble_cnn_weight
from src.pipeline.inference.consistency_guard import check_consistency, ConsistencyResult
from src.utils.model_loader import validate_feature_schema
from src.utils.signal import apply_superclass_normalization, validate_ecg_signal
from src.xai.pipeline import PredictResult, ExplanationResult, explain as xai_explain


# Default paths
DEFAULT_CNN_CHECKPOINT = Path("checkpoints/ecgcnn_superclass.pt")
DEFAULT_XGB_DIR = Path("logs/xgb_superclass")
DEFAULT_THRESHOLDS = Path("artifacts/thresholds_superclass.json")
DEFAULT_LOCALIZATION_CHECKPOINT = Path("checkpoints/ecgcnn_localization.pt")
DEFAULT_LOCALIZATION_THRESHOLDS = None # Use default 0.5 for now, or implement optimization later


def get_primary_label(probs: Dict[str, float], thresholds: Dict[str, float]) -> Tuple[str, float]:
    """
    Determine primary label using MI-first-then-priority rule.
    
    Args:
        probs: Dict of class -> probability
        thresholds: Dict of class -> threshold
        
    Returns:
        (primary_label, confidence)
    """
    # 1. MI first (highest priority for clinical importance)
    if probs.get("MI", 0) >= thresholds.get("MI", 0.5):
        return "MI", probs["MI"]
    
    # 2. Other pathologies in priority order
    for cls in ["STTC", "CD", "HYP"]:
        if probs.get(cls, 0) >= thresholds.get(cls, 0.5):
            return cls, probs[cls]
    
    # 3. If no pathology detected, return NORM
    # NORM probability = 1 - max(pathology probs)
    max_pathology = max(probs.get(cls, 0) for cls in SUPERCLASS_LABELS)
    norm_prob = 1.0 - max_pathology
    return "NORM", norm_prob


def load_cnn_model(checkpoint_path: Path, device: torch.device) -> nn.Module:
    """Load superclass CNN via shared safe loader (same as API startup)."""
    from src.utils.model_loader import load_model_safe

    model, _meta = load_model_safe(checkpoint_path, "superclass", str(device))
    return model


def load_localization_model(checkpoint_path: Path, device: torch.device) -> nn.Module:
    """Load trained MI localization model."""
    if not checkpoint_path.exists():
        return None
        
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Config must match training (64 filters, 0.5 dropout)
    config = ECGCNNConfig(num_filters=64, dropout=0.5)
    model = ECGCNN(config, num_classes=len(MI_LOCALIZATION_REGIONS))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model.to(device)
    
    return model


def load_xgb_models(xgb_dir: Path) -> Dict[str, Any]:
    """Load XGBoost OVR models and calibrators."""
    from xgboost import XGBClassifier
    
    models = {}
    calibrators = {}
    scaler = None
    
    # Load scaler
    scaler_path = xgb_dir / "scaler.joblib"
    if scaler_path.exists():
        scaler = joblib.load(scaler_path)
    
    # Load per-class models
    for cls in SUPERCLASS_LABELS:
        cls_dir = xgb_dir / cls
        
        # Load model
        model_path = cls_dir / "xgb_model.json"
        if model_path.exists():
            model = XGBClassifier()
            model.load_model(model_path)
            models[cls] = model
        
        # Load calibrator
        calibrator_path = cls_dir / "calibrator.joblib"
        if calibrator_path.exists():
            calibrators[cls] = joblib.load(calibrator_path)
    
    return {"models": models, "calibrators": calibrators, "scaler": scaler}


def load_thresholds(thresholds_path: Path) -> Dict[str, float]:
    """Load optimized thresholds."""
    if not thresholds_path.exists():
        # Default thresholds
        return {cls: 0.5 for cls in SUPERCLASS_LABELS}
    
    with open(thresholds_path) as f:
        data = json.load(f)
    
    return data.get("thresholds", {cls: 0.5 for cls in SUPERCLASS_LABELS})


def load_ecg_signal(input_path: Path) -> np.ndarray:
    """Load ECG signal from various formats (CLI)."""
    from src.utils.signal_io import load_ecg_array_from_path

    return load_ecg_array_from_path(input_path)


def ensure_channel_first(signal: np.ndarray) -> np.ndarray:
    """Ensure signal is (channels, timesteps) format."""
    if signal.ndim == 1:
        signal = signal.reshape(1, -1)
    
    # Heuristic: 12-lead ECG, if first dim is 12, it's already channel-first
    if signal.shape[0] == 12:
        return signal
    if signal.shape[1] == 12:
        return signal.T
    
    # Default: assume (timesteps, channels) and transpose
    if signal.shape[0] > signal.shape[1]:
        return signal.T
    
    return signal


def core_predict(
    signal: np.ndarray,
    cnn_model: nn.Module,
    xgb_data: Dict[str, Any],
    thresholds: Dict[str, float],
    device: torch.device,
    localization_model: Optional[nn.Module] = None,
    binary_model: Optional[nn.Module] = None,
    ensemble_weight: Optional[float] = None,
    localization_threshold: float = 0.5,
    feature_schema: Optional[Dict[str, Any]] = None,
) -> PredictResult:
    """
    Pure inference: CNN + XGB ensemble, labels, consistency guard, localization.

    No XAI, plotting, or manifest writing.
    """
    if ensemble_weight is None:
        ensemble_weight = get_ensemble_cnn_weight()
    signal, _input_meta = validate_ecg_signal(signal)
    # Localization CNN is trained on raw wfdb amplitudes (channel-first only).
    signal_for_localization = np.asarray(signal, dtype=np.float32).copy()
    signal = apply_superclass_normalization(signal)

    with torch.no_grad():
        signal_tensor = torch.as_tensor(signal, dtype=torch.float32).unsqueeze(0).to(device)
        cnn_logits = cnn_model(signal_tensor)
        cnn_probs = torch.sigmoid(cnn_logits).cpu().numpy()[0]

    cnn_probs_dict = {cls: float(cnn_probs[i]) for i, cls in enumerate(SUPERCLASS_LABELS)}

    embeddings = None
    if xgb_data.get("models"):
        with torch.no_grad():
            embeddings = cnn_model.backbone(signal_tensor).cpu().numpy()

    xgb_probs_dict: Dict[str, float] = {}
    if xgb_data.get("models") and embeddings is not None:
        if feature_schema is not None:
            validate_feature_schema(embeddings.shape, feature_schema)

        xgb_embeddings = embeddings
        if xgb_data.get("scaler") is not None:
            xgb_embeddings = xgb_data["scaler"].transform(embeddings)

        for cls in SUPERCLASS_LABELS:
            if cls in xgb_data["models"]:
                model = xgb_data["models"][cls]
                raw_prob = model.predict_proba(xgb_embeddings)[0, 1]

                if cls in xgb_data.get("calibrators", {}):
                    calibrator = xgb_data["calibrators"][cls]
                    if isinstance(calibrator, IsotonicRegression):
                        prob = calibrator.predict([raw_prob])[0]
                    else:
                        prob = calibrator.predict_proba([[raw_prob]])[0, 1]
                else:
                    prob = raw_prob

                xgb_probs_dict[cls] = float(prob)

    if xgb_probs_dict:
        w = ensemble_weight
        ensemble_probs = {
            cls: w * cnn_probs_dict[cls] + (1 - w) * xgb_probs_dict.get(cls, cnn_probs_dict[cls])
            for cls in SUPERCLASS_LABELS
        }
    else:
        ensemble_probs = cnn_probs_dict

    predicted_labels = [
        cls for cls in SUPERCLASS_LABELS
        if ensemble_probs[cls] >= thresholds.get(cls, 0.5)
    ]

    primary_label, primary_confidence = get_primary_label(ensemble_probs, thresholds)
    norm_prob = 1.0 - max(ensemble_probs.values())

    consistency_result: Optional[ConsistencyResult] = None
    if binary_model is not None:
        try:
            with torch.no_grad():
                binary_logits = binary_model(signal_tensor)
                binary_mi_prob = float(torch.sigmoid(binary_logits).cpu().numpy().flatten()[0])
            consistency_result = check_consistency(
                superclass_mi_prob=ensemble_probs.get("MI", 0.0),
                binary_mi_prob=binary_mi_prob,
                superclass_threshold=thresholds.get("MI", 0.5),
                binary_threshold=0.5,
            )
        except Exception as e:
            print(f"Warning: Consistency check failed: {e}")
            consistency_result = None

    localization_result = None
    if localization_model and "MI" in predicted_labels:
        with torch.no_grad():
            loc_tensor = torch.as_tensor(
                signal_for_localization, dtype=torch.float32
            ).unsqueeze(0).to(device)
            loc_logits = localization_model(loc_tensor)
            loc_probs = torch.sigmoid(loc_logits).cpu().numpy()[0]

        localization_result = {
            region: float(prob)
            for region, prob in zip(MI_LOCALIZATION_REGIONS, loc_probs)
        }
        detected_regions = [
            region for region, prob in localization_result.items()
            if prob >= localization_threshold
        ]
        localization_result["predicted_regions"] = detected_regions

    return PredictResult(
        signal=signal,
        signal_tensor=signal_tensor,
        cnn_probs=cnn_probs_dict,
        xgb_probs=xgb_probs_dict,
        ensemble_probs=ensemble_probs,
        embeddings=embeddings,
        predicted_labels=predicted_labels,
        primary_label=primary_label,
        primary_confidence=primary_confidence,
        norm_prob=norm_prob,
        thresholds=thresholds,
        ensemble_weight=ensemble_weight,
        localization=localization_result,
        consistency=consistency_result,
    )


def generate_explanation(
    predict_result: PredictResult,
    cnn_model: nn.Module,
    xgb_data: Dict[str, Any],
    run_dir: Optional[Path],
    sample_id: str,
    sanity_check: bool = False,
    save_plot: Optional[Path] = None,
) -> ExplanationResult:
    """Delegate XAI coordination to the deep XAI pipeline module."""
    return xai_explain(
        predict_result=predict_result,
        cnn_model=cnn_model,
        xgb_data=xgb_data,
        run_dir=run_dir,
        sample_id=sample_id,
        sanity_check=sanity_check,
        save_plot=save_plot,
    )


def predict(
    signal: np.ndarray,
    cnn_model: nn.Module,
    xgb_data: Dict[str, Any],
    thresholds: Dict[str, float],
    localization_model: Optional[nn.Module],
    device: torch.device,
    binary_model: Optional[nn.Module] = None,
    ensemble_weight: Optional[float] = None,
    localization_threshold: float = 0.5,
    explain: bool = False,
    sanity_check: bool = False,
    save_plot: Optional[Path] = None,
    run_dir: Optional[Path] = None,
    sample_id: str = "sample",
    feature_schema: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Run multi-label prediction (backward-compatible wrapper).

    Composes core_predict() and optionally generate_explanation().
    """
    predict_result = core_predict(
        signal=signal,
        cnn_model=cnn_model,
        xgb_data=xgb_data,
        thresholds=thresholds,
        device=device,
        localization_model=localization_model,
        binary_model=binary_model,
        ensemble_weight=ensemble_weight,
        localization_threshold=localization_threshold,
        feature_schema=feature_schema,
    )

    explanation_result = None
    if explain:
        exp = generate_explanation(
            predict_result=predict_result,
            cnn_model=cnn_model,
            xgb_data=xgb_data,
            run_dir=run_dir,
            sample_id=sample_id,
            sanity_check=sanity_check,
            save_plot=save_plot,
        )
        explanation_result = exp.to_explanation_dict()

    ensemble_probs = predict_result.ensemble_probs
    consistency_result = predict_result.consistency

    return {
        "mode": "multilabel-superclass",
        "multi": {
            "probabilities": {**ensemble_probs, "NORM": predict_result.norm_prob},
            "predicted_labels": (
                predict_result.predicted_labels
                if predict_result.predicted_labels
                else ["NORM"]
            ),
            "thresholds": thresholds,
        },
        "primary": {
            "label": predict_result.primary_label,
            "confidence": predict_result.primary_confidence,
            "rule": "MI-first-then-priority",
        },
        "mi_localization": predict_result.localization,
        "explanation": explanation_result,
        "sources": {
            "cnn": predict_result.cnn_probs,
            "xgb": predict_result.xgb_probs if predict_result.xgb_probs else None,
            "ensemble": ensemble_probs,
        },
        "run_id": run_dir.name if run_dir else None,
        "run_dir": str(run_dir) if run_dir else None,
        "consistency": consistency_result.to_dict() if consistency_result else None,
        "ensemble_weight": predict_result.ensemble_weight,
    }


def main():

    parser = argparse.ArgumentParser(description="Multi-label Superclass Inference")
    parser.add_argument("--input", type=Path, required=True,
                        help="Path to ECG signal (.npz or .npy)")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output path for results (default: stdout)")
    parser.add_argument("--cnn-checkpoint", type=Path, default=DEFAULT_CNN_CHECKPOINT)
    parser.add_argument("--xgb-dir", type=Path, default=DEFAULT_XGB_DIR)
    parser.add_argument("--thresholds", type=Path, default=DEFAULT_THRESHOLDS)
    parser.add_argument("--localization-checkpoint", type=Path, default=DEFAULT_LOCALIZATION_CHECKPOINT)
    parser.add_argument("--ensemble-weight", type=float, default=None,
                        help="CNN weight in ensemble (default: from thresholds artifact)")
    parser.add_argument("--explain", action="store_true", help="Generate Unified XAI explanation")
    parser.add_argument("--sanity-check", action="store_true", help="Run XAI sanity checks")
    parser.add_argument("--save-plot", type=Path, default=None, help="Path to save explanation plot")
    parser.add_argument("--device", type=str, default=None)
    
    args = parser.parse_args()
    ensemble_w = args.ensemble_weight if args.ensemble_weight is not None else get_ensemble_cnn_weight()
    
    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {device}")
    
    # Load models
    print("Loading models...")
    cnn_model = load_cnn_model(args.cnn_checkpoint, device)
    xgb_data = load_xgb_models(args.xgb_dir)
    thresholds = load_thresholds(args.thresholds)
    
    localization_model = load_localization_model(args.localization_checkpoint, device)
    if localization_model:
        print("Loaded MI Localization model.")
    else:
        print("Warning: MI Localization model not found, skipping localization.")
    
    # Load signal
    print(f"Loading signal from {args.input}...")
    signal = load_ecg_signal(args.input)
    
    # Predict
    print("Running inference...")
    result = predict(
        signal, cnn_model, xgb_data, thresholds, localization_model, device,
        ensemble_weight=ensemble_w,
        explain=args.explain,
        sanity_check=args.sanity_check,
        save_plot=args.save_plot,
    )
    
    # Add metadata
    result["versions"] = {
        "cnn_checkpoint": str(args.cnn_checkpoint),
        "xgb_dir": str(args.xgb_dir),
        "thresholds_file": str(args.thresholds),
    }
    
    # Output
    output_json = json.dumps(result, indent=2)
    
    if args.output:
        with open(args.output, "w") as f:
            f.write(output_json)
        print(f"\nResults saved to {args.output}")
    else:
        print("\n" + "=" * 60)
        print("PREDICTION RESULT")
        print("=" * 60)
        print(output_json)
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Primary: {result['primary']['label']} ({result['primary']['confidence']:.3f})")
    print(f"Multi-label: {result['multi']['predicted_labels']}")
    if result.get("mi_localization"):
        print(f"Loc: {result['mi_localization']['predicted_regions']}")


if __name__ == "__main__":
    main()

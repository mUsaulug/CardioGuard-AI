"""
SHAP Explainer for XGBoost One-vs-Rest Models.

Generates per-class SHAP values for multi-label superclass prediction.
Each OVR model gets its own SHAP explanation.

Usage:
    from src.xai.shap_ovr import explain_ovr_models
    shap_results = explain_ovr_models(ovr_models, X_sample)
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional
from pathlib import Path

import numpy as np

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

from xgboost import XGBClassifier


# Class order for OVR models
PATHOLOGY_CLASSES = ["MI", "STTC", "CD", "HYP"]


def explain_single_model(
    model: XGBClassifier,
    X: np.ndarray,
    class_name: str,
    max_samples: int = 100,
) -> Dict[str, Any]:
    """
    Generate SHAP explanation for a single binary XGBoost model.
    
    Args:
        model: Trained XGBClassifier
        X: Feature matrix (n_samples, n_features)
        class_name: Name of the class this model predicts
        max_samples: Maximum samples for SHAP computation
        
    Returns:
        Dictionary with SHAP values and summary statistics
    """
    if not SHAP_AVAILABLE:
        return {"error": "SHAP not installed", "class": class_name}
    
    # Subsample if needed
    if len(X) > max_samples:
        indices = np.random.choice(len(X), max_samples, replace=False)
        X_sample = X[indices]
    else:
        X_sample = X
    
    # Create explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    
    # For binary classifier, shap_values might be a list [neg, pos] or single array
    if isinstance(shap_values, list):
        # Use positive class
        shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    
    # Compute feature importance (mean absolute SHAP)
    feature_importance = np.abs(shap_values).mean(axis=0)
    
    return {
        "class": class_name,
        "shap_values": shap_values,
        "expected_value": float(explainer.expected_value) if np.isscalar(explainer.expected_value) else float(explainer.expected_value[1]),
        "feature_importance": feature_importance,
        "n_samples": len(X_sample),
        "n_features": X_sample.shape[1],
    }


def explain_ovr_models(
    models: Dict[str, XGBClassifier],
    X: np.ndarray,
    max_samples: int = 100,
    class_order: List[str] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Generate SHAP explanations for all OVR models.
    
    Args:
        models: Dictionary of class_name -> XGBClassifier
        X: Feature matrix
        max_samples: Maximum samples per model
        class_order: Order of classes to explain
        
    Returns:
        Dictionary of class_name -> SHAP explanation
    """
    class_order = class_order or PATHOLOGY_CLASSES
    results = {}
    
    for cls in class_order:
        if cls in models:
            print(f"  Computing SHAP for {cls}...")
            results[cls] = explain_single_model(models[cls], X, cls, max_samples)
        else:
            results[cls] = {"error": f"Model not found for {cls}"}
    
    return results


def get_top_features(
    shap_result: Dict[str, Any],
    feature_names: List[str] = None,
    top_k: int = 10,
) -> List[Dict[str, Any]]:
    """
    Get top-K most important features for a class.
    
    Args:
        shap_result: Result from explain_single_model
        feature_names: Optional feature names
        top_k: Number of top features to return
        
    Returns:
        List of {feature, importance, rank} dicts
    """
    if "error" in shap_result:
        return []
    
    importance = shap_result["feature_importance"]
    n_features = len(importance)
    
    # Sort by importance
    sorted_idx = np.argsort(importance)[::-1][:top_k]
    
    results = []
    for rank, idx in enumerate(sorted_idx):
        name = feature_names[idx] if feature_names and idx < len(feature_names) else f"feature_{idx}"
        results.append({
            "rank": rank + 1,
            "feature": name,
            "feature_index": int(idx),
            "importance": float(importance[idx]),
        })
    
    return results


def explain_single_sample(
    models: Dict[str, XGBClassifier],
    X_single: np.ndarray,
    relevant_classes: List[str] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Explain a single sample for relevant classes only.
    
    Args:
        models: OVR models
        X_single: Single sample features (1, n_features) or (n_features,)
        relevant_classes: Only explain these classes (default: all)
        
    Returns:
        Dictionary of class -> explanation
    """
    if not SHAP_AVAILABLE:
        return {"error": "SHAP not installed"}
    
    if X_single.ndim == 1:
        X_single = X_single.reshape(1, -1)
    
    relevant_classes = relevant_classes or PATHOLOGY_CLASSES
    results = {}
    
    for cls in relevant_classes:
        if cls not in models:
            continue
        
        model = models[cls]
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_single)
        
        if isinstance(shap_values, list):
            shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        
        results[cls] = {
            "shap_values": shap_values[0],  # Single sample
            "expected_value": float(explainer.expected_value) if np.isscalar(explainer.expected_value) else float(explainer.expected_value[1]),
            "prediction_contribution": float(shap_values[0].sum()),
        }
    
    return results


def save_shap_summary(
    shap_results: Dict[str, Dict[str, Any]],
    output_dir: Path,
    feature_names: List[str] = None,
    top_k: int = 20,
) -> None:
    """
    Save SHAP summary to files.
    
    Args:
        shap_results: Results from explain_ovr_models
        output_dir: Directory to save files
        feature_names: Optional feature names
        top_k: Number of top features per class
    """
    import json
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    summary = {}
    
    for cls, result in shap_results.items():
        if "error" in result:
            summary[cls] = {"error": result["error"]}
            continue
        
        top_features = get_top_features(result, feature_names, top_k)
        
        summary[cls] = {
            "n_samples": result["n_samples"],
            "n_features": result["n_features"],
            "expected_value": result["expected_value"],
            "top_features": top_features,
        }
        
        # Save SHAP values as npz
        np.savez(
            output_dir / f"shap_{cls}.npz",
            shap_values=result["shap_values"],
            feature_importance=result["feature_importance"],
        )
    
    # Save summary JSON
    with open(output_dir / "shap_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"SHAP results saved to {output_dir}")

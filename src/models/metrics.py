"""Metric utilities for model evaluation."""

from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize


def compute_classification_metrics(y_true: np.ndarray, y_logits: np.ndarray) -> Dict[str, float]:
    """Compute ROC-AUC, PR-AUC, F1, and accuracy from logits."""

    y_true = np.asarray(y_true)
    y_logits = np.asarray(y_logits)
    if y_logits.ndim > 1 and y_logits.shape[1] > 1:
        return compute_multiclass_metrics(y_true, y_logits)
    y_probs = 1 / (1 + np.exp(-y_logits))
    y_pred = (y_probs >= 0.5).astype(int)

    roc_auc = roc_auc_score(y_true, y_probs)
    pr_auc = average_precision_score(y_true, y_probs)
    f1 = f1_score(y_true, y_pred)
    accuracy = float((y_pred == y_true).mean())

    return {
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "f1": float(f1),
        "accuracy": accuracy,
    }


def compute_multiclass_metrics(y_true: np.ndarray, y_logits: np.ndarray) -> Dict[str, float]:
    """Compute macro ROC-AUC, PR-AUC, F1, and accuracy for multi-class logits."""

    y_true = np.asarray(y_true)
    y_logits = np.asarray(y_logits)
    num_classes = y_logits.shape[1]
    y_probs = np.exp(y_logits - y_logits.max(axis=1, keepdims=True))
    y_probs = y_probs / y_probs.sum(axis=1, keepdims=True)
    y_pred = np.argmax(y_probs, axis=1)

    y_true_bin = label_binarize(y_true, classes=np.arange(num_classes))
    roc_auc = roc_auc_score(y_true_bin, y_probs, average="macro", multi_class="ovr")
    pr_auc = average_precision_score(y_true_bin, y_probs, average="macro")
    f1 = f1_score(y_true, y_pred, average="macro")
    accuracy = float((y_pred == y_true).mean())

    return {
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "f1": float(f1),
        "accuracy": accuracy,
    }


# Class names for multi-label superclass (NORM is derived)
SUPERCLASS_NAMES = ["MI", "STTC", "CD", "HYP"]


def compute_multilabel_metrics(
    y_true: np.ndarray,
    y_logits: np.ndarray,
    class_names: list = None,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute metrics for multi-label classification (4 labels: MI, STTC, CD, HYP).
    
    Args:
        y_true: Ground truth labels, shape (n_samples, n_classes)
        y_logits: Model logits, shape (n_samples, n_classes)
        class_names: Optional list of class names
        threshold: Threshold for binary prediction
        
    Returns:
        Dictionary with macro/micro AUROC, AUPRC, F1, and per-class metrics
    """
    y_true = np.asarray(y_true)
    y_logits = np.asarray(y_logits)
    
    if class_names is None:
        class_names = SUPERCLASS_NAMES
    
    # Sigmoid to get probabilities
    y_probs = 1 / (1 + np.exp(-y_logits))
    y_pred = (y_probs >= threshold).astype(int)
    
    n_classes = y_true.shape[1]
    
    # Per-class metrics
    per_class = {}
    valid_classes_auroc = []
    valid_classes_auprc = []
    
    for i in range(n_classes):
        class_name = class_names[i] if i < len(class_names) else f"class_{i}"
        y_true_i = y_true[:, i]
        y_prob_i = y_probs[:, i]
        y_pred_i = y_pred[:, i]
        
        # Skip if only one class present (can't compute AUC)
        if len(np.unique(y_true_i)) < 2:
            per_class[class_name] = {
                "auroc": None,
                "auprc": None,
                "f1": float(f1_score(y_true_i, y_pred_i, zero_division=0)),
                "support": int(y_true_i.sum()),
            }
            continue
        
        auroc = roc_auc_score(y_true_i, y_prob_i)
        auprc = average_precision_score(y_true_i, y_prob_i)
        f1_val = f1_score(y_true_i, y_pred_i, zero_division=0)
        
        per_class[class_name] = {
            "auroc": float(auroc),
            "auprc": float(auprc),
            "f1": float(f1_val),
            "support": int(y_true_i.sum()),
        }
        valid_classes_auroc.append(auroc)
        valid_classes_auprc.append(auprc)
    
    # Macro metrics (average over classes)
    macro_auroc = float(np.mean(valid_classes_auroc)) if valid_classes_auroc else 0.0
    macro_auprc = float(np.mean(valid_classes_auprc)) if valid_classes_auprc else 0.0
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    
    # Micro metrics (flatten all)
    micro_f1 = float(f1_score(y_true.ravel(), y_pred.ravel(), zero_division=0))
    
    # Exact match (all labels correct)
    exact_match = float((y_pred == y_true).all(axis=1).mean())
    
    # Hamming accuracy (per-label accuracy)
    hamming_acc = float((y_pred == y_true).mean())
    
    return {
        "macro_auroc": macro_auroc,
        "macro_auprc": macro_auprc,
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "exact_match": exact_match,
        "hamming_accuracy": hamming_acc,
        "per_class": per_class,
    }

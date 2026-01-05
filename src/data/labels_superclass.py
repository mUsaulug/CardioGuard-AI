"""
CardioGuard-AI Superclass Label Module

Multi-label labeling with derived NORM for 5 superclasses (NORM, MI, STTC, CD, HYP).
NORM is NOT extracted from data - it's derived as: NORM=1 iff (MI=0 & STTC=0 & CD=0 & HYP=0)

This ensures NORM never co-occurs with any pathology class.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

from src.config import DIAGNOSTIC_SUPERCLASSES


# Pathology classes (NORM is derived, not in this list)
PATHOLOGY_CLASSES = ["MI", "STTC", "CD", "HYP"]
NUM_PATHOLOGY_CLASSES = 4


def extract_pathology_superclasses(
    scp_codes: Dict[str, float],
    scp_df: pd.DataFrame,
    min_likelihood: float = 0.0
) -> List[str]:
    """
    Extract ONLY pathology superclasses (MI, STTC, CD, HYP) from a record.
    
    NORM is NOT extracted here - it will be derived later.
    
    Args:
        scp_codes: Dictionary of {code: likelihood}
        scp_df: SCP statements DataFrame with diagnostic_class column
        min_likelihood: Minimum likelihood threshold
        
    Returns:
        List of pathology superclasses present (subset of ["MI", "STTC", "CD", "HYP"])
    """
    diagnostic_scp = scp_df[scp_df["diagnostic"] == 1.0]
    
    superclasses = set()
    for code, likelihood in scp_codes.items():
        if likelihood > min_likelihood and code in diagnostic_scp.index:
            superclass = diagnostic_scp.loc[code, "diagnostic_class"]
            # Only add pathology classes - skip NORM
            if pd.notna(superclass) and superclass in PATHOLOGY_CLASSES:
                superclasses.add(superclass)
    
    return list(superclasses)


def extract_y_multi4(
    df: pd.DataFrame,
    scp_df: pd.DataFrame,
    min_likelihood: float = 0.0
) -> np.ndarray:
    """
    Extract 4-class multi-hot labels (MI, STTC, CD, HYP).
    
    NORM is NOT included - use derive_norm() to compute it.
    
    Args:
        df: PTB-XL metadata DataFrame with scp_codes column
        scp_df: SCP statements DataFrame
        min_likelihood: Minimum likelihood threshold
        
    Returns:
        y_multi4: (N, 4) array with multi-hot labels [MI, STTC, CD, HYP]
    """
    n_samples = len(df)
    y_multi4 = np.zeros((n_samples, NUM_PATHOLOGY_CLASSES), dtype=np.float32)
    
    for i, scp_codes in enumerate(df["scp_codes"]):
        superclasses = extract_pathology_superclasses(scp_codes, scp_df, min_likelihood)
        for cls in superclasses:
            if cls in PATHOLOGY_CLASSES:
                idx = PATHOLOGY_CLASSES.index(cls)
                y_multi4[i, idx] = 1.0
    
    return y_multi4


def derive_norm(y_multi4: np.ndarray) -> np.ndarray:
    """
    Derive NORM label from pathology labels.
    
    NORM = 1 iff ALL pathology labels are 0.
    
    Args:
        y_multi4: (N, 4) array with multi-hot labels [MI, STTC, CD, HYP]
        
    Returns:
        norm: (N,) array with derived NORM labels
    """
    # NORM = 1 iff no pathology
    any_pathology = y_multi4.sum(axis=1) > 0
    norm = (~any_pathology).astype(np.float32)
    return norm


def add_superclass_labels_derived(
    df: pd.DataFrame,
    scp_df: pd.DataFrame,
    min_likelihood: float = 0.0
) -> pd.DataFrame:
    """
    Add superclass labels with derived NORM.
    
    This is the CORRECT way to label for multi-label training:
    - Extracts pathology labels (MI, STTC, CD, HYP) from data
    - Derives NORM: NORM=1 iff no pathology present
    
    Args:
        df: PTB-XL metadata DataFrame
        scp_df: SCP statements DataFrame
        min_likelihood: Minimum likelihood threshold
        
    Returns:
        DataFrame with added columns:
        - 'superclass_pathologies': list of pathology superclasses
        - 'label_MI', 'label_STTC', 'label_CD', 'label_HYP': binary columns
        - 'label_NORM': derived binary column
        - 'y_multi4': (4,) array for each row
    """
    df = df.copy()
    
    # Extract y_multi4
    y_multi4 = extract_y_multi4(df, scp_df, min_likelihood)
    
    # Derive NORM
    norm = derive_norm(y_multi4)
    
    # Add pathology list
    df["superclass_pathologies"] = [
        [PATHOLOGY_CLASSES[j] for j in range(4) if y_multi4[i, j] == 1]
        for i in range(len(df))
    ]
    
    # Add binary columns
    for j, cls in enumerate(PATHOLOGY_CLASSES):
        df[f"label_{cls}"] = y_multi4[:, j].astype(int)
    
    df["label_NORM"] = norm.astype(int)
    
    # Add y_multi4 as list (for easier access)
    df["y_multi4"] = [y_multi4[i] for i in range(len(df))]
    
    # Also add full superclass list WITH derived NORM
    def get_full_superclass_list(pathologies: List[str], is_norm: int) -> List[str]:
        if is_norm == 1:
            return ["NORM"]
        return pathologies
    
    df["diagnostic_superclass_derived"] = [
        get_full_superclass_list(p, n) 
        for p, n in zip(df["superclass_pathologies"], df["label_NORM"])
    ]
    
    return df


def compute_cooccurrence_derived(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute co-occurrence matrix using DERIVED superclass labels.
    
    Args:
        df: DataFrame with 'label_NORM', 'label_MI', etc. columns
        
    Returns:
        Co-occurrence matrix (5x5) as DataFrame
    """
    all_classes = ["NORM", "MI", "STTC", "CD", "HYP"]
    cooccur = np.zeros((5, 5), dtype=int)
    
    for _, row in df.iterrows():
        for i, c1 in enumerate(all_classes):
            for j, c2 in enumerate(all_classes):
                if row[f"label_{c1}"] == 1 and row[f"label_{c2}"] == 1:
                    cooccur[i, j] += 1
    
    return pd.DataFrame(cooccur, index=all_classes, columns=all_classes)


def compute_pos_weight_train(
    y_multi4: np.ndarray
) -> np.ndarray:
    """
    Compute pos_weight for BCEWithLogitsLoss from training labels.
    
    pos_weight[i] = n_negative[i] / n_positive[i]
    
    Args:
        y_multi4: (N, 4) training labels
        
    Returns:
        pos_weight: (4,) array
    """
    n_samples = len(y_multi4)
    pos_counts = y_multi4.sum(axis=0)
    neg_counts = n_samples - pos_counts
    
    # Avoid division by zero
    pos_weight = np.where(pos_counts > 0, neg_counts / pos_counts, 1.0)
    
    return pos_weight.astype(np.float32)


def build_label_report(
    df: pd.DataFrame,
    y_multi4: np.ndarray,
    prefix: str = ""
) -> Dict[str, Any]:
    """
    Build comprehensive label statistics report.
    
    Args:
        df: DataFrame with derived labels
        y_multi4: (N, 4) multi-hot labels
        prefix: Optional prefix for split name
        
    Returns:
        Report dictionary with all statistics
    """
    n_samples = len(df)
    norm = derive_norm(y_multi4)
    
    # Per-class stats
    class_stats = {}
    for j, cls in enumerate(PATHOLOGY_CLASSES):
        pos = int(y_multi4[:, j].sum())
        class_stats[cls] = {
            "positive": pos,
            "negative": n_samples - pos,
            "pos_rate": pos / n_samples if n_samples > 0 else 0,
            "pos_weight": (n_samples - pos) / pos if pos > 0 else 1.0
        }
    
    # NORM (derived) stats
    norm_pos = int(norm.sum())
    class_stats["NORM (derived)"] = {
        "positive": norm_pos,
        "negative": n_samples - norm_pos,
        "pos_rate": norm_pos / n_samples if n_samples > 0 else 0,
        "note": "derived, not trained"
    }
    
    # Multi-label stats
    labels_per_sample = y_multi4.sum(axis=1)
    multi_label_count = int((labels_per_sample > 1).sum())
    
    report = {
        "prefix": prefix,
        "n_samples": n_samples,
        "class_stats": class_stats,
        "multi_label": {
            "avg_labels_per_sample": float(labels_per_sample.mean()),
            "samples_with_multiple_labels": multi_label_count,
            "samples_with_no_pathology": int((labels_per_sample == 0).sum())
        }
    }
    
    return report


def verify_norm_cooccurrence(df: pd.DataFrame) -> Tuple[bool, str]:
    """
    Verify that derived NORM never co-occurs with any pathology.
    
    This is a critical assertion - if it fails, there's a label bug.
    
    Args:
        df: DataFrame with 'label_NORM', 'label_MI', etc. columns
        
    Returns:
        (passed, message)
    """
    # NORM should never co-occur with any pathology
    norm_samples = df[df["label_NORM"] == 1]
    
    for cls in PATHOLOGY_CLASSES:
        overlap = (norm_samples[f"label_{cls}"] == 1).sum()
        if overlap > 0:
            return False, f"NORM-{cls} overlap: {overlap} samples (should be 0)"
    
    return True, f"PASS: NORM does not co-occur with any pathology (NORM count: {len(norm_samples)})"


# Quick test function
def quick_sanity_check(df: pd.DataFrame, scp_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Run quick sanity checks on the labeling.
    
    Returns dict with all check results.
    """
    # Add derived labels
    df_labeled = add_superclass_labels_derived(df, scp_df)
    
    # Check NORM co-occurrence
    passed, msg = verify_norm_cooccurrence(df_labeled)
    
    # Compute co-occurrence
    cooccur = compute_cooccurrence_derived(df_labeled)
    
    return {
        "norm_check_passed": passed,
        "norm_check_message": msg,
        "cooccurrence_matrix": cooccur,
        "norm_count": int(df_labeled["label_NORM"].sum()),
        "total_samples": len(df_labeled)
    }

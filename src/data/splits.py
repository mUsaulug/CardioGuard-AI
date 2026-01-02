"""
CardioGuard-AI Data Splitting Module

Functions for patient-level train/val/test splitting.
Uses PTB-XL's strat_fold to prevent data leakage between patients.
"""

from typing import List, Tuple

import numpy as np
import pandas as pd

from src.config import PTBXLConfig


def get_standard_split(
    df: pd.DataFrame,
    train_folds: List[int] = None,
    val_folds: List[int] = None,
    test_folds: List[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get standard train/val/test split using PTB-XL's strat_fold.
    
    Default split follows PTB-XL benchmark:
    - Train: folds 1-8
    - Validation: fold 9
    - Test: fold 10
    
    Args:
        df: PTB-XL metadata DataFrame with strat_fold column
        train_folds: List of fold numbers for training (default [1-8])
        val_folds: List of fold numbers for validation (default [9])
        test_folds: List of fold numbers for testing (default [10])
        
    Returns:
        Tuple of (train_indices, val_indices, test_indices) as numpy arrays
        Indices refer to DataFrame index values (ecg_id)
        
    Example:
        >>> train_idx, val_idx, test_idx = get_standard_split(df)
        >>> X_train = df.loc[train_idx]
        >>> print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
    """
    if train_folds is None:
        train_folds = [1, 2, 3, 4, 5, 6, 7, 8]
    if val_folds is None:
        val_folds = [9]
    if test_folds is None:
        test_folds = [10]
    
    train_mask = df["strat_fold"].isin(train_folds)
    val_mask = df["strat_fold"].isin(val_folds)
    test_mask = df["strat_fold"].isin(test_folds)
    
    train_indices = df[train_mask].index.values
    val_indices = df[val_mask].index.values
    test_indices = df[test_mask].index.values
    
    return train_indices, val_indices, test_indices


def get_split_from_config(
    df: pd.DataFrame,
    config: PTBXLConfig
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get train/val/test split using configuration settings.
    
    Args:
        df: PTB-XL metadata DataFrame
        config: PTBXLConfig instance
        
    Returns:
        Tuple of (train_indices, val_indices, test_indices)
    """
    return get_standard_split(
        df,
        train_folds=config.train_folds,
        val_folds=config.val_folds,
        test_folds=config.test_folds
    )


def verify_no_patient_leakage(
    df: pd.DataFrame,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    test_indices: np.ndarray
) -> bool:
    """
    Verify that no patient appears in multiple splits.
    
    Args:
        df: PTB-XL metadata DataFrame with patient_id column
        train_indices: Training set indices
        val_indices: Validation set indices
        test_indices: Test set indices
        
    Returns:
        True if no leakage (all splits have disjoint patients)
        
    Raises:
        ValueError: If patient leakage is detected
    """
    train_patients = set(df.loc[train_indices, "patient_id"].dropna())
    val_patients = set(df.loc[val_indices, "patient_id"].dropna())
    test_patients = set(df.loc[test_indices, "patient_id"].dropna())
    
    train_val_overlap = train_patients & val_patients
    train_test_overlap = train_patients & test_patients
    val_test_overlap = val_patients & test_patients
    
    if train_val_overlap:
        raise ValueError(
            f"Patient leakage between train and val: {len(train_val_overlap)} patients"
        )
    
    if train_test_overlap:
        raise ValueError(
            f"Patient leakage between train and test: {len(train_test_overlap)} patients"
        )
    
    if val_test_overlap:
        raise ValueError(
            f"Patient leakage between val and test: {len(val_test_overlap)} patients"
        )
    
    return True


def get_split_statistics(
    df: pd.DataFrame,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    test_indices: np.ndarray,
    label_column: str = None
) -> dict:
    """
    Get statistics about the data split.
    
    Args:
        df: PTB-XL metadata DataFrame
        train_indices: Training set indices
        val_indices: Validation set indices
        test_indices: Test set indices
        label_column: Optional label column for class distribution
        
    Returns:
        Dictionary with split statistics
    """
    total = len(df)
    
    stats = {
        "total_samples": total,
        "train": {
            "samples": len(train_indices),
            "percentage": len(train_indices) / total * 100,
            "unique_patients": df.loc[train_indices, "patient_id"].nunique()
        },
        "val": {
            "samples": len(val_indices),
            "percentage": len(val_indices) / total * 100,
            "unique_patients": df.loc[val_indices, "patient_id"].nunique()
        },
        "test": {
            "samples": len(test_indices),
            "percentage": len(test_indices) / total * 100,
            "unique_patients": df.loc[test_indices, "patient_id"].nunique()
        },
        "total_unique_patients": df["patient_id"].nunique()
    }
    
    # Add label distribution if specified
    if label_column and label_column in df.columns:
        for split_name, indices in [
            ("train", train_indices),
            ("val", val_indices),
            ("test", test_indices)
        ]:
            label_counts = df.loc[indices, label_column].value_counts().to_dict()
            stats[split_name]["label_distribution"] = label_counts
    
    return stats


def filter_split_by_label(
    df: pd.DataFrame,
    indices: np.ndarray,
    label_column: str,
    valid_labels: List[int]
) -> np.ndarray:
    """
    Filter split indices to only include samples with valid labels.
    
    Args:
        df: PTB-XL metadata DataFrame
        indices: Original split indices
        label_column: Label column name
        valid_labels: List of valid label values to keep
        
    Returns:
        Filtered indices array
    """
    split_df = df.loc[indices]
    valid_mask = split_df[label_column].isin(valid_labels)
    return split_df[valid_mask].index.values

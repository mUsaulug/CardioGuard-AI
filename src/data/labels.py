"""
CardioGuard-AI Label Mapping Module

Functions for extracting and mapping diagnostic labels from PTB-XL data.
Supports both binary (MI vs NORM) and 5-class (NORM/MI/STTC/CD/HYP) tasks.
"""

from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd

from src.config import MI_CODES, DIAGNOSTIC_SUPERCLASSES


def get_mi_codes() -> List[str]:
    """
    Get all SCP codes that indicate Myocardial Infarction.
    
    Returns:
        List of MI-related SCP code names
    """
    return MI_CODES.copy()


def extract_codes_above_threshold(
    scp_codes: Dict[str, float],
    min_likelihood: float = 0.0
) -> Set[str]:
    """
    Extract SCP codes from a record that exceed the likelihood threshold.
    
    Args:
        scp_codes: Dictionary of {code: likelihood} from a single record
        min_likelihood: Minimum likelihood to consider (default 0 = include all)
        
    Returns:
        Set of code names above threshold
    """
    return {code for code, likelihood in scp_codes.items() 
            if likelihood > min_likelihood}


def has_mi_code(scp_codes: Dict[str, float], min_likelihood: float = 0.0) -> bool:
    """
    Check if a record has any MI-related code above threshold.
    
    Args:
        scp_codes: Dictionary of {code: likelihood}
        min_likelihood: Minimum likelihood threshold
        
    Returns:
        True if any MI code is present above threshold
    """
    codes_present = extract_codes_above_threshold(scp_codes, min_likelihood)
    mi_codes_set = set(MI_CODES)
    return bool(codes_present & mi_codes_set)


def has_norm_code(scp_codes: Dict[str, float], min_likelihood: float = 0.0) -> bool:
    """
    Check if a record has NORM code above threshold.
    
    Args:
        scp_codes: Dictionary of {code: likelihood}
        min_likelihood: Minimum likelihood threshold
        
    Returns:
        True if NORM is present above threshold
    """
    return scp_codes.get("NORM", 0) > min_likelihood


def add_binary_mi_labels(
    df: pd.DataFrame,
    scp_df: pd.DataFrame,
    min_likelihood: float = 0.0
) -> pd.DataFrame:
    """
    Add binary MI vs NORM labels to the metadata DataFrame.
    
    Label scheme:
    - 1: MI positive (has any MI code)
    - 0: NORM (has NORM code, no MI codes)
    - -1: Excluded (ambiguous or neither)
    
    Args:
        df: PTB-XL metadata DataFrame with scp_codes column
        scp_df: SCP statements DataFrame (for validation)
        min_likelihood: Minimum likelihood threshold
        
    Returns:
        DataFrame with added 'label_mi_norm' column
        
    Example:
        >>> df = add_binary_mi_labels(df, scp_df)
        >>> print(df['label_mi_norm'].value_counts())
    """
    df = df.copy()
    
    def assign_label(scp_codes: dict) -> int:
        is_mi = has_mi_code(scp_codes, min_likelihood)
        is_norm = has_norm_code(scp_codes, min_likelihood)
        
        if is_mi and not is_norm:
            return 1  # Pure MI
        elif is_norm and not is_mi:
            return 0  # Pure NORM
        else:
            return -1  # Ambiguous (both or neither)
    
    df["label_mi_norm"] = df["scp_codes"].apply(assign_label)
    
    return df


def aggregate_diagnostic_superclass(
    scp_codes: Dict[str, float],
    scp_df: pd.DataFrame,
    min_likelihood: float = 0.0
) -> List[str]:
    """
    Aggregate diagnostic superclasses for a single record.
    
    Maps each SCP code to its superclass (NORM, MI, STTC, CD, HYP)
    and returns unique superclasses present.
    
    Args:
        scp_codes: Dictionary of {code: likelihood}
        scp_df: SCP statements DataFrame with diagnostic_class column
        min_likelihood: Minimum likelihood threshold
        
    Returns:
        List of unique diagnostic superclasses present
    """
    # Get diagnostic-only codes from scp_statements
    diagnostic_scp = scp_df[scp_df["diagnostic"] == 1.0]
    
    superclasses = []
    for code, likelihood in scp_codes.items():
        if likelihood > min_likelihood and code in diagnostic_scp.index:
            superclass = diagnostic_scp.loc[code, "diagnostic_class"]
            if pd.notna(superclass):
                superclasses.append(superclass)
    
    return list(set(superclasses))


def add_superclass_labels(
    df: pd.DataFrame,
    scp_df: pd.DataFrame,
    min_likelihood: float = 0.0
) -> pd.DataFrame:
    """
    Add diagnostic superclass list to each record.
    
    Args:
        df: PTB-XL metadata DataFrame with scp_codes column
        scp_df: SCP statements DataFrame
        min_likelihood: Minimum likelihood threshold
        
    Returns:
        DataFrame with added 'diagnostic_superclass' column (list of superclasses)
    """
    df = df.copy()
    
    df["diagnostic_superclass"] = df["scp_codes"].apply(
        lambda x: aggregate_diagnostic_superclass(x, scp_df, min_likelihood)
    )
    
    return df


def add_5class_labels(
    df: pd.DataFrame,
    scp_df: pd.DataFrame,
    min_likelihood: float = 0.0,
    multi_label: bool = True
) -> pd.DataFrame:
    """
    Add 5-class diagnostic labels (NORM, MI, STTC, CD, HYP).
    
    Args:
        df: PTB-XL metadata DataFrame with scp_codes column
        scp_df: SCP statements DataFrame
        min_likelihood: Minimum likelihood threshold
        multi_label: If True, add multi-hot encoding. If False, single dominant label.
        
    Returns:
        DataFrame with added label columns:
        - If multi_label: 'label_NORM', 'label_MI', etc. (binary each)
        - If not multi_label: 'label_5class' (single class index)
    """
    # First get superclass lists
    df = add_superclass_labels(df, scp_df, min_likelihood)
    
    if multi_label:
        # Multi-hot encoding
        for i, superclass in enumerate(DIAGNOSTIC_SUPERCLASSES):
            df[f"label_{superclass}"] = df["diagnostic_superclass"].apply(
                lambda x: 1 if superclass in x else 0
            )
    else:
        # Single label - take first superclass or -1 if ambiguous
        def get_single_label(superclasses: List[str]) -> int:
            if len(superclasses) == 1:
                return DIAGNOSTIC_SUPERCLASSES.index(superclasses[0])
            elif len(superclasses) == 0:
                return -1  # No diagnostic class
            else:
                # Multiple classes - take first in order of priority
                for sc in DIAGNOSTIC_SUPERCLASSES:
                    if sc in superclasses:
                        return DIAGNOSTIC_SUPERCLASSES.index(sc)
                return -1
        
        df["label_5class"] = df["diagnostic_superclass"].apply(get_single_label)
    
    return df


def get_label_statistics(
    df: pd.DataFrame,
    label_column: str = "label_mi_norm"
) -> Dict[str, int]:
    """
    Get label distribution statistics.
    
    Args:
        df: DataFrame with label column
        label_column: Name of the label column
        
    Returns:
        Dictionary with label counts
    """
    counts = df[label_column].value_counts().to_dict()
    
    # Add percentage
    total = len(df)
    stats = {
        "total": total,
        "counts": counts,
        "percentages": {k: v / total * 100 for k, v in counts.items()}
    }
    
    return stats


def filter_valid_samples(
    df: pd.DataFrame,
    label_column: str = "label_mi_norm",
    exclude_value: int = -1
) -> pd.DataFrame:
    """
    Filter out samples with excluded/invalid labels.
    
    Args:
        df: DataFrame with label column
        label_column: Name of the label column
        exclude_value: Value to exclude (default -1)
        
    Returns:
        Filtered DataFrame with only valid samples
    """
    return df[df[label_column] != exclude_value].copy()

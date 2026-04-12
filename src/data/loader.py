"""
CardioGuard-AI Metadata Loader Module

Functions for loading and parsing PTB-XL metadata files.
Optimized for memory efficiency with large datasets.
"""

import ast
from pathlib import Path
from typing import Union

import pandas as pd


def load_ptbxl_metadata(
    path: Union[str, Path],
    parse_scp_codes: bool = True
) -> pd.DataFrame:
    """
    Load PTB-XL database metadata from CSV.
    
    Args:
        path: Path to ptbxl_database.csv
        parse_scp_codes: If True, parse the scp_codes column from string to dict
        
    Returns:
        DataFrame with ECG metadata, indexed by ecg_id
        
    Example:
        >>> df = load_ptbxl_metadata("path/to/ptbxl_database.csv")
        >>> print(df.shape)  # (21837, 27)
        >>> print(df.columns.tolist())
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Metadata file not found: {path}")
    
    # Load CSV with ecg_id as index
    df = pd.read_csv(path, index_col="ecg_id")
    
    # Parse scp_codes from string representation to dict
    if parse_scp_codes:
        df["scp_codes"] = df["scp_codes"].apply(_parse_scp_codes)
    
    return df


def load_scp_statements(path: Union[str, Path]) -> pd.DataFrame:
    """
    Load SCP statements mapping from CSV.
    
    This file maps SCP codes to their properties:
    - description: Human-readable description
    - diagnostic: 1.0 if diagnostic statement
    - form: 1.0 if form-related
    - rhythm: 1.0 if rhythm-related
    - diagnostic_class: Superclass (NORM, MI, STTC, CD, HYP)
    - diagnostic_subclass: Subclass for finer granularity
    
    Args:
        path: Path to scp_statements.csv
        
    Returns:
        DataFrame indexed by SCP code name
        
    Example:
        >>> scp = load_scp_statements("path/to/scp_statements.csv")
        >>> print(scp.loc["IMI", "diagnostic_class"])  # "MI"
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"SCP statements file not found: {path}")
    
    # Load with first column as index (unnamed, contains SCP codes)
    df = pd.read_csv(path, index_col=0)
    
    return df


def get_diagnostic_codes(scp_df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter SCP statements to only diagnostic codes.
    
    Args:
        scp_df: Full SCP statements DataFrame from load_scp_statements()
        
    Returns:
        DataFrame with only diagnostic (not form/rhythm) codes
    """
    return scp_df[scp_df["diagnostic"] == 1.0].copy()


def _parse_scp_codes(scp_string: str) -> dict:
    """
    Parse SCP codes string to dictionary.
    
    The scp_codes column contains string representations of dicts like:
    "{'NORM': 100.0, 'SR': 0.0}"
    
    Args:
        scp_string: String representation of SCP codes dict
        
    Returns:
        Parsed dictionary {code: likelihood}
    """
    try:
        return ast.literal_eval(scp_string)
    except (ValueError, SyntaxError):
        return {}

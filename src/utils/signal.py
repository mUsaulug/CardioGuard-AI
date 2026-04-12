"""
Signal Processing Utilities for CardioGuard-AI.

Centralized utility functions for ECG signal handling.
"""

from __future__ import annotations

import numpy as np


def ensure_channel_first(signal: np.ndarray) -> np.ndarray:
    """
    Ensure ECG signal is in (channels, timesteps) format.
    
    PTB-XL standard: 12 leads, ~1000 timesteps (10 sec @ 100Hz)
    
    Args:
        signal: ECG signal array, either (C, T) or (T, C)
        
    Returns:
        Signal in (channels, timesteps) format
        
    Raises:
        ValueError: If signal cannot be interpreted as 12-lead ECG
    """
    if signal.ndim == 1:
        # Single lead - reshape to (1, T)
        signal = signal.reshape(1, -1)
    
    if signal.ndim != 2:
        raise ValueError(f"Expected 2D signal, got shape {signal.shape}")
    
    # Heuristic for 12-lead ECG
    if signal.shape[0] == 12:
        return signal
    if signal.shape[1] == 12:
        return signal.T
    
    # Fallback: assume (T, C) if first dim is larger
    if signal.shape[0] > signal.shape[1]:
        return signal.T
    
    return signal


def normalize_signal(
    signal: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    eps: float = 1e-8
) -> np.ndarray:
    """
    Normalize signal using channel-wise mean and std.
    
    Args:
        signal: ECG signal (C, T) or (T, C)
        mean: Channel means (C,)
        std: Channel stds (C,)
        eps: Small value to avoid division by zero
        
    Returns:
        Normalized signal in same format as input
    """
    signal = ensure_channel_first(signal)
    
    # Broadcast: (C, 1) for channel-wise normalization
    mean = np.asarray(mean).reshape(-1, 1)
    std = np.asarray(std).reshape(-1, 1)
    
    return (signal - mean) / (std + eps)

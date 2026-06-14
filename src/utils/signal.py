"""
Signal Processing Utilities for CardioGuard-AI.

Centralized utility functions for ECG signal handling.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

DEFAULT_NORM_STATS_NPZ = Path("logs/superclass_cnn/normalization_stats.npz")
DEFAULT_NORM_STATS_JSON = Path("features_out/superclass_feature_config.json")
EXPECTED_LEADS = 12
PTBXL_SAMPLE_RATE_HZ = 100
PTBXL_EXPECTED_TIMESTEPS = 1000
PTBXL_DURATION_SEC = 10.0
MIN_TIMESTEPS = 200
MAX_TIMESTEPS = 10_000
MAX_ABS_AMPLITUDE_WARN = 25.0
_NORM_EPS = 1e-8


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


def validate_ecg_signal(
    arr: np.ndarray,
    *,
    sample_rate_hz: int = PTBXL_SAMPLE_RATE_HZ,
    expected_timesteps: int = PTBXL_EXPECTED_TIMESTEPS,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Validate a 12-lead ECG array for inference.

    Returns:
        (validated float32 array in channel-first order, metadata dict)

    Raises:
        ValueError: invalid shape, non-finite values, or unusable signal
    """
    if not isinstance(arr, np.ndarray):
        raise ValueError("ECG signal must be a numpy array")

    signal = ensure_channel_first(np.asarray(arr, dtype=np.float64))

    if signal.ndim != 2:
        raise ValueError(
            f"ECG must be 2-dimensional (leads × timesteps), got shape {signal.shape}"
        )

    n_leads, n_steps = int(signal.shape[0]), int(signal.shape[1])
    if n_leads != EXPECTED_LEADS:
        raise ValueError(
            f"ECG must have {EXPECTED_LEADS} leads, got {n_leads}. "
            f"Expected ({EXPECTED_LEADS}, timesteps) or (timesteps, {EXPECTED_LEADS})."
        )

    if n_steps < MIN_TIMESTEPS or n_steps > MAX_TIMESTEPS:
        raise ValueError(
            f"Timestep count {n_steps} out of range [{MIN_TIMESTEPS}, {MAX_TIMESTEPS}]"
        )

    if not np.all(np.isfinite(signal)):
        non_finite = int(signal.size - np.isfinite(signal).sum())
        raise ValueError(f"ECG contains {non_finite} non-finite values (NaN/Inf)")

    amp_max = float(np.max(np.abs(signal)))
    amp_std = float(np.std(signal))
    if amp_std < 1e-9:
        raise ValueError("ECG signal is flat (zero variance) — cannot analyze")

    quality_flags: list[str] = []
    if n_steps != expected_timesteps:
        quality_flags.append(
            f"timesteps_not_ptbxl_standard:got_{n_steps}_expected_{expected_timesteps}"
        )
    if amp_max > MAX_ABS_AMPLITUDE_WARN:
        quality_flags.append(f"amplitude_high:max_abs_{amp_max:.2f}")
    if amp_max < 1e-4:
        quality_flags.append(f"amplitude_very_low:max_abs_{amp_max:.2e}")

    validated = signal.astype(np.float32)
    duration_sec = round(n_steps / sample_rate_hz, 3)

    meta: Dict[str, Any] = {
        "shape": [n_leads, n_steps],
        "leads": n_leads,
        "timesteps": n_steps,
        "sample_rate_hz": sample_rate_hz,
        "duration_sec": duration_sec,
        "dtype": str(validated.dtype),
        "amplitude_max": amp_max,
        "amplitude_std": amp_std,
        "quality_flags": quality_flags or None,
        "validation": {
            "valid": True,
            "ptbxl_standard": n_steps == expected_timesteps,
        },
    }
    return validated, meta


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


@lru_cache(maxsize=1)
def load_superclass_norm_stats(
    stats_npz: str = str(DEFAULT_NORM_STATS_NPZ),
    stats_json: str = str(DEFAULT_NORM_STATS_JSON),
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load per-channel z-score stats used during superclass CNN training.

    Resolution order:
    1. logs/superclass_cnn/normalization_stats.npz (training output)
    2. features_out/superclass_feature_config.json (feature extraction export)

    Raises:
        FileNotFoundError: neither source exists
        ValueError: stats are not 12-lead shaped
    """
    npz_path = Path(stats_npz)
    if npz_path.exists():
        payload = np.load(npz_path)
        mean = np.asarray(payload["mean"], dtype=np.float64).reshape(-1)
        std = np.asarray(payload["std"], dtype=np.float64).reshape(-1)
    else:
        json_path = Path(stats_json)
        if not json_path.exists():
            raise FileNotFoundError(
                "Superclass normalization stats not found. Expected "
                f"{npz_path} or {json_path}. Re-run CNN training or feature extraction."
            )
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
        mean = np.asarray(data["normalization_mean"], dtype=np.float64).reshape(-1)
        std = np.asarray(data["normalization_std"], dtype=np.float64).reshape(-1)

    if mean.shape[0] != EXPECTED_LEADS or std.shape[0] != EXPECTED_LEADS:
        raise ValueError(
            f"Expected {EXPECTED_LEADS} lead stats, got mean={mean.shape}, std={std.shape}"
        )
    if np.any(std <= 0):
        raise ValueError("Normalization std must be positive for all leads")

    return mean, std


def apply_superclass_normalization(signal: np.ndarray) -> np.ndarray:
    """Channel-first z-score using training-derived superclass stats."""
    mean, std = load_superclass_norm_stats()
    return normalize_signal(signal, mean, std, eps=_NORM_EPS)

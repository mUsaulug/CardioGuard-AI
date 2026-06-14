"""
Shared ECG file I/O for API uploads and CLI inference scripts.

Consolidates .npy/.npz parsing (R3-06) with validation at API boundary.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

from src.utils.signal import validate_ecg_signal


def _array_from_npz_file(path: Path) -> np.ndarray:
    data = np.load(path, allow_pickle=False)
    if "signal" in data:
        signal = np.asarray(data["signal"]).copy()
    elif "X" in data:
        signal = np.asarray(data["X"]).copy()
    else:
        keys = list(data.files)
        if not keys:
            raise ValueError(f"No arrays in NPZ: {path}")
        signal = np.asarray(data[keys[0]]).copy()
    data.close()
    return signal


def load_ecg_array_from_path(path: Path) -> np.ndarray:
    """Load raw ECG array from path (.npy, .npz, .csv/.txt). No validation."""
    suffix = path.suffix.lower()
    if suffix == ".npz":
        signal = _array_from_npz_file(path)
    elif suffix == ".npy":
        signal = np.load(path, allow_pickle=False)
    elif suffix in {".csv", ".txt"}:
        signal = np.loadtxt(path, delimiter=",")
    else:
        raise ValueError(f"Unsupported format: {suffix}")
    return np.asarray(signal)


def load_ecg_from_path(path: Path, validate: bool = True) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Load ECG from disk; optionally validate shape/finiteness."""
    signal = load_ecg_array_from_path(path)
    if validate:
        return validate_ecg_signal(signal)
    return signal, {}


def load_ecg_from_bytes(
    file_content: bytes,
    filename: str,
    validate: bool = True,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Parse uploaded bytes (.npy/.npz) and optionally validate."""
    suffix = Path(filename).suffix.lower()
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(file_content)
            tmp_path = Path(tmp.name)

        if suffix == ".npz":
            signal = _array_from_npz_file(tmp_path)
        elif suffix == ".npy":
            signal = np.load(tmp_path, allow_pickle=False)
        else:
            raise ValueError(f"Unsupported file format: {filename}")
    finally:
        if tmp_path is not None and tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass

    if validate:
        return validate_ecg_signal(signal)
    return np.asarray(signal), {}

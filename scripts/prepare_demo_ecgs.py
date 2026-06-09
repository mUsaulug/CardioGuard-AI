#!/usr/bin/env python3
"""
Prepare demo ECG fixtures for local API/UI testing.

Uses existing repo samples (no PTB-XL download required):
  - sample.npy          (12, 1000) general validation export
  - test_mi_sample.npz  (1000, 12) MI-oriented sample

Outputs:
  tests/fixtures/ecg/*.npy / *.npz
  tests/fixtures/ecg/manifest.json
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tests" / "fixtures" / "ecg"


def _copy_npy(src: Path, dst: Path) -> dict:
    arr = np.load(src)
    if arr.ndim != 2:
        raise ValueError(f"{src}: expected 2D array, got {arr.shape}")
    shutil.copy2(src, dst)
    return {
        "file": dst.name,
        "format": "npy",
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "notes": "Channels-first (12, 1000) expected by API",
    }


def _copy_npz(src: Path, dst: Path) -> dict:
    data = np.load(src)
    key = "signal" if "signal" in data.files else data.files[0]
    arr = data[key]
    shutil.copy2(src, dst)
    return {
        "file": dst.name,
        "format": "npz",
        "array_key": key,
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "notes": "API auto-transposes to (12, timesteps) if needed",
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    entries = []

    sample_npy = ROOT / "sample.npy"
    if sample_npy.exists():
        entries.append(_copy_npy(sample_npy, OUT / "general_sample.npy"))

    mi_npz = ROOT / "test_mi_sample.npz"
    if mi_npz.exists():
        entries.append(_copy_npz(mi_npz, OUT / "mi_sample.npz"))

    # Extra alias at repo root for Docker/tests
    if sample_npy.exists() and not (ROOT / "sample.npy").stat().st_mtime_ns:
        pass  # already at root

    manifest = {
        "description": "CardioGuard-AI local demo ECG fixtures",
        "source": "Existing repo exports (no CSV conversion; PTB-XL optional for more)",
        "api_usage": "POST /predict/superclass?explain=true&full=true with multipart file",
        "samples": entries,
    }

    manifest_path = OUT / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Wrote {len(entries)} fixture(s) to {OUT}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()

"""
Shared XAI run manifest writer (API + localization paths).

Unifies manifest.json schema for backend artifact serving (R3-04).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def discover_visual_artifacts(run_dir: Path) -> List[Dict[str, str]]:
    """Scan visuals/*.png under run_dir."""
    artifacts: List[Dict[str, str]] = []
    visuals_dir = run_dir / "visuals"
    if not visuals_dir.exists():
        return artifacts
    for png in visuals_dir.glob("*.png"):
        artifacts.append({
            "type": "report_png",
            "path": f"visuals/{png.name}",
            "mime": "image/png",
        })
    return artifacts


def write_run_manifest(
    run_dir: Path,
    sample_id: str,
    task: str,
    artifacts: List[Dict[str, str]],
    sanity: Optional[Dict[str, Any]] = None,
    highlights: Optional[Any] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Write manifest.json with a consistent schema for all production XAI paths.

    Returns path to manifest.json.
    """
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest: Dict[str, Any] = {
        "run_id": run_dir.name,
        "created_at": datetime.now(timezone.utc).isoformat() + "Z",
        "task": task,
        "sample_id": sample_id,
        "artifacts": artifacts,
        "sanity": sanity,
        "highlights": highlights,
    }
    if extra:
        manifest.update(extra)

    manifest_path = run_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest_path

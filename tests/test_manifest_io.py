"""Tests for shared XAI manifest writer (R3-04)."""

import json
from pathlib import Path

from src.xai.manifest_io import write_run_manifest


def test_write_run_manifest_schema(tmp_path: Path):
  run_dir = tmp_path / "run_001"
  manifest_path = write_run_manifest(
    run_dir=run_dir,
    sample_id="sample_001",
    task="localization",
    artifacts=[{"type": "narrative_md", "path": "text/x.md", "mime": "text/markdown"}],
    sanity={"status": "PASS"},
    highlights=None,
  )

  assert manifest_path.exists()
  data = json.loads(manifest_path.read_text(encoding="utf-8"))
  assert data["run_id"] == "run_001"
  assert data["task"] == "localization"
  assert data["sample_id"] == "sample_001"
  assert data["artifacts"][0]["type"] == "narrative_md"
  assert data["sanity"] == {"status": "PASS"}
  assert data["created_at"].endswith("Z")

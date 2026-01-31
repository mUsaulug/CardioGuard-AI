# Phase 2: API Contracts & Security

**Generated Date:** 2026-01-31
**Source:** `src/backend/main.py`

## 1. Endpoint Inventory

| Method | Path | Request | Response | Error Codes |
| :--- | :--- | :--- | :--- | :--- |
| **POST** | `/predict/superclass` | `file` (Upload), `ensemble_weight` (Query), `explain` (Query) | `SuperclassPredictionResponse` | 400, 413 (Too Large), 500, 503 |
| **POST** | `/predict/mi-localization` | `file` (Upload), `threshold` (Query), `explain` (Query) | `MILocalizationResponse` | 400, 413, 500, 503 |
| **GET** | `/runs/{run_id}/{file_path}` | `run_id` (Path), `file_path` (Path) | `FileResponse` (Stream) | 400 (Invalid ID/Traversal), 404 |
| **GET** | `/health` | - | `HealthResponse` | 200 |
| **GET** | `/ready` | - | `ReadyResponse` | 200 |

## 2. Data Models (Pydantic)

### 2.1 Superclass Response
Defined in `src/backend/main.py`:
```python
class SuperclassPredictionResponse(BaseModel):
    mode: str = "multilabel-superclass"
    probabilities: PredictionProbabilities  # MI, STTC, CD, HYP, NORM
    predicted_labels: List[str]             # e.g., ["MI", "STTC"]
    thresholds: Dict[str, float]            # e.g., {"MI": 0.5, ...}
    primary: PrimaryPrediction              # { label, confidence, rule }
    sources: SourceProbabilities            # { cnn, xgb, ensemble }
    xai: Optional[XAIInfo]                  # { enabled, run_id, artifacts[] }
```

### 2.2 XAI Info & Artifacts
```python
class XAIArtifact(BaseModel):
    type: str   # "report_png", "narrative_md"
    name: str   # filename
    url: str    # /runs/RunID/path/to/file
    mime: str
```

## 3. Security Controls

### 3.1 Path Traversal Protection
In `serve_xai_artifact`:
- **Regex Validation:** `run_id` must match `^[a-zA-Z0-9_-]+$`.
- **Path Resolution:** Uses `path.resolve()` and checks `.relative_to(RUNS_DIR)`.
- **Evidence:**
  ```python
  # src/backend/main.py:417
  try:
      target_resolved.relative_to(base_resolved)
  except ValueError:
      raise HTTPException(400, "Path traversal not allowed")
  ```

### 3.2 Input Validation
- **File Size:** Manual check `if len(content) > 10 * 1024 * 1024` (10MB limit).
- **Format:** Supports `.npz` and `.npy`, parses using `tempfile` to avoid memory exhaustion (though `read()` loads into RAM first).

## 4. Findings
- **Robust Contracts:** Pydantic models ensure strict schema validation.
- **Security Awareness:** Explicit path traversal checks and ID validation prevent unauthorized file access.
- **Fail-Safe:** The `load_models` and `startup_event` ensure the API doesn't serve requests if models/thresholds are invalid.

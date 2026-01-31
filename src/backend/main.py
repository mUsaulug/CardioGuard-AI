"""
CardioGuard-AI FastAPI Backend.

REST API for multi-label superclass ECG prediction.

ARCHITECTURE RULES:
- Backend does NOT generate XAI artifacts
- Pipeline predict() is the ONLY source of inference and XAI
- Backend reads manifest.json from run_dir and returns artifact URLs
- No inline CNN/XGB/ensemble code in endpoints

Endpoints:
- POST /predict/superclass - Multi-label prediction (explain=true for XAI)
- POST /predict/mi-localization - MI localization (explain=true for XAI)
- GET /runs/{run_id}/{file_path} - Serve XAI artifacts
- GET /health - Health check
- GET /ready - Readiness check

Usage:
    uvicorn src.backend.main:app --reload --port 8000
"""

from __future__ import annotations

import json
import hashlib
import re
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

import numpy as np

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field


# =============================================================================
# Configuration
# =============================================================================

RUNS_DIR = Path("reports/xai/runs")
RUN_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+$")


# =============================================================================
# Pydantic Models
# =============================================================================

class PredictionProbabilities(BaseModel):
    """Probabilities for each class."""
    MI: float = Field(..., description="MI probability")
    STTC: float = Field(..., description="STTC probability")
    CD: float = Field(..., description="CD probability")
    HYP: float = Field(..., description="HYP probability")
    NORM: float = Field(..., description="Derived NORM probability")


class PrimaryPrediction(BaseModel):
    """Primary (single) label prediction."""
    label: str = Field(..., description="Primary predicted label")
    confidence: float = Field(..., description="Confidence score")
    rule: str = Field(default="MI-first-then-priority", description="Selection rule")


class SourceProbabilities(BaseModel):
    """Probabilities from each model source."""
    cnn: Dict[str, float] = Field(..., description="CNN probabilities")
    xgb: Optional[Dict[str, float]] = Field(None, description="XGBoost probabilities")
    ensemble: Dict[str, float] = Field(..., description="Ensemble probabilities")


class VersionInfo(BaseModel):
    """Model version information."""
    model_hash: str = Field(..., description="Hash of model checkpoint")
    threshold_hash: str = Field(..., description="Hash of threshold config")
    api_version: str = Field(default="1.1.0", description="API version")
    timestamp: str = Field(..., description="Prediction timestamp")


class XAIArtifact(BaseModel):
    """Single XAI artifact descriptor."""
    type: str = Field(..., description="Artifact type")
    name: str = Field(..., description="Filename")
    url: str = Field(..., description="URL to fetch artifact")
    mime: str = Field(..., description="MIME type")


class XAIInfo(BaseModel):
    """XAI response info."""
    enabled: bool = Field(..., description="Whether XAI was generated")
    run_id: Optional[str] = Field(None, description="XAI run identifier")
    run_dir: Optional[str] = Field(None, description="Relative path to run directory")
    artifacts: List[XAIArtifact] = Field(default=[], description="List of artifacts")
    highlights: Optional[List[Dict[str, Any]]] = Field(None, description="Top activation windows")
    sanity: Optional[Dict[str, Any]] = Field(None, description="Sanity check results")


class SuperclassPredictionResponse(BaseModel):
    """Full superclass prediction response."""
    mode: str = Field(default="multilabel-superclass", description="Prediction mode")
    probabilities: PredictionProbabilities
    predicted_labels: List[str] = Field(..., description="Labels exceeding threshold")
    thresholds: Dict[str, float] = Field(..., description="Per-class thresholds")
    primary: PrimaryPrediction
    sources: SourceProbabilities
    versions: VersionInfo
    xai: Optional[XAIInfo] = Field(None, description="XAI artifacts info")


class MILocalizationResponse(BaseModel):
    """MI localization prediction response."""
    mi_detected: bool = Field(..., description="Whether MI was detected")
    regions: List[str] = Field(default=[], description="Predicted MI regions")
    probabilities: Dict[str, float] = Field(default={}, description="Per-region probabilities")
    label_space: str = Field(default="ptbxl_derived_anatomical_v1")
    labels: List[str] = Field(default=["AMI", "ASMI", "ALMI", "IMI", "LMI"])
    mapping_source: str = Field(default="src/data/mi_localization.py")
    mapping_fingerprint: str = Field(default="8ab274e06afa1be8")
    localization_head_type: str = Field(default="classification_5")
    xai: Optional[XAIInfo] = Field(None, description="XAI artifacts info")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: str


class ReadyResponse(BaseModel):
    """Readiness check response."""
    ready: bool
    models_loaded: Dict[str, bool]
    message: str


# =============================================================================
# Application State
# =============================================================================

class AppState:
    """Application state for loaded models."""
    
    def __init__(self):
        self.superclass_model = None
        self.binary_model = None
        self.localization_model = None
        self.xgb_models = {}
        self.calibrators = {}
        self.scaler = None
        self.feature_schema = None
        self.thresholds = {}
        self.model_hashes = {}
        self.threshold_hash = ""
        self.loaded = False
        self.xgb_data = None
        self.device = None
    
    def load_models(
        self,
        superclass_checkpoint: Path = Path("checkpoints/ecgcnn_superclass.pt"),
        localization_checkpoint: Path = Path("checkpoints/ecgcnn_localization.pt"),
        xgb_dir: Path = Path("logs/xgb_superclass"),
        thresholds_path: Path = Path("artifacts/thresholds_superclass.json"),
    ):
        """Load all models."""
        import torch
        import joblib
        from xgboost import XGBClassifier
        from src.utils.model_loader import load_model_safe
        
        self.device = torch.device("cpu")
        
        # Superclass (required)
        if superclass_checkpoint.exists():
            self.superclass_model, meta = load_model_safe(
                superclass_checkpoint, "superclass", self.device
            )
            self.model_hashes["superclass"] = meta["checkpoint_hash"]
            print(f"Superclass model loaded (schema: {meta['schema']})")
        else:
            raise RuntimeError(f"Required superclass checkpoint not found: {superclass_checkpoint}")
        
        # Localization (optional)
        if localization_checkpoint.exists():
            self.localization_model, meta = load_model_safe(
                localization_checkpoint, "mi_localization", self.device
            )
            self.model_hashes["localization"] = meta["checkpoint_hash"]
            print(f"Localization model loaded")
        
        # XGBoost
        xgb_models_dict = {}
        calibrators_dict = {}
        scaler_obj = None
        
        if xgb_dir.exists():
            schema_path = xgb_dir / "feature_schema.json"
            if schema_path.exists():
                with open(schema_path) as f:
                    self.feature_schema = json.load(f)
                print(f"XGBoost feature schema loaded: {self.feature_schema['feature_count']} features")
            
            for cls in ["MI", "STTC", "CD", "HYP"]:
                model_path = xgb_dir / cls / "xgb_model.json"
                if model_path.exists():
                    model = XGBClassifier()
                    model.load_model(model_path)
                    xgb_models_dict[cls] = model
                    self.xgb_models[cls] = model
                
                calibrator_path = xgb_dir / cls / "calibrator.joblib"
                if calibrator_path.exists():
                    calibrators_dict[cls] = joblib.load(calibrator_path)
                    self.calibrators[cls] = calibrators_dict[cls]
            
            scaler_path = xgb_dir / "scaler.joblib"
            if scaler_path.exists():
                scaler_obj = joblib.load(scaler_path)
                self.scaler = scaler_obj
        
        self.xgb_data = {
            "models": xgb_models_dict,
            "calibrators": calibrators_dict,
            "scaler": scaler_obj
        }
        
        # Thresholds (required)
        if thresholds_path.exists():
            with open(thresholds_path) as f:
                data = json.load(f)
            self.thresholds = data.get("thresholds", {})
            with open(thresholds_path, "rb") as f:
                self.threshold_hash = hashlib.md5(f.read()).hexdigest()[:8]
        else:
            raise RuntimeError(f"Required thresholds not found: {thresholds_path}")
        
        self.loaded = True
        print(f"Models loaded: Superclass=OK, Localization={self.localization_model is not None}, XGB={len(self.xgb_models)}")


state = AppState()


# =============================================================================
# FastAPI App
# =============================================================================

app = FastAPI(
    title="CardioGuard-AI",
    description="Multi-label ECG Classification API",
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Load models on startup (fail-closed)."""
    from src.utils.checkpoint_validation import (
        validate_all_checkpoints,
        CheckpointMismatchError,
        MappingDriftError,
    )
    
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Fail-closed checkpoint validation
    print("Validating checkpoints...")
    try:
        results = validate_all_checkpoints(strict=True)
        print("Checkpoint validation passed!")
        for task, result in results.items():
            if isinstance(result, dict) and result.get("valid"):
                print(f"  {task}: out_dim={result.get('out_dim')} ✓")
    except (CheckpointMismatchError, MappingDriftError) as e:
        raise RuntimeError(f"CRITICAL: Checkpoint validation failed: {e}")
    except FileNotFoundError as e:
        print(f"Warning: Some checkpoints missing: {e}")
    
    # Load models (fail-closed)
    print("Loading models...")
    state.load_models()  # Raises RuntimeError if required files missing
    print("Models loaded successfully!")


# =============================================================================
# Utility Functions (NO XAI GENERATION)
# =============================================================================

def parse_ecg_file(file_content: bytes, filename: str) -> np.ndarray:
    """Parse uploaded ECG file with temp cleanup."""
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=Path(filename).suffix, delete=False) as tmp:
            tmp.write(file_content)
            tmp_path = Path(tmp.name)
        
        if filename.endswith(".npz"):
            data = np.load(tmp_path)
            if "signal" in data:
                signal = data["signal"]
            elif "X" in data:
                signal = data["X"]
            else:
                signal = data[list(data.keys())[0]]
        elif filename.endswith(".npy"):
            signal = np.load(tmp_path)
        else:
            raise HTTPException(400, f"Unsupported file format: {filename}")
    finally:
        if tmp_path and tmp_path.exists():
            tmp_path.unlink()
    
    # Ensure (channels, timesteps) format
    if signal.ndim == 1:
        signal = signal.reshape(1, -1)
    if signal.shape[0] != 12:
        if signal.shape[1] == 12:
            signal = signal.T
        elif signal.shape[0] > signal.shape[1]:
            signal = signal.T
    
    return signal.astype(np.float32)


def build_xai_info_from_manifest(run_id: str, run_dir: Path) -> Optional[XAIInfo]:
    """
    Build XAIInfo by READING manifest.json (NOT generating artifacts).
    This is the ONLY XAI-related function in backend.
    """
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        return XAIInfo(enabled=False)
    
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    
    artifacts = []
    for artifact in manifest.get("artifacts", []):
        rel_path = artifact.get("path", "")
        artifacts.append(XAIArtifact(
            type=artifact.get("type", "unknown"),
            name=Path(rel_path).name,
            url=f"/runs/{run_id}/{rel_path}",
            mime=artifact.get("mime", "application/octet-stream")
        ))
    
    return XAIInfo(
        enabled=True,
        run_id=run_id,
        run_dir=str(run_dir),
        artifacts=artifacts,
        highlights=manifest.get("highlights"),
        sanity=manifest.get("sanity")
    )


# =============================================================================
# Health Endpoints
# =============================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow().isoformat(),
    )


@app.get("/ready", response_model=ReadyResponse)
async def readiness_check():
    """Readiness check."""
    models_status = {
        "superclass": state.superclass_model is not None,
        "localization": state.localization_model is not None,
        "xgb": len(state.xgb_models) > 0,
        "thresholds": len(state.thresholds) > 0,
    }
    ready = state.loaded
    return ReadyResponse(
        ready=ready,
        models_loaded=models_status,
        message="Ready" if ready else "Not ready",
    )


# =============================================================================
# Static Artifact Serving (Secure)
# =============================================================================

@app.get("/runs/{run_id}/{file_path:path}")
async def serve_xai_artifact(run_id: str, file_path: str):
    """Serve XAI artifact files with path traversal protection."""
    # Validate run_id format
    if not RUN_ID_PATTERN.match(run_id):
        raise HTTPException(400, "Invalid run_id format")
    
    # Resolve paths
    base_resolved = RUNS_DIR.resolve()
    target_path = RUNS_DIR / run_id / file_path
    target_resolved = target_path.resolve()
    
    # Path traversal check using is_relative_to
    try:
        target_resolved.relative_to(base_resolved)
    except ValueError:
        raise HTTPException(400, "Path traversal not allowed")
    
    if not target_resolved.exists():
        raise HTTPException(404, "Artifact not found")
    
    if target_resolved.is_dir():
        raise HTTPException(400, "Cannot serve directory")
    
    suffix = target_resolved.suffix.lower()
    media_types = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".md": "text/markdown",
        ".json": "application/json",
        ".csv": "text/csv",
        ".npz": "application/octet-stream",
    }
    return FileResponse(target_resolved, media_type=media_types.get(suffix, "application/octet-stream"))


# =============================================================================
# Prediction Endpoints (NO INLINE INFERENCE - calls pipeline only)
# =============================================================================

@app.post("/predict/superclass", response_model=SuperclassPredictionResponse)
async def predict_superclass(
    file: UploadFile = File(...),
    ensemble_weight: float = Query(0.5, ge=0.0, le=1.0),
    explain: bool = Query(False, description="Generate XAI artifacts"),
    sanity_check: bool = Query(False, description="Run XAI sanity checks"),
):
    """
    Multi-label superclass prediction.
    
    Pipeline does ALL inference and XAI generation.
    Backend only maps result to response.
    """
    from src.pipeline.inference.run_inference_superclass import predict as pipeline_predict
    from src.xai.reporting import generate_run_id
    
    if not state.loaded:
        raise HTTPException(503, "Models not loaded")
    
    content = await file.read()
    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(413, "File too large (max 10MB)")
    
    try:
        signal = parse_ecg_file(content, file.filename)
    except Exception as e:
        raise HTTPException(400, f"Could not parse file: {e}")
    
    sample_id = Path(file.filename).stem if file.filename else "api_sample"
    
    # Prepare run_dir for explain=true
    run_id = None
    run_dir = None
    if explain:
        run_id = generate_run_id("api", "multiclass")
        run_dir = RUNS_DIR / run_id
    
    # PIPELINE DOES ALL WORK - no inline inference here
    try:
        result = pipeline_predict(
            signal=signal,
            cnn_model=state.superclass_model,
            xgb_data=state.xgb_data,
            thresholds=state.thresholds,
            localization_model=state.localization_model,
            device=state.device,
            ensemble_weight=ensemble_weight,
            explain=explain,
            sanity_check=sanity_check,
            run_dir=run_dir,
            sample_id=sample_id,
        )
    except Exception as e:
        raise HTTPException(500, f"Prediction failed: {e}")
    
    # Read XAI info from manifest (if explain=true, pipeline wrote it)
    xai_info = None
    if explain and run_dir and run_dir.exists():
        xai_info = build_xai_info_from_manifest(run_id, run_dir)
    
    # Map pipeline result to response
    multi = result.get("multi", {})
    probs = multi.get("probabilities", {})
    sources = result.get("sources", {})
    primary = result.get("primary", {})
    
    return SuperclassPredictionResponse(
        mode="multilabel-superclass",
        probabilities=PredictionProbabilities(
            MI=probs.get("MI", 0),
            STTC=probs.get("STTC", 0),
            CD=probs.get("CD", 0),
            HYP=probs.get("HYP", 0),
            NORM=probs.get("NORM", 0),
        ),
        predicted_labels=multi.get("predicted_labels", ["NORM"]),
        thresholds=multi.get("thresholds", state.thresholds),
        primary=PrimaryPrediction(
            label=primary.get("label", "NORM"),
            confidence=primary.get("confidence", 0.5),
            rule=primary.get("rule", "MI-first-then-priority"),
        ),
        sources=SourceProbabilities(
            cnn=sources.get("cnn", {}),
            xgb=sources.get("xgb"),
            ensemble=sources.get("ensemble", {}),
        ),
        versions=VersionInfo(
            model_hash=state.model_hashes.get("superclass", ""),
            threshold_hash=state.threshold_hash,
            timestamp=datetime.utcnow().isoformat(),
        ),
        xai=xai_info,
    )


@app.post("/predict/mi-localization", response_model=MILocalizationResponse)
async def predict_mi_localization(
    file: UploadFile = File(...),
    threshold: float = Query(0.5, ge=0.0, le=1.0),
    explain: bool = Query(False, description="Generate XAI artifacts"),
):
    """
    MI localization prediction.
    
    Pipeline does ALL inference and XAI generation.
    Backend only maps result to response.
    """
    from src.pipeline.inference.run_inference_localization import predict as pipeline_predict_localization
    from src.xai.reporting import generate_run_id
    from src.data.mi_localization import MI_LOCALIZATION_REGIONS
    from src.utils.checkpoint_validation import MI_LOCALIZATION_FINGERPRINT
    
    if state.localization_model is None:
        raise HTTPException(503, "MI localization model not loaded")
    
    content = await file.read()
    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(413, "File too large (max 10MB)")
    
    try:
        signal = parse_ecg_file(content, file.filename)
    except Exception as e:
        raise HTTPException(400, f"Could not parse file: {e}")
    
    sample_id = Path(file.filename).stem if file.filename else "api_sample"
    
    run_id = None
    run_dir = None
    if explain:
        run_id = generate_run_id("api", "localization")
        run_dir = RUNS_DIR / run_id
    
    # PIPELINE DOES ALL WORK
    try:
        result = pipeline_predict_localization(
            signal=signal,
            model=state.localization_model,
            device=state.device,
            threshold=threshold,
            explain=explain,
            run_dir=run_dir,
            sample_id=sample_id,
        )
    except Exception as e:
        raise HTTPException(500, f"Prediction failed: {e}")
    
    xai_info = None
    if explain and run_dir and run_dir.exists():
        xai_info = build_xai_info_from_manifest(run_id, run_dir)
    
    return MILocalizationResponse(
        mi_detected=result.get("mi_detected", False),
        regions=result.get("regions", []),
        probabilities=result.get("probabilities", {}),
        label_space="ptbxl_derived_anatomical_v1",
        labels=MI_LOCALIZATION_REGIONS,
        mapping_source="src/data/mi_localization.py",
        mapping_fingerprint=MI_LOCALIZATION_FINGERPRINT,
        localization_head_type="classification_5",
        xai=xai_info,
    )


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

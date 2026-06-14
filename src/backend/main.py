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

import os
import json
import hashlib
import re
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone

import numpy as np
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from src.config import get_ensemble_cnn_weight
from src.backend.llm_proxy import (
    FREE_MODEL_CHAIN,
    allow_client_llm_key,
    llm_available,
    llm_proxy_enabled,
    proxy_chat_completion,
    server_key_configured,
)


# =============================================================================
# Configuration
# =============================================================================

RUNS_DIR = Path("reports/xai/runs")
RUN_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+$")
CLIENT_LOG_DIR = Path("logs")
CLIENT_LOG_FILE = CLIENT_LOG_DIR / "client-events.jsonl"

_DEFAULT_CORS_ORIGINS = (
    "http://localhost:3000,http://localhost:5173,http://localhost:8080,"
    "http://127.0.0.1:3000,http://127.0.0.1:5173,http://127.0.0.1:8080"
)


def _debug_endpoints_enabled() -> bool:
    return os.getenv("ENABLE_DEBUG_ENDPOINTS", "0") == "1"


def _resolve_cors_origins() -> tuple[list[str], bool]:
    """Return (origins, allow_credentials). Wildcard disables credentials."""
    raw = os.environ.get("CORS_ORIGINS", _DEFAULT_CORS_ORIGINS).split(",")
    origins = [o.strip() for o in raw if o.strip()]
    if "*" in origins:
        if len(origins) == 1:
            return ["*"], False
        origins = [o for o in origins if o != "*"]
    return origins, True


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
    api_version: str = Field(default="1.2.0", description="API version")
    timestamp: str = Field(..., description="Prediction timestamp")


class ExplanationInfo(BaseModel):
    """Inline XAI explanation summary for frontend consumption."""
    narrative: str = Field(default="", description="Unified XAI narrative")
    coherence_score: float = Field(..., description="Grad-CAM vs SHAP coherence")
    sanity_passed: Optional[bool] = Field(None, description="Sanity check passed (null=skipped)")
    gradcam_summary: str = Field(default="", description="Grad-CAM text summary")
    shap_summary: str = Field(default="", description="SHAP text summary")
    dominant_source: str = Field(default="", description="Dominant evidence source")
    conflicts: List[str] = Field(default=[], description="Conflicting evidence notes")


class LocalizationInline(BaseModel):
    """Inline MI localization from superclass predict path."""
    mi_detected: bool = Field(..., description="Whether MI localization is active")
    regions: List[str] = Field(default=[], description="Predicted MI regions")
    probabilities: Dict[str, float] = Field(default={}, description="Per-region probabilities")
    labels: List[str] = Field(default=[], description="Region label codes")
    labels_tr: Dict[str, str] = Field(default={}, description="Turkish region labels")


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


class ConsistencyInfo(BaseModel):
    """Model agreement check result (optional)."""
    agreement: str = Field(..., description="AGREE_MI/AGREE_NO_MI/DISAGREE_TYPE_1/DISAGREE_TYPE_2")
    triage_level: str = Field(..., description="HIGH/LOW/REVIEW")
    superclass_mi_prob: float = Field(..., description="MI prob from superclass model")
    binary_mi_prob: float = Field(..., description="MI prob from binary model")
    superclass_mi_decision: bool = Field(..., description="Superclass MI decision")
    binary_mi_decision: bool = Field(..., description="Binary MI decision")
    warnings: List[str] = Field(default=[], description="Disagreement warnings")


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
    consistency: Optional[ConsistencyInfo] = Field(None, description="Model agreement check")
    explanation: Optional[ExplanationInfo] = Field(None, description="Inline XAI explanation")
    localization: Optional[LocalizationInline] = Field(None, description="Inline MI localization")
    glossary: Dict[str, str] = Field(default={}, description="Turkish clinical glossary")
    airesult: Optional[Dict[str, Any]] = Field(None, description="Canonical AIResult v1.0 (full=true)")
    latency_ms: Optional[float] = Field(None, description="Server-side inference latency in milliseconds")


class MILocalizationResponse(BaseModel):
    """MI localization prediction response."""
    mi_detected: bool = Field(..., description="Whether MI was detected")
    regions: List[str] = Field(default=[], description="Predicted MI regions")
    probabilities: Dict[str, float] = Field(default={}, description="Per-region probabilities")
    label_space: str = Field(default="ptbxl_derived_anatomical_v1")
    labels: List[str] = Field(default=["AMI", "ASMI", "ALMI", "IMI", "LMI"])
    labels_tr: Dict[str, str] = Field(default={}, description="Turkish region labels")
    mapping_source: str = Field(default="src/data/mi_localization.py")
    mapping_fingerprint: str = Field(default="8ab274e06afa1be8")
    localization_head_type: str = Field(default="classification_5")
    versions: Optional[VersionInfo] = Field(None, description="Model version metadata")
    glossary: Dict[str, str] = Field(default={}, description="Turkish clinical glossary")
    xai: Optional[XAIInfo] = Field(None, description="XAI artifacts info")
    latency_ms: Optional[float] = Field(None, description="Server-side inference latency in milliseconds")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: str


class ReadyResponse(BaseModel):
    """Readiness check response."""
    ready: bool
    models_loaded: Dict[str, bool]
    message: str
    degraded: bool = False
    degraded_models: List[str] = Field(default_factory=list)


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
        self.degraded = False
        self.degraded_models: List[str] = []
    
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
        from src.utils.signal import load_superclass_norm_stats
        
        self.device = torch.device("cpu")

        # Superclass normalization stats (required — must match CNN training)
        mean, std = load_superclass_norm_stats()
        print(f"Superclass normalization stats loaded: {len(mean)} leads")
        _ = (mean, std)  # fail-closed validation only
        
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
        
        # Binary MI model (optional - for consistency guard)
        binary_checkpoint = Path("checkpoints/ecgcnn.pt")
        if binary_checkpoint.exists():
            self.binary_model, meta = load_model_safe(
                binary_checkpoint, "binary", self.device
            )
            self.model_hashes["binary"] = meta["checkpoint_hash"]
            print("Binary MI model loaded (for consistency guard)")
        
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

        required_xgb = ["MI", "STTC", "CD", "HYP"]
        missing_xgb = [cls for cls in required_xgb if cls not in xgb_models_dict]
        if missing_xgb:
            raise RuntimeError(
                f"Required XGB OVR models missing: {missing_xgb} (dir={xgb_dir})"
            )
        if self.feature_schema is None:
            raise RuntimeError(f"Required XGB feature schema not found: {xgb_dir / 'feature_schema.json'}")
        if scaler_obj is None:
            raise RuntimeError(f"Required XGB scaler not found: {xgb_dir / 'scaler.joblib'}")
        
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

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup (fail-closed)."""
    from src.utils.checkpoint_validation import (
        validate_checkpoint_task,
        validate_localization_fingerprint,
        CheckpointMismatchError,
        MappingDriftError,
    )

    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    state.degraded = False
    state.degraded_models = []

    required_checkpoints = [
        (Path("checkpoints/ecgcnn_superclass.pt"), "superclass"),
    ]
    optional_checkpoints = [
        (Path("checkpoints/ecgcnn.pt"), "binary"),
        (Path("checkpoints/ecgcnn_localization.pt"), "mi_localization"),
    ]

    print("Validating checkpoints...")
    try:
        for path, task in required_checkpoints:
            validate_checkpoint_task(path, task, strict=True)
            print(f"  {task}: required ✓")

        for path, task in optional_checkpoints:
            try:
                validate_checkpoint_task(path, task, strict=True)
                print(f"  {task}: optional ✓")
            except FileNotFoundError:
                state.degraded = True
                state.degraded_models.append(task)
                print(f"  {task}: missing (degraded mode)")

        loc_path = Path("checkpoints/ecgcnn_localization.pt")
        if loc_path.exists():
            validate_localization_fingerprint(strict=True)

        print("Checkpoint validation passed!")
    except (CheckpointMismatchError, MappingDriftError) as e:
        raise RuntimeError(f"CRITICAL: Checkpoint validation failed: {e}")
    except FileNotFoundError as e:
        raise RuntimeError(f"CRITICAL: Required checkpoint missing: {e}")

    print("Loading models...")
    state.load_models()
    print("Models loaded successfully!")

    yield  # App runs here

app = FastAPI(
    title="CardioGuard-AI",
    description="Multi-label ECG Classification API",
    version="1.2.0",
    lifespan=lifespan,
)

CORS_ORIGINS, CORS_ALLOW_CREDENTIALS = _resolve_cors_origins()

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=CORS_ALLOW_CREDENTIALS,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Utility Functions (NO XAI GENERATION)
# =============================================================================

def parse_ecg_file(file_content: bytes, filename: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Parse and validate uploaded ECG file. Returns (signal, validation_meta)."""
    from src.utils.signal_io import load_ecg_from_bytes

    return load_ecg_from_bytes(file_content, filename, validate=True)


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
        timestamp=datetime.now(timezone.utc).isoformat(),
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
        message="Ready (degraded)" if ready and state.degraded else ("Ready" if ready else "Not ready"),
        degraded=state.degraded,
        degraded_models=list(state.degraded_models),
    )


# =============================================================================
# LLM proxy (OpenRouter — R3-05)
# =============================================================================


class LlmStatusResponse(BaseModel):
    proxy_enabled: bool
    server_key_configured: bool
    allow_client_key: bool
    available: bool
    default_models: List[str]


class LlmChatMessage(BaseModel):
    role: str
    content: str


class LlmChatRequest(BaseModel):
    model: str
    messages: List[LlmChatMessage]
    max_tokens: int = Field(default=600, ge=1, le=4096)
    temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    stream: bool = False


@app.get("/api/llm/status", response_model=LlmStatusResponse)
async def llm_status():
    """Report whether LLM proxy can serve chat (server key or dev client key)."""
    return LlmStatusResponse(
        proxy_enabled=llm_proxy_enabled(),
        server_key_configured=server_key_configured(),
        allow_client_key=allow_client_llm_key(),
        available=llm_available(),
        default_models=list(FREE_MODEL_CHAIN),
    )


@app.post("/api/llm/chat")
async def llm_chat(
    request: LlmChatRequest,
    x_openrouter_key: Optional[str] = Header(default=None, alias="X-OpenRouter-Key"),
):
    """Proxy chat completions to OpenRouter (stream or JSON)."""
    body = {
        "model": request.model,
        "messages": [m.model_dump() for m in request.messages],
        "max_tokens": request.max_tokens,
        "temperature": request.temperature,
        "stream": request.stream,
    }
    return await proxy_chat_completion(body, x_openrouter_key)


# =============================================================================
# Client debug log (browser → agent-readable file)
# =============================================================================


class ClientLogEvent(BaseModel):
    ts: str
    level: str = "info"
    category: str = "ui"
    message: str
    meta: Optional[Dict[str, Any]] = None


@app.post("/debug/client-log")
async def append_client_log(event: ClientLogEvent):
    """Append one frontend debug event to logs/client-events.jsonl."""
    if not _debug_endpoints_enabled():
        raise HTTPException(404, "Not found")
    CLIENT_LOG_DIR.mkdir(parents=True, exist_ok=True)
    line = json.dumps(event.model_dump(), ensure_ascii=False) + "\n"
    with open(CLIENT_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line)
    return {"ok": True}


@app.get("/debug/client-log")
async def read_client_log(tail: int = Query(50, ge=1, le=500)):
    """Return recent frontend debug events (for local QA / agent inspection)."""
    if not _debug_endpoints_enabled():
        raise HTTPException(404, "Not found")
    if not CLIENT_LOG_FILE.exists():
        return {"events": [], "count": 0, "file": str(CLIENT_LOG_FILE)}
    lines = [ln for ln in CLIENT_LOG_FILE.read_text(encoding="utf-8").splitlines() if ln.strip()]
    slice_lines = lines[-tail:]
    events = [json.loads(ln) for ln in slice_lines]
    return {"events": events, "count": len(events), "file": str(CLIENT_LOG_FILE)}


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
    ensemble_weight: float = Query(
        default=get_ensemble_cnn_weight(),
        ge=0.0,
        le=1.0,
        description="CNN weight in ensemble (XGB weight = 1 - this). Default from thresholds artifact.",
    ),
    explain: bool = Query(False, description="Generate XAI artifacts"),
    sanity_check: bool = Query(False, description="Run XAI sanity checks"),
    full: bool = Query(False, description="Include canonical AIResult v1.0 payload"),
):
    """
    Multi-label superclass prediction.
    
    Pipeline does ALL inference and XAI generation.
    Backend only maps result to response.
    """
    import uuid

    from src.contracts.airesult_mapper import derive_input_meta, map_predict_output_to_airesult
    from src.contracts.api_mapper import (
        build_glossary_subset,
        map_explanation_info,
        map_localization_inline,
    )
    from src.pipeline.inference.run_inference_superclass import predict as pipeline_predict
    from src.xai.reporting import generate_run_id
    
    if not state.loaded:
        raise HTTPException(503, "Models not loaded")
    
    content = await file.read()
    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(413, "File too large (max 10MB)")
    
    try:
        signal, input_meta = await run_in_threadpool(parse_ecg_file, content, file.filename)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(400, f"Could not parse file: {e}")
    
    sample_id = Path(file.filename).stem if file.filename else "api_sample"
    
    # Prepare run_dir for explain=true
    run_id = None
    run_dir = None
    save_plot = None
    if explain:
        run_id = generate_run_id("api", "multiclass")
        run_dir = RUNS_DIR / run_id
        (run_dir / "visuals").mkdir(parents=True, exist_ok=True)
        save_plot = run_dir / "visuals" / f"{sample_id}_report.png"
    
    # PIPELINE DOES ALL WORK - no inline inference here
    import time as _time

    _t0 = _time.perf_counter()
    try:
        result = await run_in_threadpool(
            pipeline_predict,
            signal=signal,
            cnn_model=state.superclass_model,
            xgb_data=state.xgb_data,
            thresholds=state.thresholds,
            localization_model=state.localization_model,
            device=state.device,
            binary_model=state.binary_model,
            ensemble_weight=ensemble_weight,
            explain=explain,
            sanity_check=sanity_check,
            run_dir=run_dir,
            sample_id=sample_id,
            save_plot=save_plot,
            feature_schema=state.feature_schema,
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(500, f"Prediction failed: {e}")
    latency_ms = round((_time.perf_counter() - _t0) * 1000.0, 1)
    
    # Read XAI info from manifest (if explain=true, pipeline wrote it)
    xai_info = None
    if explain and run_dir and run_dir.exists():
        xai_info = build_xai_info_from_manifest(run_id, run_dir)
    
    # Map pipeline result to response
    multi = result.get("multi", {})
    probs = multi.get("probabilities", {})
    sources = result.get("sources", {})
    primary = result.get("primary", {})
    predicted_labels = multi.get("predicted_labels", ["NORM"])

    explanation_mapped = map_explanation_info(result.get("explanation"))
    explanation_info = (
        ExplanationInfo(**explanation_mapped) if explanation_mapped else None
    )

    mi_detected = "MI" in predicted_labels
    localization_mapped = map_localization_inline(
        result.get("mi_localization"),
        mi_detected=mi_detected,
    )
    localization_info = (
        LocalizationInline(**localization_mapped) if localization_mapped else None
    )

    airesult_payload = None
    if full:
        airesult_payload = map_predict_output_to_airesult(
            predict_out=result,
            case_id=str(uuid.uuid4()),
            sample_id=sample_id,
            run_dir=run_dir,
            input_meta=derive_input_meta(signal_path=Path(file.filename) if file.filename else None, validation_meta=input_meta),
        )

    return SuperclassPredictionResponse(
        mode="multilabel-superclass",
        probabilities=PredictionProbabilities(
            MI=probs.get("MI", 0),
            STTC=probs.get("STTC", 0),
            CD=probs.get("CD", 0),
            HYP=probs.get("HYP", 0),
            NORM=probs.get("NORM", 0),
        ),
        predicted_labels=predicted_labels,
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
            timestamp=datetime.now(timezone.utc).isoformat(),
        ),
        xai=xai_info,
        consistency=ConsistencyInfo(**result["consistency"]) if result.get("consistency") else None,
        explanation=explanation_info,
        localization=localization_info,
        glossary=build_glossary_subset(),
        airesult=airesult_payload,
        latency_ms=latency_ms,
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
    from src.config import MI_LOCALIZATION_FINGERPRINT, MI_LOCALIZATION_LABELS_TR, GLOSSARY
    
    if state.localization_model is None:
        raise HTTPException(503, "MI localization model not loaded")
    
    content = await file.read()
    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(413, "File too large (max 10MB)")
    
    try:
        signal, _input_meta = await run_in_threadpool(parse_ecg_file, content, file.filename)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(400, f"Could not parse file: {e}")
    
    sample_id = Path(file.filename).stem if file.filename else "api_sample"
    
    run_id = None
    run_dir = None
    if explain:
        run_id = generate_run_id("api", "localization")
        run_dir = RUNS_DIR / run_id
    
    # PIPELINE DOES ALL WORK
    import time as _time
    _t0 = _time.perf_counter()
    try:
        result = await run_in_threadpool(
            pipeline_predict_localization,
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
    latency_ms = round((_time.perf_counter() - _t0) * 1000.0, 1)

    xai_info = None
    if explain and run_dir and run_dir.exists():
        xai_info = build_xai_info_from_manifest(run_id, run_dir)
    
    return MILocalizationResponse(
        mi_detected=result.get("mi_detected", False),
        regions=result.get("regions", []),
        probabilities=result.get("probabilities", {}),
        labels_tr=MI_LOCALIZATION_LABELS_TR,
        label_space="ptbxl_derived_anatomical_v1",
        labels=MI_LOCALIZATION_REGIONS,
        mapping_source="src/data/mi_localization.py",
        mapping_fingerprint=MI_LOCALIZATION_FINGERPRINT,
        localization_head_type="classification_5",
        versions=VersionInfo(
            model_hash=state.model_hashes.get("localization", ""),
            threshold_hash=state.threshold_hash,
            timestamp=datetime.now(timezone.utc).isoformat(),
        ),
        glossary={k: GLOSSARY.get(k, k) for k in ("MI", "NORM")},
        xai=xai_info,
        latency_ms=latency_ms,
    )


# =============================================================================
# Static Frontend (Docker production)
# =============================================================================

# Serve pre-built client assets when available (TanStack Start: dist/client)
_frontend_dist = Path("frontend/dist/client")
if not _frontend_dist.exists():
    _frontend_dist = Path("frontend/dist")
_assets_dir = _frontend_dist / "assets"
_index_html = _frontend_dist / "index.html"
if _assets_dir.exists():
    app.mount("/assets", StaticFiles(directory=str(_assets_dir)), name="frontend-assets")

if _index_html.exists():

    @app.get("/")
    async def serve_frontend_index():
        return FileResponse(_index_html)

    @app.get("/{spa_path:path}")
    async def serve_frontend_spa(spa_path: str):
        """SPA fallback for client routes (exclude API paths)."""
        if spa_path.startswith(
            ("predict", "health", "ready", "runs", "debug", "assets", "api", "docs", "openapi.json", "redoc")
        ):
            raise HTTPException(404, "Not found")
        return FileResponse(_index_html)


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

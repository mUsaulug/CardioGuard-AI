# Phase 2: API Contracts & Security — Kapsamlı Analiz

**Generated Date:** 2026-01-31
**Kaynak:** `src/backend/main.py` (614 satır)
**Framework:** FastAPI + Pydantic

---

## 1. API Endpoint Envanteri

CardioGuard-AI Backend, 5 ana endpoint sunar:

| # | Method | Path | Açıklama | Auth |
| :---: | :--- | :--- | :--- | :---: |
| 1 | **POST** | `/predict/superclass` | Multi-label MI/STTC/CD/HYP tahmini | ❌ |
| 2 | **POST** | `/predict/mi-localization` | 5-bölge MI lokalizasyonu | ❌ |
| 3 | **GET** | `/runs/{run_id}/{file_path:path}` | XAI artifact dosyası sunma | ❌ |
| 4 | **GET** | `/health` | Liveness probe | ❌ |
| 5 | **GET** | `/ready` | Readiness probe (model durumu) | ❌ |

> **Not:** Şu anda authentication/authorization yok. Üretim deployment için JWT veya API key eklenmelidir.

---

## 2. Endpoint Detayları

### 2.1 POST /predict/superclass

**Ana prediction endpoint.** EKG sinyali yükleyip multi-label tahmin alır.

#### Request

| Parametre | Tür | Konum | Zorunlu | Açıklama |
| :--- | :--- | :--- | :---: | :--- |
| `file` | File | Body (multipart) | ✅ | `.npz` veya `.npy` formatında EKG sinyali |
| `ensemble_weight` | float | Query | ❌ | CNN/XGB ağırlığı (default: 0.5) |
| `explain` | bool | Query | ❌ | XAI artifact üretimi (default: false) |

#### Response Schema

**Pydantic Model:** `SuperclassPredictionResponse`

```python
class SuperclassPredictionResponse(BaseModel):
    mode: str = "multilabel-superclass"
    
    probabilities: PredictionProbabilities
    # {MI: 0.85, STTC: 0.23, CD: 0.12, HYP: 0.08, NORM: 0.15}
    
    predicted_labels: List[str]
    # ["MI"] — Threshold aşan sınıflar
    
    thresholds: Dict[str, float]
    # {MI: 0.5, STTC: 0.5, CD: 0.5, HYP: 0.5}
    
    primary: PrimaryPrediction
    # {label: "MI", confidence: 0.85, rule: "priority_order"}
    
    sources: SourceProbabilities
    # {cnn: {...}, xgb: {...}, ensemble: {...}}
    
    versions: VersionInfo
    # {model_hash: "abc123", api_version: "1.1.0", ...}
    
    xai: Optional[XAIInfo]
    # explain=true ise artifact bilgileri
```

#### Örnek Response (JSON)

```json
{
  "mode": "multilabel-superclass",
  "probabilities": {
    "MI": 0.8523,
    "STTC": 0.2341,
    "CD": 0.1205,
    "HYP": 0.0891,
    "NORM": 0.1477
  },
  "predicted_labels": ["MI"],
  "thresholds": {
    "MI": 0.5,
    "STTC": 0.5,
    "CD": 0.5,
    "HYP": 0.5
  },
  "primary": {
    "label": "MI",
    "confidence": 0.8523,
    "rule": "priority_order"
  },
  "sources": {
    "cnn": {"MI": 0.8412, "STTC": 0.2100, ...},
    "xgb": {"MI": 0.8634, "STTC": 0.2582, ...},
    "ensemble": {"MI": 0.8523, ...}
  },
  "versions": {
    "model_hash": "a1b2c3d4",
    "threshold_hash": "e5f6g7h8",
    "api_version": "1.1.0",
    "timestamp": "2026-01-31T03:20:00Z"
  },
  "xai": {
    "enabled": true,
    "run_id": "run_20260131_032000_abc123",
    "run_dir": "reports/xai/runs/run_20260131_032000_abc123",
    "artifacts": [
      {
        "type": "report_png",
        "name": "sample__report.png",
        "url": "/runs/run_20260131_032000_abc123/visuals/sample__report.png",
        "mime": "image/png"
      },
      {
        "type": "narrative_md",
        "name": "sample__narrative.md",
        "url": "/runs/run_20260131_032000_abc123/text/sample__narrative.md",
        "mime": "text/markdown"
      }
    ],
    "highlights": [...],
    "sanity": {"status": "PASS", ...}
  }
}
```

#### Hata Kodları

| HTTP Code | Durum | Açıklama |
| :---: | :--- | :--- |
| 400 | Bad Request | Geçersiz dosya formatı, bozuk veri |
| 413 | Payload Too Large | Dosya >10MB |
| 500 | Internal Server Error | Model hatası, beklenmeyen exception |
| 503 | Service Unavailable | Modeller yüklenmemiş (startup hatası) |

---

### 2.2 POST /predict/mi-localization

MI tespit edildikten sonra anatomik bölge lokalizasyonu.

#### Request

| Parametre | Tür | Konum | Zorunlu | Açıklama |
| :--- | :--- | :--- | :---: | :--- |
| `file` | File | Body | ✅ | EKG sinyali |
| `threshold` | float | Query | ❌ | Bölge tespit threshold'u (default: 0.5) |
| `explain` | bool | Query | ❌ | XAI üretimi |

#### Response Schema

```python
class MILocalizationResponse(BaseModel):
    mi_detected: bool
    regions: List[str]          # ["IMI", "ALMI"]
    probabilities: LocalizationProbabilities
    # {AMI: 0.12, ASMI: 0.08, ALMI: 0.72, IMI: 0.85, LMI: 0.15}
    
    label_space: str            # "5-region"
    labels: List[str]           # ["AMI", "ASMI", "ALMI", "IMI", "LMI"]
    mapping_source: str         # "src/data/mi_localization.py"
    localization_head_type: str # "multi-label-sigmoid"
    
    xai: Optional[XAIInfo]
```

---

### 2.3 GET /runs/{run_id}/{file_path}

XAI artifact dosyalarını güvenli şekilde sunar.

#### Path Parameters

| Parametre | Kısıtlama | Açıklama |
| :--- | :--- | :--- |
| `run_id` | Regex: `^[a-zA-Z0-9_-]+$` | XAI run tanımlayıcısı |
| `file_path` | Path traversal korumalı | Dosya yolu (ör: `visuals/report.png`) |

#### Güvenlik Kontrolleri

**Kaynak:** `src/backend/main.py` L405-430

```python
@app.get("/runs/{run_id}/{file_path:path}")
async def serve_xai_artifact(run_id: str, file_path: str):
    # 1. Run ID format validation
    if not re.match(r"^[a-zA-Z0-9_-]+$", run_id):
        raise HTTPException(400, "Invalid run_id format")
    
    # 2. Path construction
    base_path = RUNS_DIR / run_id
    target_path = base_path / file_path
    
    # 3. Path traversal protection
    base_resolved = base_path.resolve()
    target_resolved = target_path.resolve()
    
    try:
        target_resolved.relative_to(base_resolved)
    except ValueError:
        raise HTTPException(400, "Path traversal not allowed")
    
    # 4. File existence check
    if not target_resolved.exists():
        raise HTTPException(404, "Artifact not found")
    
    # 5. MIME type detection
    mime_type = mimetypes.guess_type(str(target_resolved))[0]
    
    return FileResponse(target_resolved, media_type=mime_type)
```

---

### 2.4 GET /health

Basit liveness probe.

```python
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }
```

---

### 2.5 GET /ready

Model yükleme durumunu kontrol eder.

```python
@app.get("/ready")
async def readiness_check():
    return {
        "ready": models_loaded,
        "models_loaded": {
            "superclass": superclass_model is not None,
            "localization": localization_model is not None,
            "xgb": len(xgb_models) == 4,
            "thresholds": thresholds is not None
        },
        "message": "All models loaded" if models_loaded else "Models not ready"
    }
```

---

## 3. Pydantic Veri Modelleri

### 3.1 Olasılık Modelleri

```python
class PredictionProbabilities(BaseModel):
    MI: float = Field(..., ge=0, le=1)
    STTC: float = Field(..., ge=0, le=1)
    CD: float = Field(..., ge=0, le=1)
    HYP: float = Field(..., ge=0, le=1)
    NORM: float = Field(..., ge=0, le=1)

class LocalizationProbabilities(BaseModel):
    AMI: float = Field(..., ge=0, le=1)
    ASMI: float = Field(..., ge=0, le=1)
    ALMI: float = Field(..., ge=0, le=1)
    IMI: float = Field(..., ge=0, le=1)
    LMI: float = Field(..., ge=0, le=1)
```

### 3.2 XAI Modelleri

```python
class XAIArtifact(BaseModel):
    type: str       # "report_png", "narrative_md", "shap_bar", ...
    name: str       # Dosya adı
    url: str        # Relative URL: /runs/{id}/path
    mime: str       # "image/png", "text/markdown"

class XAIInfo(BaseModel):
    enabled: bool
    run_id: Optional[str]
    run_dir: Optional[str]
    artifacts: List[XAIArtifact]
    highlights: Optional[List[dict]]  # Top features
    sanity: Optional[dict]            # Sanity check results
```

---

## 4. Request Validation

### 4.1 Dosya Boyutu Kontrolü

```python
# src/backend/main.py L235
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

content = await file.read()
if len(content) > MAX_FILE_SIZE:
    raise HTTPException(413, f"File too large. Max: {MAX_FILE_SIZE} bytes")
```

### 4.2 Dosya Format Kontrolü

```python
# Supported formats
ALLOWED_EXTENSIONS = {".npz", ".npy"}

filename = file.filename or ""
ext = Path(filename).suffix.lower()
if ext not in ALLOWED_EXTENSIONS:
    raise HTTPException(400, f"Unsupported format. Allowed: {ALLOWED_EXTENSIONS}")
```

### 4.3 NumPy Parsing

```python
import tempfile
import numpy as np

with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
    tmp.write(content)
    tmp_path = tmp.name

try:
    if ext == ".npz":
        data = np.load(tmp_path)
        signal = data[data.files[0]]  # İlk array'i al
    else:
        signal = np.load(tmp_path)
finally:
    os.unlink(tmp_path)
```

---

## 5. Güvenlik Analizi

### 5.1 Güvenlik Kontrolleri Özeti

| Kontrol | Durum | Kaynak |
| :--- | :--- | :--- |
| Input validation (size) | ✅ | `main.py` L235 |
| Input validation (format) | ✅ | `main.py` L240 |
| Path traversal protection | ✅ | `main.py` L417 |
| Run ID format validation | ✅ | `main.py` L405 |
| SQL injection | N/A | Veritabanı yok |
| XSS | N/A | JSON API |
| CORS | ⚠️ | Konfigüre edilmeli |
| Rate limiting | ❌ | Yok |
| Authentication | ❌ | Yok |

### 5.2 Path Traversal Koruması (Detay)

Saldırgan şu URL'i denerse:
```
GET /runs/../../../etc/passwd
```

Sistem bunu engeller:

```python
run_id = "../../../etc"  # Saldırı girişimi

# Regex kontrolü başarısız olur
if not re.match(r"^[a-zA-Z0-9_-]+$", run_id):
    raise HTTPException(400, "Invalid run_id format")
# ↑ ".." ve "/" karakterleri reddedilir
```

Eğer regex atlatılsa bile `relative_to()` kontrolü devreye girer:
```python
base = Path("/app/reports/xai/runs/valid_id")
target = Path("/etc/passwd")  # resolve() sonrası

target.relative_to(base)  # ValueError fırlatır!
```

---

## 6. Startup & Shutdown

### 6.1 Startup Event (Fail-Closed)

**Kaynak:** `src/backend/main.py` L80-110

```python
@app.on_event("startup")
async def startup_event():
    global superclass_model, localization_model, xgb_models, thresholds
    
    logger.info("Starting CardioGuard-AI Backend...")
    
    # 1. Checkpoint validation
    validation_result = validate_all_checkpoints()
    if not validation_result["valid"]:
        logger.error(f"Checkpoint validation failed: {validation_result}")
        raise RuntimeError("FATAL: Checkpoint validation failed!")
    
    # 2. Load models
    try:
        superclass_model = load_superclass_model()
        localization_model = load_localization_model()
        xgb_models = load_xgb_models()
        thresholds = load_thresholds()
    except Exception as e:
        logger.error(f"Model loading failed: {e}")
        raise RuntimeError(f"FATAL: Cannot load models: {e}")
    
    logger.info("All models loaded successfully!")
```

**Önem:** Model yüklenemezse API **HİÇ BAŞLAMAZ**. Bu "fail-closed" yaklaşımı, sıradan bir hatanın sessizce yoksayılmasını önler.

### 6.2 Shutdown Event

```python
@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down CardioGuard-AI Backend...")
    # Cleanup (if needed)
```

---

## 7. Frontend-Backend Kontrat Uyumu

### 7.1 TypeScript Tanımları vs Pydantic

| Backend (Python) | Frontend (TypeScript) | Uyum |
| :--- | :--- | :---: |
| `SuperclassPredictionResponse` | `SuperclassResponse` | ✅ |
| `MILocalizationResponse` | `LocalizationResponse` | ✅ |
| `PredictionProbabilities` | `SuperclassProbabilities` | ✅ |
| `LocalizationProbabilities` | `LocalizationProbabilities` | ✅ |
| `XAIInfo` | `XaiSchema` | ✅ |
| `XAIArtifact` | `Artifact` | ✅ |
| `VersionInfo` | `Versions` | ✅ |

**Kaynak Karşılaştırması:**

Python (`main.py`):
```python
class PredictionProbabilities(BaseModel):
    MI: float
    STTC: float
    CD: float
    HYP: float
    NORM: float
```

TypeScript (`types.ts`):
```typescript
export interface SuperclassProbabilities {
    MI: number;
    STTC: number;
    CD: number;
    HYP: number;
    NORM: number;
}
```

---

## 8. Örnek API Kullanımı

### 8.1 cURL ile Tahmin

```bash
# Superclass prediction with XAI
curl -X POST "http://localhost:8000/predict/superclass?explain=true" \
     -F "file=@sample.npz" \
     -H "accept: application/json"

# Expected response: 200 OK with SuperclassPredictionResponse JSON
```

### 8.2 JavaScript/Fetch ile Kullanım

```javascript
const formData = new FormData();
formData.append('file', ecgFile);

const response = await fetch(
    'http://localhost:8000/predict/superclass?explain=true',
    {
        method: 'POST',
        body: formData
    }
);

const result = await response.json();
console.log(result.primary.label);  // "MI"
console.log(result.xai.artifacts);  // [{type: "report_png", url: "..."}]
```

---

## 9. Özet: API Güçlü ve Zayıf Yönler

| Kategori | Güçlü Yön | Eksik / Risk |
| :--- | :--- | :--- |
| **Kontrat** | Strict Pydantic validation | - |
| **Type Safety** | Frontend/Backend %100 uyumlu | - |
| **Security** | Path traversal protection | Auth/Rate limiting yok |
| **Reliability** | Fail-closed startup | - |
| **Documentation** | - | OpenAPI spec güncellenmeli |

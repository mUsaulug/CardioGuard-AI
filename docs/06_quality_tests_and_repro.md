# Phase 6: Quality, Tests & Reproducibility — Kapsamlı Analiz

**Generated Date:** 2026-01-31
**Test Framework:** Pytest
**Test Directory:** `tests/`

---

## 1. Test Suite Envanteri

CardioGuard-AI, kritik modülleri kapsayan 11 test dosyası içerir:

| Test Dosyası | Satır | Kapsam | Kritiklik |
| :--- | :---: | :--- | :---: |
| `test_consistency_guard.py` | 177 | Model tutarlılık kontrolü | ⭐⭐⭐ |
| `test_airesult_mapper.py` | ~350 | Backend/Frontend kontrat dönüşümleri | ⭐⭐⭐ |
| `test_checkpoint_validation.py` | ~200 | Model checkpoint doğrulama | ⭐⭐⭐ |
| `test_artifacts.py` | ~200 | XAI artifact üretimi | ⭐⭐ |
| `test_xai_visualization.py` | ~150 | Görselleştirme fonksiyonları | ⭐⭐ |
| `test_data.py` | ~300 | Veri yükleme ve split | ⭐⭐ |
| `test_gradcam.py` | ~50 | Grad-CAM temel işlevsellik | ⭐ |
| `test_model.py` | ~50 | Model instantiation | ⭐ |
| `test_xgb_pipeline.py` | ~50 | XGBoost pipeline | ⭐ |
| `test_checkpoint_utils.py` | ~50 | Checkpoint yardımcıları | ⭐ |
| `__init__.py` | 1 | Modül tanımlayıcı | - |

---

## 2. Kritik Test Analizleri

### 2.1 Consistency Guard Tests

**Kaynak:** `tests/test_consistency_guard.py` (177 satır)

Bu dosya, `ConsistencyGuard` modülünün tüm senaryolarını test eder:

```python
# === Anlaşma Senaryoları ===

def test_agree_mi():
    """Her iki model de MI tespit ediyor."""
    result = check_consistency(
        superclass_probs={"MI": 0.8, "STTC": 0.2, "CD": 0.1, "HYP": 0.05},
        binary_mi_prob=0.85,
        thresholds={"MI": 0.5}
    )
    assert result.agreement_type == AgreementType.AGREE_MI
    assert result.triage_level == "STANDARD"

def test_agree_no_mi():
    """Her iki model de MI tespit etmiyor."""
    result = check_consistency(
        superclass_probs={"MI": 0.2, "STTC": 0.8, "CD": 0.3, "HYP": 0.1},
        binary_mi_prob=0.1,
        thresholds={"MI": 0.5}
    )
    assert result.agreement_type == AgreementType.AGREE_NO_MI

# === Anlaşmazlık Senaryoları ===

def test_disagree_type_1():
    """
    Superclass MI=True, Binary MI=False
    → Low confidence MI, ek inceleme gerekebilir
    """
    result = check_consistency(
        superclass_probs={"MI": 0.6},  # Threshold üstü
        binary_mi_prob=0.3,            # Threshold altı
        thresholds={"MI": 0.5}
    )
    assert result.agreement_type == AgreementType.DISAGREE_TYPE_1
    assert result.triage_level == "ELEVATED"  # Dikkat gerekli

def test_disagree_type_2():
    """
    Superclass MI=False, Binary MI=True
    → Superclass kaçırmış olabilir, kritik
    """
    result = check_consistency(
        superclass_probs={"MI": 0.3},  # Threshold altı
        binary_mi_prob=0.8,            # Threshold üstü
        thresholds={"MI": 0.5}
    )
    assert result.agreement_type == AgreementType.DISAGREE_TYPE_2
    assert result.triage_level == "CRITICAL"  # Acil inceleme

# === Lokalizasyon Tetikleme ===

def test_should_run_localization_on_mi():
    """MI tespit edildiğinde lokalizasyon çalışmalı."""
    result = should_run_localization(
        consistency_result=ConsistencyResult(
            agreement_type=AgreementType.AGREE_MI,
            triage_level="STANDARD"
        )
    )
    assert result is True

def test_should_not_run_localization_on_no_mi():
    """MI yok ise lokalizasyon çalışmamalı."""
    result = should_run_localization(
        consistency_result=ConsistencyResult(
            agreement_type=AgreementType.AGREE_NO_MI,
            triage_level="STANDARD"
        )
    )
    assert result is False
```

> ⚠️ **Kritik Not:** Bu testler BAŞARILI geçiyor, ancak `run_inference_superclass.py` içinde `check_consistency()` **çağrılmıyor**. Testler modülü doğruluyor, entegrasyonu değil.

### 2.2 Checkpoint Validation Tests

**Kaynak:** `tests/test_checkpoint_validation.py`

Model checkpoint'larının doğruluğunu ve beklenen çıktı boyutlarını test eder:

```python
def test_validate_superclass_checkpoint():
    """Superclass model checkpoint'ı doğru formatta mı?"""
    result = validate_checkpoint(
        checkpoint_path="checkpoints/ecgcnn_superclass.pt",
        expected_output_dim=4,  # MI, STTC, CD, HYP
        expected_input_channels=12
    )
    assert result["valid"] is True
    assert result["output_dim"] == 4

def test_validate_localization_checkpoint():
    """Localization model checkpoint'ı doğru formatta mı?"""
    result = validate_checkpoint(
        checkpoint_path="checkpoints/ecgcnn_localization.pt",
        expected_output_dim=5,  # AMI, ASMI, ALMI, IMI, LMI
        expected_input_channels=12
    )
    assert result["valid"] is True

def test_validate_nonexistent_checkpoint():
    """Olmayan checkpoint için hata dönmeli."""
    result = validate_checkpoint("checkpoints/nonexistent.pt")
    assert result["valid"] is False
    assert "not found" in result["error"]

def test_validate_corrupted_checkpoint():
    """Bozuk checkpoint için hata dönmeli."""
    # Geçici bozuk dosya oluştur
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        f.write(b"corrupted data")
        path = f.name
    
    result = validate_checkpoint(path)
    assert result["valid"] is False
    os.unlink(path)
```

### 2.3 Data & Split Tests

**Kaynak:** `tests/test_data.py`

Veri yükleme ve patient-level split doğrulaması:

```python
def test_standard_split_sizes():
    """Standart split boyutları doğru mu?"""
    df = load_ptbxl_metadata()
    train_idx, val_idx, test_idx = get_standard_split(df)
    
    total = len(df)
    assert len(train_idx) / total > 0.75  # ~80%
    assert len(val_idx) / total > 0.08    # ~10%
    assert len(test_idx) / total > 0.08   # ~10%

def test_no_patient_leakage():
    """Aynı hasta birden fazla split'te olmamalı."""
    df = load_ptbxl_metadata()
    train_idx, val_idx, test_idx = get_standard_split(df)
    
    # Bu fonksiyon leakage varsa ValueError fırlatır
    assert verify_no_patient_leakage(df, train_idx, val_idx, test_idx) is True

def test_strat_fold_usage():
    """PTB-XL strat_fold sütunu kullanılıyor mu?"""
    df = load_ptbxl_metadata()
    train_idx, val_idx, test_idx = get_standard_split(df)
    
    train_folds = df.loc[train_idx, "strat_fold"].unique()
    val_folds = df.loc[val_idx, "strat_fold"].unique()
    test_folds = df.loc[test_idx, "strat_fold"].unique()
    
    assert set(train_folds) == {1, 2, 3, 4, 5, 6, 7, 8}
    assert set(val_folds) == {9}
    assert set(test_folds) == {10}
```

---

## 3. Test Çalıştırma

### 3.1 Tüm Testler

```bash
cd CardioGuard-AI

# Tüm testleri çalıştır
pytest tests/ -v

# Sadece belirli bir test dosyası
pytest tests/test_consistency_guard.py -v

# Coverage ile
pytest tests/ --cov=src --cov-report=html
```

### 3.2 Beklenen Çıktı

```
==================== test session starts ====================
platform win32 -- Python 3.10.x, pytest-7.x.x
collected 47 items

tests/test_consistency_guard.py::test_agree_mi PASSED
tests/test_consistency_guard.py::test_agree_no_mi PASSED
tests/test_consistency_guard.py::test_disagree_type_1 PASSED
tests/test_consistency_guard.py::test_disagree_type_2 PASSED
...
==================== 47 passed in 12.34s ====================
```

---

## 4. Kod Kalitesi

### 4.1 Type Hints

Proje genelinde type hint kullanımı **yüksek**:

```python
# Örnek: run_inference_superclass.py
def predict(
    signal: np.ndarray,
    cnn_model: nn.Module,
    xgb_models: Dict[str, Any],
    thresholds: Dict[str, float],
    device: torch.device,
    explain: bool = False,
    run_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    ...
```

### 4.2 Docstrings

Çoğu fonksiyon docstring içerir:

```python
def get_primary_label(predicted_labels: List[str], probabilities: Dict) -> Dict:
    """
    Klinik önceliğe göre birincil etiketi seçer.
    
    Args:
        predicted_labels: Threshold aşan sınıf listesi
        probabilities: Sınıf olasılıkları
        
    Returns:
        Dict with label, confidence, rule
    """
```

### 4.3 Error Handling

Backend'de spesifik HTTP exception'lar:

```python
if len(content) > MAX_FILE_SIZE:
    raise HTTPException(
        status_code=413,
        detail=f"File too large. Max: {MAX_FILE_SIZE} bytes"
    )
```

Pipeline'da spesifik ValueError'lar:

```python
if signal.shape[0] != 12 and signal.shape[1] != 12:
    raise ValueError(f"Expected 12-lead ECG, got shape {signal.shape}")
```

---

## 5. Reproducibility (Tekrar Üretilebilirlik)

### 5.1 Random Seed

**Kaynak:** `src/config.py` ve `logs/*/training_results.json`

```python
# config.py
random_seed: int = 42

# Eğitim scriptlerinde:
np.random.seed(config.random_seed)
torch.manual_seed(config.random_seed)
```

**Kanıt (`logs/superclass_cnn/training_results.json`):**
```json
{
  "args": {
    "seed": 42
  }
}
```

### 5.2 Dependency Locking

| Dosya | Amaç |
| :--- | :--- |
| `requirements.txt` | Python bağımlılıkları (minimal) |
| `frontend/package-lock.json` | Node.js bağımlılıkları (exact versions) |

> ⚠️ **Not:** `requirements.txt` sadece paket isimlerini içeriyor, version pin yok. Reproducibility için `requirements.txt` pinlenmeli veya `pyproject.toml` + `poetry.lock` kullanılmalı.

### 5.3 Data Paths

**Kaynak:** `src/config.py`

```python
@dataclass
class PTBXLConfig:
    data_root: Path = field(
        default_factory=lambda: Path("physionet.org/files/ptb-xl/1.0.3")
    )
```

---

## 6. E2E Demo Adımları

### 6.1 Kurulum (One-Time)

```bash
# 1. Repository clone
git clone https://github.com/user/CardioGuard-AI.git
cd CardioGuard-AI

# 2. Python environment
python -m venv .venv
.venv\Scripts\activate  # Windows

# 3. Python dependencies
pip install -r requirements.txt
pip install fastapi uvicorn  # requirements.txt'te eksik

# 4. Frontend dependencies
cd frontend
npm install
cd ..
```

### 6.2 Backend Başlatma

```bash
uvicorn src.backend.main:app --host 0.0.0.0 --port 8000 --reload

# Beklenen çıktı:
# INFO:     Validating checkpoints...
# INFO:     Checkpoint validation passed!
# INFO:     Superclass model loaded
# INFO:     XGBoost models loaded: 4
# INFO:     Uvicorn running on http://0.0.0.0:8000
```

### 6.3 CLI Inference Testi

```bash
# Örnek sinyal ile tahmin
python -m src.pipeline.inference.run_inference_superclass \
    --input sample.npy \
    --explain \
    --output predictions/test_result.json

# Beklenen çıktı:
# Loading models...
# Running inference...
# Generating XAI artifacts...
# Result saved to predictions/test_result.json
# XAI artifacts: reports/xai/runs/run_20260131_...
```

### 6.4 API Testi (cURL)

```bash
# Health check
curl http://localhost:8000/health
# {"status": "healthy", "timestamp": "2026-01-31T03:20:00Z"}

# Readiness check
curl http://localhost:8000/ready
# {"ready": true, "models_loaded": {...}, "message": "All models loaded"}

# Prediction
curl -X POST "http://localhost:8000/predict/superclass?explain=true" \
     -F "file=@sample.npz" \
     -H "accept: application/json"
# {
#   "mode": "multilabel-superclass",
#   "probabilities": {"MI": 0.85, ...},
#   "primary": {"label": "MI", "confidence": 0.85, "rule": "priority_order"},
#   "xai": {"enabled": true, "artifacts": [...]}
# }
```

### 6.5 Örnek Input/Output

**Input (`sample.npy`):**
- Shape: `(12, 1000)` — 12-lead, 10 saniye @100Hz
- Format: float32

**Output (JSON):**
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
  "xai": {
    "enabled": true,
    "run_id": "run_20260131_032000_abc123",
    "artifacts": [
      {
        "type": "report_png",
        "name": "sample__report.png",
        "url": "/runs/run_20260131_032000_abc123/visuals/sample__report.png",
        "mime": "image/png"
      }
    ]
  }
}
```

---

## 7. Test Kapsamı Analizi

### 7.1 Kapsanan Alanlar

| Alan | Test Dosyası | Durum |
| :--- | :--- | :---: |
| Consistency Guard | `test_consistency_guard.py` | ✅ Kapsamlı |
| Checkpoint Validation | `test_checkpoint_validation.py` | ✅ Kapsamlı |
| Data Splitting | `test_data.py` | ✅ Kapsamlı |
| AI Result Mapping | `test_airesult_mapper.py` | ✅ Kapsamlı |
| XAI Visualization | `test_xai_visualization.py` | ✅ Orta |
| Artifacts | `test_artifacts.py` | ✅ Orta |
| Grad-CAM | `test_gradcam.py` | ⚠️ Minimal |
| Model | `test_model.py` | ⚠️ Minimal |

### 7.2 Eksik Test Alanları

| Alan | Durum | Öneri |
| :--- | :--- | :--- |
| E2E Integration | ❌ Yok | `test_e2e_prediction.py` ekle |
| API Endpoints | ❌ Yok | `test_api_endpoints.py` ekle |
| Frontend | ❌ Yok | Jest/Vitest ile component testleri |
| Performance | ❌ Yok | Benchmark testleri ekle |

---

## 8. Özet: Kalite Güçlü ve Zayıf Yönler

| Kategori | Güçlü Yön | Zayıf Yön / Eksik |
| :--- | :--- | :--- |
| **Unit Tests** | Kritik modüller kapsanmış | E2E eksik |
| **Type Safety** | Yüksek type hint kullanımı | - |
| **Reproducibility** | Seed=42 kullanılıyor | Version pinning eksik |
| **Documentation** | Docstrings mevcut | - |
| **Error Handling** | Spesifik exceptions | - |
| **CI/CD** | - | GitHub Actions yok |

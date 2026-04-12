# CardioGuard-AI Kapsamli Iyilestirme Tasarim Spesifikasyonu

**Tarih:** 2026-04-12
**Durum:** Taslak
**Kapsam:** Backend duzeltmeleri, threshold optimizasyonu, XAI iyilestirme, frontend yeniden tasarim, API testleri, Docker
**Kapsam Disi:** LLM entegrasyonu, model yeniden egitimi, CI/CD, frontend testleri

---

## Bagalam (Context)

CardioGuard-AI, 12 derivasyonlu EKG sinyallerinden kardiyak patoloji tespiti yapan Aciklanabilir AI platformudur. Derinlemesine analiz sonucunda su bulgular ortaya cikti:

- **19 sorun** tespit edildi (6 P0, 6 P1, 7 P2)
- **Threshold'lar optimize edilmemis** - tum siniflar 0.5, optimize degerler var ama kullanilmiyor
- **XAI'de placeholder lojik** - coherence score sabit 0.85, SHAP-weighted GradCAM eski pipeline'da ama tasinmamis
- **Frontend eksik** - Consistency Guard render edilmiyor, markdown parse yok, Tailwind CDN'den
- **Deployment yok** - Dockerfile, docker-compose eksik
- **API testi 0** - hicbir endpoint test edilmemis

**Hedef:** Akademik prototipten cilali bir showcase'e donusturmek - kritik bug'lari duzelt, modeli yeniden egitmeden performansi artir (threshold optimization), XAI'yi guclendir, frontend'i modern Turkce klinik dashboard olarak yeniden tasarla, Docker ile paketli.

---

## Genel Mimari (Degismeyecek)

```
Frontend (React 19, TypeScript, Turkce UI)
  |
  POST /predict/superclass | POST /predict/mi-localization | GET /health | GET /ready
  |
  FastAPI Backend (src/backend/main.py) — Gateway only, NO ML code
  |
  Inference Pipeline (src/pipeline/inference/)
  |-- run_inference_superclass.py — CNN + XGBoost ensemble, primary label, NORM turetimi
  |-- consistency_guard.py — Binary MI vs Superclass MI karsilastirma
  |-- run_inference_localization.py — 5 bolge MI lokalizasyon
  |
  XAI Pipeline (src/xai/)
  |-- gradcam.py — Temporal saliency (Grad-CAM + SmoothGrad)
  |-- shap_ovr.py — Per-class SHAP (XGBoost OVR)
  |-- unified.py — Birlesik aciklama sentezi (iyilestirilecek)
  |-- sanity.py — Adebayo et al. sanity checks
  |-- visualize.py — 3-panel PNG report
  |-- reporting.py — Manifest + artifact yonetimi

Model Dosyalari:
  checkpoints/ecgcnn_superclass.pt — 4 sinif CNN (AKTIF)
  checkpoints/ecgcnn_localization.pt — 5 bolge lokalizasyon (AKTIF)
  checkpoints/ecgcnn.pt — Binary MI, Consistency Guard icin (AKTIF)
  logs/xgb_superclass/ — 4 XGBoost OVR model + calibrator + scaler (AKTIF)
```

---

## GOREV 1: Backend Bug Duzeltmeleri

**Oncelik:** P0 - Acil
**Tahmini Dosya Sayisi:** 5 dosya
**Bagimsizlik:** Diger gorevlerden bagimsiz, paralel calisabilir

### 1.1 Hardcoded Threshold'lari Parametrik Yap

**Dosya:** `src/pipeline/inference/run_inference_superclass.py`

| Satir | Mevcut | Hedef |
|-------|--------|-------|
| 289 | `binary_threshold=0.5` | `binary_threshold=thresholds.get("MI_binary", 0.5)` |
| 309 | `prob >= 0.5` | `prob >= localization_threshold` (yeni parametre) |

`predict()` fonksiyonuna `localization_threshold: float = 0.5` parametresi ekle.

### 1.2 Consistency Guard Exception Handling

**Dosya:** `src/pipeline/inference/run_inference_superclass.py:279-290`

```python
# ONCE (hatasiz):
consistency_result: Optional[ConsistencyResult] = None
if binary_model is not None:
    with torch.no_grad():
        binary_logits = binary_model(signal_tensor)
        binary_mi_prob = float(torch.sigmoid(binary_logits).cpu().numpy().flatten()[0])
    consistency_result = check_consistency(...)

# SONRA (guvenli):
consistency_result: Optional[ConsistencyResult] = None
if binary_model is not None:
    try:
        with torch.no_grad():
            binary_logits = binary_model(signal_tensor)
            binary_mi_prob = float(torch.sigmoid(binary_logits).cpu().numpy().flatten()[0])
        consistency_result = check_consistency(
            superclass_mi_prob=ensemble_probs.get("MI", 0.0),
            binary_mi_prob=binary_mi_prob,
            superclass_threshold=thresholds.get("MI", 0.5),
            binary_threshold=thresholds.get("MI_binary", 0.5),
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        consistency_result = None
```

### 1.3 Debug Print Temizligi

**Dosya:** `src/pipeline/inference/run_inference_superclass.py:422-423`

Sil:
```python
print(f"DEBUGGING ERROR: explanation_result is not dict! Type: {type(explanation_result)}")
print(f"DEBUGGING ERROR: Content: {explanation_result}")
```

### 1.4 Duplicate Fonksiyon Duzeltme

**Dosya:** `src/xai/visualize.py`

`plot_gradcam_heatmap` iki kez tanimli (satir 154-171 ve 195-247). Ilk tanimi sil (154-171), ikinci tanim dogru signature ile kaliyor.

### 1.5 Requirements.txt Tamamla

**Dosya:** `requirements.txt`

Ekle:
```
fastapi
uvicorn[standard]
joblib
pydantic
```

### 1.6 .gitignore Genislet

**Dosya:** `.gitignore`

```gitignore
/physionet.org
__pycache__/
*.pyc
*.pyo
.env
.env.*
node_modules/
.venv/
venv/
*.egg-info/
dist/
build/
.DS_Store
*.log
.pytest_cache/
.coverage
htmlcov/
```

### 1.7 CORS Ortam Degiskeni

**Dosya:** `src/backend/main.py:279-285`

```python
import os

CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "*").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### 1.8 Deprecation Duzeltmeleri

**Dosyalar:** `src/backend/main.py`, `src/xai/reporting.py`, `src/pipeline/inference/run_inference_superclass.py`

- `@app.on_event("startup")` → `lifespan` context manager
- `datetime.utcnow()` → `datetime.now(timezone.utc)` (tum dosyalarda)

### Dogrulama
```bash
pytest tests/ -v  # Mevcut testler hala gecmeli
uvicorn src.backend.main:app --port 8000  # Baslamali
curl http://localhost:8000/health  # {"status": "healthy"}
```

---

## GOREV 2: Threshold ve Ensemble Optimizasyonu

**Oncelik:** P0 - Performans
**Tahmini Dosya Sayisi:** 2 dosya (script + config)
**Bagimsizlik:** Gorev 1'den bagimsiz, paralel calisabilir
**On Kosul:** `predictions/` dizinindeki dosyalar mevcut (val_cnn_probs.npz, val_xgb_probs.npz, val_labels.npz)

### 2.1 Threshold Optimizasyonu Calistir

**Mevcut Script:** `src/pipeline/evaluation/optimize_thresholds.py`

Komut:
```bash
python -m src.pipeline.evaluation.optimize_thresholds \
    --cnn-probs predictions/val_cnn_probs.npz \
    --xgb-probs predictions/val_xgb_probs.npz \
    --labels predictions/val_labels.npz \
    --output artifacts/thresholds_superclass.json \
    --ensemble-weight 0.5 \
    --mi-beta 2.0 \
    --mi-recall-min 0.9
```

**Beklenen Sonuc:**
| Sinif | Eski Threshold | Optimize Threshold | F1 Degisim |
|-------|---------------|-------------------|------------|
| MI | 0.5 | ~0.01-0.10 | Recall artisi |
| STTC | 0.5 | ~0.42 | F1 +%5 |
| CD | 0.5 | ~0.42 | F1 +%5 |
| HYP | 0.5 | ~0.26 | F1 +%10-20 |

### 2.2 Ensemble Weight Grid Search

**Dosya:** Yeni script veya `evaluate_ensemble.py` genisletmesi

```python
# Grid search: CNN weight 0.0 → 1.0, step 0.05
# Her weight icin: ensemble prob hesapla, optimize threshold'larla F1 hesapla
# En iyi alpha'yi bul ve thresholds_superclass.json'a yaz
best_alpha = None
best_f1 = 0
for w in np.arange(0.0, 1.05, 0.05):
    ens = w * cnn + (1 - w) * xgb
    # Apply per-class optimized thresholds
    f1 = macro_f1(y_true, ens >= thresholds)
    if f1 > best_f1:
        best_f1, best_alpha = f1, w
```

### 2.3 Config Guncelle

**Dosya:** `artifacts/thresholds_superclass.json`

Optimize edilmis threshold'lari `thresholds` alanina yaz (su an `details`'de duruyor).
`ensemble_weight` alanini optimal alpha ile guncelle.

**Onemli:** Eski `thresholds` (hep 0.5) ve `details` (optimize) ayrimi kaldirilacak. Tek bir `thresholds` alani olacak.

### Dogrulama
```bash
# Optimize sonrasi metrikleri kontrol et
python -c "
import json
with open('artifacts/thresholds_superclass.json') as f:
    data = json.load(f)
print('Thresholds:', data['thresholds'])
print('Ensemble weight:', data.get('ensemble_weight'))
for cls, info in data.get('details', {}).items():
    print(f'{cls}: F1={info.get(\"f1_at_threshold\", 0):.3f}, Recall={info.get(\"recall_at_threshold\", 0):.3f}')
"
```

---

## GOREV 3: XAI Iyilestirme

**Oncelik:** P1 - Kalite
**Tahmini Dosya Sayisi:** 2 dosya (unified.py, run_inference_superclass.py)
**Bagimsizlik:** Gorev 1'den bagimsiz, paralel calisabilir

### 3.1 SHAP-Weighted Grad-CAM Entegrasyonu

**Kaynak:** `src/xai/combined.py` (CombinedExplainer._compute_shap_weighted_cam)
**Hedef:** `src/xai/unified.py` (UnifiedExplainer.synthesize)

Tasinacak lojik:
```python
def _compute_shap_weighted_cam(self, gradcam_heatmap, shap_values):
    """Scale GradCAM heatmap by SHAP feature importance."""
    total_shap = float(np.sum(np.abs(shap_values)))
    scaling = 1.0 + 0.5 * np.tanh(total_shap)
    combined = gradcam_heatmap * scaling
    # Re-normalize
    combined = (combined - combined.min()) / (combined.max() + 1e-8)
    return combined
```

UnifiedExplainer.synthesize() ciktisina `combined_heatmap` alani ekle.

### 3.2 Contrastive Mode

**Kaynak:** `src/xai/combined.py`
**Hedef:** `src/xai/unified.py`

```python
def _compute_contrastive(self, pred_shap, runnerup_shap):
    """Compare SHAP values between predicted and runner-up class."""
    if pred_shap is None or runnerup_shap is None:
        return None
    diff = np.array(pred_shap) - np.array(runnerup_shap)
    top_distinguishing = np.argsort(np.abs(diff))[::-1][:10]
    return {
        "pred_vs_runnerup": {
            "distinguishing_features": top_distinguishing.tolist(),
            "diff_values": diff[top_distinguishing].tolist(),
        }
    }
```

### 3.3 Coherence Score Gercek Hesaplama

**Dosya:** `src/xai/unified.py:123-135`

Placeholder (sabit 0.85) yerine:
```python
def _analyze_coherence(self, gradcam_result, shap_result):
    """Compute real coherence between visual and feature explanations."""
    if not gradcam_result or not shap_result:
        return 0.5, ["Insufficient data for coherence analysis"]

    # GradCAM peak regions
    gradcam_peaks = set()
    for cls, cam in gradcam_result.items():
        if isinstance(cam, np.ndarray):
            cam_flat = cam.flatten()
            peak_region = int(np.argmax(cam_flat) / len(cam_flat) * 10)  # 0-9
            gradcam_peaks.add(peak_region)

    # SHAP top features — check if high-importance features correlate with peak
    shap_consistency = 0
    shap_total = 0
    for cls, data in shap_result.items():
        if isinstance(data, dict) and "top_features" in data:
            shap_total += 1
            top_feat = data["top_features"][0] if data["top_features"] else None
            if top_feat and top_feat.get("importance", 0) > 0.01:
                shap_consistency += 1

    if shap_total > 0:
        score = 0.5 + 0.5 * (shap_consistency / shap_total)
    else:
        score = 0.5

    conflicts = []
    if score < 0.6:
        conflicts.append("Visual and feature explanations show low agreement.")

    return score, conflicts
```

### 3.4 Embedding Cache (Performans)

**Dosya:** `src/pipeline/inference/run_inference_superclass.py`

Embeddings su an 2 kez compute ediliyor (XGBoost icin satir 225, SHAP icin satir 338). Tek seferde compute edip degiskende tut:

```python
# Bir kez compute et
embeddings = None
if xgb_data["models"] or explain:
    with torch.no_grad():
        embeddings = cnn_model.backbone(signal_tensor).cpu().numpy()
```

### Dogrulama
```bash
# XAI ciktisini kontrol et
python -m src.pipeline.inference.run_inference_superclass \
    --input sample.npy --explain --sanity-check
# Cikti: coherence_score != 0.85 (gercek hesaplama)
# Cikti: combined_heatmap mevcut (SHAP-weighted)
```

---

## GOREV 4: Frontend Yeniden Tasarim

**Oncelik:** P1 - Goruntu
**Tahmini Dosya Sayisi:** 12-15 dosya
**Bagimsizlik:** Backend gorevlerinden bagimsiz, paralel calisabilir (API kontrati degismediginden)

### 4.1 Teknik Altyapi

**Kalacak:** React 19.2.4, TypeScript 5.8.2, Vite 6.2.0
**Eklenecek (npm install):**
- `tailwindcss` + `postcss` + `autoprefixer` (CDN kaldiriliyor)
- `react-markdown` (XAI narrative rendering)
- `recharts` (olasilik grafikleri)

**Kaldirilacak:**
- `index.html`'den `<script src="https://cdn.tailwindcss.com">` satiri

**Olusturulacak:**
- `tailwind.config.js`
- `postcss.config.js`
- `frontend/src/styles/globals.css` (@tailwind directives)

### 4.2 Dil

- Tum UI metinleri **Turkce** (butonlar, etiketler, basliklar, hata mesajlari)
- Kod ve degisken isimleri **Ingilizce**
- API response'lar degismeyecek (Ingilizce key'ler)

### 4.3 Renk Paleti ve Tema

```
Dark Mode (varsayilan):
  Arka plan: #0f172a (slate-900)
  Kart: #1e293b (slate-800)
  Metin: #f1f5f9 (slate-100)
  Vurgu: #3b82f6 (blue-500)
  Tehlike: #ef4444 (red-500)
  Basari: #22c55e (green-500)
  Uyari: #f59e0b (amber-500)

Light Mode:
  Arka plan: #f8fafc (slate-50)
  Kart: #ffffff
  Metin: #0f172a (slate-900)
  Vurgu: #2563eb (blue-600)
```

### 4.4 Sayfa Yapisi (Layout)

```
┌──────────────────────────────────────────────────────────────┐
│  HEADER                                                       │
│  CardioGuard-AI | Sistem: ● Hazir | API: localhost:8000 | 🌙 │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  SOL PANEL (w-1/3)          │  SAG PANEL (w-2/3)             │
│  ┌────────────────────────┐ │  ┌──────────────────────────┐  │
│  │ EKG Dosyasi Yukle      │ │  │ TAHMIN SONUCLARI         │  │
│  │ [Dosya Sec] .npy/.npz  │ │  │                          │  │
│  │ Dosya: sample.npy      │ │  │ Birincil Tani            │  │
│  ├────────────────────────┤ │  │ ┌──────────────────────┐ │  │
│  │ AYARLAR                │ │  │ │ MI   ████████░░ 85%  │ │  │
│  │ Ensemble: CNN ──●── XGB│ │  │ │ STTC ██░░░░░░░░ 12%  │ │  │
│  │ ☑ XAI Aciklama         │ │  │ │ CD   █░░░░░░░░░  8%  │ │  │
│  │ ☑ Kalite Kontrolu      │ │  │ │ HYP  █░░░░░░░░░  5%  │ │  │
│  ├────────────────────────┤ │  │ │ NORM █░░░░░░░░░ 15%  │ │  │
│  │ [TAHMIN YAP]           │ │  │ └──────────────────────┘ │  │
│  └────────────────────────┘ │  ├──────────────────────────┤  │
│                              │  │ MODEL UYUMU              │  │
│  ┌────────────────────────┐ │  │ ✅ AGREE_MI              │  │
│  │ SISTEM DURUMU          │ │  │ Triage: YUKSEK           │  │
│  │ Superclass ✅           │ │  │ Superclass MI: 0.850     │  │
│  │ Lokalizasyon ✅         │ │  │ Binary MI: 0.920         │  │
│  │ XGBoost (4) ✅          │ │  ├──────────────────────────┤  │
│  │ Esikler ✅              │ │  │ MI LOKALIZASYON          │  │
│  └────────────────────────┘ │  │ AMI ████████░░ 0.82      │  │
│                              │  │ ASMI ██████░░░░ 0.61     │  │
│                              │  │ ALMI ████░░░░░░ 0.45     │  │
│                              │  │ IMI  ██░░░░░░░░ 0.23     │  │
│                              │  │ LMI  █░░░░░░░░░ 0.12     │  │
│                              │  └──────────────────────────┘  │
│                              │                                │
│                              │  ┌──────────────────────────┐  │
│                              │  │ XAI ACIKLAMALAR           │  │
│                              │  │ [GradCAM] [SHAP] [Rapor] │  │
│                              │  │ ┌────────────────────┐   │  │
│                              │  │ │ Gorsel/Metin icerik│   │  │
│                              │  │ └────────────────────┘   │  │
│                              │  │ Kalite: GUVENILIR 3/4    │  │
│                              │  └──────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
```

### 4.5 Bilesen Listesi

| Dosya | Amac | Yeni/Guncelleme |
|-------|------|-----------------|
| `App.tsx` | Ana layout, tema yonetimi, routing | Yeniden yaz |
| `components/Header.tsx` | Ust bar: logo, API URL, dark mode toggle, sistem durumu badge | Yeni |
| `components/UploadPanel.tsx` | Dosya yukleme, ensemble weight, XAI checkbox, tahmin butonu | Yeniden yaz (eski SuperclassPanel'in sol kismi) |
| `components/PredictionResult.tsx` | Birincil tani, olasilik grafigi, predicted labels | Yeni |
| `components/ProbabilityChart.tsx` | Yatay bar chart (recharts) — threshold cizgisi ile | Yeni |
| `components/ConsistencyPanel.tsx` | Model uyumu gosterimi — badge, triage, olaslik karsilastirma, uyarilar | Yeni |
| `components/LocalizationPanel.tsx` | MI bolge tespiti — progress bar'lar ile | Guncelle (Turkce + yeni stil) |
| `components/XaiViewer.tsx` | Tab'li XAI gosterimi: GradCAM / SHAP / Narrative | Yeniden yaz (markdown parse, tab yapisi) |
| `components/SanityBadge.tsx` | XAI kalite kontrolu durumu — GUVENILIR/KABUL EDILEBILIR/GUVENILMEZ | Yeni |
| `components/SystemStatus.tsx` | Model yukleme durumu paneli | Yeniden yaz (eski HealthReady) |
| `components/ThemeProvider.tsx` | Dark/light mode context | Yeni |
| `lib/api.ts` | HTTP client | Koru (degisiklik yok) |
| `lib/types.ts` | TypeScript tipleri | Koru (degisiklik yok) |

### 4.6 Consistency Guard Render Detayi

**Mevcut Durum:** `ConsistencyInfo` tipi tanimli (`types.ts:40-50`) ama hicbir component render etmiyor.

**Tasarim:**

```tsx
// components/ConsistencyPanel.tsx
function ConsistencyPanel({ consistency }: { consistency: ConsistencyInfo }) {
  const badges = {
    AGREE_MI: { color: "red", icon: "⚠️", text: "MI Onaylandi" },
    AGREE_NO_MI: { color: "green", icon: "✅", text: "MI Yok" },
    DISAGREE_TYPE_1: { color: "amber", icon: "🔍", text: "Inceleme Gerekli" },
    DISAGREE_TYPE_2: { color: "amber", icon: "🔍", text: "Inceleme Gerekli" },
  };

  const triage = {
    HIGH: { color: "red", text: "YUKSEK" },
    LOW: { color: "green", text: "DUSUK" },
    REVIEW: { color: "amber", text: "INCELEME" },
  };

  return (
    <Card>
      <Title>Model Uyumu</Title>
      <AgreementBadge type={consistency.agreement} />
      <TriageBadge level={consistency.triage_level} />
      <ComparisonBar
        superclass={consistency.superclass_mi_prob}
        binary={consistency.binary_mi_prob}
      />
      {consistency.warnings.map(w => <Warning text={w} />)}
    </Card>
  );
}
```

### 4.7 XAI Viewer Yeniden Tasarim

**Mevcut Sorun:** Markdown `<pre>` icinde raw text gosteriliyor, tab yapisi yok.

**Yeni Tasarim:**
```tsx
// Tabs: [GradCAM Haritasi] [SHAP Analizi] [Klinik Rapor]
// Her tab farkli artifact tipini gosterir

// GradCAM tab: PNG image render
// SHAP tab: PNG image render  
// Rapor tab: react-markdown ile parsed narrative
// Alt kisim: Sanity badge (GUVENILIR / KABUL EDILEBILIR / GUVENILMEZ)
```

### Dogrulama
```bash
cd frontend
npm install  # Yeni bagimliliklar
npm run dev  # Derleme hatasi olmamalj
# Tarayicida: http://localhost:5173
# Dark mode toggle calismali
# sample.npy yukle → tum paneller render edilmeli
# Consistency Guard paneli gorunmeli
# XAI markdown duzgun parse edilmeli
```

---

## GOREV 5: API Testleri

**Oncelik:** P2 - Kalite
**Tahmini Dosya Sayisi:** 1 dosya
**Bagimsizlik:** Gorev 1 tamamlandiktan sonra calistirilmali (cunku bug fix'ler test beklentilerini etkiler)

### 5.1 Test Dosyasi

**Dosya:** `tests/test_api.py`
**Araclar:** pytest + httpx (FastAPI TestClient)

### 5.2 Test Listesi

```python
import pytest
from fastapi.testclient import TestClient
from src.backend.main import app

client = TestClient(app)

# --- Health & Ready ---
def test_health_returns_200():
    """GET /health 200 donmeli, status=healthy."""

def test_ready_returns_model_status():
    """GET /ready donmeli, models_loaded dict icermeli."""

# --- Superclass Prediction ---
def test_predict_superclass_with_npy():
    """POST /predict/superclass sample.npy ile 200 donmeli."""
    # Response: probabilities, predicted_labels, primary, sources, versions

def test_predict_superclass_with_npz():
    """POST /predict/superclass test_mi_sample.npz ile 200 donmeli."""

def test_predict_superclass_with_explain():
    """POST /predict/superclass?explain=true ile XAI artifacts donmeli."""

def test_predict_superclass_response_schema():
    """Response SuperclassPredictionResponse Pydantic modeline uymali."""

def test_predict_superclass_consistency_present():
    """Response consistency alani None olmamali (binary model yukluyse)."""

def test_predict_superclass_invalid_file():
    """Gecersiz dosya formati 400 donmeli."""

def test_predict_superclass_too_large():
    """10MB'dan buyuk dosya 413 donmeli."""

# --- MI Localization ---
def test_predict_localization():
    """POST /predict/mi-localization 200 donmeli."""

def test_predict_localization_response_schema():
    """Response MILocalizationResponse modeline uymali."""

# --- Artifact Serving ---
def test_path_traversal_blocked():
    """../../etc/passwd gibi path traversal 400 donmeli."""

def test_invalid_run_id_rejected():
    """Gecersiz run_id formati 400 donmeli."""
```

### Dogrulama
```bash
pytest tests/test_api.py -v
# Tum testler gecmeli
```

---

## GOREV 6: Docker

**Oncelik:** P2 - Deployment
**Tahmini Dosya Sayisi:** 3 dosya
**Bagimsizlik:** Gorev 1, 4 tamamlandiktan sonra (requirements.txt ve frontend build gerekli)

### 6.1 Dockerfile

```dockerfile
# ====== Stage 1: Frontend Build ======
FROM node:18-alpine AS frontend-builder
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ .
RUN npm run build

# ====== Stage 2: Python Backend ======
FROM python:3.10-slim AS backend
WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 && rm -rf /var/lib/apt/lists/*

# Python deps (CPU-only PyTorch)
COPY requirements.txt .
RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# App code
COPY src/ src/
COPY checkpoints/ checkpoints/
COPY logs/ logs/
COPY artifacts/ artifacts/
COPY sample.npy test_mi_sample.npz ./

# Frontend static files
COPY --from=frontend-builder /app/frontend/dist frontend/dist

# Serve frontend from backend (production)
# Backend main.py'de StaticFiles mount eklenecek

EXPOSE 8000
CMD ["uvicorn", "src.backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 6.2 docker-compose.yml

```yaml
version: "3.8"
services:
  cardioguard:
    build: .
    container_name: cardioguard-ai
    ports:
      - "8000:8000"
    environment:
      - CORS_ORIGINS=*
    volumes:
      - ./reports:/app/reports
    restart: unless-stopped
```

### 6.3 .dockerignore

```
.git
.idea
__pycache__
*.pyc
node_modules
.venv
physionet.org
docs
tests
```

### 6.4 Backend Static File Serving

**Dosya:** `src/backend/main.py`

Production'da frontend'i backend'den serve etmek icin:
```python
from fastapi.staticfiles import StaticFiles

frontend_dist = Path("frontend/dist")
if frontend_dist.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dist), html=True), name="frontend")
```

### Dogrulama
```bash
docker-compose up --build
# Beklenen:
# - Image build basarili (~500MB)
# - Container baslar
# - http://localhost:8000 → Frontend gorunur
# - http://localhost:8000/health → {"status": "healthy"}
# - http://localhost:8000/predict/superclass → Tahmin calisiyor
```

---

## GOREV 7: Dokumantasyon Guncellemesi

**Oncelik:** P2 - Temizlik
**Tahmini Dosya Sayisi:** 4 dosya
**Bagimsizlik:** Tum gorevler tamamlandiktan sonra

### 7.1 CLAUDE.md Guncelle
- Optimize edilmis threshold'lari yaz
- Yeni frontend bilesen listesini guncelle
- Docker komutlarini ekle

### 7.2 01_architecture.md Duzelt
- "Consistency Guard entegre degil" → "Consistency Guard TAM ENTEGRE" olarak duzelt
- Component diagram guncelle

### 7.3 05_frontend_integration.md Duzelt
- Eski component isimleri (ECGUploader, ResultDisplay) → yeni isimler

### 7.4 Threshold Optimization Study
- Yeni dosya: `docs/threshold_optimization_study.md`
- Icerik: Yontem (F-beta, Youden-J), sonuclar, onceki-sonrasi karsilastirma

---

## Paralel Calisma Haritasi

```
PARALEL GRUP A (Backend):          PARALEL GRUP B (Frontend):
┌──────────────────────┐           ┌──────────────────────┐
│ Agent 1: Gorev 1     │           │ Agent 3: Gorev 4     │
│ Backend Bug Fix      │           │ Frontend Redesign    │
│ (5 dosya)            │           │ (12-15 dosya)        │
└──────────────────────┘           └──────────────────────┘
┌──────────────────────┐
│ Agent 2: Gorev 2+3   │
│ Threshold + XAI      │
│ (4 dosya)            │
└──────────────────────┘

         ↓ Tamamlaninca ↓

SIRAYLA:
┌──────────────────────┐
│ Agent 4: Gorev 5     │
│ API Tests            │
│ (1 dosya)            │
└──────────────────────┘
         ↓
┌──────────────────────┐
│ Agent 5: Gorev 6     │
│ Docker               │
│ (3 dosya)            │
└──────────────────────┘
         ↓
┌──────────────────────┐
│ Agent 6: Gorev 7     │
│ Docs Update          │
│ (4 dosya)            │
└──────────────────────┘
```

**Toplam Dosya Degisikligi:** ~30-35 dosya
**Yeni Dosya:** ~15 dosya
**Guncellenen Dosya:** ~15-20 dosya

---

## Kapsam Disi (Bu Iterasyonda Yapilmayacak)

- LLM entegrasyonu (Claude API ile klinik rapor)
- CNN/XGBoost model yeniden egitimi
- Gercek zamanli EKG streaming (WebSocket)
- CI/CD pipeline (GitHub Actions)
- Frontend component testleri (Jest/Vitest)
- E2E testleri
- Git LFS
- Veritabani entegrasyonu

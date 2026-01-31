# Phase 0: Repository Map & Discovery

**Generated Date:** 2026-01-31
**Auditor:** Antigravity (Technical Auditor AI)
**Methodology:** Exhaustive file system traversal + static code analysis.

---

## 1. Project Identity

**Proje Adı:** CardioGuard-AI
**Amaç:** 12 derivasyonlu (12-lead) EKG sinyallerinden Miyokard Enfarktüsü (MI / Kalp Krizi) tespiti ve anatomik lokalizasyonu.
**Hedef Kullanıcı:** Klinisyenler, kardiyologlar, acil servis hekimleri.

Bu proje, derin öğrenme (CNN) ve geleneksel makine öğrenmesi (XGBoost) yöntemlerini birleştiren bir "ensemble" yaklaşımı kullanmaktadır. Ayrıca, kararların açıklanabilirliği için XAI (Explainable AI) modülleri entegre edilmiştir.

---

## 2. Üst Seviye Dizin Yapısı

Aşağıdaki tablo, proje kök dizinindeki her klasör ve kritik dosyanın rolünü açıklamaktadır:

| Dizin/Dosya | Tür | Açıklama | Kritiklik |
| :--- | :--- | :--- | :---: |
| `src/` | Klasör | **Ana kaynak kodu.** Tüm Python modülleri burada. Backend, Pipeline, Models, XAI alt klasörlerini içerir. | ⭐⭐⭐ |
| `frontend/` | Klasör | React + Vite tabanlı kullanıcı arayüzü. TypeScript ile yazılmış. | ⭐⭐ |
| `tests/` | Klasör | Pytest test suite. Kritik modüller için birim testleri. | ⭐⭐ |
| `checkpoints/` | Klasör | Eğitilmiş model ağırlıkları (`.pt` dosyaları). Örn: `ecgcnn_superclass.pt` | ⭐⭐⭐ |
| `logs/` | Klasör | XGBoost modelleri ve eğitim logları. Her sınıf için ayrı alt klasör (`MI/`, `STTC/` vb.) | ⭐⭐ |
| `artifacts/` | Klasör | Threshold JSON dosyaları ve diğer konfigürasyonlar. | ⭐⭐ |
| `reports/` | Klasör | XAI çıktıları ve analiz raporları. `reports/xai/runs/` altında artifact'lar saklanır. | ⭐⭐ |
| `data/` | Klasör | Meta veri ve veri seti split bilgileri. | ⭐ |
| `physionet.org/` | Klasör | PTB-XL veri seti (PhysioNet'ten indirilmiş ham EKG kayıtları). | ⭐⭐ |
| `docs/` | Klasör | Dokümantasyon dosyaları (bu raporlar dahil). | ⭐ |
| `scripts/` | Klasör | Yardımcı scriptler (veri hazırlama, dönüştürme vb.). | ⭐ |
| `requirements.txt` | Dosya | Python bağımlılıkları listesi. | ⭐⭐ |
| `sample.npy` | Dosya | Test amaçlı örnek EKG sinyali. | ⭐ |
| `test_mi_sample.npz` | Dosya | MI pozitif test örneği. | ⭐ |

---

## 3. Kaynak Kod Yapısı (`src/`)

`src/` klasörü, projenin kalbidir. İçerdiği modüller ve sorumlulukları:

### 3.1 `src/backend/` — API Katmanı

| Dosya | Satır Sayısı | Sorumluluk |
| :--- | :---: | :--- |
| `main.py` | 614 | FastAPI uygulaması. Tüm HTTP endpoint'leri burada tanımlanır. Model yüklemesi, istek doğrulaması ve yanıt formatlama işlemlerini yapar. **Kritik:** Bu dosya hiçbir ML inference kodu içermez; sadece Pipeline'ı çağırır. |
| `__init__.py` | 1 | Modül tanımlayıcı. |

**Mimari Kararı:** Backend, bir "Gateway" gibi davranır. İş mantığı (inference) burada yapılmaz. Bu sayede:
- Backend bağımsız olarak test edilebilir.
- Pipeline değişiklikleri Backend'i etkilemez.
- "Fail-closed" güvenlik modeli uygulanabilir.

### 3.2 `src/pipeline/` — İş Mantığı

Bu klasör, inference, training ve evaluation mantığını barındırır.

```
src/pipeline/
├── inference/           # Çıkarım (inference) scriptleri
│   ├── run_inference_superclass.py    # ⭐ ANA ORKESTRATOR
│   ├── run_inference_localization.py  # MI lokalizasyonu
│   ├── run_inference_binary.py        # Binary MI sınıflandırma
│   ├── consistency_guard.py           # Model tutarlılık kontrolü
│   └── generate_validation_predictions.py
├── training/            # Eğitim scriptleri
│   ├── train_superclass_cnn.py        # CNN eğitimi
│   ├── train_superclass_xgb_ovr.py    # XGBoost OVR eğitimi
│   ├── train_mi_localization.py       # Lokalizasyon modeli
│   └── run_experiment.py
├── evaluation/          # Değerlendirme
│   └── run_comprehensive_test.py
├── features/            # Feature extraction
├── xai/                 # XAI entegrasyonu
└── utils/               # Yardımcı fonksiyonlar
```

**`run_inference_superclass.py` Detayları (591 satır):**
- **Ana Fonksiyon:** `predict()` — Tüm inference mantığını orkestre eder.
- **Model Yükleme:** `load_cnn_model()`, `load_xgb_models()`, `load_localization_model()`
- **Preprocessing:** `ensure_channel_first()` — Sinyal formatını `(12, T)` şekline getirir.
- **XAI Üretimi:** `explain=True` parametresi ile Grad-CAM ve SHAP üretir.
- **Manifest Yazımı:** `_write_manifest()` — XAI artifact'larını disk'e yazar.

### 3.3 `src/models/` — Model Tanımları

| Dosya | İçerik |
| :--- | :--- |
| `cnn.py` | `ECGCNNConfig`, `ECGBackbone`, `ECGCNN` sınıfları. EfficientNet benzeri 1D CNN mimarisi. |

**Mimari Detayları:**
- **Backbone:** Birden fazla konvolüsyon bloğu içerir. Her blok: Conv1D → BatchNorm → ReLU → Dropout.
- **Head:** Global Average Pooling → Dense → Sigmoid (multi-label için).
- **Konfigürasyon:** `num_filters=64`, `dropout=0.5` varsayılan değerler.

### 3.4 `src/xai/` — Açıklanabilir AI

| Dosya | Satır | Sorumluluk |
| :--- | :---: | :--- |
| `gradcam.py` | 188 | Grad-CAM implementasyonu. `GradCAM` sınıfı ve `generate_relevant_gradcam()` fonksiyonu. |
| `unified.py` | 159 | `UnifiedExplainer` sınıfı. Grad-CAM ve SHAP sonuçlarını birleştirip klinik narrative üretir. |
| `sanity.py` | ~100 | `XAISanityChecker` — XAI çıktılarının anlamlı olup olmadığını doğrular. |
| `shap_ovr.py` | ~80 | XGBoost modelleri için SHAP değerleri hesaplar. |
| `visualize.py` | ~200 | Görselleştirme fonksiyonları (12-lead plot, heatmap overlay vb.). |
| `reporting.py` | ~50 | `generate_run_id()` ve raporlama yardımcıları. |

### 3.5 `src/contracts/` — Veri Kontratları

Sistem genelinde tutarlılık sağlamak için tanımlanmış veri yapıları:

| Dosya | İçerik |
| :--- | :--- |
| `airesult_mapper.py` | Backend ve Frontend arasında veri dönüşümü. |

### 3.6 `src/utils/` — Yardımcı Modüller

| Dosya | Sorumluluk |
| :--- | :--- |
| `model_loader.py` | Güvenli model yükleme (`load_model_safe()`). Hash doğrulaması yapar. |
| `checkpoint_validation.py` | Checkpoint'ların beklenen çıktı boyutlarıyla eşleştiğini doğrular. |
| `signal.py` | Sinyal işleme yardımcıları. |

---

## 4. Frontend Yapısı (`frontend/`)

```
frontend/
├── index.html          # HTML şablonu
├── index.tsx           # React entry point
├── package.json        # Node.js bağımlılıkları
├── vite.config.ts      # Vite bundler konfigürasyonu
├── tsconfig.json       # TypeScript konfigürasyonu
├── components/         # React bileşenleri
│   └── (4 dosya)
└── lib/                # Paylaşılan yardımcılar
    ├── api.ts          # API çağrı fonksiyonları
    └── types.ts        # TypeScript tip tanımları
```

**Teknoloji Stack'i:**
- **Framework:** React 19.2.4 (en güncel major versiyon)
- **Build Tool:** Vite 6.2.0 (hızlı HMR ve bundling)
- **Dil:** TypeScript 5.8.2 (strict mode)
- **Styling:** Vanilla CSS (TailwindCSS yok)

**`lib/types.ts` (100 satır):**
Bu dosya, Backend Pydantic modellerinin birebir TypeScript karşılığını içerir. Örneğin:
- `SuperclassResponse` ↔ `SuperclassPredictionResponse`
- `LocalizationResponse` ↔ `MILocalizationResponse`
- `XaiSchema` ↔ `XAIInfo`

---

## 5. Test Suite (`tests/`)

| Test Dosyası | Satır | Kapsam |
| :--- | :---: | :--- |
| `test_consistency_guard.py` | 177 | `ConsistencyGuard` modülünün tüm senaryoları (Agree, Disagree Type 1/2). |
| `test_artifacts.py` | ~200 | XAI artifact üretimi ve manifest yazımı. |
| `test_checkpoint_validation.py` | ~200 | Model checkpoint doğrulaması. |
| `test_data.py` | ~300 | Veri yükleme ve split mantığı. |
| `test_airesult_mapper.py` | ~350 | Kontrat dönüşümleri. |
| `test_xai_visualization.py` | ~150 | Görselleştirme fonksiyonları. |
| `test_gradcam.py` | ~50 | Grad-CAM temel işlevselliği. |
| `test_model.py` | ~50 | Model instantiation. |
| `test_xgb_pipeline.py` | ~50 | XGBoost pipeline testi. |

**Test Çalıştırma:**
```bash
cd CardioGuard-AI
pytest tests/ -v
```

---

## 6. Bağımlılıklar

### 6.1 Python (`requirements.txt`)

```
numpy          # Sayısal hesaplamalar
pandas         # Veri manipülasyonu
torch          # PyTorch (derin öğrenme)
scikit-learn   # ML yardımcıları (scaler, calibration)
xgboost        # Gradient boosting
wfdb           # PhysioNet veri formatı okuyucu
tabulate       # Tablo formatlama
tqdm           # İlerleme çubuğu
shap           # SHAP değerleri
matplotlib     # Görselleştirme
scipy          # Bilimsel hesaplama
```

> ⚠️ **Eksiklik:** `fastapi` ve `uvicorn` `requirements.txt`'te listelenmemiş, ancak kod bunları kullanıyor. Bu bir dokümantasyon eksikliğidir.

### 6.2 Node.js (`package.json`)

```json
{
  "dependencies": {
    "react": "^19.2.4",
    "react-dom": "^19.2.4"
  },
  "devDependencies": {
    "vite": "^6.2.0",
    "typescript": "~5.8.2",
    "@vitejs/plugin-react": "^5.0.0",
    "@types/node": "^22.14.0"
  }
}
```

---

## 7. Çalıştırma Komutları

### 7.1 Backend API Başlatma

```bash
# Kök dizinden:
uvicorn src.backend.main:app --host 0.0.0.0 --port 8000 --reload

# Beklenen çıktı:
# Validating checkpoints...
# Checkpoint validation passed!
# Superclass model loaded (schema: ...)
# Localization model loaded
# XGBoost feature schema loaded: 64 features
# Models loaded: Superclass=OK, Localization=True, XGB=4
# INFO:     Uvicorn running on http://0.0.0.0:8000
```

### 7.2 Frontend Development Server

```bash
cd frontend
npm install
npm run dev

# Beklenen çıktı:
# VITE v6.2.0 ready in 500ms
# ➜ Local: http://localhost:5173/
```

### 7.3 CLI Inference

```bash
python -m src.pipeline.inference.run_inference_superclass \
    --input sample.npy \
    --explain \
    --output result.json
```

---

## 8. Özet Bulgular

| Kategori | Bulgu |
| :--- | :--- |
| **Mimari** | Sağlam. Backend/Pipeline ayrımı doğru uygulanmış. |
| **Kod Kalitesi** | Yüksek. Type hint'ler, docstring'ler mevcut. |
| **Bağımlılıklar** | Eksik: `fastapi`, `uvicorn` requirements.txt'te yok. |
| **Test Kapsamı** | Orta-Yüksek. Kritik modüller test edilmiş, E2E eksik. |
| **Dokümantasyon** | Bu audit öncesi yetersizdi, şimdi kapsamlı. |

---

## 9. Açık Noktalar (BULUNAMADI)

| Beklenen | Durum | Öneri |
| :--- | :--- | :--- |
| `Dockerfile` | ❌ Yok | Docker deployment için `Dockerfile` ve `docker-compose.yml` oluşturulmalı. |
| `pyproject.toml` | ❌ Yok | Modern Python projesi için `pyproject.toml` ile bağımlılık yönetimi önerilir. |
| `.env` örneği | ⚠️ Frontend'de var, Backend'de yok | Backend için de `.env.example` oluşturulmalı. |

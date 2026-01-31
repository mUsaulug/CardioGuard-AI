# CardioGuard-AI

**Açıklanabilir Yapay Zeka Destekli 12 Derivasyonlu EKG Analiz Platformu**

Çoklu Etiketli Kardiyak Patoloji Tespiti | MI Lokalizasyonu | Grad-CAM ve SHAP Açıklamaları

---

## Proje Hakkında

CardioGuard-AI, 12 derivasyonlu EKG sinyallerini analiz ederek kardiyak anormallikleri tespit eden ileri düzey bir klinik karar destek sistemidir. Geleneksel "kara kutu" modellerinden farklı olarak, CardioGuard-AI Grad-CAM ısı haritaları ve SHAP özellik analizi aracılığıyla şeffaf ve yorumlanabilir tahminler sunar.

### Temel Özellikler

| Özellik | Açıklama |
|:--------|:---------|
| Çoklu Etiket Sınıflandırma | MI, STTC, CD, HYP patolojilerinin eş zamanlı tespiti |
| MI Lokalizasyonu | 5 anatomik bölgeye lokalizasyon (AMI, ASMI, ALMI, IMI, LMI) |
| Hibrit Ensemble Mimarisi | CNN + XGBoost OVR, %50-%50 ağırlıklı ortalama |
| Açıklanabilir Yapay Zeka | Grad-CAM zamansal odak + SHAP özellik katkıları |
| Güvenlik Odaklı Tasarım | Başlangıçta doğrulama, girdi validasyonu, yol geçişi koruması |
| Birleşik Raporlama | Klinik anlatı üretimi ile XAI artifact birleştirme |

---

## Teknoloji Yığını

| Kategori | Teknoloji | Sürüm |
|:---------|:----------|:------|
| Backend | Python | 3.10+ |
| Derin Öğrenme | PyTorch | 2.x |
| API Framework | FastAPI | 0.100+ |
| Gradient Boosting | XGBoost | OVR |
| Frontend Framework | React | 19 |
| Frontend Dili | TypeScript | 5.8 |
| Build Tool | Vite | 6.2 |

---

## Sistem Mimarisi

```
+------------------------------------------------------------------+
|                         CardioGuard-AI                            |
+------------------------------------------------------------------+
|                                                                   |
|   +---------------+    +---------------+    +------------------+  |
|   |               |    |               |    |                  |  |
|   |   FRONTEND    |--->|  BACKEND API  |--->| CIKARIM MOTORU   |  |
|   |               |    |               |    |                  |  |
|   |   React 19    |    |   FastAPI     |    | PyTorch+XGBoost  |  |
|   |   TypeScript  |<---|   Pydantic    |<---| Grad-CAM+SHAP    |  |
|   |               |    |               |    |                  |  |
|   +---------------+    +---------------+    +------------------+  |
|          |                    |                     |             |
|          v                    v                     v             |
|   +----------------------------------------------------------+   |
|   |                      DOSYA SISTEMI                        |   |
|   |                                                           |   |
|   |   checkpoints/     artifacts/     reports/xai/runs/       |   |
|   |   (Model Agirliklari)  (Esikler)     (XAI Ciktilari)     |   |
|   +----------------------------------------------------------+   |
|                                                                   |
+------------------------------------------------------------------+
```

---

## Veri Akisi Diyagrami

```
+-------------------+
|                   |
|   EKG YUKLEME     |
|   (.npy / .npz)   |
|                   |
+---------+---------+
          |
          v
+---------+---------+
|                   |
|   VALIDASYON      |
|   (Format, Boyut) |
|                   |
+---------+---------+
          |
          v
+---------+---------+
|                   |
|   ON ISLEME       |
|   (12, T) format  |
|                   |
+---------+---------+
          |
          v
+---------+---------+
|                   |
|   CNN MODELI      |
|   (Superclass)    |
|                   |
+---------+---------+
          |
          +------------------+
          |                  |
          v                  v
+---------+---------+   +----+----+
|                   |   |         |
|   CNN OLASILIK    |   | EMBEDDING|
|   (4 sinif)       |   | (64 dim) |
|                   |   |         |
+---------+---------+   +----+----+
          |                  |
          |                  v
          |         +--------+--------+
          |         |                 |
          |         |   XGBOOST OVR   |
          |         |   (4 classifier)|
          |         |                 |
          |         +--------+--------+
          |                  |
          v                  v
+---------+------------------+--------+
|                                     |
|            ENSEMBLE                 |
|     (0.5 * CNN + 0.5 * XGBoost)    |
|                                     |
+---------+---------+-----------------+
          |
          v
+---------+---------+
|                   |
|   ESIK UYGULAMA   |
|   (Per-class)     |
|                   |
+---------+---------+
          |
          v
+---------+---------+
|                   |
|  MI TESPIT EDILDI |
|       MI?         |
+---------+---------+
          |
    +-----+-----+
    |           |
   EVET       HAYIR
    |           |
    v           |
+---+---+       |
|       |       |
| LOKAL.|       |
| MODEL |       |
|       |       |
+---+---+       |
    |           |
    +-----+-----+
          |
          v
+---------+---------+
|                   |
|  explain=true?    |
|                   |
+---------+---------+
          |
    +-----+-----+
    |           |
   EVET       HAYIR
    |           |
    v           |
+---+---+       |
|       |       |
|GRAD-CAM|       |
| SHAP  |       |
|UNIFIED|       |
|       |       |
+---+---+       |
    |           |
    +-----+-----+
          |
          v
+---------+---------+
|                   |
|   JSON YANIT      |
|   (Tahmin+XAI)    |
|                   |
+-------------------+
```

---

## Model Performansi

PTB-XL veri seti üzerinde eğitilmiş ve doğrulanmıştır (21,837 EKG kaydı, 18,885 hasta).

### Genel Metrikler

| Metrik | Değer |
|:-------|:------|
| Macro AUROC | 0.8998 |
| Macro AUPRC | 0.7278 |
| Macro F1 | 0.6302 |
| Micro F1 | 0.6420 |
| Hamming Accuracy | %80.51 |

### Sınıf Bazlı Performans

| Sınıf | Açıklama | AUROC | AUPRC | F1 | Destek |
|:------|:---------|:-----:|:-----:|:---:|:------:|
| MI | Miyokard Enfarktüsü | 0.9022 | 0.7795 | 0.6933 | 550 |
| STTC | ST/T Değişikliği | 0.9193 | 0.7497 | 0.6638 | 506 |
| CD | İletim Bozukluğu | 0.8923 | 0.7738 | 0.6794 | 496 |
| HYP | Hipertrofi | 0.8805 | 0.6201 | 0.4844 | 261 |

### CNN ve XGBoost Karşılaştırması

| Sınıf | CNN AUROC | XGB AUROC | Fark | Kazanan |
|:------|:---------:|:---------:|:----:|:--------|
| MI | 0.9022 | 0.9024 | +0.0002 | XGB |
| STTC | 0.9193 | 0.9218 | +0.0025 | XGB |
| CD | 0.8923 | 0.8881 | -0.0042 | CNN |
| HYP | 0.8805 | 0.8868 | +0.0063 | XGB |
| Macro | 0.8986 | 0.8998 | +0.0012 | XGB |

---

## Kurulum

### Gereksinimler

| Gereksinim | Minimum Sürüm |
|:-----------|:--------------|
| Python | 3.10+ |
| Node.js | 18+ |
| CUDA | Opsiyonel (GPU hızlandırma için) |

### Adım 1: Depoyu Klonlama

```bash
git clone https://github.com/kullanici/CardioGuard-AI.git
cd CardioGuard-AI
```

### Adım 2: Python Ortamı Kurulumu

```bash
# Sanal ortam oluşturma
python -m venv .venv

# Sanal ortamı aktifleştirme (Windows)
.venv\Scripts\activate

# Sanal ortamı aktifleştirme (Linux/Mac)
source .venv/bin/activate

# Bağımlılıkları yükleme
pip install -r requirements.txt
pip install fastapi uvicorn
```

### Adım 3: Frontend Kurulumu

```bash
cd frontend
npm install
cd ..
```

---

## Uygulamayı Çalıştırma

### Backend API Başlatma

```bash
uvicorn src.backend.main:app --host 0.0.0.0 --port 8000 --reload
```

Beklenen çıktı:

```
Validating checkpoints...
Checkpoint validation passed!
Superclass model loaded
Localization model loaded
XGBoost models loaded: 4 classifiers
Models loaded: Superclass=OK, Localization=True, XGB=4
INFO: Uvicorn running on http://0.0.0.0:8000
```

### Frontend Başlatma

Yeni bir terminal açın:

```bash
cd frontend
npm run dev
```

Beklenen çıktı:

```
VITE v6.2.0 ready in 500ms
Local: http://localhost:5173/
```

### Tarayıcıda Açma

Tarayıcınızda `http://localhost:5173` adresine gidin ve EKG dosyası yükleyin (.npy veya .npz formatında).

---

## Proje Yapısı

```
CardioGuard-AI/
|
|-- src/                              # Python kaynak kodu
|   |
|   |-- backend/                      # FastAPI REST API
|   |   |-- main.py                   # API endpoint'leri (614 satır)
|   |   +-- __init__.py
|   |
|   |-- models/                       # Sinir ağı tanımları
|   |   |-- cnn.py                    # ECGBackbone, ECGCNN (153 satır)
|   |   +-- xgb.py                    # XGBoost yardımcıları
|   |
|   |-- pipeline/                     # İş mantığı
|   |   |
|   |   |-- inference/                # Çıkarım scriptleri
|   |   |   |-- run_inference_superclass.py    # Ana orkestrator (591 satır)
|   |   |   |-- run_inference_localization.py  # MI lokalizasyonu
|   |   |   |-- run_inference_binary.py        # Binary MI
|   |   |   +-- consistency_guard.py           # Tutarlılık kontrolü
|   |   |
|   |   |-- training/                 # Eğitim scriptleri
|   |   |   |-- train_superclass_cnn.py
|   |   |   |-- train_superclass_xgb_ovr.py
|   |   |   +-- train_mi_localization.py
|   |   |
|   |   +-- evaluation/               # Değerlendirme
|   |       +-- run_comprehensive_test.py
|   |
|   |-- xai/                          # Açıklanabilir AI modülleri
|   |   |-- gradcam.py                # Grad-CAM implementasyonu (188 satır)
|   |   |-- shap_ovr.py               # XGBoost için SHAP
|   |   |-- unified.py                # Birleşik açıklayıcı (159 satır)
|   |   |-- sanity.py                 # XAI kalite kontrolü
|   |   +-- visualize.py              # Görselleştirme fonksiyonları
|   |
|   |-- contracts/                    # Veri kontratları
|   |   +-- airesult_mapper.py        # Backend-Frontend dönüşümü
|   |
|   |-- data/                         # Veri yükleme
|   |   +-- splits.py                 # Hasta bazlı split'ler
|   |
|   +-- utils/                        # Yardımcı modüller
|       |-- model_loader.py           # Güvenli model yükleme
|       |-- checkpoint_validation.py  # Checkpoint doğrulama
|       +-- signal.py                 # Sinyal işleme
|
|-- frontend/                         # React 19 + Vite + TypeScript
|   |-- index.html
|   |-- index.tsx                     # React giriş noktası
|   |-- package.json
|   |-- vite.config.ts
|   |-- tsconfig.json
|   |
|   |-- components/                   # React bileşenleri
|   |   +-- (4 dosya)
|   |
|   +-- lib/                          # Paylaşılan yardımcılar
|       |-- api.ts                    # API istemcisi
|       +-- types.ts                  # TypeScript tip tanımları (100 satır)
|
|-- checkpoints/                      # Eğitilmiş model ağırlıkları
|   |-- ecgcnn_superclass.pt          # Superclass CNN (~2.5 MB)
|   +-- ecgcnn_localization.pt        # Lokalizasyon CNN
|
|-- logs/                             # Eğitim çıktıları
|   |-- superclass_cnn/
|   |   +-- training_results.json
|   |
|   +-- xgb_superclass/               # XGBoost modelleri
|       |-- MI/
|       |   |-- xgb_model.json
|       |   +-- calibrator.joblib
|       |-- STTC/
|       |-- CD/
|       |-- HYP/
|       |-- scaler.joblib
|       +-- feature_schema.json
|
|-- artifacts/                        # Konfigürasyonlar
|   +-- thresholds_superclass.json    # Sınıf bazlı eşikler
|
|-- reports/                          # Raporlar ve XAI çıktıları
|   +-- xai/runs/                     # XAI artifact'ları
|       +-- {run_id}/
|           |-- manifest.json
|           |-- visuals/
|           +-- text/
|
|-- tests/                            # Pytest test suite
|   |-- test_consistency_guard.py
|   |-- test_airesult_mapper.py
|   |-- test_checkpoint_validation.py
|   |-- test_data.py
|   |-- test_gradcam.py
|   |-- test_xgb_pipeline.py
|   +-- (diğer testler)
|
|-- docs/                             # Dokümantasyon
|   |-- 00_repo_map.md
|   |-- 01_architecture.md
|   |-- 02_api_contracts.md
|   |-- 03_inference_pipeline.md
|   |-- 04_xai_and_artifacts.md
|   |-- 05_frontend_integration.md
|   +-- FINAL_TEKNIK_RAPOR_TR.md
|
|-- requirements.txt                  # Python bağımlılıkları
|-- sample.npy                        # Örnek EKG sinyali
+-- test_mi_sample.npz                # MI pozitif test örneği
```

---

## API Referansı

### Endpoint Listesi

| Metod | Yol | Açıklama | Hata Kodları |
|:------|:----|:---------|:-------------|
| POST | /predict/superclass | Çoklu etiket sınıflandırma | 400, 413, 500, 503 |
| POST | /predict/mi-localization | MI anatomik lokalizasyonu | 400, 500 |
| GET | /runs/{run_id}/{file_path} | XAI artifact sunumu | 400, 404 |
| GET | /health | Sağlık kontrolü | - |
| GET | /ready | Hazırlık kontrolü | - |

### Superclass Tahmin İsteği

```bash
curl -X POST "http://localhost:8000/predict/superclass" \
  -F "file=@sample.npy" \
  -F "explain=true" \
  -F "ensemble_weight=0.5"
```

### İstek Parametreleri

| Parametre | Tip | Zorunlu | Varsayılan | Açıklama |
|:----------|:----|:-------:|:-----------|:---------|
| file | UploadFile | Evet | - | EKG dosyası (.npy veya .npz) |
| explain | bool | Hayır | false | XAI artifact üretimi |
| ensemble_weight | float | Hayır | 0.5 | CNN/XGB ağırlığı (0.0-1.0) |
| sanity_check | bool | Hayır | true | XAI kalite kontrolü |

### Yanıt Şeması

```json
{
  "mode": "superclass",
  
  "probabilities": {
    "MI": 0.85,
    "STTC": 0.12,
    "CD": 0.08,
    "HYP": 0.05,
    "NORM": 0.15
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
    "confidence": 0.85,
    "rule": "priority_order"
  },
  
  "sources": {
    "cnn": {"MI": 0.82, "STTC": 0.10, "CD": 0.06, "HYP": 0.04},
    "xgb": {"MI": 0.88, "STTC": 0.14, "CD": 0.10, "HYP": 0.06},
    "ensemble": {"MI": 0.85, "STTC": 0.12, "CD": 0.08, "HYP": 0.05}
  },
  
  "versions": {
    "model_hash": "abc123def456",
    "threshold_hash": "789xyz",
    "api_version": "1.1.0",
    "timestamp": "2026-01-31T23:30:00+03:00"
  },
  
  "xai": {
    "enabled": true,
    "run_id": "20260131_233000_abc123",
    "run_dir": "reports/xai/runs/20260131_233000_abc123",
    "artifacts": [
      {
        "type": "gradcam",
        "name": "gradcam_heatmap.png",
        "url": "/runs/20260131_233000_abc123/visuals/gradcam_heatmap.png",
        "mime": "image/png"
      },
      {
        "type": "shap",
        "name": "shap_summary.png",
        "url": "/runs/20260131_233000_abc123/visuals/shap_summary.png",
        "mime": "image/png"
      },
      {
        "type": "narrative",
        "name": "unified_report.md",
        "url": "/runs/20260131_233000_abc123/text/unified_report.md",
        "mime": "text/markdown"
      }
    ],
    "sanity": {
      "status": "PASS",
      "gradcam_variance": 0.15,
      "peak_spread": 0.25
    }
  },
  
  "consistency": {
    "agreement": "AGREE_MI",
    "triage_level": "HIGH",
    "superclass_mi_prob": 0.85,
    "binary_mi_prob": 0.92,
    "superclass_mi_decision": true,
    "binary_mi_decision": true,
    "warnings": []
  }
}
```

### Hata Yanıtları

| Kod | Durum | Açıklama |
|:----|:------|:---------|
| 400 | Bad Request | Geçersiz dosya formatı veya parametre |
| 413 | Payload Too Large | Dosya boyutu 10MB'ı aşıyor |
| 500 | Internal Server Error | Tahmin sırasında hata |
| 503 | Service Unavailable | Modeller yüklenmemiş |

---

## Teknik Detaylar

### CNN Mimarisi

```
Girdi: (batch, 12, T)
        |
        v
+-------+-------+
|  Conv1d       |  12 -> 64 kanal
|  kernel=7     |  padding=3
+-------+-------+
        |
        v
+-------+-------+
|  BatchNorm1d  |  64 kanal
+-------+-------+
        |
        v
+-------+-------+
|     ReLU      |
+-------+-------+
        |
        v
+-------+-------+
|   Dropout     |  p=0.3
+-------+-------+
        |
        v
+-------+-------+
|               |
| ResidualBlock |  x4 tekrar
|   Conv1d(64)  |
|   BatchNorm   |
|   ReLU        |
|   Skip Conn.  |
|               |
+-------+-------+
        |
        v
+-------+-------+
| AdaptiveAvg   |
| Pool1d(1)     |
+-------+-------+
        |
        v
+-------+-------+
|   Flatten     |  -> (batch, 64)
+-------+-------+
        |
        v
+-------+-------+
|  Linear(64,4) |  4 sınıf çıktı
+-------+-------+
        |
        v
+-------+-------+
|   Sigmoid     |  Multi-label
+-------+-------+
        |
        v
Çıktı: (batch, 4)
MI, STTC, CD, HYP olasılıkları
```

### CNN Konfigürasyonu

| Parametre | Değer |
|:----------|:------|
| Giriş Kanalları | 12 |
| Filtre Sayısı | 64 |
| Kernel Boyutu | 7 |
| Dropout Oranı | 0.3 |
| Epoch | 50 |
| Batch Size | 64 |
| Learning Rate | 0.001 |
| Weight Decay | 0.0001 |
| En İyi Epoch | 46 |

### XGBoost OVR Yapısı

Her sınıf için ayrı binary classifier eğitilmiştir:

```
CNN Backbone Çıktısı
        |
        v
+-------+-------+
|               |
| StandardScaler|  64-dim özellik
|               |
+-------+-------+
        |
        +-------+-------+-------+
        |       |       |       |
        v       v       v       v
     +--+--+ +--+--+ +--+--+ +--+--+
     | MI  | |STTC | | CD  | | HYP |
     | XGB | | XGB | | XGB | | XGB |
     +--+--+ +--+--+ +--+--+ +--+--+
        |       |       |       |
        v       v       v       v
     +--+--+ +--+--+ +--+--+ +--+--+
     |Iso- | |Iso- | |Iso- | |Iso- |
     |tonic| |tonic| |tonic| |tonic|
     |Calib| |Calib| |Calib| |Calib|
     +--+--+ +--+--+ +--+--+ +--+--+
        |       |       |       |
        +-------+-------+-------+
                |
                v
        Kalibre Olasılıklar
```

### Ensemble Formülü

```
ensemble_prob[sınıf] = 0.5 × CNN_prob[sınıf] + 0.5 × XGB_prob[sınıf]
```

### NORM Sınıfı Türetimi

NORM sınıfı doğrudan tahmin edilmez, türetilir:

```
NORM_prob = 1.0 - max(MI_prob, STTC_prob, CD_prob, HYP_prob)
```

### Birincil Etiket Seçimi

Klinik öncelik sırasına göre seçilir:

| Öncelik | Sınıf | Gerekçe |
|:-------:|:------|:--------|
| 1 | MI | Hayatı tehdit eden acil durum |
| 2 | STTC | İskemi belirtisi olabilir |
| 3 | CD | İletim defekti, blok riski |
| 4 | HYP | Kronik durum |
| 5 | NORM | Patoloji tespit edilmedi |

### Eşik Değerleri

| Sınıf | Optimizasyon Metodu | Optimize Değer | Üretim Değeri |
|:------|:--------------------|:--------------:|:-------------:|
| MI | F-beta (beta=2.0) | 0.01 | 0.5 |
| STTC | Youden's J | 0.418 | 0.5 |
| CD | Youden's J | 0.420 | 0.5 |
| HYP | Youden's J | 0.258 | 0.5 |

---

## XAI Pipeline

### Açıklanabilirlik Bileşenleri

| Bileşen | Dosya | Açıklama |
|:--------|:------|:---------|
| Grad-CAM | gradcam.py | Zamansal saliency haritaları |
| SHAP | shap_ovr.py | XGBoost özellik katkıları |
| Unified Explainer | unified.py | Birleşik klinik anlatı |
| Sanity Checker | sanity.py | XAI kalite kontrolü |
| Visualize | visualize.py | 12-lead plot, heatmap overlay |

### Sanity Check Kriterleri

| Kontrol | Eşik | Anlam |
|:--------|:-----|:------|
| gradcam_variance | > 0.01 | Model belirli bölgelere odaklanıyor |
| peak_spread | > 0.1 | Derivasyonlar farklı ağırlıkta |

### Manifest Yapısı

Her tahmin için oluşturulan manifest.json:

| Alan | Tip | Açıklama |
|:-----|:----|:---------|
| run_id | string | Benzersiz çalışma tanımlayıcısı |
| created_at | string | ISO 8601 timestamp |
| task | string | Görev tipi (multiclass, localization) |
| sample_id | string | Örnek tanımlayıcısı |
| artifacts | array | Artifact listesi [{type, path, mime}] |
| sanity | string | Sanity check sonucu (PASS/FAIL) |
| highlights | array | Aktivasyon pencere koordinatları |

---

## Consistency Guard

Superclass MI ve Binary MI modelleri arasındaki tutarlılığı kontrol eder:

| Agreement Type | Durum | Triage |
|:---------------|:------|:-------|
| AGREE_MI | Her iki model MI tespit etti | HIGH |
| AGREE_NO_MI | Hiçbiri MI tespit etmedi | LOW |
| DISAGREE_TYPE_1 | Superclass MI+, Binary MI- | REVIEW |
| DISAGREE_TYPE_2 | Superclass MI-, Binary MI+ | REVIEW |

---

## Test

### Test Çalıştırma

```bash
# Tüm testleri çalıştır
pytest tests/ -v

# Belirli bir testi çalıştır
pytest tests/test_consistency_guard.py -v

# Coverage ile çalıştır
pytest tests/ --cov=src --cov-report=html
```

### Test Kapsamı

| Test Dosyası | Kapsam |
|:-------------|:-------|
| test_consistency_guard.py | Tutarlılık kontrol mantığı |
| test_airesult_mapper.py | Yanıt dönüştürme |
| test_checkpoint_validation.py | Model doğrulama |
| test_data.py | Veri yükleme, split'ler |
| test_gradcam.py | Grad-CAM üretimi |
| test_xgb_pipeline.py | XGBoost çıkarımı |
| test_artifacts.py | XAI artifact üretimi |
| test_xai_visualization.py | Görselleştirme |
| test_model.py | Model instantiation |

---

## Veri Seti

### PTB-XL Detayları

| Özellik | Değer |
|:--------|:------|
| Toplam Kayıt | 21,837 |
| Hasta Sayısı | 18,885 |
| Örnekleme Frekansı | 100 Hz / 500 Hz |
| Kayıt Süresi | 10 saniye |
| Derivasyon | 12-lead standart |

### Split Protokolü

| Split | Fold'lar | Oran | Örnek Sayısı |
|:------|:---------|:----:|:------------:|
| Train | 1-8 | %80 | 17,388 |
| Validation | 9 | %10 | 2,180 |
| Test | 10 | %10 | 2,180 |

### Veri Sızıntısı Önleme

Hasta bazlı split uygulanmıştır. `verify_no_patient_leakage()` fonksiyonu aynı hastanın birden fazla split'te görünmediğini doğrular.

---

## Dokümantasyon

Detaylı dokümantasyon `docs/` klasöründe mevcuttur:

| Dosya | İçerik |
|:------|:-------|
| 00_repo_map.md | Depo yapısı ve keşif |
| 01_architecture.md | Sistem mimarisi (C4 model) |
| 02_api_contracts.md | API spesifikasyonları |
| 03_inference_pipeline.md | Detaylı pipeline analizi |
| 04_xai_and_artifacts.md | XAI implementasyon detayları |
| 05_frontend_integration.md | Frontend-backend entegrasyonu |
| 06_quality_tests_and_repro.md | Test ve tekrarlanabilirlik |
| FINAL_TEKNIK_RAPOR_TR.md | Kapsamlı teknik rapor |
| MASTER_SOURCE_OF_TRUTH.md | Tüm sistem dokümantasyonu |

---

## Bağımlılıklar

### Python Bağımlılıkları

| Paket | Amaç |
|:------|:-----|
| numpy | Sayısal hesaplamalar |
| pandas | Veri manipülasyonu |
| torch | PyTorch derin öğrenme |
| scikit-learn | ML yardımcıları, kalibrasyon |
| xgboost | Gradient boosting |
| wfdb | PhysioNet veri formatı |
| shap | SHAP değerleri |
| matplotlib | Görselleştirme |
| scipy | Bilimsel hesaplama |
| fastapi | REST API framework |
| uvicorn | ASGI sunucu |
| tabulate | Tablo formatlama |
| tqdm | İlerleme çubuğu |

### Node.js Bağımlılıkları

| Paket | Sürüm | Amaç |
|:------|:------|:-----|
| react | 19.2.4 | UI framework |
| react-dom | 19.2.4 | React DOM |
| vite | 6.2.0 | Build tool |
| typescript | 5.8.2 | Tip güvenliği |

---

## Yol Haritası

| Sürüm | Dönem | Hedef |
|:------|:------|:------|
| v1.1 | Kısa Vade | Consistency Guard tam entegrasyonu |
| v1.2 | Kısa Vade | Uzman onay arayüzü |
| v2.0 | Orta Vade | RAG entegrasyonu, belirsizlik tahmini |
| v2.0 | Orta Vade | LLM ile otomatik klinik rapor |
| v2.x | Uzun Vade | Gerçek zamanlı EKG streaming |
| v2.x | Uzun Vade | Kurumsal analitik dashboard |

---

## Tıbbi Sorumluluk Reddi

**CardioGuard-AI yalnızca araştırma ve eğitim amaçlıdır.**

Bu sistem klinik ortamlarda bağımsız bir tanı aracı olarak kullanılmamalıdır. Tüm tahminler nitelikli sağlık profesyonelleri tarafından bağımsız olarak doğrulanmalıdır. Geliştiriciler, bu sistemin çıktılarına dayalı olarak alınan klinik kararlar için herhangi bir sorumluluk kabul etmez.

---

## Referanslar

| Kaynak | Açıklama |
|:-------|:---------|
| PTB-XL | PhysioNet üzerinde yayınlanan geniş EKG veri seti |
| Grad-CAM | Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks" |
| SHAP | Lundberg ve Lee, "A Unified Approach to Interpreting Model Predictions" |

---

## Lisans

Bu proje araştırma ve eğitim kullanımı için lisanslanmıştır.

---

## İletişim

Sorularınız veya geri bildirimleriniz için lütfen iletişime geçin.

---

**CardioGuard-AI** - Yapay Zeka ve Klinik Karar Verme Arasında Köprü

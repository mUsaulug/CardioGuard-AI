# CardioGuard-AI

**Aciklanabilir Yapay Zeka Destekli 12 Derivasyonlu EKG Analiz Platformu**

Coklu Etiketli Kardiyak Patoloji Tespiti | MI Lokalizasyonu | Grad-CAM ve SHAP Aciklamalari

---

## Proje Hakkinda

CardioGuard-AI, 12 derivasyonlu EKG sinyallerini analiz ederek kardiyak anormallikleri tespit eden ileri duzey bir klinik karar destek sistemidir. Geleneksel "kara kutu" modellerinden farkli olarak, CardioGuard-AI Grad-CAM isi haritalari ve SHAP ozellik analizi araciligiyla seffaf ve yorumlanabilir tahminler sunar.

### Temel Ozellikler

| Ozellik | Aciklama |
|:--------|:---------|
| Coklu Etiket Siniflandirma | MI, STTC, CD, HYP patolojilerinin es zamanli tespiti |
| MI Lokalizasyonu | 5 anatomik bolgeye lokalizasyon (AMI, ASMI, ALMI, IMI, LMI) |
| Hibrit Ensemble Mimarisi | CNN + XGBoost OVR, %50-%50 agirlikli ortalama |
| Aciklanabilir Yapay Zeka | Grad-CAM zamansal odak + SHAP ozellik katkilari |
| Consistency Guard | Superclass ve Binary MI modelleri arasinda tutarlilik kontrolu |
| Guvenlik Odakli Tasarim | Fail-closed baslangic, girdi validasyonu, yol gecisi korumasi |
| Birlesik Raporlama | Klinik anlati uretimi ile XAI artifact birlestirme |

---

## Teknoloji Yigini

| Kategori | Teknoloji | Surum |
|:---------|:----------|:------|
| Backend | Python | 3.10+ |
| Derin Ogrenme | PyTorch | 2.x |
| API Framework | FastAPI | 0.100+ |
| Gradient Boosting | XGBoost | OVR |
| Frontend Framework | React | 19.2.4 |
| Frontend Dili | TypeScript | 5.8.2 |
| Build Tool | Vite | 6.2.0 |

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

## Veri Akisi

```
+-------------------+
|   EKG YUKLEME     |
|   (.npy / .npz)   |
+---------+---------+
          |
          v
+---------+---------+
|   VALIDASYON      |
|   (Format, Boyut) |
+---------+---------+
          |
          v
+---------+---------+
|   ON ISLEME       |
|   (12, T) format  |
+---------+---------+
          |
          v
+---------+---------+
|   CNN MODELI      |
|   (Superclass)    |
+---------+---------+
          |
          +------------------+
          |                  |
          v                  v
+---------+---------+   +----+----+
|   CNN OLASILIK    |   | EMBEDDING|
|   (4 sinif)       |   | (64 dim) |
+---------+---------+   +----+----+
          |                  |
          |                  v
          |         +--------+--------+
          |         |   XGBOOST OVR   |
          |         |   (4 classifier)|
          |         +--------+--------+
          |                  |
          v                  v
+---------+------------------+--------+
|            ENSEMBLE                 |
|     (0.5 * CNN + 0.5 * XGBoost)    |
+----------------+--------------------+
                 |
                 v
+----------------+--------------------+
|        CONSISTENCY GUARD            |
|  (Superclass MI vs Binary MI)       |
|  AGREE_MI / AGREE_NO_MI / REVIEW    |
+----------------+--------------------+
                 |
                 v
+----------------+--------------------+
|          ESIK UYGULAMA              |
|          (Per-class)                |
+----------------+--------------------+
                 |
                 v
+----------------+--------------------+
|         MI TESPIT EDILDI?           |
+----------------+--------------------+
          |                |
         EVET            HAYIR
          |                |
          v                |
+--------+--------+        |
| LOKALIZASYON    |        |
| MODEL (5 bolge) |        |
+--------+--------+        |
          |                |
          +-------+--------+
                  |
                  v
+----------------+--------------------+
|          explain=true?              |
+----------------+--------------------+
          |                |
         EVET            HAYIR
          |                |
          v                |
+--------+--------+        |
| GRAD-CAM        |        |
| SHAP            |        |
| UNIFIED         |        |
| EXPLAINER       |        |
+--------+--------+        |
          |                |
          +-------+--------+
                  |
                  v
+----------------+--------------------+
|           JSON YANIT                |
|   (Tahmin + XAI + Consistency)      |
+-------------------------------------+
```

---

## Model Performansi

PTB-XL veri seti uzerinde egitilmis ve dogrulanmistir (21,837 EKG kaydi, 18,885 hasta).

### Genel Metrikler

| Metrik | Deger |
|:-------|:------|
| Macro AUROC | 0.8998 |
| Macro AUPRC | 0.7278 |
| Macro F1 | 0.6302 |
| Micro F1 | 0.6420 |
| Hamming Accuracy | %80.51 |

### Sinif Bazli Performans

| Sinif | Aciklama | AUROC | AUPRC | F1 | Destek |
|:------|:---------|:-----:|:-----:|:---:|:------:|
| MI | Miyokard Enfarktusu | 0.9022 | 0.7795 | 0.6933 | 550 |
| STTC | ST/T Degisikligi | 0.9193 | 0.7497 | 0.6638 | 506 |
| CD | Iletim Bozuklugu | 0.8923 | 0.7738 | 0.6794 | 496 |
| HYP | Hipertrofi | 0.8805 | 0.6201 | 0.4844 | 261 |

---

## Kurulum

### Gereksinimler

| Gereksinim | Minimum Surum |
|:-----------|:--------------|
| Python | 3.10+ |
| Node.js | 18+ |
| CUDA | Opsiyonel (GPU hizlandirma icin) |

### Adim 1: Depoyu Klonlama

```bash
git clone https://github.com/kullanici/CardioGuard-AI.git
cd CardioGuard-AI
```

### Adim 2: Python Ortami Kurulumu

```bash
# Sanal ortam olusturma
python -m venv .venv

# Sanal ortami aktiflestirme (Windows)
.venv\Scripts\activate

# Sanal ortami aktiflestirme (Linux/Mac)
source .venv/bin/activate

# Bagimliliklari yukleme
pip install -r requirements.txt
pip install fastapi uvicorn
```

### Adim 3: Frontend Kurulumu

```bash
cd frontend
npm install
cd ..
```

---

## Uygulamayi Calistirma

### Backend API Baslatma

```bash
uvicorn src.backend.main:app --host 0.0.0.0 --port 8000 --reload
```

Beklenen cikti:

```
Validating checkpoints...
Checkpoint validation passed!
Superclass model loaded
Localization model loaded
Binary MI model loaded (for consistency guard)
XGBoost feature schema loaded: 64 features
Models loaded: Superclass=OK, Localization=True, XGB=4
INFO: Uvicorn running on http://0.0.0.0:8000
```

### Frontend Baslatma

Yeni bir terminal acin:

```bash
cd frontend
npm run dev
```

Beklenen cikti:

```
VITE v6.2.0 ready in 500ms
Local: http://localhost:5173/
```

### Tarayicida Acma

Tarayicinizda `http://localhost:5173` adresine gidin ve EKG dosyasi yukleyin (.npy veya .npz formatinda).

---

## Proje Yapisi

```
CardioGuard-AI/
|
|-- src/                              # Python kaynak kodu
|   |
|   |-- backend/                      # FastAPI REST API
|   |   |-- main.py                   # API endpoint'leri (653 satir)
|   |   +-- __init__.py
|   |
|   |-- models/                       # Sinir agi tanimlari
|   |   |-- cnn.py                    # ECGBackbone, ECGCNN
|   |   +-- xgb.py                    # XGBoost yardimlari
|   |
|   |-- pipeline/                     # Is mantigi
|   |   |
|   |   |-- inference/                # Cikarim scriptleri
|   |   |   |-- run_inference_superclass.py    # Ana orkestrator (634 satir)
|   |   |   |-- run_inference_localization.py  # MI lokalizasyonu
|   |   |   |-- run_inference_binary.py        # Binary MI
|   |   |   +-- consistency_guard.py           # Tutarlilik kontrolu (ENTEGRE)
|   |   |
|   |   |-- training/                 # Egitim scriptleri
|   |   |   |-- train_superclass_cnn.py
|   |   |   |-- train_superclass_xgb_ovr.py
|   |   |   +-- train_mi_localization.py
|   |   |
|   |   +-- evaluation/               # Degerlendirme
|   |       +-- run_comprehensive_test.py
|   |
|   |-- xai/                          # Aciklanabilir AI modulleri
|   |   |-- gradcam.py                # Grad-CAM implementasyonu
|   |   |-- shap_ovr.py               # XGBoost icin SHAP
|   |   |-- unified.py                # Birlesik aciklayici
|   |   |-- sanity.py                 # XAI kalite kontrolu
|   |   +-- visualize.py              # Gorsellestirme fonksiyonlari
|   |
|   |-- contracts/                    # Veri kontratlari
|   |   +-- airesult_mapper.py        # Backend-Frontend donusumu
|   |
|   |-- data/                         # Veri yukleme
|   |   +-- splits.py                 # Hasta bazli split'ler
|   |
|   +-- utils/                        # Yardimci moduller
|       |-- model_loader.py           # Guvenli model yukleme
|       |-- checkpoint_validation.py  # Checkpoint dogrulama
|       +-- signal.py                 # Sinyal isleme
|
|-- frontend/                         # React 19 + Vite + TypeScript
|   |-- index.html
|   |-- index.tsx                     # React giris noktasi
|   |-- package.json
|   |-- vite.config.ts
|   |-- tsconfig.json
|   |
|   |-- components/                   # React bilesenleri
|   |
|   +-- lib/                          # Paylasilan yardilar
|       |-- api.ts                    # API istemcisi
|       +-- types.ts                  # TypeScript tip tanimlari
|
|-- checkpoints/                      # Egitilmis model agirliklari
|   |-- ecgcnn_superclass.pt          # Superclass CNN
|   |-- ecgcnn_localization.pt        # Lokalizasyon CNN
|   +-- ecgcnn.pt                     # Binary MI (Consistency Guard icin)
|
|-- logs/                             # Egitim ciktilari
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
|-- artifacts/                        # Konfigurasyonlar
|   +-- thresholds_superclass.json    # Sinif bazli esikler
|
|-- reports/                          # Raporlar ve XAI ciktilari
|   +-- xai/runs/                     # XAI artifact'lari
|       +-- {run_id}/
|           |-- manifest.json
|           |-- visuals/
|           +-- text/
|
|-- tests/                            # Pytest test suite
|   |-- test_consistency_guard.py
|   |-- test_consistency_integration.py
|   |-- test_airesult_mapper.py
|   |-- test_checkpoint_validation.py
|   |-- test_data.py
|   |-- test_gradcam.py
|   |-- test_xgb_pipeline.py
|   +-- (diger testler)
|
|-- docs/                             # Dokumantasyon
|   |-- 00_repo_map.md
|   |-- 01_architecture.md
|   |-- 02_api_contracts.md
|   |-- 03_inference_pipeline.md
|   |-- 04_xai_and_artifacts.md
|   |-- 05_frontend_integration.md
|   |-- MASTER_SOURCE_OF_TRUTH.md
|   +-- FINAL_TEKNIK_RAPOR_TR.md
|
|-- requirements.txt                  # Python bagimliliklari
|-- sample.npy                        # Ornek EKG sinyali
+-- test_mi_sample.npz                # MI pozitif test ornegi
```

---

## API Referansi

### Endpoint Listesi

| Metod | Yol | Aciklama |
|:------|:----|:---------|
| POST | /predict/superclass | Coklu etiket siniflandirma |
| POST | /predict/mi-localization | MI anatomik lokalizasyonu |
| GET | /runs/{run_id}/{file_path} | XAI artifact sunumu |
| GET | /health | Saglik kontrolu |
| GET | /ready | Hazirlik kontrolu |

### Superclass Tahmin Istegi

```bash
curl -X POST "http://localhost:8000/predict/superclass" \
  -F "file=@sample.npy" \
  -F "explain=true" \
  -F "ensemble_weight=0.5"
```

### Istek Parametreleri

| Parametre | Tip | Zorunlu | Varsayilan | Aciklama |
|:----------|:----|:-------:|:-----------|:---------|
| file | UploadFile | Evet | - | EKG dosyasi (.npy veya .npz) |
| explain | bool | Hayir | false | XAI artifact uretimi |
| ensemble_weight | float | Hayir | 0.5 | CNN/XGB agirligi (0.0-1.0) |
| sanity_check | bool | Hayir | false | XAI kalite kontrolu |

### Ornek Yanit

```json
{
  "mode": "multilabel-superclass",
  
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
    "rule": "MI-first-then-priority"
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
        "type": "report_png",
        "name": "sample_report.png",
        "url": "/runs/20260131_233000_abc123/visuals/sample_report.png",
        "mime": "image/png"
      },
      {
        "type": "narrative_md",
        "name": "sample__narrative.md",
        "url": "/runs/20260131_233000_abc123/text/sample__narrative.md",
        "mime": "text/markdown"
      }
    ],
    "sanity": {
      "status": "PASS",
      "passed_checks": 3,
      "total_checks": 3
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

---

## Consistency Guard

Sistem, Superclass MI ve Binary MI modelleri arasinda tutarlilik kontrolu yapar. Bu ozellik **tam entegre** durumundadir.

### Calisma Mantigi

1. Superclass modeli MI olasiligini hesaplar
2. Binary MI modeli bagimsiz olarak MI olasiligini hesaplar
3. Iki modelin kararlari karsilastirilir
4. Sonuc yanita eklenir

### Agreement Turleri

| Agreement Type | Durum | Triage | Aciklama |
|:---------------|:------|:-------|:---------|
| AGREE_MI | Her iki model MI tespit etti | HIGH | Yuksek guvenle MI |
| AGREE_NO_MI | Hicbiri MI tespit etmedi | LOW | Normal bulgu |
| DISAGREE_TYPE_1 | Superclass MI+, Binary MI- | REVIEW | Inceleme gerekli |
| DISAGREE_TYPE_2 | Superclass MI-, Binary MI+ | REVIEW | Inceleme gerekli |

### Kod Entegrasyonu

```
run_inference_superclass.py:
  Satir 32:  from src.pipeline.inference.consistency_guard import check_consistency
  Satir 279-290: check_consistency() cagrisi
  Satir 453: "consistency": consistency_result.to_dict()

main.py:
  Satir 207-214: Binary model yukleme
  Satir 523: binary_model=state.binary_model parametresi
```

---

## Teknik Detaylar

### CNN Mimarisi

```
Girdi: (batch, 12, T)
        |
        v
+------------------+
|  Conv1d          |  12 -> 64 kanal, kernel=7
+------------------+
        |
        v
+------------------+
|  BatchNorm1d     |  64 kanal
+------------------+
        |
        v
+------------------+
|  ReLU            |
+------------------+
        |
        v
+------------------+
|  Dropout         |  p=0.3
+------------------+
        |
        v
+------------------+
| ResidualBlock x4 |
+------------------+
        |
        v
+------------------+
| AdaptiveAvgPool  |
+------------------+
        |
        v
+------------------+
|  Flatten         |  -> (batch, 64) embedding
+------------------+
        |
        v
+------------------+
|  Linear(64, 4)   |  4 sinif ciktisi
+------------------+
        |
        v
+------------------+
|  Sigmoid         |  Multi-label olasilik
+------------------+
        |
        v
Cikti: MI, STTC, CD, HYP olasililklari
```

### Ensemble Formulu

```
ensemble_prob[sinif] = 0.5 * CNN_prob[sinif] + 0.5 * XGB_prob[sinif]
```

### NORM Sinifi Turetimi

NORM sinifi dogrudan tahmin edilmez, turetilir:

```
NORM_prob = 1.0 - max(MI_prob, STTC_prob, CD_prob, HYP_prob)
```

### Birincil Etiket Secimi (Priority Rule)

| Oncelik | Sinif | Gerekce |
|:-------:|:------|:--------|
| 1 | MI | Hayati tehdit eden acil durum |
| 2 | STTC | Iskemi belirtisi olabilir |
| 3 | CD | Iletim defekti, blok riski |
| 4 | HYP | Kronik durum |
| 5 | NORM | Patoloji tespit edilmedi |

---

## XAI Pipeline

### Bilesenler

| Bilesen | Dosya | Aciklama |
|:--------|:------|:---------|
| Grad-CAM | gradcam.py | Zamansal saliency haritalari |
| SHAP | shap_ovr.py | XGBoost ozellik katkilari |
| Unified Explainer | unified.py | Birlesik klinik anlati |
| Sanity Checker | sanity.py | XAI kalite kontrolu |
| Visualize | visualize.py | 12-lead plot, heatmap overlay |

### Sanity Check Kriterleri

| Kontrol | Esik | Anlam |
|:--------|:-----|:------|
| gradcam_variance | > 0.01 | Model belirli bolgelere odaklaniyor |
| peak_spread | > 0.1 | Derivasyonlar farkli agirlikta |

---

## Test

### Test Calistirma

```bash
# Tum testleri calistir
pytest tests/ -v

# Belirli bir testi calistir
pytest tests/test_consistency_guard.py -v

# Consistency integration testi
pytest tests/test_consistency_integration.py -v
```

### Test Kapsami

| Test Dosyasi | Kapsam |
|:-------------|:-------|
| test_consistency_guard.py | Tutarlilik kontrol mantigi (177 satir, 10 test) |
| test_consistency_integration.py | Pipeline entegrasyonu (4 test) |
| test_airesult_mapper.py | Yanit donusturme |
| test_checkpoint_validation.py | Model dogrulama |
| test_data.py | Veri yukleme, split'ler |
| test_gradcam.py | Grad-CAM uretimi |
| test_xgb_pipeline.py | XGBoost cikarimi |
| test_artifacts.py | XAI artifact uretimi |
| test_model.py | Model instantiation |

---

## Veri Seti

### PTB-XL Detaylari

| Ozellik | Deger |
|:--------|:------|
| Toplam Kayit | 21,837 |
| Hasta Sayisi | 18,885 |
| Ornekleme Frekansi | 100 Hz / 500 Hz |
| Kayit Suresi | 10 saniye |
| Derivasyon | 12-lead standart |

### Split Protokolu

| Split | Fold'lar | Oran | Ornek Sayisi |
|:------|:---------|:----:|:------------:|
| Train | 1-8 | %80 | 17,388 |
| Validation | 9 | %10 | 2,180 |
| Test | 10 | %10 | 2,180 |

### Veri Sizintisi Onleme

Hasta bazli split uygulanmistir. `verify_no_patient_leakage()` fonksiyonu ayni hastanin birden fazla split'te gorunmedigini dogrular.

---

## Dokumantasyon

Detayli dokumantasyon `docs/` klasorunde mevcuttur:

| Dosya | Icerik |
|:------|:-------|
| 00_repo_map.md | Depo yapisi ve kesif |
| 01_architecture.md | Sistem mimarisi (C4 model) |
| 02_api_contracts.md | API spesifikasyonlari |
| 03_inference_pipeline.md | Detayli pipeline analizi |
| 04_xai_and_artifacts.md | XAI implementasyon detaylari |
| 05_frontend_integration.md | Frontend-backend entegrasyonu |
| MASTER_SOURCE_OF_TRUTH.md | Tum sistem dokumantasyonu |
| FINAL_TEKNIK_RAPOR_TR.md | Kapsamli teknik rapor |

---

## Bagimliliklar

### Python Bagimliliklari

| Paket | Amac |
|:------|:-----|
| numpy | Sayisal hesaplamalar |
| pandas | Veri manipulasyonu |
| torch | PyTorch derin ogrenme |
| scikit-learn | ML yardimcilari, kalibrasyon |
| xgboost | Gradient boosting |
| wfdb | PhysioNet veri formati |
| shap | SHAP degerleri |
| matplotlib | Gorsellestirme |
| scipy | Bilimsel hesaplama |
| fastapi | REST API framework |
| uvicorn | ASGI sunucu |

### Node.js Bagimliliklari

| Paket | Surum | Amac |
|:------|:------|:-----|
| react | 19.2.4 | UI framework |
| react-dom | 19.2.4 | React DOM |
| vite | 6.2.0 | Build tool |
| typescript | 5.8.2 | Tip guvenligi |

---

## Yol Haritasi

| Surum | Donem | Hedef |
|:------|:------|:------|
| v1.1 | Tamamlandi | Consistency Guard entegrasyonu |
| v1.2 | Kisa Vade | Uzman onay arayuzu |
| v2.0 | Orta Vade | RAG entegrasyonu, belirsizlik tahmini |
| v2.0 | Orta Vade | LLM ile otomatik klinik rapor |
| v2.x | Uzun Vade | Gercek zamanli EKG streaming |
| v2.x | Uzun Vade | Kurumsal analitik dashboard |

---

## Tibbi Sorumluluk Reddi

**CardioGuard-AI yalnizca arastirma ve egitim amaclidir.**

Bu sistem klinik ortamlarda bagimsiz bir tani araci olarak kullanilmamalidir. Tum tahminler nitelikli saglik profesyonelleri tarafindan bagimsiz olarak dogrulanmalidir. Gelistiriciler, bu sistemin ciktilarina dayali olarak alinan klinik kararlar icin herhangi bir sorumluluk kabul etmez.

---

## Referanslar

| Kaynak | Aciklama |
|:-------|:---------|
| PTB-XL | PhysioNet uzerinde yayinlanan genis EKG veri seti |
| Grad-CAM | Selvaraju et al., "Visual Explanations from Deep Networks" |
| SHAP | Lundberg ve Lee, "A Unified Approach to Interpreting Model Predictions" |

---

## Lisans

Bu proje arastirma ve egitim kullanimi icin lisanslanmistir.

---

**CardioGuard-AI** - Yapay Zeka ve Klinik Karar Verme Arasinda Kopru

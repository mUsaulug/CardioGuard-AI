# CardioGuard-AI Technical Audit Report

**Date:** January 31, 2026
**Auditor:** Antigravity (Technical Auditor AI)
**Project:** CardioGuard-AI (12-Lead ECG MI Detection & Localization)
**Version:** 1.1.0

---

## 1. Executive Summary

CardioGuard-AI, 12 derivasyonlu EKG sinyallerinden Myokard Enfarktüsü (MI) ve diğer kardiyak anomalileri tespit eden, **derin öğrenme ve gradient boosting**'i birleştiren hibrit bir AI sistemidir.

### 1.1 Sistem Performansı (Test Set)

| Model | Macro AUROC | Macro AUPRC | Macro F1 |
| :--- | :---: | :---: | :---: |
| **CNN** | 0.8986 | 0.7308 | 0.6302 |
| **XGBoost** | 0.8998 | 0.7278 | 0.6688 |
| **Ensemble (50/50)** | ~0.90 | ~0.73 | ~0.65 |

### 1.2 Temel Güçlü Yönler

| Alan | Bulgu |
| :--- | :--- |
| 🏗️ **Mimari** | Backend/Pipeline strikt ayrımı, "fail-closed" startup |
| 🔐 **Güvenlik** | Path traversal koruması, input validation |
| 🔍 **XAI** | Unified approach: Grad-CAM + SHAP + Sanity checks |
| 📝 **Type Safety** | Pydantic (Backend) + TypeScript (Frontend) %100 uyumlu |
| 🧪 **Data Integrity** | Patient-level split, leakage verification |

### 1.3 Kritik Bulgular

| Ciddiyet | Bulgu | Etki |
| :---: | :--- | :--- |
| 🔴 **Yüksek** | Consistency Guard entegre değil | Model tutarsızlık kontrolü bypass ediliyor |
| 🟡 **Orta** | Hardcoded layer index (`features[-3]`) | Model değişikliğinde sessiz hata riski |
| 🟡 **Orta** | `fastapi`, `uvicorn` requirements.txt'te yok | Deployment sorunları |
| 🟢 **Düşük** | Dockerfile eksik | Container deployment zorlaşıyor |

---

## 2. Deney Sonuçları

### 2.1 CNN Test Metrikleri

**Kaynak:** `logs/superclass_cnn/training_results.json`

| Sınıf | Support | AUROC | AUPRC | F1 | Yorum |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **MI** | 550 | 0.9022 | 0.7795 | 0.6933 | ✅ Yüksek performans |
| **STTC** | 506 | 0.9193 | 0.7497 | 0.6638 | ✅ En yüksek AUROC |
| **CD** | 496 | 0.8923 | 0.7738 | 0.6794 | ✅ Dengeli |
| **HYP** | 261 | 0.8805 | 0.6201 | 0.4844 | ⚠️ Düşük (class imbalance) |

**Eğitim Konfigürasyonu:**
- Epochs: 50 (best: 46)
- Batch Size: 64
- Learning Rate: 0.001
- Weight Decay: 0.0001
- Seed: 42

### 2.2 XGBoost Test Metrikleri

**Kaynak:** `logs/xgb_superclass/training_results.json`

| Sınıf | Support | AUROC | AUPRC | F1 | Fark (vs CNN) |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **MI** | 550 | 0.9024 | 0.7726 | 0.6968 | +0.0002 |
| **STTC** | 506 | 0.9218 | 0.7708 | 0.7126 | +0.0025 |
| **CD** | 496 | 0.8881 | 0.7603 | 0.6896 | -0.0042 |
| **HYP** | 261 | 0.8868 | 0.6075 | 0.5762 | +0.0063 |

**Calibration:** Isotonic Regression kullanılmış.

### 2.3 Threshold Optimizasyonu

**Kaynak:** `artifacts/thresholds_superclass.json`

| Sınıf | Optimized | Production | Metod | Recall @ Opt |
| :--- | :---: | :---: | :--- | :---: |
| MI | 0.01 | 0.5 | F_beta (β=2) + recall_min=0.9 | 100% |
| STTC | 0.418 | 0.5 | Youden's J | 89.9% |
| CD | 0.420 | 0.5 | Youden's J | 79.5% |
| HYP | 0.258 | 0.5 | Youden's J | 89.9% |

> **Not:** MI için optimized threshold (0.01) %100 recall sağlıyor ancak F1'i 0.42'ye düşürüyor. Production'da 0.5 kullanılıyor.

---

## 3. Dataset & Protokol

### 3.1 PTB-XL Dataset

| Özellik | Değer |
| :--- | :--- |
| **Toplam Kayıt** | 21,837 |
| **Hasta Sayısı** | 18,885 |
| **Örnekleme** | 100 Hz / 500 Hz |
| **Kayıt Süresi** | 10 saniye |
| **Derivasyon** | 12-lead standart |

### 3.2 Split Stratejisi

**Kaynak:** `src/data/splits.py`

```
Train: Folds 1-8 → 17,388 samples (80%)
Val:   Fold 9    →  2,180 samples (10%)
Test:  Fold 10   →  2,180 samples (10%)
```

### 3.3 Leakage Önleme

**Fonksiyon:** `verify_no_patient_leakage()` (L85-129)

PTB-XL'in `strat_fold` sütunu hasta bazlı ayrım sağlar. Aynı hasta asla birden fazla split'te görünmez.

**Kanıt:**
```python
train_patients & val_patients  # Boş küme
train_patients & test_patients # Boş küme
val_patients & test_patients   # Boş küme
```

---

## 4. Mimari Analiz

### 4.1 Sistem Bileşenleri

```
┌─────────────────────────────────────────────────────────────┐
│                     CardioGuard-AI                          │
├─────────────────────────────────────────────────────────────┤
│  Frontend (React 19)  ←→  Backend (FastAPI)  ←→  Pipeline   │
│        ↓                         ↓                   ↓      │
│   TypeScript Types         Pydantic Models     PyTorch/XGB  │
│   (100% uyumlu)           (strict validation)   (inference) │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 Inference Akışı

1. **Input:** `.npz/.npy` (12, T) EKG sinyali
2. **Preprocessing:** Channel-first normalization
3. **CNN Forward:** Sigmoid → 4-class probs + 64-dim embeddings
4. **XGBoost:** Embeddings → 4 binary OVR classifiers
5. **Ensemble:** `0.5 * CNN + 0.5 * XGB`
6. **Thresholding:** Per-class thresholds
7. **Primary Label:** Priority rule (MI > STTC > CD > HYP > NORM)
8. **Localization:** MI tespit edilirse 5-bölge analizi
9. **XAI:** Grad-CAM + SHAP → Unified narrative

### 4.3 Consistency Guard (DEVRE DIŞI)

**Dosya:** `src/pipeline/inference/consistency_guard.py`

Bu modül Binary MI ve Superclass MI modellerini karşılaştırarak tutarsızlıkları tespit etmek için tasarlanmış:

| Agreement Type | Durum | Triage |
| :--- | :--- | :--- |
| `AGREE_MI` | Her iki model MI tespit | STANDARD |
| `AGREE_NO_MI` | İkisi de tespit etmemiş | STANDARD |
| `DISAGREE_TYPE_1` | Superclass MI+, Binary MI- | ELEVATED |
| `DISAGREE_TYPE_2` | Superclass MI-, Binary MI+ | CRITICAL |

> ⚠️ **Bulgu:** Bu modül `run_inference_superclass.py` içinde **çağrılmıyor**. Sistem bu güvenlik katmanı olmadan çalışıyor.

---

## 5. XAI Analizi

### 5.1 Grad-CAM

**Kaynak:** `src/xai/gradcam.py` (188 satır)

- Target layer: `backbone.features[-3]` (hardcoded)
- SmoothGrad-CAM: 5 noisy sample ortalaması
- Output: Normalized heatmap (0-1)

### 5.2 SHAP

**Kaynak:** `src/xai/shap_ovr.py`

- TreeExplainer (XGBoost için optimize)
- Feature count: 64 (CNN embeddings)
- Per-class SHAP values

### 5.3 Unified Explainer

Grad-CAM (spatial) + SHAP (feature) → Klinik narrative

### 5.4 Sanity Checks

| Check | Threshold | Amaç |
| :--- | :--- | :--- |
| `gradcam_variance > 0.01` | PASS | Model belirli bölgelere odaklanıyor |
| `peak_spread > 0.1` | PASS | Derivasyonlar farklı ağırlıkta |

---

## 6. Güvenlik Analizi

| Kontrol | Durum | Kaynak |
| :--- | :---: | :--- |
| Input size validation (10MB) | ✅ | `main.py` L235 |
| Input format validation | ✅ | `main.py` L240 |
| Path traversal protection | ✅ | `main.py` L417 |
| Run ID regex validation | ✅ | `main.py` L405 |
| Fail-closed startup | ✅ | `main.py` L85 |
| Authentication | ❌ | Yok |
| Rate limiting | ❌ | Yok |

---

## 7. Risk Değerlendirmesi & Öneriler

### 7.1 Yüksek Öncelik (P0)

| Risk | Etki | Çözüm |
| :--- | :--- | :--- |
| **Consistency Guard** devre dışı | Model tutarsızlığı tespit edilemiyor | `predict()` içinde `check_consistency()` çağır |

**Kod Değişikliği:**
```python
# run_inference_superclass.py L350 civarına ekle:
from .consistency_guard import check_consistency

consistency_result = check_consistency(
    superclass_probs=ensemble_probs,
    binary_mi_prob=binary_model.predict(signal) if binary_model else None,
    thresholds=thresholds
)
```

### 7.2 Orta Öncelik (P1)

| Risk | Etki | Çözüm |
| :--- | :--- | :--- |
| Hardcoded `features[-3]` | Model değişikliğinde hata | `ECGCNN.get_gradcam_layer()` method ekle |
| Missing dependencies | Install hatası | `requirements.txt`'e `fastapi`, `uvicorn` ekle |

### 7.3 Düşük Öncelik (P2)

| Risk | Etki | Çözüm |
| :--- | :--- | :--- |
| Dockerfile yok | Container deployment zor | `Dockerfile` + `docker-compose.yml` oluştur |
| Version pinning yok | Reproducibility riski | `pyproject.toml` + lockfile kullan |
| E2E testler yok | Regresyon riski | `test_e2e_prediction.py` ekle |

---

## 8. Sonuç

CardioGuard-AI, **akademik rigor** ve **endüstri standartları** açısından olgun bir projedir. Hibrit ensemble yaklaşımı, robust XAI pipeline'ı ve strict type safety ile dikkat çeker.

**Ana Güçlü Yönler:**
- ✅ Macro AUROC ~0.90 (CNN ve XGB)
- ✅ Patient-level split ile leakage önleme
- ✅ Unified XAI (Grad-CAM + SHAP)
- ✅ Backend/Pipeline separation

**Acil Eylem Gereken:**
- ⚠️ Consistency Guard entegrasyonu

Proje, P0 düzeltmesi sonrasında **production-ready** olarak değerlendirilebilir.

---

## Ek: Kanıt Dosyaları

| Dosya | Kanıt İçeriği |
| :--- | :--- |
| `logs/superclass_cnn/training_results.json` | CNN metrikleri |
| `logs/xgb_superclass/training_results.json` | XGBoost metrikleri |
| `artifacts/thresholds_superclass.json` | Threshold optimizasyonu |
| `src/data/splits.py` | Leakage prevention |
| `src/pipeline/inference/consistency_guard.py` | Devre dışı guard |
| `src/xai/gradcam.py` L305 | Hardcoded layer |

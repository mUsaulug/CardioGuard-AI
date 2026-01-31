# Phase 3: Inference Pipeline — Detaylı Analiz

**Generated Date:** 2026-01-31
**Ana Orchestrator:** `src/pipeline/inference/run_inference_superclass.py` (591 satır)
**Methodology:** Static code analysis + Training artifact inspection

---

## 1. Pipeline Genel Bakış

CardioGuard-AI inference pipeline'ı, iki farklı ML paradigmasını birleştiren hibrit bir yaklaşım kullanır:

1. **Derin Öğrenme (CNN):** Zaman serisi örüntülerini doğrudan sinyalden öğrenir.
2. **Gradient Boosting (XGBoost):** CNN'in ürettiği embedding'ler üzerinde çalışır.

Bu "late fusion" stratejisi, her iki yöntemin güçlü yanlarını birleştirir.

---

## 2. Model Mimarisi

### 2.1 CNN (Convolutional Neural Network)

**Dosya:** `src/models/cnn.py`
**Sınıf:** `ECGCNN`, `ECGBackbone`

```
Girdi: (batch, 12, T) — 12 kanal, T zaman adımı

ECGBackbone:
├── Conv1d(12, 64, kernel=7, padding=3)
├── BatchNorm1d(64)
├── ReLU
├── Dropout(0.2)
│
├── [ResidualBlock x 4]
│   ├── Conv1d(64, 64, kernel=3)
│   ├── BatchNorm1d
│   ├── ReLU
│   └── Skip Connection
│
├── AdaptiveAvgPool1d(1)  # Global Average Pooling
└── Flatten → (batch, 64)  # Embedding

Classification Head:
├── Linear(64, 4)  # 4 sınıf: MI, STTC, CD, HYP
└── Sigmoid        # Multi-label olasılık
```

**Konfigürasyon (Kanıt: `logs/superclass_cnn/training_results.json`):**
| Parametre | Değer |
| :--- | :--- |
| `epochs` | 50 |
| `batch_size` | 64 |
| `learning_rate` | 0.001 |
| `weight_decay` | 0.0001 |
| `best_epoch` | 46 |
| `pos_weight` | [2.98, 3.26, 3.46, 7.22] (class imbalance handling) |

### 2.2 XGBoost One-vs-Rest Ensemble

**Dosya:** `src/pipeline/training/train_superclass_xgb_ovr.py`
**Model Lokasyonu:** `logs/xgb_superclass/{MI,STTC,CD,HYP}/model.json`

Her sınıf için ayrı bir binary XGBoost classifier eğitilir:

```
Girdi: CNN Embedding (64 boyut)

XGBoost Pipeline (per class):
├── StandardScaler (logs/xgb_superclass/scaler.joblib)
├── XGBClassifier
│   ├── n_estimators=100
│   ├── max_depth=6
│   ├── scale_pos_weight (class-specific)
│   └── objective=binary:logistic
└── IsotonicRegression (calibrator.joblib)
    └── Platt-like probability calibration
```

**Feature Schema (Kanıt: `logs/xgb_superclass/feature_schema.json`):**
```json
{
    "version": "1.0.0",
    "feature_count": 64,
    "feature_names": ["cnn_feat_0", "cnn_feat_1", ..., "cnn_feat_63"],
    "embedder": {
        "type": "ECGCNN_backbone",
        "num_filters": 64,
        "source": "src/models/cnn.py:ECGBackbone"
    }
}
```

---

## 3. Deney Sonuçları (Experiment Results)

### 3.1 CNN Test Metrikleri

**Kaynak:** `logs/superclass_cnn/training_results.json`

| Metrik | Değer |
| :--- | :--- |
| **Macro AUROC** | **0.8986** |
| **Macro AUPRC** | **0.7308** |
| **Macro F1** | **0.6302** |
| **Micro F1** | 0.6420 |
| **Exact Match** | 44.92% |
| **Hamming Accuracy** | 80.51% |

**Per-Class Detayları:**

| Sınıf | Support | AUROC | AUPRC | F1 |
| :--- | :---: | :---: | :---: | :---: |
| **MI** | 550 | 0.9022 | 0.7795 | 0.6933 |
| **STTC** | 506 | 0.9193 | 0.7497 | 0.6638 |
| **CD** | 496 | 0.8923 | 0.7738 | 0.6794 |
| **HYP** | 261 | 0.8805 | 0.6201 | 0.4844 |

> **Yorum:** HYP (Hipertrofi) sınıfında düşük performans, düşük support (261 örnek) ve yüksek class imbalance (pos_weight=7.22) ile ilişkili.

### 3.2 XGBoost Test Metrikleri

**Kaynak:** `logs/xgb_superclass/training_results.json`

| Metrik | Değer |
| :--- | :--- |
| **Macro AUROC** | **0.8998** |
| **Macro AUPRC** | **0.7278** |

**Per-Class Detayları:**

| Sınıf | Support | AUROC | AUPRC | F1 |
| :--- | :---: | :---: | :---: | :---: |
| **MI** | 550 | 0.9024 | 0.7726 | 0.6968 |
| **STTC** | 506 | 0.9218 | 0.7708 | 0.7126 |
| **CD** | 496 | 0.8881 | 0.7603 | 0.6896 |
| **HYP** | 261 | 0.8868 | 0.6075 | 0.5762 |

### 3.3 CNN vs XGBoost Karşılaştırması

| Sınıf | CNN AUROC | XGB AUROC | Fark | Kazanan |
| :--- | :---: | :---: | :---: | :--- |
| MI | 0.9022 | 0.9024 | +0.0002 | XGB |
| STTC | 0.9193 | 0.9218 | +0.0025 | XGB |
| CD | 0.8923 | 0.8881 | -0.0042 | CNN |
| HYP | 0.8805 | 0.8868 | +0.0063 | XGB |
| **Macro** | 0.8986 | 0.8998 | +0.0012 | XGB |

> **Sonuç:** XGBoost, CNN embedding'leri üzerinde çalışarak marginal (~0.1%) iyileştirme sağlıyor. Ensemble (50/50) bu iki modelin güçlü yanlarını birleştiriyor.

---

## 4. Ensemble Mantığı

### 4.1 Ağırlıklı Ortalama

```python
# src/pipeline/inference/run_inference_superclass.py L295
ensemble_weight = 0.5  # Ayarlanabilir (API parametresi)

for cls in ["MI", "STTC", "CD", "HYP"]:
    ensemble_probs[cls] = (
        ensemble_weight * cnn_probs[cls] + 
        (1 - ensemble_weight) * xgb_probs[cls]
    )
```

**Kanıt:** `artifacts/thresholds_superclass.json` L44:
```json
"ensemble_weight": 0.5
```

### 4.2 NORM Sınıfı Türetimi

NORM (Normal) sınıfı, model tarafından doğrudan tahmin edilmez. Mantık:

```python
# Derived class
norm_prob = 1.0 - max(MI_prob, STTC_prob, CD_prob, HYP_prob)
```

Eğer hiçbir patoloji sınıfı threshold'u aşmazsa, sistem "NORM" döndürür.

---

## 5. Threshold Optimizasyonu

### 5.1 Optimizasyon Metodları

**Kaynak:** `artifacts/thresholds_superclass.json`

| Sınıf | Metod | Optimized Threshold | Üretim Threshold | Gerekçe |
| :--- | :--- | :---: | :---: | :--- |
| **MI** | F_beta (β=2.0) + recall_min=0.9 | 0.01 | 0.5 | High-recall: MI kaçırılmaması kritik |
| **STTC** | Youden's J | 0.418 | 0.5 | Sensitivite-Spesifisite dengesi |
| **CD** | Youden's J | 0.420 | 0.5 | Sensitivite-Spesifisite dengesi |
| **HYP** | Youden's J | 0.258 | 0.5 | Düşük prevalans için düşük threshold |

### 5.2 MI İçin F-Beta Optimizasyonu

MI sınıfı için özel strateji uygulanmış:

```json
"MI": {
    "threshold": 0.01,
    "method": "F_beta (beta=2.0) + recall_min=0.9",
    "score": 0.7828,
    "f1_at_threshold": 0.4193,
    "recall_at_threshold": 1.0,  // %100 recall!
    "support": 540
}
```

**Açıklama:**
- `beta=2.0`: F1 yerine F2 kullanılmış (recall'a 2x ağırlık).
- `recall_min=0.9`: Minimum %90 recall zorunlu.
- Sonuç: Threshold 0.01'e düşürülmüş, **%100 recall** elde edilmiş (ancak F1 0.42'ye düşmüş).

> ⚠️ **Üretim Notu:** Üretimde `thresholds.MI = 0.5` kullanılıyor. Optimized threshold (0.01) sadece referans için saklanıyor.

---

## 6. Veri Seti ve Split Protokolü

### 6.1 PTB-XL Veri Seti

| Özellik | Değer |
| :--- | :--- |
| **Toplam Kayıt** | 21,837 |
| **Hasta Sayısı** | 18,885 |
| **Örnekleme Frekansı** | 100 Hz / 500 Hz |
| **Kayıt Süresi** | 10 saniye |
| **Derivasyon** | 12-lead standart |

### 6.2 Split Stratejisi

**Kaynak:** `src/data/splits.py` L16-50

```python
def get_standard_split(df):
    """
    PTB-XL benchmark split:
    - Train: folds 1-8 (80%)
    - Validation: fold 9 (10%)
    - Test: fold 10 (10%)
    """
    train_folds = [1, 2, 3, 4, 5, 6, 7, 8]
    val_folds = [9]
    test_folds = [10]
```

### 6.3 Data Leakage Önleme

**Kritik Fonksiyon:** `verify_no_patient_leakage()` (`src/data/splits.py` L85-129)

```python
def verify_no_patient_leakage(df, train_idx, val_idx, test_idx):
    """
    Aynı hastanın birden fazla split'te görünmediğini doğrular.
    PTB-XL'in strat_fold sütunu bu garantiyi sağlar.
    """
    train_patients = set(df.loc[train_idx, "patient_id"])
    val_patients = set(df.loc[val_idx, "patient_id"])
    test_patients = set(df.loc[test_idx, "patient_id"])
    
    # Kesişim kontrolü
    if train_patients & val_patients:
        raise ValueError("Patient leakage between train and val!")
    if train_patients & test_patients:
        raise ValueError("Patient leakage between train and test!")
    if val_patients & test_patients:
        raise ValueError("Patient leakage between val and test!")
    
    return True
```

**Önem:** Tıbbi görüntüleme/sinyal verilerinde hasta bazlı split **zorunludur**. Aynı hastanın farklı kayıtları model tarafından "ezberlenebilir".

### 6.4 Veri Dağılımı

**Kaynak:** `logs/xgb_superclass/training_results.json`

| Split | Örnek Sayısı | MI+ | STTC+ | CD+ | HYP+ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Train | 17,388 | 4,369 | 4,078 | 3,900 | 2,115 |
| Val | 2,180 | 540 | 515 | 493 | 268 |
| Test | 2,180 | 550 | 506 | 496 | 261 |

---

## 7. Preprocessing Pipeline

### 7.1 Sinyal Normalizasyonu

**Kaynak:** `logs/superclass_cnn/normalization_stats.npz`

CNN eğitimi sırasında sinyal normalizasyonu uygulanmış:
- Per-channel mean subtraction
- Per-channel std division

### 7.2 Channel-First Dönüşümü

```python
# src/pipeline/inference/run_inference_superclass.py L209
def _ensure_channel_first(signal: np.ndarray) -> np.ndarray:
    """
    Sinyali (12, T) formatına dönüştürür.
    
    Desteklenen girdi formatları:
    - (12, T): Zaten doğru format
    - (T, 12): Transpose gerekli
    - (T,): Tek kanal, reshape gerekli
    """
    if signal.ndim == 1:
        signal = signal.reshape(1, -1)
    if signal.shape[0] == 12:
        return signal
    if signal.shape[1] == 12:
        return signal.T
    if signal.shape[0] > signal.shape[1]:
        return signal.T
    return signal
```

---

## 8. Primary Label Selection

### 8.1 Klinik Öncelik Kuralı

**Kaynak:** `src/pipeline/inference/run_inference_superclass.py` L42

```python
PRIORITY_ORDER = ["MI", "STTC", "CD", "HYP", "NORM"]

def get_primary_label(predicted_labels: List[str], probabilities: Dict) -> Dict:
    """
    Klinik önceliğe göre birincil etiketi seçer.
    
    MI en kritik bulgu olduğundan en yüksek önceliğe sahip.
    """
    for label in PRIORITY_ORDER:
        if label in predicted_labels or label == "NORM":
            return {
                "label": label,
                "confidence": probabilities.get(label, 0.0),
                "rule": "priority_order"
            }
```

### 8.2 Klinik Gerekçe

| Öncelik | Sınıf | Neden |
| :---: | :--- | :--- |
| 1 | **MI** | Hayatı tehdit eden acil durum. Kaçırılması fatal. |
| 2 | **STTC** | ST-T değişiklikleri, iskemi belirtisi olabilir. |
| 3 | **CD** | İletim defekti, blok riski. |
| 4 | **HYP** | Hipertrofi, kronik durum. |
| 5 | **NORM** | Varsayılan (patoloji yok). |

---

## 9. Localization Pipeline

### 9.1 Tetikleme Koşulu

```python
# run_inference_superclass.py L350
if localization_model is not None and "MI" in predicted_labels:
    localization_result = run_inference_localization.predict(
        signal=signal,
        model=localization_model,
        device=device,
        threshold=0.5,
        explain=explain,
        run_dir=run_dir
    )
```

### 9.2 Anatomik Bölgeler

| Kısaltma | Tam Ad | Açıklama |
| :--- | :--- | :--- |
| **AMI** | Anterior MI | Ön duvar enfarktüsü |
| **ASMI** | Anteroseptal MI | Ön-septal enfarktüs |
| **ALMI** | Anterolateral MI | Ön-lateral enfarktüs |
| **IMI** | Inferior MI | Alt duvar enfarktüsü |
| **LMI** | Lateral MI | Lateral duvar enfarktüsü |

**Model:** Aynı CNN mimarisi, 5-class multi-label çıktı.

---

## 10. Consistency Guard (DEVRE DIŞI)

### 10.1 Tasarım Amacı

`ConsistencyGuard` modülü, Binary MI modeli ile Superclass MI çıktısını karşılaştırarak tutarsızlıkları tespit etmek için tasarlanmış:

**Kaynak:** `src/pipeline/inference/consistency_guard.py`

```python
class AgreementType(Enum):
    AGREE_MI = "both_detect_mi"
    AGREE_NO_MI = "neither_detects_mi"
    DISAGREE_TYPE_1 = "superclass_mi_binary_no"  # Low confidence MI
    DISAGREE_TYPE_2 = "superclass_no_binary_mi"  # Missed by superclass
```

### 10.2 Mevcut Durum

> ⚠️ **KRİTİK BULGU:** Bu modül `run_inference_superclass.py` dosyasında **IMPORT EDİLMEMİŞ ve KULLANILMIYOR.**

**Doğrulama:**
```bash
grep -n "consistency" src/pipeline/inference/run_inference_superclass.py
# Sonuç: 0 eşleşme
```

**Etki:** Model tutarsızlık kontrolü devre dışı. Sistem bu güvenlik katmanı olmadan çalışıyor.

**Öneri:** `predict()` fonksiyonuna aşağıdaki ekleme yapılmalı:
```python
from .consistency_guard import check_consistency, should_run_localization

# After ensemble prediction
consistency_result = check_consistency(
    superclass_probs=ensemble_probs,
    binary_mi_prob=binary_mi_model.predict(signal) if binary_model else None,
    thresholds=thresholds
)
```

---

## 11. Özet: Pipeline Güçlü ve Zayıf Yönler

| Kategori | Güçlü Yön | Zayıf Yön / Risk |
| :--- | :--- | :--- |
| **Model Kalitesi** | Macro AUROC ~0.90 | HYP F1 düşük (0.48-0.58) |
| **Ensemble** | CNN+XGB complementary | Fixed weight (0.5) |
| **Threshold** | Per-class optimization | MI için aggressive (0.01 vs 0.5 üretim) |
| **Split** | Patient-level, leakage prevention | - |
| **Localization** | MI-gated triggering | - |
| **Safety** | Consistency Guard tasarlanmış | **Entegre DEĞİL** |

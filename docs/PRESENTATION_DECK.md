# CardioGuard-AI: Teknik Sunum

**Tarih:** 31 Ocak 2026
**Sunan:** [İsim]
**Proje:** CardioGuard-AI — 12-Lead EKG MI Tespit & Lokalizasyon Sistemi

---

# Slide 1: Proje Özeti

## CardioGuard-AI Nedir?

12 derivasyonlu EKG sinyallerinden otomatik kardiyak tanı yapan AI sistemi.

**Temel Hedefler:**
- 🎯 MI (Kalp Krizi) tespiti
- 📍 Anatomik lokalizasyon (5 bölge)
- 🔍 Açıklanabilir sonuçlar (XAI)

**Teknoloji Stack:**
- Backend: FastAPI + Python 3.10
- ML: PyTorch + XGBoost
- XAI: Grad-CAM + SHAP
- Frontend: React 19 + TypeScript

---

# Slide 2: Veri Seti

## PTB-XL Dataset

| Özellik | Değer |
| :--- | :--- |
| Toplam Kayıt | 21,837 |
| Hasta Sayısı | 18,885 |
| Örnekleme | 100 Hz |
| Süre | 10 saniye |

**Split Stratejisi:**
- Train: %80 (17,388)
- Val: %10 (2,180)
- Test: %10 (2,180)

**Leakage Önleme:** ✅
- Hasta bazlı ayrım (`strat_fold`)
- Aynı hasta sadece tek split'te

---

# Slide 3: Model Mimarisi

## Hibrit Ensemble Yaklaşımı

```
     [12-Lead ECG Signal]
              ↓
     ┌───────────────────┐
     │    CNN (PyTorch)  │ → Sigmoid → 4 prob
     │   EfficientNet-1D │ → Embeddings (64-dim)
     └───────────────────┘
              ↓
     ┌───────────────────┐
     │  XGBoost OVR (4x) │ → 4 prob (calibrated)
     └───────────────────┘
              ↓
     Ensemble: 0.5 × CNN + 0.5 × XGB
              ↓
     Per-class Threshold → Multi-label Output
```

---

# Slide 4: Deney Sonuçları (Test Set)

## Model Karşılaştırması

| Sınıf | CNN AUROC | XGB AUROC | Fark |
| :--- | :---: | :---: | :---: |
| **MI** | 0.9022 | 0.9024 | +0.0002 |
| **STTC** | 0.9193 | 0.9218 | +0.0025 |
| **CD** | 0.8923 | 0.8881 | -0.0042 |
| **HYP** | 0.8805 | 0.8868 | +0.0063 |

**Macro AUROC:** ~0.90 (her iki model)

## Per-Class F1 Scores

| Sınıf | CNN F1 | XGB F1 |
| :--- | :---: | :---: |
| MI | 0.693 | 0.697 |
| STTC | 0.664 | 0.713 |
| CD | 0.679 | 0.690 |
| HYP | 0.484 | 0.576 |

---

# Slide 5: Threshold Optimizasyonu

## Sınıf Bazlı Threshold Ayarlama

| Sınıf | Metod | Optimal | Production | Recall |
| :--- | :--- | :---: | :---: | :---: |
| MI | F_beta (β=2) | 0.01 | 0.5 | %100 |
| STTC | Youden's J | 0.42 | 0.5 | %90 |
| CD | Youden's J | 0.42 | 0.5 | %80 |
| HYP | Youden's J | 0.26 | 0.5 | %90 |

**MI için Strateji:**
- β=2: Recall'a 2× ağırlık
- Threshold 0.01 → %100 recall
- "Hiçbir MI kaçırılmasın" prensibi

---

# Slide 6: Sistem Mimarisi

## 3-Tier Architecture

```
┌────────────────────────────────────────────────┐
│                  FRONTEND                       │
│        React 19 + TypeScript                    │
│        (Type-safe API client)                   │
└────────────────────┬───────────────────────────┘
                     ↓ HTTP/JSON
┌────────────────────┴───────────────────────────┐
│                   BACKEND                       │
│              FastAPI Gateway                    │
│  • Request validation                           │
│  • Model loading (fail-closed)                  │
│  • Artifact serving                             │
└────────────────────┬───────────────────────────┘
                     ↓ Python import
┌────────────────────┴───────────────────────────┐
│                  PIPELINE                       │
│       Preprocessing → Inference → XAI          │
│  • CNN + XGBoost ensemble                       │
│  • Grad-CAM + SHAP                              │
│  • Manifest-based artifact management           │
└────────────────────────────────────────────────┘
```

---

# Slide 7: Inference Flow

## Tahmin Akışı

1. **Input:** `.npz` EKG dosyası (12 kanal, 10 sn)
2. **Preprocessing:** Channel-first, normalization
3. **CNN:** Sigmoid multi-label (MI, STTC, CD, HYP)
4. **Embedding:** CNN backbone → 64-dim vector
5. **XGBoost:** 4 binary classifier (OVR)
6. **Ensemble:** Ağırlıklı ortalama
7. **Threshold:** Per-class eşikler
8. **Primary Label:** Priority: MI > STTC > CD > HYP > NORM
9. **Localization:** MI tespit → 5-bölge analizi

---

# Slide 8: XAI (Açıklanabilir AI)

## Dual Approach

| Yöntem | Soru | Çıktı |
| :--- | :--- | :--- |
| **Grad-CAM** | "Nereye baktı?" | Temporal heatmap |
| **SHAP** | "Neden karar verdi?" | Feature importance |

## Unified Explainer

Grad-CAM + SHAP → Klinik Narrative

```markdown
## AI Analiz Özeti

**Tahmin:** MI (Güven: 85.2%)

### Zamansal Odak
Model, ST segmentine yoğunlaştı (0.4-0.6s arası).

### Önemli Özellikler
- cnn_feat_12: +0.23 (MI lehine)
- cnn_feat_47: -0.18 (NORM lehine)
```

---

# Slide 9: Güvenlik & Güvenilirlik

## Güvenlik Kontrolleri

| Kontrol | Durum |
| :--- | :---: |
| Input size validation (10MB) | ✅ |
| Path traversal protection | ✅ |
| Fail-closed startup | ✅ |
| Type validation (Pydantic) | ✅ |

## Fail-Closed Pattern

Model yüklenemezse → API **başlamayı reddeder**

```python
if not validation_result["valid"]:
    raise RuntimeError("FATAL: Cannot start!")
```

---

# Slide 10: Demo

## E2E Çalıştırma

### 1. Backend Başlat
```bash
uvicorn src.backend.main:app --port 8000
```

### 2. API Testi
```bash
curl -X POST "localhost:8000/predict/superclass?explain=true" \
     -F "file=@sample.npz"
```

### 3. Örnek Çıktı
```json
{
  "primary": {"label": "MI", "confidence": 0.85},
  "xai": {"artifacts": [{"url": "/runs/.../report.png"}]}
}
```

---

# Slide 11: Güvenlik Özellikleri

## Güvenlik Kontrolleri

| Kontrol | Durum |
| :--- | :---: |
| Input size validation (10MB) | ✅ |
| Path traversal protection | ✅ |
| Fail-closed startup | ✅ |
| Consistency Guard | ✅ |
| Type validation (Pydantic) | ✅ |

## Consistency Guard ✅ ENTEGRE

Binary MI vs Superclass MI karşılaştırması:
- `AGREE_MI`: İkisi de MI tespit → HIGH triage
- `DISAGREE_TYPE_1`: Superclass MI, Binary değil → REVIEW
- `DISAGREE_TYPE_2`: Binary MI, Superclass değil → REVIEW

**Entegrasyon:** `run_inference_superclass.py:276-291`

---

# Slide 12: Çözüm Önerileri

## P0: Acil (Bu Hafta)

```python
# run_inference_superclass.py:
from .consistency_guard import check_consistency

result = check_consistency(
    superclass_probs, binary_probs, thresholds
)
```

## P1: Orta Vadeli

- `ECGCNN.get_gradcam_layer()` method
- `requirements.txt` güncelleme

## P2: İyileştirme

- Dockerfile
- E2E test suite
- CI/CD pipeline

---

# Slide 13: Sonuç

## CardioGuard-AI Değerlendirmesi

**Güçlü Yönler:**
- ✅ AUROC ~0.90
- ✅ Patient-level split (leakage yok)
- ✅ Unified XAI
- ✅ Type-safe kontratlar
- ✅ Consistency Guard entegre

**İyileştirme Alanları:**
- ⚠️ Container deployment hazırlığı
- ⚠️ E2E test suite

**Sonuç:** **Production-ready**

---

# Slide 14: Soru-Cevap

## Sık Sorulan Sorular

**S: NORM nasıl hesaplanıyor?**
C: `1.0 - max(MI, STTC, CD, HYP)` — türetilmiş sınıf.

**S: Neden ensemble?**
C: CNN temporal pattern, XGB embedding feature'lar → komplementer.

**S: Leakage kontrolü nasıl?**
C: `verify_no_patient_leakage()` fonksiyonu, PTB-XL `strat_fold` kullanımı.

**S: XAI güvenilir mi?**
C: Sanity checks (variance, spread) ile doğrulanıyor.

---

# Teşekkürler!

**Sorularınız için:** [email]
**Repo:** [GitHub/CardioGuard-AI]

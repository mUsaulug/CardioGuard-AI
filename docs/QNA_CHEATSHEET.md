# CardioGuard-AI Q&A Cheatsheet

Teknik sunum ve savunma için hazırlanmış soru-cevap rehberi.

---

## 1. Mimari Sorular

### S: Backend ve Pipeline neden ayrı?

**C:** Separation of concerns prensibi. 

- **Backend:** HTTP handling, validation, static file serving. **HİÇBİR ML KODU YOKTUR.**
- **Pipeline:** Preprocessing, inference, XAI. **HİÇBİR HTTP KODU YOKTUR.**

Bu sayede:
1. Pipeline bağımsız test edilebilir (`pytest` ile)
2. Backend değişiklikleri ML'i etkilemez
3. "Fail-closed" güvenlik mümkün olur

**Kanıt:** `src/backend/main.py` sadece `pipeline.predict()` çağırır.

---

### S: Frontend backend logic'i duplicate ediyor mu?

**C:** Hayır, kesinlikle hayır.

- Frontend **sadece** visualization yapar
- Tüm threshold'lar, priority rule'ları **backend'de**
- Frontend TypeScript tipleri Backend Pydantic modellerinin **birebir kopyası**

**Kanıt:** `frontend/lib/types.ts` ↔ `src/backend/main.py` Pydantic models

---

### S: Sistem başlangıçta modeller yüklenemezse ne olur?

**C:** Sistem **başlamayı reddeder** (fail-closed).

```python
# startup_event()
if not validation_result["valid"]:
    raise RuntimeError("FATAL: Checkpoint validation failed!")
```

Bu sayede "yarı çalışan" sistem production'a çıkmaz.

**Kanıt:** `src/backend/main.py` L85-95

---

## 2. Inference Sorular

### S: NORM sınıfı nasıl hesaplanıyor?

**C:** NORM **türetilmiş** bir sınıftır, model doğrudan tahmin etmez.

```python
norm_prob = 1.0 - max(MI_prob, STTC_prob, CD_prob, HYP_prob)
```

Eğer hiçbir patoloji threshold'u aşmazsa → NORM döner.

**Kanıt:** `src/pipeline/inference/run_inference_superclass.py` (implicit logic)

---

### S: Neden hem CNN hem XGBoost kullanıyorsunuz?

**C:** **Complementary strengths** (tamamlayıcı güçler).

| Model | Güçlü Yön |
| :--- | :--- |
| CNN | Temporal pattern recognition, end-to-end learning |
| XGBoost | Tabular features üzerinde interpretable decisions |

XGBoost, CNN'in **embedding'lerini** kullanır (64-dim). Bu "neural feature extraction + classical ML" hibrit yaklaşımıdır.

**Sonuç:** Macro AUROC'ta marginal (~0.1%) ama consistent improvement.

---

### S: Ensemble weight neden 0.5?

**C:** Basit ama etkili.

- CNN AUROC: 0.8986
- XGB AUROC: 0.8998
- Fark çok küçük → eşit ağırlık makul

Alternatif: Validation set üzerinde weight optimization yapılabilir, ancak overfitting riski var.

**Kanıt:** `artifacts/thresholds_superclass.json` → `"ensemble_weight": 0.5`

---

### S: Primary label nasıl belirleniyor?

**C:** Klinik öncelik kuralı (priority rule):

```
MI > STTC > CD > HYP > NORM
```

- MI: Hayatı tehdit eden acil durum → en yüksek öncelik
- NORM: Default (patoloji yok) → en düşük öncelik

**Kanıt:** `run_inference_superclass.py` L42

---

### S: Localization ne zaman çalışır?

**C:** **Sadece MI tespit edildiğinde.**

```python
if "MI" in predicted_labels:
    run_localization(...)
```

Bu "gated execution" yaklaşımı gereksiz hesaplamayı önler.

**Kanıt:** `run_inference_superclass.py` L350

---

## 3. Veri & Eğitim Sorular

### S: Data leakage nasıl önleniyor?

**C:** **Patient-level split.**

PTB-XL'in `strat_fold` sütunu hasta bazlı ayrım sağlar:
- Aynı hasta **asla** birden fazla split'te görünmez
- Train/Val/Test tamamen disjoint

**Verification:**
```python
verify_no_patient_leakage(df, train_idx, val_idx, test_idx)
# ValueError fırlatır eğer leakage varsa
```

**Kanıt:** `src/data/splits.py` L85-129

---

### S: HYP performansı neden düşük?

**C:** **Class imbalance.**

| Sınıf | Support | pos_weight |
| :--- | :---: | :---: |
| MI | 550 | 2.98 |
| HYP | 261 | 7.22 |

HYP, diğer sınıfların yarısı kadar örneğe sahip. `pos_weight` ile dengelemeye çalışılmış ancak yeterli değil.

**Öneri:** 
- Over-sampling (SMOTE)
- Class-weighted loss function
- Data augmentation

---

### S: Threshold optimizasyonu nasıl yapıldı?

**C:** Sınıf bazlı farklı stratejiler:

| Sınıf | Metod | Neden |
| :--- | :--- | :--- |
| **MI** | F_beta (β=2) + recall_min=0.9 | Kaçırılması kritik, recall öncelikli |
| **Diğerleri** | Youden's J | Sensitivity-Specificity dengesi |

MI için optimized threshold 0.01 → **%100 recall**, ancak F1 düşük (0.42). Production'da 0.5 kullanılıyor.

**Kanıt:** `artifacts/thresholds_superclass.json`

---

## 4. XAI Sorular

### S: Grad-CAM neyi gösteriyor?

**C:** Modelin **zamansal odağını** gösterir.

- Hangi zaman dilimleri (ör: ST segment) tahmine katkıda bulundu?
- Hangi derivasyonlar daha önemli?

**Teknik:** Son konvolüsyon katmanının aktivasyonlarını, hedef sınıfa göre ağırlıklandırır.

---

### S: SHAP neyi gösteriyor?

**C:** **Feature attribution** — her özelliğin tahmini ne yönde etkilediği.

- Pozitif SHAP: Hedef sınıf lehine
- Negatif SHAP: Hedef sınıf aleyhine

XGBoost için TreeSHAP kullanılır (hızlı ve exact).

---

### S: Hardcoded `features[-3]` neden risk?

**C:** Model mimarisi değişirse yanlış katmanı hedefleyebilir.

Örneğin:
- EfficientNet → ResNet geçişi
- `features[-3]` artık Conv1d değil, ReLU olabilir

**Öneri:** `ECGCNN.get_gradcam_layer()` method encapsulation.

**Kanıt:** `run_inference_superclass.py` L305

---

### S: XAI güvenilir mi? Nasıl doğrulanıyor?

**C:** **Sanity checks** ile.

| Check | Threshold | Anlam |
| :--- | :--- | :--- |
| `gradcam_variance > 0.01` | PASS | Model belirli bölgelere odaklanıyor (flat değil) |
| `peak_spread > 0.1` | PASS | Derivasyonlar farklı ağırlıkta |

Eğer kontroller FAIL olursa → narrative'e WARNING eklenir.

**Kanıt:** `src/xai/sanity.py`

---

## 5. Güvenlik Sorular

### S: Path traversal nasıl engelleniyor?

**C:** İki katmanlı koruma:

1. **Regex validation:** `run_id` sadece `[a-zA-Z0-9_-]` kabul eder
2. **Path resolution:** `target.relative_to(base)` → ValueError fırlatır

```python
# Saldırı: /runs/../../../etc/passwd
# Sonuç: 400 "Invalid run_id format"
```

**Kanıt:** `main.py` L405-430

---

### S: Authentication var mı?

**C:** Şu anda **yok**. Production için eklenmeli:
- JWT token
- API key
- OAuth2

---

## 6. Kritik Bulgular

### S: Consistency Guard nedir ve neden sorunlu?

**C:** Binary MI model vs Superclass MI model karşılaştırması.

| Agreement | Durum | Aksiyon |
| :--- | :--- | :--- |
| `AGREE_MI` | İkisi de MI | STANDARD |
| `DISAGREE_TYPE_1` | Superclass MI+, Binary MI- | ELEVATED (inceleme) |
| `DISAGREE_TYPE_2` | Superclass MI-, Binary MI+ | CRITICAL (kaçırılmış olabilir) |

**Sorun:** Kod yazılmış, test edilmiş, ama `run_inference_superclass.py` içinde **çağrılmıyor**.

**Kanıt:** 
- Modül: `consistency_guard.py`
- Test: `test_consistency_guard.py` (PASS)
- Çağrı: YOK (`grep -n "consistency" run_inference_superclass.py` → 0 sonuç)

---

## 7. Genel Sorular

### S: Sistem production-ready mı?

**C:** **Neredeyse.**

- ✅ AUROC ~0.90
- ✅ Type-safe kontratlar
- ✅ Fail-closed startup
- ✅ Security controls
- ⚠️ Consistency Guard entegrasyonu gerekli
- ⚠️ Docker deployment hazırlanmalı

P0 düzeltmesi sonrası → **EVET, production-ready**.

---

### S: Sistem nasıl genişletilebilir?

**C:**

1. **Yeni sınıf ekleme:** 
   - CNN head'e output ekle
   - Yeni XGBoost classifier eğit
   - Threshold optimize et

2. **Yeni model:**
   - `src/models/` altına ekle
   - Pipeline'da load/predict fonksiyonları güncelle

3. **Yeni XAI yöntemi:**
   - `src/xai/` altına ekle
   - UnifiedExplainer'a entegre et

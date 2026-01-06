# CardioGuard-AI Teknik Denetim ve Sentez Raporu (Türkçe)

> **Not:** Bu rapor, depodaki gerçek artefaktlara (JSON/CSV/MD/PNG) ve kod tabanına dayanarak hazırlanmıştır. İstenilen “PROJECT_BINARY_ANALYSIS.md / PROJECT_LLM_PROMPT.md / PROJECT_MASTER_ANALYSIS.md / PROJECT_TECHNICAL_DEEP_DIVE.md” dosyaları depoda bulunamadığı için, bu raporda onların yerine mevcut kaynaklar kullanıldı. İlgili dosyaların sağlanması halinde rapor güncellenecektir.

---

## 1) Projenin Amacı ve Klinik Hedef
CardioGuard-AI, **çoklu etiketli (multi-label) ECG sınıflandırma** için klinik açıdan güvenilir bir sistemdir. Hedeflenen ana sınıflar:

- **MI (Myocardial Infarction)**
- **STTC (ST/T değişiklikleri)**
- **CD (Conduction Disturbance)**
- **HYP (Hypertrophy)**
- **NORM (Normal; türetilmiş etiket)**

Proje tasarımında üç temel klinik hedef öne çıkar:
1. **Yüksek hassasiyet (özellikle MI için)** – “MI-öncelikli” karar kuralı.
2. **Açıklanabilirlik (XAI) zorunluluğu** – Grad-CAM + SHAP + birleşik anlatı.
3. **Hibrit model yaklaşımı** – CNN (morfolojik) + XGBoost (istatistiksel) birlikte çalışır.

---

## 2) Kullanılan Veri, Veri Katmanı ve Dağılım Analizi
### 2.1 Veri Katmanı Doğrulama
`verification_output.txt` dosyası veri katmanı doğrulama çıktısını gösterir. İçerik, PTB-XL kayıtları için özet istatistikleri içerir:

- **Toplam kayıt:** 21,799 ECG
- **Hasta sayısı:** 18,869
- **SCP statement türü:** 71
- **Binary MI vs NORM dağılımı:**
  - Excluded: 6,818 (%31.3)
  - NORM: 9,513 (%43.6)
  - MI: 5,468 (%25.1)

Not: Dosya UTF-16/NULL byte içeriyor, bu nedenle metin görsel bozulma gösteriyor; ancak sayısal içerik okunabilir.

### 2.2 Sınıf Dağılımı ve Multi-label İstatistikleri
`reports/data_validation/class_distribution.json` ve `reports/superclass_labels/superclass_label_report.json`:

- **Train örnek sayısı:** 17,418
- **Ortalama etiket sayısı (train):** ~1.268
- **Birden fazla etiketli örnek:** 4,069
- **Sınıf pozitif oranları:**
  - MI: %25.14
  - STTC: %23.46
  - CD: %22.43
  - HYP: %12.17 (en dengesiz sınıf)
  - NORM: %41.70 (türetilmiş)

### 2.3 Etiket Birlikteliği (Co-occurrence)
`reports/data_validation/label_cooccurrence.csv` verisi:
- MI ve CD birlikteliği 1,794 örnek ile belirgin.
- STTC–HYP birlikteliği 1,485.
- NORM ile patoloji birlikte çok düşük (NORM, normal etiket olarak türetilmiş).

Bu birliktelik matrisi, multi-label eğitimin klinik gerçekliğe yakın olduğunu gösterir.

---

## 3) Mimari Genel Bakış

### 3.1 CNN Omurga (Backbone)
**Dosya:** `src/models/cnn.py`

- **ECGCNNConfig:** 12 kanallı giriş, 64 filtre, kernel=7, dropout=0.3
- **ECGBackbone:** 2 katmanlı Conv1D + BatchNorm + ReLU + Dropout + AdaptiveAvgPool1d
- **BinaryHead / MultiClassHead / MultiTaskECGCNN** yapıları mevcut.

Bu omurga, 12 derivasyonlu 10s ECG sinyalinden 64 boyutlu embedding çıkarır.

### 3.2 Multi-label CNN (Superclass)
**Dosya:** `src/pipeline/train_superclass_cnn.py` ve `src/pipeline/run_inference_superclass.py`

- CNN modeli 4 ana etiketi (MI, STTC, CD, HYP) tahmin eder.
- NORM etiketi türetilir: **1 - max(pathology_prob)**.
- Eğitim sırasında sınıf ağırlıkları kullanılır (`pos_weight`).

### 3.3 XGBoost OVR (One-vs-Rest)
**Dosya:** `src/pipeline/train_superclass_xgb_ovr.py`

- 4 ayrı binary XGB modeli eğitilir.
- **Platt scaling (sigmoid)** veya **isotonic** kalibrasyon uygulanabilir.
- CNN embedding’leri özellik olarak kullanılır.

### 3.4 Ensemble (Hibrit Sistem)
**Dosya:** `src/pipeline/run_inference_superclass.py`

- CNN olasılıkları + XGB olasılıkları **ağırlıklı ortalama** ile birleştirilir.
- `ensemble_weight` parametresi CNN ağırlığını belirler.

### 3.5 MI-Öncelikli Karar Kuralı
**Fonksiyon:** `get_primary_label` (`run_inference_superclass.py`)

- MI olasılığı eşik üzerindeyse **doğrudan MI** döner.
- Diğer patolojiler (STTC, CD, HYP) sırayla kontrol edilir.
- Hiçbiri eşik aşmıyorsa NORM döner.

Bu kural güvenlik odaklı (yüksek duyarlılık) tasarlanmıştır.

---

## 4) Eğitim, Doğrulama ve Test Performansı

### 4.1 CNN (Binary Pipeline)
**Dosya:** `logs/cnn/metrics.json`

- **Test ROC-AUC:** 0.9724
- **Test PR-AUC:** 0.9447
- **Test F1:** 0.8419
- **Test Accuracy:** 0.9282

Eğitim sırasında ROC-AUC 0.98+ seviyelerine ulaşıyor.

### 4.2 Superclass CNN
**Dosya:** `logs/superclass_cnn/training_results.json`

- **Macro AUROC:** 0.8986
- **Macro AUPRC:** 0.7308
- **Macro F1:** 0.6302
- **Micro F1:** 0.6420
- **Exact match:** 0.4492
- **Hamming accuracy:** 0.8051

Per-class F1:
- MI: 0.6933
- STTC: 0.6638
- CD: 0.6794
- HYP: 0.4844 (en zayıf sınıf)

### 4.3 XGBoost (Binary)
**Dosya:** `logs/xgb/metrics.json`

- **Validation:** ROC-AUC 0.9839, PR-AUC 0.9652, F1 0.9055
- **Test:** ROC-AUC 0.9763, PR-AUC 0.9497, F1 0.8664
- **Best threshold (val):** 0.80

XGB, CNN’e göre F1’de daha güçlü görünmektedir.

### 4.4 Ensemble Karşılaştırma
**Dosya:** `reports/comparison_report.md`

| Model | AUC | PR_AUC | F1 | Accuracy |
|------|-----|--------|----|----------|
| CNN | 0.9380 | 0.9215 | 0.7436 | 0.8438 |
| XGBoost | 0.9408 | 0.9254 | 0.7850 | 0.8629 |
| Ensemble (α=0.5) | 0.9414 | 0.9262 | 0.7941 | 0.8670 |
| Ensemble (α=0.15) | **0.9420** | **0.9268** | **0.8132** | **0.8765** |

**En iyi α:** 0.15 (`reports/ensemble_config.json`)

---

## 5) XAI (Explainable AI) Sistemi

### 5.1 Grad-CAM
**Dosya:** `src/xai/gradcam.py`

- CNN aktivasyonlarına dayanır.
- Çıktı olarak heatmap üretir.
- `reports/xai/gradcam_sample_*.png` çıktıları mevcut.

### 5.2 SHAP
**Dosya:** `src/xai/shap_ovr.py` ve `src/xai/shap_xgb.py`

- Embedding özellikleri üzerinde SHAP değerleri üretilir.
- `reports/xai/shap_waterfall_sample_*.png` ve `shap_summary.png` üretilmiş.

### 5.3 Unified Explainer
**Dosya:** `src/xai/unified.py`

- Grad-CAM ve SHAP çıktısını tek bir klinik anlatıda birleştirir.
- Heuristik “coherence score” (varsayılan 0.85) ile tutarlılık değerlendirmesi yapar.

### 5.4 Sanity Checker
**Dosya:** `src/xai/sanity.py`

- Randomizasyon/pertürbasyon kontrolleri ile açıklamaların tutarlılığı ölçülür.

**XAI çıktıları:** `reports/xai/` klasöründe 4 örnek ECG ve açıklama görselleri mevcuttur.

---

## 6) MI Lokalizasyonu (Anatomik Bölge Tahmini)

**Kod yolu:** `src/pipeline/run_inference_binary.py`

- Model checkpoint’inde `localization_head` varsa `MultiTaskECGCNN` kullanılır.
- Localization çıktısı sigmoidden geçirilir ve başlangıç/bitiş indeksleri üretilir (`decode_localization_bounds`).
- `plot_ecg_with_localization` ile görselleştirilir.

> **Not:** Lokalizasyon yalnızca MI varlığında otomatik sınırlandırılmıyor. Modelde localization head varsa her örnek için lokalizasyon üretilebilir. Klinik güvenlik açısından MI pozitiflik kuralı ile koşullandırmak daha güvenli olabilir.

---

## 7) Inference Akışı (Özellikle Superclass)

**Dosya:** `src/pipeline/run_inference_superclass.py`

1. ECG sinyali yüklenir (`load_ecg_signal`).
2. `ensure_channel_first` ile (12, T) formatına zorlanır.
3. CNN ile logits → sigmoid → probs.
4. XGB OVR modelleri varsa embedding çıkarılır, scaler uygulanır ve kalibre edilir.
5. CNN + XGB ensemble hesaplanır.
6. Eşiklere göre multi-label tahmin çıkarılır.
7. MI-öncelikli karar kuralı ile “primary label” seçilir.

---

## 8) Eşikler, Kalibrasyon ve Karar Stratejileri

### 8.1 Eşikler
- **Binary XGB best threshold:** 0.80 (`logs/xgb/metrics.json`)
- **Superclass inference varsayılan eşik:** 0.5 (thresholds dosyası yoksa)

### 8.2 Kalibrasyon
- XGB OVR için **Platt scaling** (sigmoid) uygulanıyor.
- `CalibratedOVRModel` sınıfı ile olasılıklar dengeleniyor.

### 8.3 MI-First Rule
- Klinik güvenlik odaklı.
- False positive artışı pahasına hassasiyet artırılıyor.

---

## 9) Klinik Değer ve Uygulama Perspektifi

- **MI tespiti** klinik olarak kritik: düşük FNR öncelikli.
- **Grad-CAM** sayesinde kardiyologlar hangi derivasyonların ve zaman aralığının etkili olduğunu görebilir.
- **SHAP** sayesinde embedding bazlı hangi istatistiksel özelliklerin etkili olduğu görülebilir.
- **Unified Explainer**, görsel ve istatistiksel açıklamayı “tıbbi anlatı” formatına çevirerek klinik kullanılabilirlik sağlar.

---

# 10) Bileşen Bazlı Denetim (İstenen Format)

Aşağıda kritik bileşenler için **Durum / Risk / Öneri / Klinik Değer** formatında değerlendirme yapılmıştır.

---

## 10.1 `ensure_channel_first` (Sinyal Şekil Kontrolü)
**Kod:** `src/pipeline/run_inference_superclass.py` ve `run_inference_binary.py`

**Durum:** **At Risk**

**Logical Flaws:**
- Heuristik yaklaşım; sadece `shape[0] == 12` veya `shape[1] == 12` kontrolü var.
- Özel durumlarda (ör. 2 kanallı, farklı sample sayıları) yanlış transpose riski.

**Optimization Tips:**
- Net veri sözleşmesi (shape=12x1000) zorunluluğu eklenmeli.
- Meta-data üzerinden lead dimension doğrulaması yapılmalı.

**Clinical Value:**
- Doğru kanal sırası olmadan Grad-CAM ve tahminler klinik olarak yanlış yönlendirici olabilir.

---

## 10.2 XGB Kalibrasyon Akışı
**Kod:** `train_superclass_xgb_ovr.py`, `run_inference_superclass.py`

**Durum:** **Robust**

**Logical Flaws:**
- `run_inference_superclass.py` yalnızca `scaler.joblib` ve `calibrator.joblib` dosyalarını arıyor; dosya isim sözleşmesi uymazsa kalibrasyon devre dışı kalabilir.

**Optimization Tips:**
- Model artefaktlarını merkezi config/manifest ile bağlamak.
- Olasılık dağılımını CNN ile hizalayan ek kalibrasyon (temperature scaling) eklenebilir.

**Clinical Value:**
- Kalibrasyon, klinisyenin olasılık değerlerini güvenle yorumlamasını sağlar.

---

## 10.3 MI Lokalizasyon Güvenliği
**Kod:** `run_inference_binary.py`

**Durum:** **Refactor Recommended**

**Logical Flaws:**
- Localization her örnek için üretilebilir; MI yoksa da lokalizasyon çıkabilir.
- Klinik anlatıda hatalı lokalizasyon riski vardır.

**Optimization Tips:**
- Lokalizasyon sadece MI eşiği geçildiğinde çalıştırılmalı.
- “MI yok → localization None” kuralı netleştirilmeli.

**Clinical Value:**
- MI lokalizasyonu, kardiyoloğun ilgili anatomik bölgeyi hızla incelemesine yardımcı olur.

---

## 10.4 UnifiedExplainer (XAI Bütünleştirici)
**Kod:** `src/xai/unified.py`

**Durum:** **At Risk**

**Logical Flaws:**
- Coherence score sabit ve heuristik (0.85). Gerçek veri ilişkisi yok.
- SHAP ve Grad-CAM çıktıları arasında gerçek ilişki kurulmamış (placeholder).

**Optimization Tips:**
- Grad-CAM lead ve SHAP feature eşleşmesi için explicit mapping.
- Coherence score’u gerçek ölçütle hesaplayan fonksiyon ekleme.

**Clinical Value:**
- Klinik rapor anlatısının güvenilirliği ve ikna ediciliği artar.

---

## 10.5 MI-First Kuralının Performans Etkisi
**Kod:** `run_inference_superclass.py`

**Durum:** **At Risk**

**Logical Flaws:**
- MI olasılığı eşik üzerindeyse diğer patolojiler göz ardı ediliyor.
- Çoklu patoloji durumlarında MI baskınlığı hatalı klinik yorum yaratabilir.

**Optimization Tips:**
- Klinik özet + multi-label rapor birlikte sunulmalı.
- MI önceliği “primary label” için kalsın ama diğer etiketler de raporda açıkça vurgulanmalı.

**Clinical Value:**
- Hasta güvenliğini artırır; fakat gereksiz MI alarmı (FP) riskini yükseltir.

---

# 11) Sonuç ve Genel Değerlendirme

CardioGuard-AI, **hibrit (CNN + XGBoost) ve açıklanabilir** bir ECG tanı sistemi olarak güçlü bir temel sunar. Özellikle:

- **Ensemble yaklaşımı** tek modele göre daha iyi performans vermektedir.
- **XAI pipeline** aktif olarak çalışmakta ve görsel açıklamalar üretmektedir.
- **MI önceliği** güvenlik odaklıdır ancak FP riski barındırır.

Geliştirme için kritik öneriler:
1. **Lokalizasyon yalnızca MI pozitiflik koşulunda çalışmalı.**
2. **XAI tutarlılığı için gerçek “coherence” ölçümü eklenmeli.**
3. **Eşikler açıkça versiyonlanmalı ve inference’de net şekilde kullanılmalı.**

---

## 12) Kullanılan Artefakt Listesi

- `logs/cnn/metrics.json`
- `logs/superclass_cnn/training_results.json`
- `logs/xgb/metrics.json`
- `reports/comparison_report.md`
- `reports/ensemble_config.json`
- `reports/data_validation/class_distribution.json`
- `reports/data_validation/label_cooccurrence.csv`
- `reports/superclass_labels/superclass_label_report.json`
- `reports/xai/*.png`
- `verification_output.txt`
- `BINARY_PIPELINE_STATUS.md`

---

## 13) Ek Notlar

- Proje, backend entegrasyonuna hazır “artifact sözleşmesi” sunuyor (`BINARY_PIPELINE_STATUS.md`).
- `run_xai_demo.py` ve ilgili XAI araçları, klinik raporlamayı destekleyecek temel görselleri üretiyor.

---

**Bu rapor, akademik makale formatına dönüştürülmeye uygun, kapsamlı bir teknik denetim ve sentez dokümanıdır.**

# Proje Klasör Haritası (File Locations)

Bu dosya, CardioGuard-AI projesindeki örnek veri, XAI çıktıları, log ve görsel dizinlerinin haritasını sunar.

---

## 1. Test ve Örnek Veri Dizinleri

### 1.1 XAI Test Örnekleri
- **Amaç:** XAI pipeline testi için hazır EKG örnekleri
- **Klasör:** `reports/xai/test_samples/`
- **İçerik:** 10 adet .npz dosyası (sample_000_normal.npz - sample_009_hyp.npz)
- **Üretim:** Veri setinden manuel seçilmiş, farklı patolojileri temsil eden örnekler

### 1.2 Kök Dizin Test Dosyası
- **Amaç:** Hızlı API/pipeline testi için tek örnek
- **Klasör:** Proje kökü (`/`)
- **Dosya:** `sample.npy`
- **İçerik:** 12x1000 boyutlu EKG sinyali

### 1.3 Özellik Çıktıları
- **Amaç:** CNN'den çıkarılan embedding özellikleri
- **Klasör:** `features_out/`
- **İçerik:** train.npz, val.npz, test.npz (her split için özellikler)
- **Üretim:** Eğitim pipeline'ı tarafından oluşturulur

### 1.4 Tahmin Çıktıları
- **Amaç:** Model tahminlerinin ara çıktıları
- **Klasör:** `predictions/`
- **İçerik:** val_cnn_probs.npz, val_xgb_probs.npz, val_labels.npz
- **Üretim:** Değerlendirme script'leri tarafından oluşturulur

---

## 2. XAI Çıktı Dizinleri

### 2.1 XAI Run Klasörleri
- **Amaç:** Her XAI çalışması için ayrı dizin
- **Klasör:** `reports/xai/runs/`
- **Alt Dizin Formatı:** `{YYYYMMDD}_{HHMMSS}__{git_hash}__{model_type}__{task_type}/`
- **Örnek:** `20260106_085756__6eb3716__ecgcnn__multiclass/`
- **Üretim:** XAI pipeline çalıştırıldığında otomatik oluşturulur

### 2.2 XAI Alt Dizin Yapısı
Her run klasörü içinde:

| Alt Dizin | İçerik | Format |
|:----------|:-------|:-------|
| `visuals/` | Grad-CAM heatmap ve özet görseller | PNG |
| `text/` | Narrative açıklama dosyaları | Markdown |
| `tensors/` | Sinyal ve aktivasyon verileri | NPZ |
| `tables/` | Tablo formatında çıktılar | CSV/JSON |

### 2.3 Manifest Dosyası
- **Amaç:** Run içeriğinin merkezi indeksi
- **Dosya:** `manifest.json` (her run klasöründe)
- **İçerik:** run_id, oluşturulma tarihi, artifact listesi, sanity check durumu

---

## 3. Model Checkpoint Dizinleri

### 3.1 Ana Checkpoint'ler
- **Amaç:** Eğitilmiş model ağırlıkları
- **Klasör:** `checkpoints/`
- **Dosyalar:**
  - `ecgcnn.pt` — Binary MI model
  - `final_superclass_model.pt` — 4 sınıflı superclass model
  - `final_mi_localization_model.pt` — 5 bölgeli lokalizasyon modeli
- **Üretim:** Eğitim script'leri tarafından kaydedilir

### 3.2 XGBoost Modelleri
- **Amaç:** Ensemble için XGBoost classifier'lar
- **Klasör:** `artifacts/xgb_ovr/`
- **Dosyalar:** Her sınıf için ayrı .joblib dosyası + calibrator'lar
- **Üretim:** XGBoost eğitim pipeline'ı tarafından

---

## 4. Log ve Metrik Dizinleri

### 4.1 Eğitim Logları
- **Amaç:** Eğitim sürecinin detaylı kayıtları
- **Klasör:** `logs/`
- **Alt Dizinler:**
  - `logs/superclass_cnn/` — Superclass CNN eğitim metrikleri
  - `logs/mi_localization/` — Lokalizasyon eğitim metrikleri
- **İçerik:** training_results.json, epoch_metrics.csv, normalization_stats.npz

### 4.2 Konfigürasyon Artifact'ları
- **Amaç:** Üretim ortamı yapılandırmaları
- **Klasör:** `artifacts/`
- **Dosyalar:**
  - `thresholds_superclass.json` — Sınıf threshold'ları
  - `feature_schema.json` — Özellik şeması
- **Üretim:** Threshold optimizasyon script'leri tarafından

---

## 5. Doğrulama Kanıt Dizini

### 5.1 Evidence Klasörü
- **Amaç:** Sistem doğrulama kanıtları
- **Klasör:** `docs/evidence/20260131_7fb4b83/`
- **İçerik:**
  - `env.txt` — Ortam bilgileri
  - `backend_start.log` — Backend başlatma logu
  - `frontend_start.log` — Frontend başlatma logu
  - `pytest.log` — Test sonuçları
  - `e2e_run.log` — E2E test adımları
  - `artifacts_snapshot/` — XAI örnek çıktıları
  - `screenshots/` — UI ekran görüntüleri (manuel eklenmeli)

---

## 6. Dokümantasyon Dizini

### 6.1 Ana Dokümanlar
- **Klasör:** `docs/`
- **Ana Dosyalar:**
  - `MASTER_SOURCE_OF_TRUTH.md` — Teknik referans dokümanı
  - `FINAL_TEKNIK_RAPOR_TR.md` — Türkçe teknik rapor
  - `PRESENTATION_DECK.md` — Sunum içeriği
  - `QNA_CHEATSHEET.md` — Soru-cevap rehberi

---

## Özet Tablo

| Kategori | Klasör | Dosya Sayısı | Boyut |
|:---------|:-------|:-------------|:------|
| Test Örnekleri | reports/xai/test_samples/ | 10 | ~20MB |
| XAI Runs | reports/xai/runs/ | 23 dizin | ~50MB |
| Checkpoints | checkpoints/ | 3 | ~1MB |
| XGBoost | artifacts/xgb_ovr/ | ~10 | ~5MB |
| Loglar | logs/ | ~5 dizin | ~2MB |
| Evidence | docs/evidence/ | 1 dizin | ~10MB |

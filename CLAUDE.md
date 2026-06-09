# CardioGuard-AI - Claude Code Project Context

## Proje Ozeti
12 derivasyonlu EKG sinyallerinden kardiyak patoloji tespiti yapan Aciklanabilir AI platformu.
PTB-XL veri seti (21,837 kayit, 18,885 hasta) uzerinde egitilmis.

## Teknoloji Yigini
- **Backend:** Python 3.10+, FastAPI, Pydantic
- **ML:** PyTorch 2.x (CNN), XGBoost (OVR), SHAP, Grad-CAM
- **Frontend:** React 19, TypeScript 5.8, Vite 6.2
- **Veri:** PTB-XL (PhysioNet), 100Hz/500Hz, 10s kayitlar

## Calistirma
```bash
# Backend
pip install -r requirements.txt && pip install fastapi uvicorn
uvicorn src.backend.main:app --host 0.0.0.0 --port 8000 --reload

# Frontend
cd frontend && npm install && npm run dev
```

## Mimari Kurallar
1. **Backend (main.py) HICBIR ML kodu icermez** - sadece pipeline.predict() cagirir
2. **Pipeline (run_inference_superclass.py) tek inference kaynagi** - tum ML mantigi burada
3. **XAI artifact'lari pipeline uretir**, backend sadece manifest.json okur ve serve eder
4. **Fail-closed:** Model validation basarisiz olursa uygulama baslamaz
5. **NORM sinifi turetilir:** `1 - max(MI, STTC, CD, HYP)` - dogrudan tahmin edilmez

## Aktif Akis (Production Path)

### API Istegi Akisi
```
Frontend -> POST /predict/superclass -> main.py -> pipeline_predict() -> JSON Response
Frontend -> POST /predict/mi-localization -> main.py -> pipeline_predict_localization() -> JSON Response
```

### Aktif Dosyalar (Calisir Durumda)
```
BACKEND:
  src/backend/main.py                    # API gateway (653 satir)

INFERENCE PIPELINE (ANA AKIS):
  src/pipeline/inference/run_inference_superclass.py  # Ana orkestrator (634 satir)
  src/pipeline/inference/run_inference_localization.py # MI lokalizasyon
  src/pipeline/inference/consistency_guard.py          # Binary vs Superclass MI karsilastirma

MODELLER:
  src/models/cnn.py                      # ECGBackbone, ECGCNN, head'ler

XAI:
  src/xai/gradcam.py                     # Grad-CAM + SmoothGrad-CAM
  src/xai/shap_ovr.py                    # XGBoost SHAP (OVR)
  src/xai/unified.py                     # Birlesik aciklama sentezi
  src/xai/sanity.py                      # XAI kalite kontrol (Adebayo et al.)
  src/xai/visualize.py                   # Plot fonksiyonlari
  src/xai/reporting.py                   # XAIReporter, manifest olusturma

CONTRACTS:
  src/contracts/airesult_mapper.py       # predict() -> AIResult v1.0 donusumu
  src/contracts/artifacts.py             # XAI artifact discovery

UTILS:
  src/utils/model_loader.py             # Schema-aware model yukleme
  src/utils/checkpoint_validation.py    # Checkpoint dogrulama

CONFIG:
  src/config.py                          # PTBXLConfig, MI_CODES, sinif tanimlari
  artifacts/thresholds_superclass.json   # Sinif bazli esikler

CHECKPOINTS:
  checkpoints/ecgcnn_superclass.pt       # 4 sinif CNN
  checkpoints/ecgcnn_localization.pt     # 5 bolge lokalizasyon CNN
  checkpoints/ecgcnn.pt                  # Binary MI (Consistency Guard)

FRONTEND:
  frontend/index.tsx                     # React entry
  frontend/components/HealthReady.tsx    # Sistem durumu
  frontend/components/SuperclassPanel.tsx # Superclass tahmin UI
  frontend/components/LocalizationPanel.tsx # MI lokalizasyon UI
  frontend/components/XaiViewer.tsx      # XAI artifact goruntuleme
  frontend/lib/api.ts                    # HTTP client
  frontend/lib/types.ts                  # TypeScript tip tanimlari
```

### Pasif / Eski Dosyalar (Dogrudan Akista Degil)
```
ESKI BINARY PIPELINE (eski mimari, farkli import yapisi):
  src/pipeline/inference/run_inference_binary.py  # Bagimsiz calisir ama API'dan cagirilmaz
  src/utils/checkpoints.py               # Eski checkpoint loader (model_loader.py kullaniliyor)
  src/utils/llm_prompt.py                # LLM prompt builder (henuz entegre degil)
  src/xai/shap_xgb.py                   # Eski SHAP (shap_ovr.py kullaniliyor)
  src/xai/summary.py                    # Eski XAI ozet
  src/xai/combined.py                   # CombinedExplainer (unified.py kullaniliyor)

EGITIM SCRIPSLERI (inference'da kullanilmaz):
  src/pipeline/training/train_superclass_cnn.py     # CNN egitimi
  src/pipeline/training/train_superclass_xgb_ovr.py # XGBoost egitimi
  src/pipeline/training/train_mi_localization.py     # Lokalizasyon egitimi
  src/pipeline/training/run_experiment.py            # Eski deney calistirma
  src/pipeline/training/run_xgb.py                   # Eski XGBoost

VERI HAZIRLAMA (egitimde kullanilir):
  src/data/loader.py                     # PTB-XL veri yukleme
  src/data/labels.py                     # Binary etiketleme
  src/data/labels_superclass.py          # Superclass etiketleme
  src/data/mi_localization.py            # MI lokalizasyon etiketleri
  src/data/signals.py                    # Sinyal isleme, SignalDataset
  src/data/splits.py                     # Hasta bazli split
  src/data/validate.py                   # Veri dogrulama
  src/data/verify_superclass_labels.py   # Etiket dogrulama

DEGERLENDIRME (offline analiz):
  src/pipeline/evaluation/run_comprehensive_test.py
  src/pipeline/evaluation/compare_models.py
  src/pipeline/evaluation/compare_metrics.py
  src/pipeline/evaluation/evaluate_ensemble.py
  src/pipeline/evaluation/optimize_thresholds.py
  src/pipeline/evaluation/test_mi_integration.py

FEATURE EXTRACTION (egitimde):
  src/pipeline/features/extract_superclass_features.py
  src/pipeline/features/run_feature_extraction.py
  src/features/extract_cnn_features.py

XAI DEMO/REPORT (offline):
  src/pipeline/xai/generate_xai_report.py
  src/pipeline/xai/run_xai_demo.py

UTILITY:
  src/pipeline/utils/inspect_checkpoint.py
  src/pipeline/utils/export_sample.py
  src/pipeline/core/data_pipeline.py
  src/models/trainer.py
  src/models/metrics.py
  src/models/xgb.py
  src/utils/signal.py
  verify_data_layer.py
```

## Duzeltilen Sorunlar (2026-04-12)
- [x] requirements.txt tamamlandi (fastapi, uvicorn, joblib, pydantic eklendi)
- [x] Coherence score gercek hesaplama ile degistirildi (placeholder 0.85 kaldirildi)
- [x] Duplicate plot_gradcam_heatmap duzeltildi
- [x] Thresholds optimize edildi (Macro F1: 0.630 -> 0.681, +%8)
- [x] CORS ortam degiskeninden okunuyor (CORS_ORIGINS)
- [x] Dockerfile + docker-compose eklendi
- [x] Frontend'de Consistency Guard render ediliyor
- [x] XAI'ye SHAP-weighted GradCAM + contrastive mode eklendi
- [x] Debug print'ler kaldirildi
- [x] Consistency Guard exception handling eklendi

## Kalan Sorunlar
1. `datetime.utcnow()` deprecated (Python 3.12+) - duzeltilecek
2. `torch.load()` cagrilarinda `weights_only=True` eksik
3. `@app.on_event("startup")` deprecated - `lifespan` kullanilmali

## Model Performansi (Optimize Edilmis Thresholds ile)
| Sinif | AUROC  | F1 (eski) | F1 (yeni) | Threshold | Destek |
|-------|--------|-----------|-----------|-----------|--------|
| MI    | 0.9022 | 0.6933    | 0.6383    | 0.16      | 550    |
| STTC  | 0.9193 | 0.6638    | **0.7277**| 0.26      | 506    |
| CD    | 0.8923 | 0.6794    | **0.7254**| 0.28      | 496    |
| HYP   | 0.8805 | 0.4844    | **0.5335**| 0.19      | 261    |
| Macro | 0.8998 | 0.6302    | **0.6810**| -         | -      |

**Not:** MI F1 dustu ama recall 0.70 -> 0.93'e cikti (klinik guvenlik onceligi).

## Ensemble Formulu
```
ensemble_prob[sinif] = 0.15 * CNN_prob + 0.85 * XGB_prob  (optimize edilmis)
NORM_prob = 1.0 - max(ensemble_probs)
```

## Oncelik Kurali (Primary Label)
MI > STTC > CD > HYP > NORM

## Docker
```bash
docker-compose up --build    # Tek komutla calistir
# http://localhost:8000      # Frontend + API
```

## Test
```bash
pytest tests/ -v             # Tum testler
pytest tests/test_api.py -v  # API endpoint testleri
```

## Agent skills

### Issue tracker

Issues live in GitHub Issues on `mUsaulug/CardioGuard-AI` — use `gh` CLI from the terminal. See `docs/agents/issue-tracker.md`.

### Triage labels

Default label vocabulary (needs-triage, needs-info, ready-for-agent, ready-for-human, wontfix). See `docs/agents/triage-labels.md`.

### Domain docs

Single-context repo — one `CONTEXT.md` + `docs/adr/` at the repo root (created lazily via `/grill-with-docs`). See `docs/agents/domain.md`.

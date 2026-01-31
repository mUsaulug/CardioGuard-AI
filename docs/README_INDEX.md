# CardioGuard-AI Documentation Index

Teknik denetim sonucu oluşturulan kapsamlı dokümantasyon.

**Denetim Tarihi:** 31 Ocak 2026
**Denetçi:** Antigravity (Technical Auditor AI)

---

## 📋 Phase Documents

Aşamalı analiz dökümanları. Her biri derinlemesine inceleme içerir.

| # | Dosya | İçerik | Sayfa |
| :---: | :--- | :--- | :---: |
| 0 | [00_repo_map.md](./00_repo_map.md) | Proje yapısı, dizin ağacı, entry points, bağımlılıklar | ~10 |
| 1 | [01_architecture.md](./01_architecture.md) | C4 diyagramları, sequence flow, component analysis | ~12 |
| 2 | [02_api_contracts.md](./02_api_contracts.md) | Endpoint detayları, Pydantic modeller, güvenlik | ~10 |
| 3 | [03_inference_pipeline.md](./03_inference_pipeline.md) | **DENEY SONUÇLARI**, threshold optimizasyonu, split | ~15 |
| 4 | [04_xai_and_artifacts.md](./04_xai_and_artifacts.md) | Grad-CAM, SHAP, Unified Explainer, sanity checks | ~10 |
| 5 | [05_frontend_integration.md](./05_frontend_integration.md) | Type definitions, API client, kontrat uyumu | ~8 |
| 6 | [06_quality_tests_and_repro.md](./06_quality_tests_and_repro.md) | Test envanteri, **E2E DEMO ADIMLARI**, reproducibility | ~12 |

---

## 📊 Final Reports

Sunum ve raporlama için hazırlanmış özet dökümanlar.

| Dosya | Amaç | Hedef Kitle |
| :--- | :--- | :--- |
| [TECHNICAL_REPORT.md](./TECHNICAL_REPORT.md) | Tam teknik denetim raporu | Jüri, Danışman |
| [PRESENTATION_DECK.md](./PRESENTATION_DECK.md) | Sunum slaytları (14 slide) | Savunma |
| [DEMO_SCRIPT.md](./DEMO_SCRIPT.md) | Canlı demo senaryosu (5-7 dk) | Demo |
| [QNA_CHEATSHEET.md](./QNA_CHEATSHEET.md) | Soru-cevap rehberi (30+ soru) | Soru-Cevap |

---

## 🔑 Kritik Bulgular Özeti

### Yüksek Öncelik (P0)

| Bulgu | Dosya | Satır |
| :--- | :--- | :--- |
| **Consistency Guard entegre değil** | `run_inference_superclass.py` | - (çağrı yok) |

### Orta Öncelik (P1)

| Bulgu | Dosya | Satır |
| :--- | :--- | :--- |
| Hardcoded layer index | `run_inference_superclass.py` | L305 |
| Missing dependencies | `requirements.txt` | - |

### Düşük Öncelik (P2)

| Bulgu | Durum |
| :--- | :--- |
| Dockerfile | Yok |
| E2E testler | Yok |
| CI/CD | Yok |

---

## 📈 Performans Özeti

| Model | Macro AUROC | Macro F1 |
| :--- | :---: | :---: |
| CNN | 0.8986 | 0.6302 |
| XGBoost | 0.8998 | 0.6688 |
| Ensemble | ~0.90 | ~0.65 |

**En İyi Per-Class:** STTC (AUROC: 0.92)
**En Zor Sınıf:** HYP (F1: 0.48-0.58)

---

## 🚀 Hızlı Başlangıç

### Backend

```bash
uvicorn src.backend.main:app --host 0.0.0.0 --port 8000
```

### API Test

```bash
curl -X POST "http://localhost:8000/predict/superclass?explain=true" \
     -F "file=@sample.npz"
```

### Frontend

```bash
cd frontend && npm run dev
```

---

## 📁 Kanıt Dosyaları

Dokümanlarda referans verilen önemli kaynak dosyalar:

| Dosya | İçerik |
| :--- | :--- |
| `logs/superclass_cnn/training_results.json` | CNN metrikleri |
| `logs/xgb_superclass/training_results.json` | XGBoost metrikleri |
| `artifacts/thresholds_superclass.json` | Threshold optimizasyonu |
| `src/data/splits.py` | Leakage prevention |
| `src/pipeline/inference/consistency_guard.py` | Devre dışı guard |
| `frontend/lib/types.ts` | Frontend tipleri |

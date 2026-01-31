# CardioGuard-AI Demo Script

**Süre:** 5-7 dakika
**Hedef:** Sistemin uçtan uca çalışmasını göstermek

---

## Scene 1: Sistem Durumu (1 dk)

### 1.1 Backend Başlatma

```bash
# Terminal 1: Backend
cd CardioGuard-AI
uvicorn src.backend.main:app --host 0.0.0.0 --port 8000
```

**Beklenen Çıktı:**
```
INFO:     Validating checkpoints...
INFO:     Checkpoint validation passed!
INFO:     Superclass model loaded (hash: a1b2c3d4)
INFO:     Localization model loaded
INFO:     XGBoost models loaded: 4
INFO:     Thresholds loaded: 4 classes
INFO:     Uvicorn running on http://0.0.0.0:8000
```

**Demo Notu:** "Bakın, sistem başlarken tüm checkpoint'ları doğruluyor. Eğer herhangi biri bozuksa, sistem başlamayı reddedecek. Bu 'fail-closed' yaklaşımı."

### 1.2 Health & Readiness Check

```bash
# Health check
curl http://localhost:8000/health
```
```json
{"status": "healthy", "timestamp": "2026-01-31T03:20:00Z"}
```

```bash
# Readiness check
curl http://localhost:8000/ready
```
```json
{
  "ready": true,
  "models_loaded": {
    "superclass": true,
    "localization": true,
    "xgb": true,
    "thresholds": true
  },
  "message": "All models loaded"
}
```

---

## Scene 2: Superclass Prediction (2 dk)

### 2.1 API Çağrısı

```bash
# XAI ile tahmin
curl -X POST "http://localhost:8000/predict/superclass?explain=true" \
     -F "file=@sample.npz" \
     -H "accept: application/json" \
     > result.json
```

### 2.2 Sonuç İnceleme

```bash
# Primary label
cat result.json | jq '.primary'
```
```json
{
  "label": "MI",
  "confidence": 0.8523,
  "rule": "priority_order"
}
```

**Demo Notu:** "Model %85 güvenle MI tespit etti. Priority rule gereği MI en yüksek önceliğe sahip."

```bash
# Tüm olasılıklar
cat result.json | jq '.probabilities'
```
```json
{
  "MI": 0.8523,
  "STTC": 0.2341,
  "CD": 0.1205,
  "HYP": 0.0891,
  "NORM": 0.1477
}
```

```bash
# Kaynak karşılaştırması
cat result.json | jq '.sources'
```
```json
{
  "cnn": {"MI": 0.8412, "STTC": 0.2100, ...},
  "xgb": {"MI": 0.8634, "STTC": 0.2582, ...},
  "ensemble": {"MI": 0.8523, ...}
}
```

**Demo Notu:** "CNN ve XGBoost'un ayrı tahminlerini görüyorsunuz. Ensemble bunların ortalaması."

---

## Scene 3: XAI Artifacts (2 dk)

### 3.1 Artifact Listesi

```bash
cat result.json | jq '.xai.artifacts'
```
```json
[
  {
    "type": "report_png",
    "name": "sample__report.png",
    "url": "/runs/run_20260131.../visuals/sample__report.png",
    "mime": "image/png"
  },
  {
    "type": "narrative_md",
    "name": "sample__narrative.md",
    "url": "/runs/run_20260131.../text/sample__narrative.md",
    "mime": "text/markdown"
  }
]
```

### 3.2 Görsel Rapor

Tarayıcıda açın:
```
http://localhost:8000/runs/run_20260131.../visuals/sample__report.png
```

**Görsel İçeriği:**
- 12-lead EKG sinyali
- Grad-CAM heatmap overlay
- Top SHAP features bar chart
- Tahmin ve güven skorları

### 3.3 Narrative

```bash
RUN_ID=$(cat result.json | jq -r '.xai.run_id')
curl "http://localhost:8000/runs/$RUN_ID/text/sample__narrative.md"
```

```markdown
## AI Analiz Özeti

**Tahmin:** MI (Güven: 85.2%)

### Zamansal Odak
Model, sinyal üzerinde 0.4-0.6 saniye aralığına (ST segment) 
yoğun şekilde odaklanmıştır.

### Özellik Katkıları
- cnn_feat_12: +0.23 (MI lehine güçlü katkı)
- cnn_feat_47: -0.18 (NORM lehine katkı)
- cnn_feat_03: +0.14 (MI lehine)

### Sanity Check: PASS ✓
```

---

## Scene 4: Güvenlik Demo (1 dk)

### 4.1 Path Traversal Testi

```bash
# Kötü niyetli istek
curl "http://localhost:8000/runs/../../../etc/passwd"
```

**Beklenen Yanıt:**
```json
{"detail": "Invalid run_id format"}
```

**Demo Notu:** "Sistem path traversal saldırılarını engelliyor. Run ID sadece alfanümerik karakterler kabul ediyor."

### 4.2 Büyük Dosya Reddi

```bash
# 15MB dosya oluştur (test için)
dd if=/dev/zero of=large.npz bs=1M count=15
curl -X POST "http://localhost:8000/predict/superclass" -F "file=@large.npz"
```

**Beklenen Yanıt:**
```json
{"detail": "File too large. Max: 10485760 bytes"}
```

---

## Scene 5: CLI Inference (Opsiyonel, 1 dk)

```bash
# Doğrudan pipeline kullanımı
python -m src.pipeline.inference.run_inference_superclass \
    --input sample.npy \
    --explain \
    --output predictions/demo_result.json

# Çıktı
cat predictions/demo_result.json | jq '.primary'
```

---

## Kapanış

**Gösterilen Özellikler:**
1. ✅ Fail-closed startup
2. ✅ Health/Readiness probes
3. ✅ Multi-label prediction (CNN + XGBoost ensemble)
4. ✅ XAI artifact generation (Grad-CAM + SHAP)
5. ✅ Güvenlik kontrolleri

**Demo Soruları için hazır ol:**
- "NORM nasıl hesaplanıyor?" → `1 - max(pathology)`
- "Neden iki model?" → Complementary strengths
- "XAI güvenilir mi?" → Sanity checks ile doğrulanıyor

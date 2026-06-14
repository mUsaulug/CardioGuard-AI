# WP-04: Feature schema inference validation (P1)

**Tip:** AFK | **Blocked by:** WP-01 | **Tahmini:** 0.5 oturum

## Problem

Backend startup `logs/xgb_superclass/feature_schema.json` yüklüyor ama `validate_feature_schema()` inference sırasında **çağrılmıyor**. Embedding boyutu drift → sessiz yanlış XGB/SHAP.

## Okunacak dosyalar

1. `src/utils/model_loader.py` — `validate_feature_schema`, `load_feature_schema`
2. `src/backend/main.py` — `load_models`, `AppState.feature_schema`
3. `src/pipeline/inference/run_inference_superclass.py` — CNN embedding → XGB
4. `logs/xgb_superclass/feature_schema.json` (varsa)

## Yapılacaklar

1. `core_predict()` içinde embedding üretildikten sonra:
   ```python
   validate_feature_schema(embedding.shape[-1], feature_schema)
   ```
2. Startup'ta schema yoksa warn vs fail — superclass XGB varsa **fail-closed**
3. Unit test: yanlış boyut → exception

## Kabul kriterleri

- [ ] Inference embedding dim ≠ schema → hata
- [ ] Test eklendi
- [ ] Mevcut happy path bozulmadı

```bash
.venv/bin/python -m pytest tests/ -q -k "schema or xgb"
```

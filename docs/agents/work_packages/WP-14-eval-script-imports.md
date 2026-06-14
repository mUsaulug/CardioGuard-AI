# WP-14: Kırık eval script importları (P1)

**Tip:** AFK | **Blocked by:** — 

## Problem

Offline scriptler yanlış import path kullanıyor — çalıştırılamıyor:

| Dosya | Yanlış | Doğru |
|-------|--------|-------|
| `run_comprehensive_test.py` | `src.pipeline.run_inference_superclass` | `src.pipeline.inference.run_inference_superclass` |
| `generate_xai_report.py` | aynı | aynı |
| `generate_validation_predictions.py` | `src.pipeline.train_superclass_xgb_ovr` | `src.pipeline.training.train_superclass_xgb_ovr` |

## Yapılacaklar

1. Import'ları düzelt
2. Docstring `python -m ...` path'lerini güncelle
3. `tests/test_evaluation_imports.py` — modüller import edilebilmeli (smoke)

## Kabul kriterleri

- [ ] Üç script import hatasız
- [ ] Smoke test geçer

```bash
.venv/bin/python -c "from src.pipeline.evaluation import run_comprehensive_test"
.venv/bin/python -m pytest tests/test_evaluation_imports.py -q
```

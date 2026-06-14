# WP-02: Ensemble weight tek kaynak (P0)

**Tip:** AFK | **Blocked by:** — | **Tahmini:** 1 oturum

## Problem

Ensemble ağırlığı üç yerde farklı:

| Kaynak | Değer |
|--------|-------|
| `artifacts/thresholds_superclass.json` | `ensemble_weight: 0.15` (optimize) |
| API `main.py` default | 0.15 |
| Pipeline CLI `run_inference_superclass.py` | 0.5 |
| Frontend demo `useAnalysisSession.ts` | 0.85 |

Eşikler 0.15'te optimize edildi; farklı α → yanlış kararlar.

## Okunacak dosyalar

1. `artifacts/thresholds_superclass.json`
2. `src/backend/main.py` — `ensemble_weight` query param
3. `src/pipeline/inference/run_inference_superclass.py` — `core_predict`, CLI argparse
4. `src/contracts/airesult_mapper.py` — hardcoded `ensemble_best_alpha: 0.15`
5. `frontend/src/hooks/useAnalysisSession.ts`, `frontend/src/routes/index.tsx`
6. `src/config.py`

## Yapılacaklar

1. `src/config.py`'ye sabit ekle veya thresholds JSON'dan oku:
   ```python
   DEFAULT_ENSEMBLE_WEIGHT = 0.15  # veya json'dan
   ```

2. Pipeline CLI default → config/thresholds dosyasından

3. Backend startup'ta thresholds yüklenirken `ensemble_weight` field'ını da `AppState`'e al

4. `airesult_mapper.py`: gerçek kullanılan weight'i response'a yaz

5. Frontend:
   - Demo default slider → 0.15 (veya 0.85 = XGB ağırlığı değil — **CNN weight** olduğuna dikkat)
   - Slider etiketi: "XGB %85 · CNN %15" (mevcut) — değer 0.15 olmalı

6. Test: API default param verilmeden predict → ensemble 0.15 CNN ağırlığı kullanıldığını doğrula

## Dokunma

- Threshold sayıları (MI 0.16 vb.) — sadece weight birleştirme
- `frontend-legacy/`

## Kabul kriterleri

- [ ] API, CLI, frontend demo aynı default (0.15 CNN)
- [ ] AIResult/versions gerçek weight'i yansıtır
- [ ] `tests/test_api.py`'ye default ensemble testi eklendi
- [ ] README/CLAUDE ensemble satırı doğru (WP-17'de de güncellenir)

## Doğrulama

```bash
.venv/bin/python -m pytest tests/test_api.py -q
cd frontend && npm test && npx tsc --noEmit
```

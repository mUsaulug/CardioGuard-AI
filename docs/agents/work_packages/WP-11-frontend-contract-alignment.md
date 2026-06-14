# WP-11: Frontend ↔ backend contract hizalama (P1)

**Tip:** AFK | **Blocked by:** — 

## Problem

İki mapper drift:
- Python: `src/contracts/frontend_context.py` — `xaiArtifacts`, `runId`, `latencyMs` yok
- TS: `frontend/src/lib/mapResultToContext.ts` — bunları ekliyor
- `TechnicalDetails.tsx` sahte hash gösteriyor

## Okunacak dosyalar

1. `src/contracts/frontend_context.py`
2. `frontend/src/lib/mapResultToContext.ts`
3. `frontend/src/lib/types.ts` — `AnalysisContext`
4. `tests/test_frontend_contract_coverage.py`
5. `frontend/src/lib/mapResultToContext.test.ts`
6. `frontend/src/components/evidence/TechnicalDetails.tsx`

## Yapılacaklar

1. Python `build_frontend_context()` veya validator'a eksik alanları ekle (veya dokümante et ki TS-only kalabilir)
2. `tests/test_frontend_contract_coverage.py` — TS-only alanları fixture JSON'da assert et
3. `TechnicalDetails`: `api.versions.model_hash`, `threshold_hash`, `api_version` göster
4. `mapResultToContext`'e `versions` map ekle

## Kabul kriterleri

- [ ] Contract testleri Python+TS alanları kapsar
- [ ] TechnicalDetails gerçek version/hash
- [ ] pytest + vitest geçer

```bash
.venv/bin/python -m pytest tests/test_frontend_contract_coverage.py -q
cd frontend && npm test
```

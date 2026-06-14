# WP-05: Async route'larda sync inference (P0)

**Tip:** AFK | **Blocked by:** — | **Tahmini:** 0.5 oturum

## Problem

`async def predict_superclass` doğrudan sync `pipeline_predict()` çağırıyor — PyTorch+XGB+XAI event loop'u bloklar.

## Okunacak dosyalar

1. `src/backend/main.py` — `/predict/superclass`, `/predict/mi-localization`
2. FastAPI `run_in_threadpool` veya `asyncio.to_thread` docs

## Yapılacaklar

1. Her predict endpoint'te:
   ```python
   from starlette.concurrency import run_in_threadpool
   result = await run_in_threadpool(
       pipeline_predict, signal, **kwargs
   )
   ```
2. Upload parse (`parse_ecg_file`) de thread pool'a alınabilir (np.load I/O)
3. `/health`, `/ready` sync kalabilir

## Kabul kriterleri

- [ ] Predict route'ları `await run_in_threadpool(...)` kullanır
- [ ] `tests/test_api.py` geçer
- [ ] Davranış değişmedi (aynı JSON shape)

## Doğrulama

```bash
.venv/bin/python -m pytest tests/test_api.py -q
# Manuel: iki eşzamanlı curl — ikisi de yanıt vermeli
```

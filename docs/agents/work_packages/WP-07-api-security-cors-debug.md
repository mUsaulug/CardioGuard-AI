# WP-07: API güvenlik — CORS ve debug log (P1)

**Tip:** AFK | **Blocked by:** — 

## Problem

1. `CORS_ORIGINS` default `"*"` + `allow_credentials=True` — geçersiz/tehlikeli
2. `docker-compose.yml` CORS'ta `*` var
3. `/debug/client-log` GET/POST auth yok — production'da açık kalabilir

## Okunacak dosyalar

1. `src/backend/main.py` — CORS middleware, debug endpoints
2. `docker-compose.yml`
3. `frontend/src/lib/sessionDebugLog.ts`

## Yapılacaklar

1. CORS:
   - Default: `http://localhost:3000,http://localhost:5173` (credentials ile uyumlu)
   - `*` + credentials kombinasyonunu reddet veya credentials=False when wildcard

2. Debug endpoints:
   ```python
   ENABLE_DEBUG_ENDPOINTS = os.getenv("ENABLE_DEBUG_ENDPOINTS", "0") == "1"
   ```
   - Kapalıyken POST/GET → 404
   - Dev: `ENABLE_DEBUG_ENDPOINTS=1`

3. `np.load`: `allow_pickle=False` upload path'te

4. Test: debug endpoint kapalıyken 404

## Kabul kriterleri

- [ ] Prod-safe CORS default
- [ ] Debug log env-gated
- [ ] Test eklendi
- [ ] `docker-compose` wildcard kaldırıldı

```bash
.venv/bin/python -m pytest tests/test_api.py -q
```

# WP-18: Docker frontend mimarisi (P0 — HITL)

**Tip:** HITL (insan kararı gerekli) | **Blocked by:** — 

## Problem

`docker-compose up` README'de "tek komut" diyor ama:
- TanStack Start → `frontend/dist/client` + `dist/server/server.js`
- FastAPI sadece `/assets` mount — **SPA/SSR UI serve edilmiyor**

## Karar seçenekleri (insan seçmeli)

### A) Static SPA export
- Vite/TanStack static build → `index.html` + assets
- FastAPI: `StaticFiles(directory="frontend/dist/client", html=True)` on `/`
- **Artı:** Tek container, basit
- **Eksi:** SSR özellikleri kaybolabilir

### B) İki servis compose
- `api`: uvicorn :8000
- `web`: Nitro/node SSR :3000
- nginx veya CORS ile birleştir
- **Artı:** Mevcut TanStack Start korunur
- **Eksi:** İki container, daha karmaşık

### C) Multi-stage Dockerfile
- Node stage: `npm run build`
- Python stage: copy dist + uvicorn
- Seçime göre A veya B

## Agent yapabilir (karar sonrası)

1. Multi-stage Dockerfile
2. `docker-compose.yml` güncelle
3. Smoke test: container'da `/health` + `/` HTML 200

## Bu paketi başlatmadan önce

Kullanıcıya sor: **A mı B mi?**

Karar gelene kadar agent **implementasyon yapmasın** — sadece spike/prototype (WP prototype skill) opsiyonel.

## Referans dosyalar

- `Dockerfile`, `docker-compose.yml`
- `frontend/vite.config.ts`, `frontend/package.json` scripts
- `src/backend/main.py` static mount bölümü

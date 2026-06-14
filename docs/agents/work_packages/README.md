# Teknik Borç — Agent İş Paketleri (Yerel)

> **GitHub Issues kullanılmıyor** — işler bu klasörde tanımlı. Repo remote'u ayarlanana kadar issue açma/push yapma.

## Nasıl kullanılır

1. `README.md` (bu dosya) → sıra ve bağımlılıklar
2. Her `WP-XX-*.md` → **tek bir agent oturumunda** tamamlanacak iş
3. Agent başlamadan önce `CLAUDE.md` mimari kurallarını oku
4. Bitince: `pytest tests/ -q` + `cd frontend && npm test && npx tsc --noEmit`
5. **Commit/push kullanıcı söylemeden yapma**

## Mimari kurallar (tüm paketler için)

1. `src/backend/main.py` içinde ML kodu **yok** — sadece `pipeline_predict()` çağrısı
2. Tek inference kaynağı: `src/pipeline/inference/run_inference_superclass.py`
3. XAI artifact'ları pipeline üretir; backend manifest okur/serve eder
4. Fail-closed: model validation başarısız → uygulama başlamaz
5. NORM türetilir: `1 - max(MI, STTC, CD, HYP)`
6. `frontend-legacy/` **dokunma** (arşiv)
7. Ensemble production default: **0.15 CNN / 0.85 XGB** (`artifacts/thresholds_superclass.json`)

## Bağımlılık grafiği

```
Sprint A (Doğruluk)
  WP-01 Normalizasyon ──┬──► WP-04 Feature schema
  WP-02 Ensemble tek kaynak ──► WP-03 ECG validation
                        └──► (frontend slider sync)

Sprint B (Prod)
  WP-05 Async thread pool
  WP-06 CI workflow
  WP-07 Güvenlik (CORS + debug log)
  WP-08 Fail-closed startup

Sprint C (Frontend)
  WP-09 Backend health UX
  WP-10 Demo vs live ayrımı
  WP-11 Contract hizalama ──► WP-12 Zod validation
  WP-13 LLM timeout + banner fix

Sprint D (XAI / test / docs)
  WP-14 Eval import fix
  WP-15 XAI sanity + baseline
  WP-16 Frontend test genişletme
  WP-17 Dokümantasyon sync

HITL (insan kararı)
  WP-18 Docker frontend mimarisi
```

## İş paketi indeksi

| ID | Dosya | Öncelik | Tip | Bağımlılık |
|----|-------|---------|-----|------------|
| WP-01 | [WP-01-inference-normalization.md](./WP-01-inference-normalization.md) | P0 | AFK | — |
| WP-02 | [WP-02-ensemble-single-source.md](./WP-02-ensemble-single-source.md) | P0 | AFK | — |
| WP-03 | [WP-03-ecg-validation.md](./WP-03-ecg-validation.md) | P1 | AFK | WP-01 önerilir |
| WP-04 | [WP-04-feature-schema-validation.md](./WP-04-feature-schema-validation.md) | P1 | AFK | WP-01 |
| WP-05 | [WP-05-async-inference-threadpool.md](./WP-05-async-inference-threadpool.md) | P0 | AFK | — |
| WP-06 | [WP-06-github-actions-ci.md](./WP-06-github-actions-ci.md) | P0 | AFK | — |
| WP-07 | [WP-07-api-security-cors-debug.md](./WP-07-api-security-cors-debug.md) | P1 | AFK | — |
| WP-08 | [WP-08-fail-closed-startup.md](./WP-08-fail-closed-startup.md) | P1 | AFK | — |
| WP-09 | [WP-09-frontend-backend-health.md](./WP-09-frontend-backend-health.md) | P1 | AFK | — |
| WP-10 | [WP-10-demo-vs-live-mode.md](./WP-10-demo-vs-live-mode.md) | P1 | AFK | — |
| WP-11 | [WP-11-frontend-contract-alignment.md](./WP-11-frontend-contract-alignment.md) | P1 | AFK | — |
| WP-12 | [WP-12-api-zod-validation.md](./WP-12-api-zod-validation.md) | P1 | AFK | WP-11 |
| WP-13 | [WP-13-llm-timeout-ux.md](./WP-13-llm-timeout-ux.md) | P1 | AFK | — |
| WP-14 | [WP-14-eval-script-imports.md](./WP-14-eval-script-imports.md) | P1 | AFK | — |
| WP-15 | [WP-15-xai-sanity-baseline.md](./WP-15-xai-sanity-baseline.md) | P1 | AFK | — |
| WP-16 | [WP-16-frontend-test-expansion.md](./WP-16-frontend-test-expansion.md) | P2 | AFK | WP-11 |
| WP-17 | [WP-17-documentation-sync.md](./WP-17-documentation-sync.md) | P2 | AFK | WP-02 |
| WP-18 | [WP-18-docker-frontend-architecture.md](./WP-18-docker-frontend-architecture.md) | P0 | HITL | insan kararı |

Tam envanter: [`../technical_debt_inventory.md`](../technical_debt_inventory.md)

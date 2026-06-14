# Uçtan uca review — WP-01 → WP-08 (2026-06-09)

Orchestrator + 2 paralel review agent. Yerel doğrulama çalıştırıldı.

## Özet karar

| Katman | Sonuç | Not |
|--------|-------|-----|
| Backend (WP-01,02,05,06,07,08) | **PASS** | Mimari kurallar korunmuş |
| CI / Docker | **PASS** | Workflow + compose uyumlu |
| Frontend (WP-02 kısmi) | **Kısmi** | Live path OK; demo modu sorunlu (WP-10) |
| Bilinen regresyon | **1 test** | XAI sanity — WP-15 |

**Genel:** Backend batch merge-ready (XAI hariç). Frontend demo/contract backlog WP-09–11'de.

---

## Test kanıtı (yerel)

```
Backend:  pytest tests/ -q --ignore=test_data -k "not test_explain_produces..."
           → TÜMÜ GEÇTİ (XAI test hariç)

Frontend:  npm test → 5/5 | npx tsc --noEmit → OK
```

Değişen dosyalar: 10 modified + 4 yeni (`.github/`, work packages, test dosyaları). **Commit/push yok.**

---

## WP bazlı review

### WP-01 — CNN normalizasyon — PASS
- `load_superclass_norm_stats()` → npz veya JSON fallback
- `apply_superclass_normalization()` `core_predict()` içinde CNN öncesi
- Startup'ta stats doğrulanır (eksikse fail-closed)
- Test: `tests/test_signal_normalization.py`

**Açık risk:** Binary/localization modelleri superclass-normalize edilmiş tensör kullanıyor; standalone localization route normalize etmiyor.

### WP-02 — Ensemble tek kaynak — PASS
- `get_ensemble_cnn_weight()` → `artifacts/thresholds_superclass.json` (0.15)
- API, pipeline, CLI, AIResult mapper uyumlu
- Review blocker düzeltildi: `predict()` return'a `ensemble_weight` eklendi
- Test: default ensemble math + `full=true` custom weight

**Not:** `loadDemo()` içindeki `ensemble: 0.15` mock path'te kullanılmıyor (WP-10).

### WP-05 — Async thread pool — PASS
- `run_in_threadpool` → parse + her iki predict route
- Event loop bloklanmıyor

### WP-06 — GitHub Actions CI — PASS
- Backend: Python 3.12, torch CPU, pytest subset
- Frontend: npm ci, vitest, tsc
- `ENABLE_DEBUG_ENDPOINTS=1` test env'de

### WP-07 — CORS + debug güvenlik — PASS
- CORS default localhost; `*` → credentials kapalı
- `/debug/client-log` → `ENABLE_DEBUG_ENDPOINTS=1` yoksa 404
- `allow_pickle=False` upload path
- docker-compose wildcard kaldırıldı

### WP-08 — Fail-closed startup — PASS
- Zorunlu: superclass checkpoint, XGB OVR (4 sınıf), schema, scaler, thresholds
- Opsiyonel: binary, localization → `degraded: true`
- `/ready` → `degraded` + `degraded_models`
- Test: `test_startup_failclosed.py`

---

## Mimari uyum kontrolü

| Kural | Durum |
|-------|-------|
| `main.py` ML kodu yok | ✅ |
| Tek inference kaynağı `run_inference_superclass.py` | ✅ |
| NORM türetilir | ✅ |
| Fail-closed startup | ✅ |
| XAI pipeline üretir, backend serve eder | ✅ (XAI sanity bug ayrı) |

---

## Frontend + contract (henüz WP değil)

| Konu | Durum | Hedef WP |
|------|-------|----------|
| Live upload ensemble slider (0.85 XGB → 0.15 CNN) | ✅ | — |
| Demo modu gerçek upload'ı mock'a çeviriyor | ❌ | WP-10 |
| `loadDemo()` ensemble değeri ölü kod | ⚠️ | WP-10 |
| Welcome `/ready` health yok | ❌ | WP-09 |
| `AnalysisContext` contract drift | ❌ | WP-11 |
| Vitest sadece mapper (5 test) | ⚠️ | WP-16 |

---

## Açık blocker'lar (sıradaki iş)

1. **WP-15** — `test_explain_produces_*` XAI sanity (`sanity.py` NoneType)
2. **WP-10** — Demo vs live mod ayrımı
3. **WP-09/11** — Health UX + contract sync
4. **Norm tutarlılığı** — binary/localization input scaling (yeni ticket adayı)

---

## Review agent sonuçları

- [Backend review](8a60265d-96e9-42fe-bdc4-0182a24a2ee7): WP-01–08 PASS
- [Frontend+CI review](d85c234b-dcc1-45cb-bad3-343633c9d033): CI PASS, frontend FAIL (demo/contract)
- [WP-01–06 review](0d980055-0b0b-42ad-b7d6-9036f1b88aa3): WP-02 blocker düzeltildi

---

## Orchestrator onayı

WP-01 → WP-08 backend hedefleri karşılandı, testler geçiyor, mimari ihlal yok. Commit öncesi tek zorunlu bilinen fail: XAI sanity (CI'da skip). Frontend demo/contract maddeleri ayrı WP olarak kuyrukta.

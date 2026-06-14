# Agent orchestration progress

| WP | Title | Status | Verified | Notes |
|----|-------|--------|----------|-------|
| WP-01 | Inference CNN normalization | ✅ done | ✅ pytest | `signal.py` + `core_predict` + startup validate |
| WP-02 | Ensemble single source | ✅ done | ✅ pytest | `get_ensemble_cnn_weight()` from thresholds artifact |
| WP-03 | ECG input validation | ✅ done | ✅ pytest | `validate_ecg_signal` + real AIResult shape |
| WP-05 | Async thread pool | ✅ done | ✅ pytest* | `run_in_threadpool` on parse + predict routes |
| WP-06 | CI workflow | ✅ done | ✅ local | `.github/workflows/ci.yml` (untracked until commit) |
| WP-07 | CORS + debug security | ✅ done | ✅ pytest | prod-safe CORS, debug env gate |
| WP-08 | Fail-closed startup | ✅ done | ✅ pytest | required vs optional checkpoints, `/ready` degraded |
| WP-10 | Demo vs live mode | ✅ done | ✅ vitest | Settings demo ≠ mock analysis |
| WP-15 | XAI sanity baseline | ✅ done | ✅ pytest | None heatmap + GradCAM cleanup |

### Implementation batch (2026-06-14 — validation fixes, no commit)

| Area | Changes |
|------|---------|
| WP-01/03 | Standalone MI localization: `validate_ecg_signal` only (raw amplitudes — Tur 4 R3-02) |
| WP-04 | `validate_feature_schema` in `core_predict`; API passes `feature_schema` |
| WP-06/18 | Multi-stage `Dockerfile` (Node build + API); SPA `index.html` routes in `main.py` |
| WP-09 | Welcome backend health poll; analyze disabled when backend down |
| WP-11 | `mapResultToContext` versions; `TechnicalDetails` real hashes |
| WP-14 | Eval import paths fixed; `tests/test_evaluation_imports.py` |
| WP-02/17 | Eval scripts default ensemble from `get_ensemble_cnn_weight()` |
| Config | `DEFAULT_MIN_LIKELIHOOD = 0.0` SSOT |
| WP-15 | `extract_sanity` maps RELIABLE/ACCEPTABLE → PASS |
| Frontend | Inference timeout 120s; `useAnalysisSession` file guard |

### Implementation batch 2 (2026-06-14 continued)

| Area | Changes |
|------|---------|
| WP-12 | Zod `superclassSchema.ts` + parse in `predictSuperclass` |
| WP-16 | +10 vitest (schema, storage, openrouter) → 19 total |
| WP-11/09 | MI-localization: `versions`, `latency_ms`, `labels_tr`, `glossary` |
| Data | `validate.py` co-occurrence + pathology column fix |
| XAI | `generate_xai_report` ensemble SSOT; `reporting.py` docstring |
| Docs | `technical_debt_inventory.md` validation sync section |

**Verified:** pytest + npm test + tsc (see below)

### Implementation batch 3 — Tur 4 (2026-06-14, R3 validation fixes)

| ID | Changes |
|----|---------|
| R3-02 | Localization CNN fed **raw** tensor in `core_predict` + standalone `predict` (matches training) |
| R3-FIX-05 | Frontend `/ready` poll + degraded UX (`fetchBackendStatus`, Welcome status rows) |
| R3-06 | `src/utils/signal_io.py`; API `parse_ecg_file` + CLI `load_ecg_signal` consolidated |
| R3-01 | CLI `load_cnn_model` → `load_model_safe` (same as API) |
| R3-04 | `src/xai/manifest_io.py` shared writer for pipeline + localization paths |
| R3-03 | CI workflow present locally; commit pending user request |
| R3-05 | `src/backend/llm_proxy.py` + `/api/llm/*`; frontend via backend proxy (no browser→OpenRouter) |

**Tests:** `test_localization_raw_inference.py`, `test_signal_io.py`, `test_manifest_io.py`, `test_api.py` LLM proxy

## E2E review (2026-06-09)

Tam uçtan uca review: `E2E_REVIEW.md` — backend PASS, frontend kısmi, 1 XAI test blocker (WP-15).

## Review log

### WP-01 (2026-06-09)
- **Implemented:** `load_superclass_norm_stats`, `apply_superclass_normalization`, wired in `core_predict`, backend startup check
- **Fallback:** `features_out/superclass_feature_config.json` when npz missing
- **Tests:** `tests/test_signal_normalization.py`

### WP-02 (2026-06-09)
- **Implemented:** `get_ensemble_cnn_weight()` in `config.py`, pipeline/API/CLI defaults, demo frontend 0.15
- **Tests:** `test_predict_superclass_default_ensemble_weight`

### WP-05 (2026-06-09)
- **Implemented:** `run_in_threadpool` for `parse_ecg_file` + both predict endpoints
- **Tests:** 15/16 `test_api.py` pass; `test_explain_produces_*` pre-existing XAI sanity bug (WP-15)

### WP-06 (2026-06-09)
- **Implemented:** `.github/workflows/ci.yml` — backend pytest + frontend vitest/tsc
- **CI skips:** `test_data.py` (PTB-XL), XAI sanity regression (WP-15)

### WP-07 (2026-06-09)
- **Implemented:** CORS default localhost origins; `*` disables credentials; `ENABLE_DEBUG_ENDPOINTS` gate; `allow_pickle=False`
- **Tests:** `test_client_debug_log_disabled_returns_404`

### WP-08 (2026-06-09)
- **Implemented:** Required superclass + XGB fail-closed; optional binary/localization → `degraded`; `/ready` fields
- **Tests:** `tests/test_startup_failclosed.py`, `test_ready_reports_degraded`

### WP-03 (2026-06-09)
- **Implemented:** `validate_ecg_signal()` — 12 lead, finite, amplitude, PTB-XL meta
- **Wired:** `parse_ecg_file`, `core_predict`, `derive_input_meta`, API 400 on invalid
- **Tests:** `tests/test_ecg_validation.py`

### WP-10 (2026-06-09)
- **Implemented:** `shouldUseMockAnalysis()` — Settings demo modu yalnızca LLM; gerçek upload → backend
- **Tests:** `frontend/src/lib/analysisMode.test.ts`

### WP-15 (2026-06-09)
- **Implemented:** NORM/boş Grad-CAM sanity fix, `_resolve_sanity_gradcam_target`, GradCAM `cleanup()`
- **Tests:** `test_explain_produces_*` geçiyor, `tests/test_xai_sanity.py`

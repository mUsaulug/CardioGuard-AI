# CardioGuard-AI — Tur 3 Doğrulama Raporu

**Tarih:** 2026-06-14  
**Orchestrator:** Tur 3 validation (V-R3-01 … V-R3-06 + V-R3-FIX + V-R3-15)  
**Kaynak:** [ROUND-3_VALIDATION_PROMPT.md](./ROUND-3_VALIDATION_PROMPT.md), [VALIDATION_REPORT.md](./VALIDATION_REPORT.md), [PROGRESS.md](./work_packages/PROGRESS.md)  
**Mod:** Readonly — kod değiştirilmedi, commit yapılmadı

---

## Executive summary

1. **6/6 “açık” madde hâlâ geçerli** — Tur 2 birçok semptomu düzeltti ama kök nedenlerin çoğu duruyor.
2. **En kritik risk R3-02 (S4):** Localization modeli **raw** sinyalle eğitildi; inference (standalone + embedded) **superclass z-score** uyguluyor — klinik tahmin drift riski.
3. **R3-03 (S3):** `.github/workflows/ci.yml` dosyası var ve geçerli ama **git’te untracked** → remote CI çalışmıyor; PROGRESS WP-06 ✅ iddiası **yanlış**.
4. **7/7 Tur 2 fix kodda mevcut** — R3-FIX-01…07 DOĞRULANDI; R3-FIX-05 `/ready` poll etmez (KISMEN kabul edilebilir).
5. **R3-01 dual loader (S2)** ve **R3-06 signal I/O dup (S2)** bakım/drift riski; aktif production crash değil.
6. **R3-05 OpenRouter (S2 prod)** bilinçli tradeoff — API key tarayıcıda; demo modu LLM’i kesiyor.

---

## Özet tablo

| ID | Durum | Şiddet | Efor | Kısa kanıt |
|----|-------|--------|------|------------|
| **R3-01** | DOĞRULANDI | **S2** | M | API `ECGCNN` vs CLI `MultiLabelECGCNN` |
| **R3-02** | DOĞRULANDI | **S4** | HITL | Train raw; inference z-score |
| **R3-03** | DOĞRULANDI | **S3** | XS | `ci.yml` untracked |
| **R3-04** | DOĞRULANDI | **S2** | M | 3 manifest writer, farklı şemalar |
| **R3-05** | DOĞRULANDI | **S2** (prod) / S0 (demo) | M | `openrouter.ts:5,189` browser fetch |
| **R3-06** | DOĞRULANDI | **S2** | S | `signal_io.py` yok; 3+ I/O kopyası |

### Fix teyitleri

| Fix ID | Durum | Şiddet (kalan) | Kanıt |
|--------|-------|----------------|-------|
| **R3-FIX-01** | DOĞRULANDI | R3-02 train gap **S4** | `run_inference_localization.py:51-53` |
| **R3-FIX-02** | DOĞRULANDI | S0 | `core_predict` `:216-217`; `main.py:688` |
| **R3-FIX-03** | DOĞRULANDI | S0 | `cardioguard.ts:104`; `superclassSchema.ts` |
| **R3-FIX-04** | DOĞRULANDI | S0* | Multi-stage `Dockerfile`; SPA `main.py:863-876` |
| **R3-FIX-05** | KISMEN | S1 | `/health` poll var; `/ready` yok |
| **R3-FIX-06** | DOĞRULANDI | S0 | `MILocalizationResponse` + handler `:829-846` |
| **R3-FIX-07** | DOĞRULANDI | S0 | Import paths düzeltildi; smoke test var |

\*R3-FIX-04: Yerel `frontend/dist/client/index.html` yok (pre-build); Docker build stage üretir.

---

## Açık maddeler — detay

### R3-01 — Dual superclass loader

**Durum:** DOĞRULANDI | **Şiddet:** S2 | **Efor:** M

| Path | Yüklenen sınıf |
|------|----------------|
| API startup | `load_model_safe` → `ECGCNN` |
| CLI inference | `load_cnn_model` → `MultiLabelECGCNN` |

**Kanıt:**

```258:260:src/backend/main.py
            self.superclass_model, meta = load_model_safe(
                superclass_checkpoint, "superclass", self.device
            )
```

```145:147:src/utils/model_loader.py
        msd = normalize_state_dict_keys(msd)
        model = ECGCNN(config, num_classes=out_dim)
```

```69:74:src/pipeline/inference/run_inference_superclass.py
def load_cnn_model(checkpoint_path: Path, device: torch.device) -> MultiLabelECGCNN:
    ...
    model = MultiLabelECGCNN(config)
```

**Risk:** Key remap ile çalışıyor olabilir; audit/type drift, CLI vs API regression test eksikliği.

**Fix yönü:** Tek loader (`load_model_safe`) veya her yerde `MultiLabelECGCNN` + shared factory.

---

### R3-02 — Localization train raw vs inference z-score

**Durum:** DOĞRULANDI | **Şiddet:** S4 | **Efor:** HITL

**Eğitim — z-score YOK:**

```65:66:src/pipeline/training/train_mi_localization.py
        # Ensure channel first (C, T)
        signal = ensure_channel_first(signal)
```

`rg "normalize|apply_superclass" train_mi_localization.py` → **0 match**

**Inference — z-score VAR (Tur 2 fix sonrası unified inference):**

```51:53:src/pipeline/inference/run_inference_localization.py
    signal, _ = validate_ecg_signal(signal)
    signal = apply_superclass_normalization(signal)
```

```199:200:src/pipeline/inference/run_inference_superclass.py
    signal, _input_meta = validate_ecg_signal(signal)
    signal = apply_superclass_normalization(signal)
```

Embedded loc `:275` aynı z-scored `signal_tensor` kullanır.

**Klinik risk:** Model raw PTB-XL dağılımında eğitildi; production inference superclass z-score stats uygular → bölge olasılıkları drift edebilir.

**Fix yönü (HITL karar):**

| Seçenek | Açıklama | Efor |
|---------|----------|------|
| **A** | Retrain localization z-score stats ile | L |
| **B** | Inference’ı raw’a çek (superclass norm kaldır loc path’ten) | M — R3-FIX-01 geri alınır |
| **C** | Her ikisini align + regression test | L |

**R3-FIX-01 ilişkisi:** Inference path unify edildi (REDDEDİLDİ “hâlâ split” iddiası standalone vs embedded için); **train≠inference kök nedeni açık**.

---

### R3-03 — CI untracked

**Durum:** DOĞRULANDI | **Şiddet:** S3 | **Efor:** XS

**Git çıktısı:**

```
git status .github/workflows/ci.yml
→ Untracked files: .github/workflows/ci.yml

git ls-files .github/workflows/
→ (boş)
```

Workflow içeriği geçerli: Python 3.12, `pytest --ignore=tests/test_data.py`, `npm test`, `tsc`.

**PROGRESS çelişkisi:** `PROGRESS.md:9` “WP-06 ✅ done” — dosya commit edilmediği için **yanlış iddia**.

**Fix:** `git add .github/workflows/ci.yml` + push → S0.

---

### R3-04 — XAI manifest 3 writer

**Durum:** DOĞRULANDI | **Şiddet:** S2 | **Efor:** M

| Writer | Dosya | Şema farkı |
|--------|-------|------------|
| API production | `xai/pipeline.py:281-331` | `artifacts[]`, `sanity`, `highlights` |
| Batch offline | `xai/reporting.py:257-311` | `files{}`, `cards.jsonl`, `n_samples` |
| Localization | `run_inference_localization.py:182-238` | `task=localization`, `sanity=null` |

`rg "_write_manifest" src/` → 2 dosya, 3 writer fonksiyonu.

**API path `cards.jsonl` yazmıyor** — `rg "cards.jsonl" src/xai/pipeline.py` → 0 match.

**Tur 2 kısmi fix:** `extract_sanity` RELIABLE mapping eklendi:

```217:220:src/contracts/artifacts.py
    if status in ("RELIABLE", "ACCEPTABLE"):
        status = "PASS"
    elif status == "UNRELIABLE":
        status = "FAIL"
```

**Unify stratejisi:** Minimum = shared `write_manifest_v1()` adapter; full = tek schema + `cards.jsonl` veya manifest `highlights` producer.

---

### R3-05 — OpenRouter tarayıcıdan direkt

**Durum:** DOĞRULANDI | **Şiddet:** S2 (prod) / S0 (demo) | **Efor:** M

```5:5:frontend/src/lib/openrouter.ts
const ENDPOINT = "https://openrouter.ai/api/v1/chat/completions";
```

```50:50:frontend/src/lib/openrouter.ts
    Authorization: `Bearer ${apiKey}`,
```

```189:189:frontend/src/lib/openrouter.ts
  const res = await fetch(ENDPOINT, {
```

**Key storage:**

```53:58:frontend/src/lib/storage.ts
export function getApiKey(): string {
  ...
  const stored = localStorage.getItem(APIKEY_KEY);
  ...
  const envKey = import.meta.env.VITE_OPENROUTER_API_KEY as string | undefined;
```

**Demo ayrımı:** `getDemoMode()` LLM’i kesiyor (`useAnalysisSession.ts:128,385`).

**Prod risk:** Key XSS/localStorage exposure; rate limit client-side only.

**Fix:** Backend proxy + server-side key (M).

---

### R3-06 — Signal I/O duplication

**Durum:** DOĞRULANDI | **Şiddet:** S2 | **Efor:** S

```
test -f src/utils/signal_io.py → MISSING
```

| Fonksiyon | Dosya | Farklar |
|-----------|-------|---------|
| `parse_ecg_file` | `main.py:427-463` | bytes upload, `validate_ecg_signal`, `allow_pickle=False` |
| `load_ecg_signal` | `run_inference_superclass.py:143-159` | path-based, no validate |
| `load_ecg_signal` | `run_inference_binary.py:70-81` | + csv/txt desteği |

**Minimum fix:** `src/utils/signal_io.py` — `load_ecg_from_bytes`, `load_ecg_from_path`, shared npz key logic.

---

## Fix teyitleri (R3-FIX-01 … 07) — detay

### R3-FIX-01 — Standalone localization validate + z-score

**DOĞRULANDI** — `run_inference_localization.py:51-53`. Embedded path zaten `:199-200`. **Train gap (R3-02) devam ediyor.**

### R3-FIX-02 — validate_feature_schema inference

**DOĞRULANDI:**

```216:217:src/pipeline/inference/run_inference_superclass.py
        if feature_schema is not None:
            validate_feature_schema(embeddings.shape, feature_schema)
```

```688:688:src/backend/main.py
            feature_schema=state.feature_schema,
```

### R3-FIX-03 — Zod API parse

**DOĞRULANDI:** `cardioguard.ts:102-104` → `parseSuperclassApiResponse(data)`; schema `frontend/src/lib/api/superclassSchema.ts`.

### R3-FIX-04 — Docker multi-stage + SPA routes

**DOĞRULANDI:**

- `Dockerfile:4-10` Node build stage; `COPY --from=frontend-build .../dist`
- `main.py:863-876` `@app.get("/")` + SPA catch-all when `index.html` exists

Yerel dist: `index.html MISSING` (henüz `npm run build` yapılmamış veya dist gitignore).

### R3-FIX-05 — Backend health UX

**KISMEN:**

```76:80:frontend/src/routes/index.tsx
  useEffect(() => {
    ...
      const ok = await testBackendConnection(getBackendUrl());
```

```111:117:frontend/src/lib/api/cardioguard.ts
    const res = await fetch(`${normalizeBaseUrl(backendUrl)}/health`, ...
```

```206:206:frontend/src/routes/index.tsx
            <Button ... disabled={backendOk === false}>
```

- ✅ `/health` poll, dinamik status, analyze disable
- ❌ `/ready` poll yok (degraded mode görünmez)
- ⚠️ “Açıklama motoru aktif” hâlâ statik yeşil (`index.tsx:230`)

### R3-FIX-06 — MI-localization response parity

**DOĞRULANDI** — schema genişletildi:

```179:193:src/backend/main.py
class MILocalizationResponse(BaseModel):
    ...
    labels_tr: Dict[str, str] = ...
    versions: Optional[VersionInfo] = ...
    glossary: Dict[str, str] = ...
    latency_ms: Optional[float] = ...
```

Handler `:829-846` dolduruyor. `full`/`airesult` query hâlâ yok (superclass-only) — kabul edilebilir scope.

### R3-FIX-07 — Eval import paths

**DOĞRULANDI:**

```23:33:src/pipeline/evaluation/run_comprehensive_test.py
from src.pipeline.inference.run_inference_superclass import (...)
from src.pipeline.inference.run_inference_binary import (...)
```

`grep "from src.pipeline.run_inference"` → **0 match** (eski path yok).

`tests/test_evaluation_imports.py:6-15` smoke test mevcut.

Import smoke (bu ortam): pandas eksik → ModuleNotFoundError; path hatası değil.

---

## Yeni bulgular (R3-NEW)

| ID | Şiddet | Bulgu | Kanıt |
|----|--------|-------|-------|
| **R3-NEW-01** | S2 | PROGRESS WP-06 “done” ama ci.yml untracked | `PROGRESS.md:9` vs `git status` |
| **R3-NEW-02** | S2 | Python 3.10 Docker vs 3.12 CI | `Dockerfile:13` vs `ci.yml:16` |
| **R3-NEW-03** | S1 | Welcome “Açıklama motoru aktif” statik yeşil | `index.tsx:230` |
| **R3-NEW-04** | S1 | `/ready` degraded mode UI’da yok | health-only poll |
| **R3-NEW-05** | S2 | API path hâlâ `cards.jsonl` / `top_windows` üretmiyor | `pipeline.py` grep |
| **R3-NEW-06** | S1 | `_ensure_channel_first` dead code loc script’te | `run_inference_localization.py:100+` (artık validate path kullanılıyor) |

---

## Öncelikli fix sırası (doğrulama sonrası)

1. **R3-02 / WP-01 HITL** — Localization retrain vs inference raw kararı (S4)
2. **R3-03 / WP-06** — `ci.yml` commit + push (XS → S0)
3. **R3-01** — Dual loader consolidation (S2, M)
4. **R3-04 / WP-15** — Manifest schema unify (S2, M)
5. **R3-05** — OpenRouter backend proxy (prod, M)
6. **R3-06** — `signal_io.py` konsolidasyon (S, S)

---

## Tur 4 önerisi (implementasyon — bu oturumda YAPILMADI)

| Sıra | WP / konu | Gerekçe |
|------|-----------|---------|
| 1 | WP-06 commit | R3-03 tek commit |
| 2 | Localization norm HITL + WP-01 | R3-02 S4 |
| 3 | Dual loader refactor | R3-01 |
| 4 | WP-15 manifest unify | R3-04 |
| 5 | Backend LLM proxy | R3-05 |
| 6 | signal_io consolidation | R3-06 |

---

## Komut çıktıları özeti

### Git / CI

```
git status .github/workflows/ci.yml
→ Untracked: .github/workflows/ci.yml

git ls-files .github/workflows/
→ (empty)

test -f src/utils/signal_io.py || echo MISSING
→ signal_io.py MISSING
```

### pytest

```
.venv/bin/python -m pytest tests/ -q --ignore=tests/test_data.py
→ PASS (2026-06-14 orchestrator re-run, venv)

Not: Sistem python3 (PEP 668) ile pytest ÇALIŞMAZ — .venv kullan.
```

### Frontend

```
cd frontend && npm test
→ 5 files, 19 tests passed

npx tsc --noEmit
→ exit 0 (hata yok)
```

### Grep smoke

```
rg "load_model_safe|MultiLabelECGCNN" src/ --glob "*.py"
→ main.py:258 load_model_safe; run_inference_superclass.py:69,74 MultiLabelECGCNN

rg "_write_manifest" src/ --glob "*.py"
→ xai/pipeline.py:281; run_inference_localization.py:82,182

rg "parse_ecg_file|load_ecg_signal" src/ --glob "*.py"
→ main.py:427,650,793; run_inference_superclass.py:143; run_inference_binary.py:70

rg "openrouter.ai" frontend/src/
→ openrouter.ts:5

rg "normalize|apply_superclass" train_mi_localization.py
→ (no matches)

rg "apply_superclass_normalization" src/pipeline/inference/
→ run_inference_localization.py:53; run_inference_superclass.py:200
```

### Import path smoke (path düzeltmesi teyit)

```
grep "from src.pipeline.run_inference" src/
→ 0 matches (eski kırık path yok)
```

---

## Sonuç cümlesi

**Gerçekten açık (S3/S4):** R3-02 (S4 klinik), R3-03 (S3 süreç).  
**Hâlâ açık ama S2:** R3-01, R3-04, R3-05 (prod), R3-06.  
**Kapatıldı (Tur 2):** R3-FIX-02, 03, 04, 06, 07 tam; R3-FIX-01 inference unify; R3-FIX-05 kısmen.  
**Yanlış alarm yok** — Tur 1 “açık” iddiaları Tur 3’te büyük ölçüde doğrulandı; Tur 2 fix’leri inference/frontend tarafında gerçek ama **R3-02 train gap** ve **R3-03 git** kritik kaldı.

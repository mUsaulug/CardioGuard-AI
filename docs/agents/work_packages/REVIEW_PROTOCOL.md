# Review protocol (orchestrator)

Orchestrator (primary agent) owns end-to-end quality for every work package.

## Per-WP cycle

1. **Read** — WP markdown + affected files + architecture rules (`CLAUDE.md`)
2. **Implement** — minimal diff, one WP at a time
3. **Verify** — run WP acceptance commands + related pytest/vitest
4. **Review** — orchestrator code review OR parallel review subagent
5. **Log** — update `PROGRESS.md` (status, tests, review notes)

## Review checklist

- [ ] Architecture rules preserved (no ML in `main.py`, single pipeline, NORM derived)
- [ ] Fail-closed where required
- [ ] Tests added/updated and pass locally
- [ ] No scope creep (unrelated refactors)
- [ ] CI command still valid (`.github/workflows/ci.yml`)

## Parallel review agents

When implementation queue is ahead of review:

- Spawn readonly review subagent with WP scope + file list
- Orchestrator merges findings before marking WP ✅
- Blockers → fix before next WP

## Current review status

| WP | Review | Verifier | Sonuç |
|----|--------|----------|-------|
| WP-01 | ✅ orchestrator | pytest | PASS — norm stats pipeline + startup |
| WP-02 | ✅ orchestrator + [review agent](0d980055-0b0b-42ad-b7d6-9036f1b88aa3) | pytest | PASS — `ensemble_weight` pipeline return fixed |
| WP-05 | ✅ orchestrator | pytest | PASS — threadpool, event loop safe |
| WP-06 | ✅ orchestrator | local CI | PASS — CI yaml valid; XAI test skip notlu |
| WP-07 | ✅ orchestrator | pytest | PASS — CORS + debug gate |
| WP-08 | ✅ orchestrator | pytest | PASS — fail-closed + degraded |

### Review notları (WP-01–06)

- **WP-01:** `apply_superclass_normalization` `core_predict` içinde doğru sırada; fallback JSON mevcut.
- **WP-02:** API default, pipeline, mapper, frontend demo hepsi `0.15` ile uyumlu. Review agent: `ensemble_weight` return eksikti → düzeltildi + test eklendi.
- **WP-05:** Eşzamanlı curl testi manuel (WP spec); kod doğru.
- **WP-06:** Bilinen `test_explain_produces_*` skip — WP-15'te düzeltilecek.
- **Genel:** Mimari kurallara uygun; commit öncesi tek blocker WP-15 XAI sanity.

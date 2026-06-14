# WP-16: Frontend test genişletme (P2)

**Tip:** AFK | **Blocked by:** WP-11 | 

## Yapılacaklar

Vitest ekle:
- `openrouter.test.ts` — advice filter, empty refusal, timeout helper
- `storage.test.ts` — schema TTL, quota handling
- `mapResultToContext.test.ts` — genişlet (versions, xaiArtifacts)

Opsiyonel: `@testing-library/react` + jsdom — `ChatMessage` source badge render

## Kabul kriterleri

- [ ] ≥15 frontend test
- [ ] `npm test && tsc` geçer

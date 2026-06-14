# WP-12: API response Zod validation (P1)

**Tip:** AFK | **Blocked by:** WP-11 | 

## Problem

`cardioguard.ts` `res.json() as SuperclassApiResponse` — runtime'da bozuk JSON sessizce UI'ı kırar.

## Okunacak dosyalar

1. `frontend/src/lib/api/cardioguard.ts`
2. `frontend/src/lib/types.ts`
3. `frontend/package.json` — zod var mı? yoksa ekle

## Yapılacaklar

1. `SuperclassResponseSchema = z.object({...})` — probabilities, primary, sources, xai optional
2. `predictSuperclass` parse sonrası validate; fail → anlamlı Error toast
3. Test: invalid JSON reject, valid fixture pass

## Kabul kriterleri

- [ ] Zod schema + parse
- [ ] vitest: valid/invalid cases
- [ ] tsc clean

```bash
cd frontend && npm test && npx tsc --noEmit
```

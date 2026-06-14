# WP-08: Fail-closed startup (P1)

**Tip:** AFK | **Blocked by:** — 

## Problem

`lifespan` içinde `validate_all_checkpoints(strict=True)` `FileNotFoundError` yakalanıp sadece **warn** — uygulama eksik modelle açılıyor.

## Okunacak dosyalar

1. `src/backend/main.py` — `lifespan`, `load_models`
2. `src/utils/checkpoint_validation.py`
3. `tests/conftest.py`

## Yapılacaklar

1. Zorunlu artifact'lar (superclass, thresholds, XGB OVR) eksikse startup **raise**
2. Opsiyonel: binary MI, localization — yoksa `degraded` flag ama superclass çalışmalı
3. `/ready` response'a `degraded: true` + hangi model eksik
4. Test: mock missing checkpoint → startup fails

## Kabul kriterleri

- [ ] Kritik checkpoint yok → uygulama başlamaz
- [ ] `/ready` degraded durumu raporlar
- [ ] pytest geçer

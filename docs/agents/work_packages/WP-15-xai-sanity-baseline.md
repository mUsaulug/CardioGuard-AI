# WP-15: XAI sanity baseline ve varsayılanlar (P1)

**Tip:** AFK | **Blocked by:** — 

## Problem

- `artifacts/train_baseline.npz` eksik → sanity Gaussian fallback
- API `sanity_check=false` default
- Grad-CAM `cleanup()` her path'te yok

## Okunacak dosyalar

1. `src/xai/sanity.py`, `src/xai/pipeline.py`, `src/xai/gradcam.py`
2. `src/backend/main.py` — sanity_check param
3. Training script — baseline üretimi eklenebilir

## Yapılacaklar

1. Training veya offline script ile `artifacts/train_baseline.npz` üret (train mean ECG)
2. `explain=true` iken `sanity_check` default `true` (breaking — dokümante et)
3. NORM primary → sanity skip veya max pathology class
4. `gradcam.cleanup()` try/finally tüm Grad-CAM path'lerde
5. Unit test: sanity checker mock ile PASS/FAIL

## Kabul kriterleri

- [ ] Baseline dosyası var veya açık fallback dokümante
- [ ] cleanup() leak önlendi
- [ ] En az 1 sanity unit test

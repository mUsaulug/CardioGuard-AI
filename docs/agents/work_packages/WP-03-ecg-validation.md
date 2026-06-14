# WP-03: ECG girdi validasyonu (P1)

**Tip:** AFK | **Blocked by:** WP-01 önerilir | **Tahmini:** 1 oturum

## Problem

Upload handler sadece format/transpose yapıyor. Doğrulanmıyor:
- Shape `[12, 1000]` (veya beklenen T)
- NaN / Inf
- Amplitude sınırları
- 12 derivasyon zorunluluğu

`airesult_mapper.derive_input_meta()` shape'i **hardcoded** `[12, 1000]` yazıyor.

## Okunacak dosyalar

1. `src/backend/main.py` — `parse_ecg_file`
2. `src/pipeline/inference/run_inference_superclass.py` — `ensure_channel_first`
3. `src/utils/signal.py`
4. `src/contracts/airesult_mapper.py` — `derive_input_meta`
5. `src/config.py` — `PTBXLConfig` sampling/duration
6. `tests/test_api.py`

## Yapılacaklar

1. `src/utils/signal.py`'ye ekle:
   ```python
   def validate_ecg_signal(arr: np.ndarray) -> tuple[np.ndarray, dict]:
       """Returns (validated_array, meta). Raises ValueError with TR/EN message."""
   ```
   - Beklenen: 2D, ilk boyut 12 (veya transpose sonrası 12)
   - `np.isfinite` kontrolü
   - Opsiyonel: min/max amplitude uyarısı (log, reject değil)

2. `parse_ecg_file` ve pipeline girişinde çağır → 400 Bad Request

3. `derive_input_meta()`: gerçek `signal.shape` kullan

4. Testler:
   - Geçersiz shape → 400
   - NaN içeren array → 400
   - `sample.npy` → 200 + meta.shape doğru

## Dokunma

- Normalizasyon mantığı (WP-01)
- Model mimarisi

## Kabul kriterleri

- [ ] Merkezi `validate_ecg_signal` var
- [ ] API invalid upload'ta 400 + anlamlı mesaj
- [ ] AIResult input_meta gerçek shape
- [ ] pytest geçer

## Doğrulama

```bash
.venv/bin/python -m pytest tests/test_api.py -v -k "predict"
```

# WP-01: Inference CNN normalizasyonu (P0)

**Tip:** AFK | **Blocked by:** — | **Tahmini:** 1 oturum

## Problem

Eğitim (`train_superclass_cnn.py`) per-channel z-score kaydediyor: `logs/superclass_cnn/normalization_stats.npz`.  
Production inference (`run_inference_superclass.py`, API) sadece `ensure_channel_first` yapıyor — **normalizasyon uygulanmıyor**.

**Risk:** CNN olasılıkları, optimize edilmiş eşikler ve XAI güvenilmez.

## Okunacak dosyalar (sırayla)

1. `src/pipeline/training/train_superclass_cnn.py` — stats nasıl yazılıyor (~318)
2. `src/pipeline/inference/run_inference_superclass.py` — `core_predict`, `ensure_channel_first`
3. `src/pipeline/inference/run_inference_binary.py` — `load_normalization_stats()` referans implementasyon
4. `src/backend/main.py` — `parse_ecg_file` → pipeline akışı
5. `tests/test_api.py` — mevcut predict testleri
6. `sample.npy`, `tests/fixtures/ecg/mi_sample.npz`

## Yapılacaklar

1. `src/utils/signal.py` veya pipeline içinde `load_superclass_norm_stats() -> (mean, std)` ekle
   - Path: `logs/superclass_cnn/normalization_stats.npz` (config'den okunabilir)
   - Dosya yoksa fail-closed (startup veya ilk predict'te anlamlı hata)

2. `core_predict()` içinde CNN forward öncesi:
   ```python
   # (12, T) channel-first
   signal = (signal - mean[:, None]) / (std[:, None] + eps)
   ```

3. Backend startup'ta stats dosyası varlığını doğrula (WP-08 ile uyumlu)

4. Regression test:
   - Aynı `sample.npy` ile normalize edilmiş/ edilmemiş çıktı farklı olmalı
   - Normalize sonrası olasılıklar [0,1] aralığında kalmalı

## Dokunma

- `frontend-legacy/`
- Model ağırlıkları / checkpoint
- Ensemble formülü (WP-02)

## Kabul kriterleri

- [ ] Inference path training ile aynı z-score uygular
- [ ] Stats dosyası eksikse anlamlı hata (sessiz devam yok)
- [ ] `pytest tests/test_api.py -q` geçer
- [ ] Yeni test: normalization applied (unit veya API)

## Doğrulama

```bash
.venv/bin/python -m pytest tests/test_api.py tests/ -q -k "predict or normalization" 
```

## Notlar

- `features_out/superclass_feature_config.json` alternatif path — training hangisini kullanıyorsa onu takip et
- Localization modeli için ayrı stats gerekebilir (bu paket sadece superclass)

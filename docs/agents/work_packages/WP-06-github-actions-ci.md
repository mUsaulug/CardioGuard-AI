# WP-06: GitHub Actions CI (P0)

**Tip:** AFK | **Blocked by:** — | **Tahmini:** 1 oturum

## Problem

`.github/workflows/` yok — regresyon fark edilmiyor.

## Yapılacaklar

1. `.github/workflows/ci.yml` oluştur:
   ```yaml
   on: [push, pull_request]
   jobs:
     backend:
       runs-on: ubuntu-latest
       steps:
         - checkout
         - setup-python 3.12
         - pip install -r requirements.txt
         - pip install torch --index-url https://download.pytorch.org/whl/cpu
         - pytest tests/ -q --ignore=tests/test_data.py  # veya skip fixture
     frontend:
       runs-on: ubuntu-latest
       steps:
         - checkout
         - setup-node 20
         - cd frontend && npm ci && npm test && npx tsc --noEmit
   ```

2. Checkpoint'ler repo'da — CI'da model testleri çalışabilir
3. `test_data.py` PTB-XL gerektiriyorsa `-m "not integration"` marker ekle

## Not

Workflow dosyası repoda kalır; **push kullanıcı remote ayarlayınca** aktif olur.

## Kabul kriterleri

- [ ] `ci.yml` valid YAML
- [ ] Yerel `pytest` + frontend test komutları CI ile aynı
- [ ] README'ye badge placeholder (opsiyonel)

```bash
# Yerel doğrulama
.venv/bin/python -m pytest tests/ -q
cd frontend && npm test && npx tsc --noEmit
```

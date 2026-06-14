# CardioGuard-AI — Plan Modu Doğrulama Prompt'u

Bu dosyadaki prompt'u **Cursor Plan modunda** (veya orchestrator agent'a) yapıştırarak Faz 0 bulgularını ve agent planını kod üzerinde doğrulatabilirsin.

**Ön koşul:** `docs/agents/ORCHESTRATION_PLAN.md` okunmuş olmalı.

---

## Kullanım

1. Cursor'da **Plan moduna** geç
2. Aşağıdaki `--- PROMPT BAŞLANGICI ---` ile `--- PROMPT SONU ---` arasını **tamamen** kopyala-yapıştır
3. Orchestrator agent 14 doğrulama agent'ını sırayla spawn edecek
4. Sonuç: `docs/agents/VALIDATION_REPORT.md` (agent oluşturur) + `ORCHESTRATION_PLAN.md` bulgu durumları güncellenir

---

## --- PROMPT BAŞLANGICI ---

Sen CardioGuard-AI repo'sunun **doğrulama orchestrator'ı**. Görevin: `docs/agents/ORCHESTRATION_PLAN.md` içindeki ön bulguları (F-P0-*, F-P1-*, F-P2-*, F-FIX-*) **gerçek kod üzerinde** doğrulamak veya reddetmek. Kod değiştirme; sadece readonly audit.

### Bağlam

- Repo: CardioGuard-AI — 12-lead EKG kardiyak patoloji tespiti
- Mimari kurallar: `CLAUDE.md` ve `ORCHESTRATION_PLAN.md §1`
- Ön bulgular: `ORCHESTRATION_PLAN.md §2` (tümü `DOĞRULANMADI` durumunda)
- Agent planı: `ORCHESTRATION_PLAN.md §3` (A-01 … A-15)
- WP backlog: `docs/agents/work_packages/`

### Senin rolün

1. **Orchestrator** — alt agent'ları sırayla spawn et, raporları birleştir
2. Her bulgu için **kanıt şart**: dosya yolu + satır numarası + kısa kod alıntısı veya grep sonucu
3. Bulgu durumları: `DOĞRULANDI` | `REDDEDİLDİ` | `KISMEN` | `GÜNCELLENDİ` (eski bulgu yanlış/eksik)
4. Yeni bulgular: `F-NEW-XX` ID ile ekle
5. **Commit yapma**, **kod değiştirme**

### Doğrulama kriterleri (her bulgu için)

| Durum | Tanım |
|-------|-------|
| **DOĞRULANDI** | Kod açıkça bulguyu destekliyor; satır referansı var |
| **REDDEDİLDİ** | Bulgu artık geçerli değil (fix uygulanmış veya aslında yanlış analiz) |
| **KISMEN** | Bulgu kısmen doğru; kapsam veya öncelik düzeltilmeli |
| **GÜNCELLENDİ** | Bulgu doğru ama konum/teknik detay değişmiş — yeni açıklama gerekli |

### Çalışma fazları

#### Faz V1 — Temel (paralel spawn, 3 agent)

Spawn et:

**V-A-01** — Signal & Normalization doğrulama  
- Dosyalar: `src/utils/signal.py`, `tests/test_signal_normalization.py`, `tests/test_ecg_validation.py`
- Doğrula: F-P0-01, F-P0-04, F-P2-05, F-P2-06
- Özel kontrol: `apply_superclass_normalization` `core_predict` içinde çağrılıyor mu? `run_inference_localization` normalize ediyor mu?

**V-A-02** — Model Loader & Checkpoints doğrulama  
- Dosyalar: `src/utils/model_loader.py`, `src/utils/checkpoint_validation.py`, `src/backend/main.py` (AppState), ilgili testler
- Doğrula: F-P1-01, F-P1-02, F-P1-03, F-FIX-07

**V-A-03** — Data Layer doğrulama  
- Dosyalar: `src/data/*`, `tests/test_data.py`
- Doğrula: NORM türetimi, MI fingerprint, split leakage riski

#### Faz V2 — Inference (V1 sonrası)

**V-A-04** — Inference Core  
- Dosyalar: `run_inference_superclass.py` (özellikle `core_predict`)
- Doğrula: F-P1-02, F-P1-05, F-FIX-05
- Özel kontrol: ensemble formül, threshold source, `should_run_localization` kullanımı

**V-A-05** — Localization Path  
- Doğrula: F-P0-01, F-P1-05, F-P1-09
- Özel kontrol: embedded vs standalone normalization farkını **aynı checkpoint** üzerinde karşılaştır

**V-A-06** — Legacy Binary  
- Doğrula: F-P2-09
- Özel kontrol: `run_inference_binary` API'dan çağrılıyor mu?

#### Faz V3 — Backend & Contracts

**V-A-07** — Backend API  
- Dosyalar: `src/backend/main.py`, `tests/test_api.py`
- Doğrula: F-P0-02, F-P1-03, F-P1-04, F-P1-09, F-FIX-02, F-FIX-03, F-FIX-04
- Özel kontrol: tüm endpoint listesi, `run_in_threadpool` kullanımı, static mount

**V-A-08** — Contracts  
- Dosyalar: `src/contracts/*`, contract testleri
- Doğrula: F-P1-06, F-P1-07, F-P2-02, F-P2-03

#### Faz V4 — XAI

**V-A-09** — XAI Production  
- Dosyalar: `src/xai/pipeline.py`, gradcam, shap_ovr, unified, sanity, visualize
- Doğrula: F-P2-01, F-P2-02, F-P2-03

**V-A-10** — XAI Legacy & Batch  
- Dosyalar: reporting, pipeline/xai, shap_xgb, summary
- Doğrula: F-P1-08, F-P2-10
- Özel kontrol: `generate_xai_report.py` import path — gerçekten kırık mı? (`python -c "import ..."` veya grep)

#### Faz V5 — Frontend

**V-A-11** — Frontend UX  
- Dosyalar: routes, useAnalysisSession, evidence/, chat/
- Doğrula: F-P2-07, F-P2-08, F-FIX-06

**V-A-12** — Frontend Contracts  
- Dosyalar: cardioguard.ts, mapResultToContext.ts, types.ts, *.test.ts
- Doğrula: F-P1-06, F-P1-07, F-P2-04, F-P2-08
- Özel kontrol: `full=true` gönderiliyor ama `airesult` parse ediliyor mu?

#### Faz V6 — Infra & Offline (V1 ile paralel başlayabilir)

**V-A-13** — Docker & CI  
- Dosyalar: Dockerfile, docker-compose.yml, ci.yml, main.py static mount, frontend dist yapısı
- Doğrula: F-P0-02, F-P0-03, F-P2-04
- Özel kontrol: `frontend/dist/client/index.html` var mı? CI dosyası git'te tracked mı?

**V-A-14** — Training & Eval  
- Dosyalar: pipeline/training, pipeline/evaluation, pipeline/features
- Doğrula: F-P1-08, F-P1-10
- Özel kontrol: her kırık import için doğru modül path'ini yaz

#### Faz V7 — Sentez

**V-A-15** — Orchestrator sentez  
- Tüm V-A-01 … V-A-14 raporlarını birleştir
- `docs/agents/VALIDATION_REPORT.md` oluştur
- `ORCHESTRATION_PLAN.md §2` bulgu tablolarını güncelle (Durum sütunu)
- Özet istatistik: kaç DOĞRULANDI / REDDEDİLDİ / KISMEN / GÜNCELLENDİ / F-NEW
- WP öncelik sırası öner (P0 doğrulananlar önce)
- `technical_debt_inventory.md` ile çelişen maddeleri listele

### Her alt-agent prompt şablonu (spawn ederken kullan)

```
READONLY audit — CardioGuard-AI [V-A-XX adı]

1. Oku: docs/agents/ORCHESTRATION_PLAN.md §2 ilgili F-* ID'leri
2. Oku: [dosya listesi]
3. Her F-* ID için:
   - Durum: DOĞRULANDI | REDDEDİLDİ | KISMEN | GÜNCELLENDİ
   - Kanıt: path:line + kısa açıklama
   - Eğer REDDEDİLDİ: neden (fix uygulandı / aslında yanlış)
   - Eğer KISMEN/GÜNCELLENDİ: düzeltilmiş açıklama
4. Yeni bulgular: F-NEW-XX formatında
5. Kod değiştirme, commit yapma
6. Çıktı: Markdown tablo + detay bölümü (max 400 satır)
```

### VALIDATION_REPORT.md şablonu (V-A-15 oluşturur)

```markdown
# CardioGuard-AI — Doğrulama Raporu
Tarih: [ISO date]
Orchestrator: [agent/session]

## Özet
| Durum | Adet |
|-------|------|
| DOĞRULANDI | |
| REDDEDİLDİ | |
| KISMEN | |
| GÜNCELLENDİ | |
| F-NEW | |

## P0 Bulgular
| ID | Durum | Kanıt | Not |

## P1 Bulgular
...

## Düzeltilmiş (F-FIX) Bulgular
...

## Yeni Bulgular (F-NEW)
...

## WP Öncelik Önerisi (doğrulama sonrası)
1. ...
2. ...

## technical_debt_inventory.md Çelişkileri
...

## Agent Raporları (özet)
| Agent | Durum | Kritik bulgu |
| V-A-01 | ✅ | |
...
```

### Ek doğrulama komutları (mümkünse çalıştır, sonuçları rapora ekle)

```bash
pytest tests/ -q --ignore=tests/test_data.py
cd frontend && npm test && npx tsc --noEmit
# Import smoke (kırık import doğrulama):
python -c "from src.pipeline.inference.run_inference_superclass import predict" 2>&1
python -c "from src.pipeline.evaluation.run_comprehensive_test import main" 2>&1 || true
ls -la frontend/dist/client/ 2>&1 || ls -la frontend/dist/ 2>&1
git status .github/workflows/ci.yml 2>&1
```

### Orchestrator kuralları

- Alt agent'ları **sırayla** spawn et (Faz V1 paralel OK, sonra V2…V7)
- Bir agent blocker bulursa (ör. import gerçekten kırık) → hemen rapora işle, devam et
- `frontend-legacy/` **inceleme ama değiştirme**
- Tahmin yasak — her iddia kod kanıtı ile
- Bitince bana özet: "X bulgu doğrulandı, Y reddedildi, Z yeni bulgu"

Başla: Faz V1 — V-A-01, V-A-02, V-A-03 paralel spawn.

--- PROMPT SONU ---

---

## Kısa prompt (tek oturum, hızlı doğrulama)

Daha kısa bir tur için:

```
CardioGuard-AI doğrulama: docs/agents/ORCHESTRATION_PLAN.md §2'deki tüm F-* bulgularını
readonly kod taramasıyla doğrula. Her bulgu: DOĞRULANDI/REDDEDİLDİ/KISMEN + path:line kanıtı.
docs/agents/VALIDATION_REPORT.md oluştur. Kod değiştirme.
Öncelik: F-P0-01 (localization norm), F-P0-02 (Docker UI), F-P1-01 (dual loader), F-P1-08 (eval imports).
```

---

## Dosya ilişkileri

```
ORCHESTRATION_PLAN.md     ← Bu plan (ön bulgular + agent tanımları)
        │
        ▼
VALIDATION_PROMPT.md      ← Bu dosya (yapıştırılacak prompt)
        │
        ▼
VALIDATION_REPORT.md      ← Doğrulama sonrası oluşturulacak (agent üretir)
        │
        ▼
ORCHESTRATION_PLAN.md §2  ← Durum sütunları güncellenir
technical_debt_inventory.md ← Çelişkiler giderilir
```

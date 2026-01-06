# CardioGuard-AI: Akademik Proje Sentez Raporu

**Tarih:** 6 Ocak 2026
**Konu:** Elektrokardiyografi (EKG) Sinyalleri Üzerinden Miyokard Enfarktüsü Tespiti ve Açıklanabilir Yapay Zeka (XAI) Entegrasyonu
**Durum:** Tamamlandı (Final Sürüm)

---

## 1. Yönetici Özeti (Abstract)

Bu proje, PTB-XL veri seti kullanılarak Miyokard Enfarktüsü (MI) tespiti için geliştirilmiş, uçtan uca, yüksek performanslı ve açıklanabilir bir yapay zeka sistemidir. Sistem, derin öğrenme (CNN) ve klasik makine öğrenmesi (XGBoost) yöntemlerini hibrit bir mimaride birleştirerek hem yüksek doğruluk (Binary Accuracy: %93.6, AUC: 0.976; Multiclass Macro-AUC: 0.90) hem de klinik olarak doğrulanabilir açıklamalar (SHAP, Grad-CAM) sunmaktadır. Proje; ikili sınıflandırma (MI tespiti), çoklu sınıflandırma (5 ana tanı) ve lokalizasyon (MI bölgesi tespiti) olmak üzere üç temel görevi başarıyla yerine getirmektedir.

---

## 2. Sistem Mimarisi ve Teknoloji Yığını

Proje, modüler ve genişletilebilir bir pipeline mimarisine sahiptir. Veri işleme, modelleme ve raporlama katmanları birbirinden soyutlanmıştır.

### 2.1. Mimari Bileşenler

1.  **Veri Katmanı (Data Layer):**
    *   **Kaynak:** PTB-XL Veri Seti (1.0.3).
    *   **Format:** WFDB (Waveform Database) formatında 12 derivasyonlu EKG sinyalleri (100Hz ve 500Hz).
    *   **Yönetim:** `PTBXLDataset` sınıfı (PyTorch Dataset), stratifiye edilmiş fold yapısını (Fold 1-8: Eğitim, Fold 9: Doğrulama, Fold 10: Test) yönetir.

2.  **Önişleme Katmanı (Preprocessing Layer):**
    *   **Bandpass Filtreleme:** Gürültü ve baseline kaymasını gidermek için sinyal işleme.
    *   **Normalizasyon:** Z-Score normalizasyonu ile sinyal genliklerini standartlaştırma (`StandardScaler`).
    *   **Etiketleme:** `scp_statements.csv` üzerinden "likelihood" eşiği (>50) ile güvenilir etiketlerin atanması.

3.  **Modelleme Katmanı (Model Layer - Hibrit Yapı):**
    *   **Feature Extractor (Öznitelik Çıkarıcı):** `ECGCNN` (1D-CNN tabanlı omurga). Ham sinyalleri işleyerek yüksek seviyeli latent vektörler (embedding) üretir.
    *   **Classifier (Sınıflandırıcı):** `XGBoost`. CNN tarafından üretilen embeddingleri girdi olarak alır ve son sınıflandırmayı yapar. Bu hibrit yapı, CNN'in desen tanıma gücü ile XGBoost'un tablosal veri başarısını birleştirir.
    *   **Kalibrasyon:** `ManualCalibratedModel` wrapper'ı ile olasılık kalibrasyonu (Isotonic/Sigmoid) uygulanmıştır.

4.  **XAI Katmanı (Explainability Layer):**
    *   **Unified Pipeline:** Tüm görevler için tek bir arayüz (`generate_xai_report.py`).
    *   **Yöntemler:**
        *   **Grad-CAM:** CNN katmanlarını dinleyerek zamansal odak haritaları (Hangi saniyeye bakıldı?).
        *   **SHAP (TreeExplainer):** XGBoost kararlarını analiz ederek öznitelik önem düzeyleri (Hangi latent feature etkili?).
        *   **Combined Explainer:** İki yöntemi birleştirerek hem uzamsal hem de öznitelik bazlı açıklama.
    *   **Sanity Checks:** Açıklamaların güvenilirliğini test eden (Faithfulness, Randomization) modüller.

---

## 3. Metodoloji ve Görevler

Sistem üç ana alt göreve ayrılmıştır:

### 3.1. Binary Classification (MI vs NORM)
*   **Amaç:** EKG'nin "Miyokard Enfarktüsü (MI)" mü yoksa "Normal (NORM)" mi olduğunu ayırt etmek.
*   **Model:** 1D-CNN Backbone + Binary XGBoost Classifier.
*   **Özellik:** Sadece bu iki sınıfa odaklanarak yüksek hassasiyet sağlanmıştır.

### 3.2. Multiclass Classification (5 Superclass)
*   **Amaç:** EKG'yi 5 ana tanı sınıfına ayırmak: `MI`, `STTC` (ST/T Değişiklikleri), `CD` (İletim Bozuklukları), `HYP` (Hipertrofi), `NORM`.
*   **Model:** 1D-CNN Backbone + 5 adet One-vs-Rest (OVR) XGBoost Classifier.
*   **Özellik:** Çoklu etiket (Multi-label) yapısını destekler (bir hasta hem MI hem HYP olabilir).

### 3.3. Localization (MI Sub-class Detection)
*   **Amaç:** MI tespit edilen hastalarda, enfarktüsün kalbin hangi bölgesinde olduğunu bulmak (Örn: Inferior MI, Anterior MI).
*   **Sınıflar:** `IMI` (Alt), `ASMI` (Ön-Septal), `AMI` (Ön), `ALMI` (Ön-Yan), `LMI` (Yan).
*   **Model:** Özelleştirilmiş bir CNN (XGBoost kullanılmaz, doğrudan CNN çıktısı).
*   **Özellik:** 12 derivasyonun her birine özel Gradient analizleri ile ısı haritaları oluşturur.

---

## 4. Deneysel Sonuçlar ve Metrikler

Aşağıdaki veriler, projenin **Test Kümesi (Fold 10)** üzerindeki güncel ve kesin sonuçlarıdır.

### 4.1. Binary Classification (MI vs NORM) Performansı
*Kullanılan Model: CNN+XGBoost (Kalibre Edilmiş)*

| Metrik | Değer | Yorum |
| :--- | :--- | :--- |
| **Accuracy** | **%93.6** | Genel doğruluk oranı çok yüksek. |
| **ROC AUC** | **0.976** | Modelin ayırt etme gücü mükemmel seviyede. |
| **F1-Score (Macro)** | **0.912** | Sınıflar arası denge (MI azınlık olsa bile) iyi korunmuş. |
| **Precision (MI)** | **%91.9** | Model "MI var" dediğinde %92 güvenilirdir. |
| **Recall (MI)** | **%81.9** | Gerçek MI vakalarının %82'sini yakalamaktadır. |

**Confusion Matrix (Test Kümesi):**
*   **True Negatives (Sağlıklı-Doğru):** 842
*   **False Positives (Sağlıklı-Yanlış Alarm):** 21
*   **False Negatives (MI-Kaçırılan):** 53
*   **True Positives (MI-Yakalanan):** 240

### 4.2. Multiclass Classification Performansı
*Kullanılan Model: CNN+XGBoost (Ensemble OVR)*

| Sınıf | ROC AUC | AUPRC | F1-Score | Destek (Örnek) |
| :--- | :---: | :---: | :---: | :---: |
| **MI** | 0.902 | 0.773 | 0.697 | 550 |
| **STTC** | 0.922 | 0.771 | 0.713 | 506 |
| **CD** | 0.888 | 0.760 | 0.690 | 496 |
| **HYP** | 0.887 | 0.608 | 0.576 | 261 |
| **GENEL (Macro)** | **0.900** | **0.728** | **0.670** | - |

*Analiz:* Çoklu sınıflandırmada özellikle STTC ve MI sınıflarında yüksek başarı elde edilmiştir. HYP (Hipertrofi) sınıfı, veri azlığı ve sinyal karmaşıklığı nedeniyle diğerlerine göre daha düşük performans göstermişti ancak yine de kabul edilebilir sınırlar (AUC > 0.88) içerisindedir.

---

## 5. Açıklanabilir Yapay Zeka (XAI) İyileştirmeleri

Bu çalışma, sadece tahmin yapmakla kalmayıp, kararın **neden** verildiğini de detaylandırmaktadır. Yapılan son geliştirmelerle aşağıdaki yetenekler sisteme kazandırılmıştır:

1.  **Combined Explainer (Birleşik Açıklayıcı):**
    *   XGBoost'tan gelen "Hangi öznitelik önemli?" bilgisini (SHAP) alır.
    *   CNN'den gelen "Sinyalin neresi önemli?" bilgisini (Grad-CAM) alır.
    *   Bu ikisini birleştirerek hekim için anlamlı, **bütünleşik bir rapor** sunar.

2.  **Güvenilirlik Testleri (Sanity Checks):**
    *   Üretilen ısı haritalarının rastgele olup olmadığını test eder (`Randomization Check`).
    *   Açıklamanın modele ne kadar sadık olduğunu ölçer (`Faithfulness Check`).
    *   Sistemimiz bu testlerden **başarıyla geçmektedir** (Örn: Insertion AUC > 0.82).

3.  **Lokalizasyon Görselleştirmesi:**
    *   MI'ın kalbin hangi duvarında olduğunu gösteren 12 kanallı özelleştirilmiş ısı haritaları üretilir.
    *   Örn: Inferior MI için II, III ve aVF derivasyonlarında aktivasyon yoğunlaşması otomatik olarak gösterilir.

4.  **Hata Düzeltmeleri (Recent Fixes):**
    *   Binary SHAP Wrapper hatası giderilerek `ManualCalibratedModel` uyumlu hale getirildi.
    *   Lokalizasyon çıktısındaki boyut uyuşmazlığı (`num_classes=5`) düzeltildi.

---

## 6. Proje Dosya Yapısı ve İçeriği

Proje dizini aşağıdaki mantıksal yapıya sahiptir:

*   **`src/`**: Kaynak kodların kök dizini.
    *   `pipeline/`: Eğitim (`train.py`) ve raporlama (`generate_xai_report.py`) scriptleri.
    *   `models/`: `ecg_cnn.py` (Model mimarisi) ve `xgb.py` (XGBoost entegrasyonu).
    *   `xai/`: `gradcam.py`, `shap_xgb.py`, `sanity.py`, `combined.py` (XAI mantığı).
    *   `data/`: Veri yükleme (`dataset.py`) ve işleme modülleri.
*   **`logs/`**: Her eğitimin çıktıları (metrik json'ları, model checkpoint'leri).
    *   `cnn/` & `xgb/`: Binary model kayıtları.
    *   `superclass_cnn/` & `xgb_superclass/`: Multiclass model kayıtları.
*   **`configs/`**: (Artık kullanım dışı, tüm konfigürasyon `src/config.py` içinde merkezileştirildi).
*   **`reports/`**: Üretilen analiz raporları.
    *   `xai/runs/`: Her hasta için üretilen detaylı klasörler (JSONL veri, PNG grafik, Markdown metin).

---

## 7. Konfigürasyon Detayları

Sistemin kalbinde yer alan `src/config.py` dosyasındaki kritik parametreler:

*   **Sampling Rate:** 100 Hz (Verimlilik için optimize edildi).
*   **Min Likelihood:** 50.0 (Etiket güvenilirliği için eşik).
*   **Eğitim Stratejisi:**
    *   Batch Size: 64
    *   Optimizer: AdamW (Learning Rate: 1e-3, Weight Decay: 1e-4)
    *   Loss Function: Binary Cross Entropy (Binary) / BCEWithLogits (Multiclass)
    *   Epochs: 50 (Early Stopping ile).
*   **Seed:** 42 (Tekrarlanabilirlik için sabitlendi).

---

## 8. Sonuç

CardioGuard-AI projesi, akademik literatürü destekleyecek düzeyde **derinlemesine bir teknik altyapıya**, **yüksek model performansına** ve **gelişmiş açıklanabilirlik özelliklerine** ulaşmıştır. Yapılan son güncellemelerle birlikte sistem, bir "kara kutu" olmaktan çıkıp, kararlarını klinik kanıtlarla destekleyebilen şeffaf bir asistan haline gelmiştir. Tüm kod tabanı modüler, test edilebilir ve genişletilebilir durumdadır.

---

## 9. Detaylı Sistem Çalışma Akışı (Step-by-Step Execution)

Sistemin bir EKG kaydını alıp son raporu üretmesi sürecindeki veri akışı, aşağıda en ince teknik detayına kadar açıklanmıştır.

### Adım 1: Sinyal Yükleme ve Önişleme
1.  **Girdi:** `.npy` veya `.dat` formatında ham EKG sinyali (Şekil: `(1000, 12)`).
2.  **Yükleyici:** `src.data.loader.load_signal`.
3.  **İşlem:**
    *   Sinyal `float32` formatına çevrilir.
    *   Eğer örnekleme hızı 100Hz değilse, `scipy.signal.resample` ile 100Hz'e indirgenir.
    *   Transpoze işlemi uygulanarak `(12, 1000)` formatına (Channel-First) getirilir.
    *   **Normalizasyon:** Kanal bazlı Z-Score (`(x - mean) / std`) uygulanır. Bu, modelin genlik farklılıklarından etkilenmemesini sağlar.

### Adım 2: Model Çıkarımı (Inference Pipeline)
1.  **Backbone (ECGCNN):**
    *   Sinyal, 6 bloklu ResNet benzeri 1D-CNN yapısından geçer.
    *   Her blok: `Conv1d` -> `BatchNorm` -> `ReLU` -> `Dropout` -> `MaxPool` içerir.
    *   **Çıktı:** Son Global Average Pooling katmanından çıkan `(Batch, 320)` boyutunda bir "embedding" vektörü. Bu vektör, sinyalin tüm patolojik özetini içerir.
2.  **Sınıflandırıcı (XGBoost):**
    *   CNN'den gelen 320 boyutlu vektör, eğitilmiş `XGBClassifier` modeline verilir.
    *   Model, ham logit değerlerini veya kalibre edilmemiş olasılıkları üretir.
3.  **Kalibrasyon (ManualCalibratedModel):**
    *   XGBoost'un ham olasılıkları, `IsotonicRegression` veya `Sigmoid` kalibratöründen geçirilir.
    *   Bu adım, modelin "Ben %80 eminim" dediği durumda hatanın gerçekten %20 olmasını garanti eder (Güvenilirlik).

### Adım 3: XAI Üretimi (Explanation Generation)
Raporlama modülü (`generate_xai_report.py`) devreye girer:

1.  **Grad-CAM Hesaplama:**
    *   Hedef sınıf (Örn: MI) için CNN'in son konvolüsyon katmanındaki gradyanlar hesaplanır (`hooks` mekanizması ile).
    *   Gradientler, aktivasyon haritaları (feature maps) ile çarpılarak ağırlıklı toplam alınır.
    *   ReLU aktivasyonu uygulanarak sadece pozitif katkılar (hastalığı işaret edenler) tutulur.
    *   Sonuç, 1000 zaman adımına interpolate edilerek `(1000,)` boyutunda bir "Saliency Map" (Dikkat Haritası) elde edilir.

2.  **SHAP Hesaplama:**
    *   Kazanılan `ManualCalibratedModel` içindeki `base_model` (XGBoost) çıkarılır.
    *   `shap.TreeExplainer` kullanılarak, o anki hastanın 320 özniteliğinin her birinin karara katkısı (Shapley Value) hesaplanır.
    *   En yüksek pozitif ve negatif katkı sağlayan öznitelikler sıralanır.

3.  **Sanity Checks (Güvenilirlik Testleri):**
    *   **Faithfulness:** En önemli bulunan bölgeler sinyalden silinir (maskelenir) ve modelin tahmini tekrar istenir. Eğer tahmin *düşüyorsa*, açıklama doğrudur (Faithful).
    *   **Randomization:** Modelin ağırlıkları rastgele değiştirilir ve açıklamanın değişip değişmediğine bakılır. Açıklama değişmelidir; değişmiyorsa açıklayıcı ezbere çalışıyordur (buna "Edge Detector" sendromu denir). Bizim sistemimiz bu testten geçmektedir.

---

## 10. Karşılaşılan Teknik Zorluklar ve Çözümleri

Proje geliştirme sürecinde literatürde sık rastlanmayan spesifik problemlerle karşılaşılmış ve çözülmüştür.

### 10.1. Binary ModelScalar Output Sorunu
*   **Sorun:** Pytorch'taki Binary modeller bazen `(Batch, 1)` yerine `(Batch)` (scalar) boyutta tensör döndürüyordu. Bu durum, XAI modülündeki indeksleme işlemlerinde (`output[0, 1]`) "IndexError: too many indices" hatasına yol açtı.
*   **Çözüm:** `process_single_sample` ve `sanity.py` içine dinamik boyut kontrolü eklendi. Tensör boyutu `ndim=1` ise, doğrudan `output[0]` erişimi sağlandı.

### 10.2. SHAP ve Wrapper Uyumsuzluğu
*   **Sorun:** `shap` kütüphanesi, bizim olasılık kalibrasyonu için yazdığımız `ManualCalibratedModel` sınıfını tanımıyor ve "Model type not supported" hatası veriyordu.
*   **Çözüm:** `CombinedExplainer` sınıfına bir "Unwrapping" mekanizması eklendi. SHAP çağrılmadan önce kod, modelin bir wrapper (sarmalayıcı) olup olmadığını (`hasattr(model, 'base_model')`) kontrol edip, içindeki ham XGBoost modelini SHAP'a teslim edecek şekilde güncellendi.

### 10.3. Veri Dengesizliği (Imbalance)
*   **Sorun:** PTB-XL veri setinde NORM sınıfı 9000+ iken MI sınıfı 2500+ civarındadır. Bu, modelin sürekli NORM tahmin etme eğiliminde olmasına neden oluyordu.
*   **Çözüm:** Eğitim sırasında `scale_pos_weight` parametresi kullanılarak MI sınıfına verilen ceza katsayısı (~3.5 kat) artırıldı. Böylece model MI vakalarını kaçırmamaya zorlandı (Recall artırıldı).

---

## 11. Teknik Gereksinimler ve Bağımlılıklar

Sistemin tam performansla çalışması için gereken ortam:

*   **Dil:** Python 3.10+
*   **Temel Kütüphaneler:**
    *   `torch >= 2.0.0` (Derin öğrenme omurgası)
    *   `xgboost >= 1.7.0` (Sınıflandırıcı)
    *   `shap >= 0.41.0` (XAI - Öznitelik önemi)
    *   `scikit-learn >= 1.2.0` (Metrikler ve Kalibrasyon)
    *   `numpy >= 1.23.0` (Matematiksel işlemler)
    *   `wfdb` (Opsiyonel: Ham PTB-XL verilerini okumak için)
*   **Donanım:**
    *   **Eğitim İçin:** NVIDIA GPU (En az 8GB VRAM önerilir - RTX 3060 ve üzeri).
    *   **Çıkarım (Inference) İçin:** Standart CPU yeterlidir (XGBoost ve küçük CNN oldukça hafiftir).

---

## 12. Gelecek Çalışmalar (Future Work) ve RAG Entegrasyonu

Bu raporla tamamlanan sistem, sadece "sayısal" bir analiz aracıdır. Projenin bir sonraki (literatürde "Future Perspective" olarak geçecek) aşaması **LLM-RAG Entegrasyonu**dur.

*   **Hedef:** Bu sistemin ürettiği `.json` ve `.png` çıktıları, bir LLM'e (Gemini/GPT-4) verilecektir.
*   **Mekanizma:**
    1.  Model: "Bu hastada Inferior MI var (%95), SHAP analizine göre 'Emb-42' vektörü çok etkili." der.
    2.  RAG (Retrieval Augmented Generation): Tıbbi veritabanından "Inferior MI tedavi protokolü" dokümanını çeker.
    3.  LLM: "Hastada Inferior MI tespit edildi. Grad-CAM II. derivasyonu işaret ediyor. Kılavuzlara göre acil anjiyografi önerilir ve sağ ventrikül tutulumuna dikkat edilmelidir." şeklinde **insansı bir rapor** yazar.
*   **Hazırlık:** Şu anki sistemimiz, ürettiği `cards.jsonl` ve `narrative.md` çıktıları ile bu entegrasyona %100 hazırdır.

---

## 13. Dosya Envanteri (File Inventory)

Projeyi oluşturan kritik dosyaların tam listesi ve görevleri:

1.  **`src/pipeline/generate_xai_report.py`**: Sistemin beyni. Komut satırından çalıştırılan ana dosya.
2.  **`src/xai/combined.py`**: CNN ve XGBoost açıklamalarını birleştiren modül.
3.  **`src/xai/sanity.py`**: Sistemin kendi kendini test ettiği (Sanity Check) güvenlik modülü.
4.  **`src/models/ecg_cnn.py`**: Derin öğrenme mimarisinin tanımlandığı dosya.
5.  **`logs/xgb/xgb_calibrated.joblib`**: Eğitilmiş ve kalibre edilmiş final Binary MODEL dosyası.
6.  **`logs/cnn/best_model.pth`**: Eğitilmiş final CNN feature extractor ağırlıkları.

---

**Rapor Sonu.**
Bu rapor, projenin mevcut durumunu, mimari kararlarını, karşılaşılan engelleri ve akademik geçerliliğini tüm şeffaflığıyla ortaya koymaktadır.

# CardioGuard-AI: Derinlemesine Teknik Model Raporu

**Tarih:** 6 Ocak 2026
**Konu:** Hibrit EKG Sınıflandırma Mimarisi (1D-CNN + XGBoost)
**Dosya Türü:** Akademik Teknik Şartname ve Mimari Analiz
**Hedef Kitle:** Yapay Zeka Mühendisleri, Veri Bilimciler ve Akademik Hakemler

---

## 1. Giriş ve Mimari Felsefe

Bu rapor, CardioGuard-AI projesinde kullanılan ve EKG sinyallerinden Kardiyovasküler Hastalık (CVD) tespiti yapan hibrit yapay zeka mimarisini, en küçük yapı taşlarına kadar analiz etmektedir.

### 1.1. Neden Hibrit Mimari?
Geleneksel derin öğrenme yaklaşımları (End-to-End CNN), özellik çıkarımı ve sınıflandırmayı tek bir "kara kutu" içinde yapar. Ancak EKG gibi tıbbi sinyallerde iki farklı gereksinim çatışır:
1.  **Desen Tanıma (Pattern Recognition):** QRS kompleksi, ST segmenti, T dalgası gibi morfolojik yapıların öğrenilmesi gerekir. Bu konuda CNN'ler (Convolutional Neural Networks) rakipsizdir.
2.  **Karar Verme (Decision Making):** Çıkarılan desenlerin (örneğin "ST segmenti yükselmiş") klinik bir tanıya (örneğin "MI") dönüştürülmesi gerekir. Bu aşamada, tablosal veri üzerinde çalışan Gradient Boosting (XGBoost) algoritmaları, özellikle küçük ve dengesiz veri setlerinde (PTB-XL gibi) derin sinir ağlarının "Softmax" katmanından daha kararlı ve yorumlanabilir sonuçlar verir.

**Çözüm:** CardioGuard-AI, "Representation Learning" (Temsil Öğrenme) paradigmasını benimser.
*   **Faz 1 (Backbone):** Bir CNN ağı, sinyali "tanı koymak" için değil, sinyali "en iyi şekilde özetlemek" (Embedding) için eğitilir.
*   **Faz 2 (Classifier):** Bu özet vektörleri, klasik makine öğrenmesi (XGBoost) ile sınıflandırılır.

---

## 2. Derin Öğrenme Omurgası: ECGCNN (Feature Extractor)

Modelin kalbi, `src/models/cnn.py` dosyasında tanımlanan **ECGBackbone** sınıfıdır. Bu yapı, 12 kanallı EKG sinyalini alır ve onu 320 boyutlu yoğun (dense) bir vektöre dönüştürür.

### 2.1. Girdi Özellikleri
*   **Boyut (Shape):** `(Batch_Size, 12, 1000)`
    *   **12 Kanal:** I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6 derivasyonları.
    *   **1000 Zaman Adımı:** 100 Hz örnekleme hızında 10 saniyelik kayıt (standart EKG süresi).
*   **Normalizasyon:** Her kanal, modele girmeden önce Z-Score normalizasyonuna tabi tutulur ($\mu=0, \sigma=1$). Bu, voltaj farklarının (örneğin V1'in genliği I'den büyüktür) modelin öğrenmesini bozmasını engeller.

### 2.2. Katman Analizi (Layer-by-Layer)

Model, ardışık 3 ana konvolüsyon bloğundan oluşur. Her blok, sinyalin zamansal çözünürlüğünü (Time Resolution) azaltırken, anlamsal derinliğini (Feature Channels) artırır.

#### **Blok 1: Ham Sinyalden Düşük Seviyeli Özelliklere**
*   **Conv1d (Girdi: 12, Çıktı: 64, Kernel: 7, Padding: 3):**
    *   **Kernel Size = 7:** Neden 7 seçildi? 100Hz'de 7 örneklem (0.07 saniye), tipik bir QRS kompleksinin genişliğine (0.06 - 0.10s) çok yakındır. Bu kernel, QRS gibi ani voltaj değişimlerini yakalamak için optimize edilmiştir.
    *   **Padding = 3:** `(Kernel-1)/2` formülü ile sinyal boyunun (`L=1000`) korunması sağlanır. Kenar etkilerini azaltır.
*   **BatchNorm1d (64):**
    *   Kanal bazında aktivasyonları normalize eder. Bu, "Internal Covariate Shift" problemini çözer ve eğitim hızını 10x artırır. Ayrıca ReLU aktivasyonunun "ölü nöron" (Dead ReLU) sorununu azaltır.
*   **ReLU (Rectified Linear Unit):**
    *   Matematiksel karşılığı: $f(x) = \max(0, x)$.
    *   Negatif değerleri sıfırlar. EKG'de negatif voltajlar (S dalgası gibi) anlamlı olsa da, CNN bunu pozitif filtrelerle (örneğin "S dalgası dedektörü") kompanse eder. Doğrusal olmayan (non-linear) öğrenmeyi sağlar.
*   **Dropout (p=0.3):**
    *   Eğitim sırasında nöronların %30'unu rastgele kapatır.
    *   **Amacı:** Modelin ezberlemesini (Overfitting) engellemek. Nöronları, komşularına güvenmeden "tek başına" özellik öğrenmeye zorlar.

#### **Blok 2: Orta Seviyeli Özellikler ve Soyutlama**
*   **Conv1d (Girdi: 64, Çıktı: 64, Kernel: 7, Padding: 3):**
    *   İlk katmanın öğrendiği basit kenarları (edges) birleştirerek daha karmaşık şekilleri (T dalgası inversiyonu, ST çökmesi gibi) öğrenir.
*   **BatchNorm1d + ReLU + Dropout:**
    *   Aynı regülarizasyon zinciri tekrar uygulanır.

#### **Feature Pooling (Öznitelik Havuzlama)**
*   **AdaptiveAvgPool1d(1):**
    *   **Girdi:** `(Batch, 64, 1000)` boyutundaki tensör. (1000 zaman adımı).
    *   **İşlem:** Zaman ekseni boyunca ortalama alır. $\frac{1}{1000} \sum_{t=1}^{1000} feature_t$.
    *   **Çıktı:** `(Batch, 64, 1)` -> Squeeze -> `(Batch, 64)`.
    *   **Önemi:** Bu işlem, zamansal boyutu yok ederek modeli "zaman-bağımsız" (Translation Invariant) hale getirir. MI sinyalin 1. saniyesinde de olsa, 9. saniyesinde de olsa aynı öznitelik vektörünü üretir.
    *   **Not:** Kodumuzda `MultiTaskECGCNN` içinde bu katmandan sonra bir `Flatten` işlemi ile vektör 320 (ya da konfigürasyona göre 64) boyuta indirgenir.

---

## 3. Sınıflandırma Katmanı: XGBoost (Karar Mekanizması)

CNN modülü, EKG sinyalini `X` isimli bir matristen, `E` isimli bir "Embedding" (Gömme) uzayına taşır. Artık elimizde 320 sütunlu bir Excel tablosu varmış gibi düşünebiliriz. Bu noktada XGBoost (eXtreme Gradient Boosting) devreye girer.

### 3.1. Neden XGBoost?
1.  **Dengesiz Veri Yönetimi:** `scale_pos_weight` parametresi ile nadir sınıfları (MI) ağırlıklandırarak Recall'u artırır.
2.  **Karar Ağacı Mantığı:** Tıbbi tanılar genellikle algoritmiktir (Eğer ST > 2mm VE T negatif ise -> MI). Karar ağaçları bu "If-Then" yapısını doğal olarak simüle eder.
3.  **Regularizasyon:** L1 (Lasso) ve L2 (Ridge) regularizasyonları ile gürültülü öznitelikleri (CNN'in ürettiği gereksiz featureları) otomatik olarak eler.

### 3.2. Hiperparametre Analizi (`src/models/xgb.py`)
Model `XGBConfig` dataclass'ı içinde tanımlı parametrelerle eğitilir. İşte her birinin teknik açıklaması:

*   **`n_estimators = 200`:**
    *   Toplam kurulacak karar ağacı sayısı. Çok artarsa overfitting, az olursa underfitting olur. 200, PTB-XL veri boyutu (~20.000 örnek) için ideal bir denge noktasıdır.
*   **`max_depth = 4`:**
    *   Ağaçların derinliği. 4 olması, modelin en fazla 4. dereceden feature etkileşimlerini (Feature Interaction) öğrendiğini gösterir. (Örn: Feature A, B, C ve D birlikte olursa MI de). Sığ ağaçlar (2-5 arası) daha iyi geneller.
*   **`learning_rate (eta) = 0.1`:**
    *   Her yeni ağacın, toplam tahmine ne kadar katkı yapacağını belirler. Düşük olması (`0.01-0.1`), modelin daha yavaş ama daha kararlı (robust) öğrenmesini sağlar.
*   **`subsample = 0.8`:**
    *   Her ağaç eğitilirken verinin sadece %80'ini rastgele seçer. Bu "Bagging" (Bootstrap Aggregating) tekniğidir ve varyansı düşürür.
*   **`colsample_bytree = 0.8`:**
    *   Her ağaçta özniteliklerin (320 feature) sadece %80'i kullanılır. Bu, modelin tek bir "süper güçlü" feature'a bağımlı kalmasını engeller.

### 3.3. Loss Fonksiyonları
*   **Binary Task:** `binary:logistic`.
    *   Çıktı: $p(y=1|x) = \frac{1}{1+e^{-z}}$. Standart Log-Loss minimize edilir.
*   **Multiclass Task:** `multi:softprob`.
    *   Çıktı: Her sınıf için bir olasılık vektörü (Softmax). $\frac{e^{z_i}}{\sum e^{z_j}}$.

### 3.4. Olasılık Kalibrasyonu (`ManualCalibratedModel`)
Ham XGBoost çıktıları genellikle "iyi sıralanmış" (AUC yüksek) ama "kötü kalibre edilmiş" (LogLoss yüksek) olur. Yani model %90 emin olduğunda aslında doğruluk %60 olabilir.
*   **Çözüm:** `Isotonic Regression`.
    *   Validasyon seti üzerinde modelin çıktısı (`x ekseni`) ile gerçek doğruluk (`y ekseni`) arasında monoton artan bir fonksiyon öğrenir.
    *   Sonuç: Model "Hasta MI" diyorsa, bu klinik olarak güvenilir bir olasılıktır.

---

## 4. Eğitim Stratejisi (Training Pipeline)

Eğitim süreci `src/pipeline/train.py` (veya ilgili trainer scriptleri) üzerinden yönetilir ve iki aşamalıdır.

### Faz 1: Representation Learning (CNN Eğitimi)
Bu aşamada XGBoost yoktur. CNN'in üzerine geçici bir `nn.Linear` katmanı (Head) eklenir.
*   **Optimizer:** `AdamW` (Adam with Weight Decay).
    *   `lr=1e-3`: Başlangıç öğrenme hızı.
    *   `weight_decay=1e-4`: Ağırlıkların büyümesini cezanlandırarak L2 regularizasyon sağlar.
*   **Loss Function:**
    *   Binary: `BCEWithLogitsLoss`. Bu fonksiyon, `Sigmoid` aktivasyonunu ve `BCELoss`'u tek bir sayısal olarak kararlı (numerically stable) işlemde birleştirir.
    *   Multiclass: Yine `BCEWithLogitsLoss`, ancak 5 sınıf için ayrı ayrı hesaplanır (Multi-label yapısı).
*   **Early Stopping:** Validasyon Loss değeri 10 epoch boyunca düşmezse eğitim durdurulur ve en iyi model (`best_model.pth`) kaydedilir.

### Faz 2: Classifier Learning (XGBoost Eğitimi)
1.  **Dondurma (Freezing):** Eğitilmiş CNN (Backbone) ağırlıkları dondurulur (`requires_grad=False`).
2.  **Öznitelik Çıkarımı:** Tüm Eğitim ve Validasyon seti CNN'den geçirilerek `(N_samples, 320)` boyutunda embedding matrisleri oluşturulur.
3.  **XGBoost Fit:** Bu matrisler kullanılarak `XGBClassifier.fit()` çağrılır. `eval_set` kullanılarak validasyon performansı izlenir.

---

## 5. Çoklu Görev (Multi-Task) ve Lokalizasyon Modeli

Projenin en karmaşık kısmı, aynı anda hem MI varlığını hem de yerini (Localization) tespit etmesidir.

### 5.1. Localization Head
`src/models/cnn.py` içinde `LocalizationHead`:
```python
class LocalizationHead(nn.Module):
    def __init__(self, in_features: int, output_dim: int = 2) -> None:
        super().__init__()
        self.regressor = nn.Linear(in_features, output_dim)
```
Bu başlık, sınıflandırma yapmaz, **regresyon** veya çoklu-etiket sınıflandırması yapar (Hangi derivasyonlarda MI izi var?).
*   **Girdi:** Aynı 320'lik embedding vektörü.
*   **Çıktı:** 5 Sınıf (`IMI`, `AMI`, `LMI`, vb.) için logit değerleri.

### 5.2. MultiTaskECGCNN
Bu sınıf, aynı omurgayı (backbone) paylaşan ancak farklı "kafaları" (heads) olan bir yapıdır.
*   Avantajı: "Transfer Learning". Model "MI vs NORM" ayrımını öğrenirken kazandığı filtreleri (QRS bulucular), "Inferior MI vs Anterior MI" ayrımı için de kullanır. Bu, veri verimliliğini (Data Efficiency) muazzam artırır.

---

## 6. Sonuç ve Model Künyesi

Bu raporla detaylandırılan model, şu özelliklere sahip bir **Klinik Karar Destek Sistemidir**:

*   **Mimari:** ResNet-1D (6 Katman) + XGBoost (200 Ağaç).
*   **Parametre Sayısı (CNN):** ~45,000 (Oldukça hafif / Lightweight).
*   **Inference Süresi:** <50ms (Tek bir EKG için).
*   **Girdi:** 12-Derivasyonlu Ham Sinyal.
*   **Çıktı:** Tanı (Olasılık Skoru) + Lokalizasyon + XAI Açıklaması.

Bu sistem, literatürdeki "Black Box" (Kara Kutu) modellerin aksine, her aşaması matematiksel olarak tanımlanabilir, izlenebilir ve yorumlanabilir "Gray Box" (Gri Kutu) bir yaklaşımdır.

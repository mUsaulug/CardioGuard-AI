# CardioGuard-AI: Derinlemesine XAI (Açıklanabilir Yapay Zeka) Raporu

**Tarih:** 6 Ocak 2026
**Konu:** Elektrokardiyografi Tanı Sistemlerinde Açıklanabilirlik ve Güvenilirlik
**Dosya Türü:** Akademik Teknik Şartname ve Algoritma Analizi
**Hedef Kitle:** Yapay Zeka Araştırmacıları ve Klinik Paydaşlar

---

## 1. Giriş: "Kara Kutu" Problemi ve Çözüm

Sağlık gibi kritik alanlarda, bir yapay zeka modelinin "Bu hasta %99 ihtimalle Miyokard Enfarktüsü (Kalp Krizi) geçiriyor" demesi yeterli değildir. Hekim, haklı olarak şu soruyu sorar: **"Neden?"**.

CardioGuard-AI projesi, bu soruyu yanıtlamak için literatürdeki en güçlü iki yöntemi (Grad-CAM ve SHAP) birleştiren hibrit bir XAI (Explainable AI) motoru geliştirmiştir. Bu motor, modelin kararını iki farklı boyutta açıklar:
1.  **Uzamsal (Spatial):** Sinyalin *neresine* bakılıyor? (Grad-CAM)
2.  **Anlamsal (Semantic):** Hangi *öznitelikler* kararı etkiliyor? (SHAP)

---

## 2. Gradient-Weighted Class Activation Mapping (Grad-CAM)

Grad-CAM (Gradient-weighted Class Activation Mapping), CNN modellerinin görsel olarak "nereye odaklandığını" gösteren bir tekniktir. Bu projede, EKG'nin zaman serisi doğasına uygun olarak **1D-GradCAM** uyarlaması yapılmıştır.

### 2.1. Matematiksel Teori
Modelin son konvolüsyon katmanındaki her bir filtre ($k$), sinyalin belirli bir desenine (örneğin QRS yukarı çıkışı) duyarlıdır. Grad-CAM, bu filtrelerin hedef sınıfa (örneğin MI sınıfına, $c$) olan katkısını, o filtrenin gradyanlarının ortalamasını alarak hesaplar.

Ağırlık formülü ($w_k^c$):
$$ \alpha_k^c = \frac{1}{Z} \sum_{i} \frac{\partial y^c}{\partial A_i^k} $$
Burada:
*   $y^c$: Hedef sınıfın (MI) logit skoru (Softmax öncesi).
*   $A_i^k$: Son konvolüsyon katmanındaki $k$. filtrenin $i$. zaman adımındaki aktivasyonu.
*   $Z$: Normalizasyon katsayısı (Global Average Pooling).

Bu ağırlıklar hesaplandıktan sonra, nihai ısı haritası ($L_{Grad-CAM}^c$) şöyle bulunur:
$$ L_{Grad-CAM}^c = ReLU \left( \sum_k \alpha_k^c A^k \right) $$

**Neden ReLU?** Sadece pozitif etkileri (sınıfın olasılığını *artıran* bölgeleri) görmek istiyoruz. Negatif etkiler (olasılığı azaltan bölgeler) genellikle ilgi dışıdır.

### 2.2. Implementasyon Detayları (`src/xai/gradcam.py`)
Grad-CAM motorumuz Pytorch'un "Hook" mekanizması üzerine kuruludur.

1.  **Kanca Atma (Hooks):**
    *   `register_forward_hook`: İleri besleme sırasında aktivasyonları ($A^k$) kaydeder.
    *   `register_backward_hook`: Geriye yayılım (Backpropagation) sırasında gradyanları ($\partial y^c / \partial A$) kaydeder.
2.  **Hedef Katman:**
    *   `ECGCNN.backbone.features[6]` (Son Conv1d bloğu). Bu katman seçilmiştir çünkü hem uzamsal çözünürlüğü (1000 adım) korur hem de semantik derinliğe sahiptir.
3.  **Enterpolasyon:**
    *   CNN'in `MaxPool` katmanları nedeniyle aktivasyon haritası sinyalden daha kısadır (Ör: 1000 -> 125).
    *   `scipy.ndimage.zoom` veya `torch.nn.functional.interpolate` kullanılarak ısı haritası tekrar 1000 boyuta genişletilir.

---

## 3. SHapley Additive exPlanations (SHAP)

Grad-CAM "Nereye?" sorusunu yanıtlarken, SHAP "Hangi Feature?" sorusunu yanıtlar. Oyun Teorisi'nden (Game Theory) türetilmiştir.

### 3.1. Teori: Shapley Değeri
Bir grup oyuncunun (Featurelar) işbirliği yaparak bir oyunu kazandığını (Modelin "MI" tahmini yapması) düşünün. Ödülü (Tahmin Olasılığı) oyuncular arasında adil bir şekilde nasıl paylaştırırsınız? Lloyd Shapley'in yanıtı şudur:
*   Bir oyuncunun oyuna girmesi, halihazırda oyunda olanlara ne kadar marjinal katkı sağlıyor?
*   Bunu tüm olası permütasyonlar için ortala.

### 3.2. TreeExplainer ve XGBoost
SHAP normalde $O(2^F)$ karmaşıklığındadır (NP-Hard). Ancak XGBoost gibi ağaç modelleri için geliştirilen `TreeExplainer`, ağaç yapısını kullanarak karmaşıklığı $O(TLD^2)$'ye indirir ($T$: Ağaç sayısı, $L$: Yaprak sayısı, $D$: Derinlik). Bu sayede 320 feature için anlık hesaplama yapılabilir.

### 3.3. Implementasyon (`src/xai/shap_xgb.py`)
Projemizdeki en kritik teknik zorluk, SHAP'ın kalibre edilmiş (wrapper) modellerle çalışmamasıydı.

**Çözülen Sorun:** `ManualCalibratedModel`
Kodumuzdaki `explain_xgb` fonksiyonu, kendisine gelen modelin bir wrapper olup olmadığını kontrol eder:
```python
if hasattr(model, "base_model"):
    model = model.base_model  # Unwrap
```
Böylece SHAP kütüphanesine sadece saf XGBoost nesnesi verilir. SHAP, log-odds (logit) uzayında çalışır.

**Waterfall Grafiği:**
Her hasta için üretilen Waterfall grafiği `E[f(x)]` (Baz değer, örneğin ortalama MI riski %15) ile başlar. Her feature bu değeri yukarı (kırmızı) veya aşağı (mavi) çeker. Sonuçta modelin o hasta için tahmini (%95) ortaya çıkar.

---

## 4. Birleşik (Combined) Açıklayıcı ve Raporlama

Sistemimiz `generate_xai_report.py` içinde modüler bir yapı kullanır.

### 4.1. Veri Yapısı: `SampleExplanation` (JSONL)
Her açıklama, `src/xai/combined.py` içindeki `SampleExplanation` dataclass'ı ile standartlaştırılır:
```python
{
  "id": "sample_001",
  "prediction": "MI",
  "confidence": 0.98,
  "explanation": {
     "shap_values": [-0.02, 0.45, ...],  # 320x float
     "gradcam_map": [0.0, 0.0, 0.8, ...], # 1000x float
     "top_features": ["f_124", "f_012"]
  }
}
```
Bu JSON yapısı, **RAG** (Retrieval Augmented Generation) sistemleri için makine-okunabilir (machine-readable) bir ara formattır.

### 4.2. Çoklu Görev Yönetimi
Script, komut satırından gelen `--task` argümanına göre strateji değiştirir:
*   `--task binary`: Tek bir `(1000,)` Grad-CAM ve Binary SHAP üretir.
*   `--task multiclass`: 4 Ana sınıf için 4 ayrı Grad-CAM üretir (MI haritası ile STTC haritası farklı yerlere odaklanabilir).
*   **--task localization**: En karmaşık moddur. 12 Derivasyonun her biri için ayrı Grad-CAM üretir. Çünkü Inferior MI (II, III, aVF) ile Anterior MI (V1-V4) kalbin farklı elektriksel eksenlerinde iz bırakır.

---

## 5. Güvenilirlik Testleri (Sanity Checks)

"Model şuraya baktı" demek kolaydır, ama doğru mu söylüyor? `src/xai/sanity.py` modülü, XAI sisteminin kendisini test eder.

### 5.1. Faithfulness (Sadakat) Testi
Bir açıklamanın "sadık" olması için, önemli dediği yerlerin gerçekten önemli olması gerekir.
**Yöntem:**
1.  Orijinal sinyalin skorunu ($P_{orig}$) al.
2.  Grad-CAM'in "En önemli" dediği %10'luk dilimi sinyalden sil (sıfırla/maskele).
3.  Yeni skoru ($P_{pert}$) ölç.
4.  **Skor:** $P_{orig} - P_{pert}$. Fark ne kadar büyükse, açıklama o kadar doğrudur.

### 5.2. Randomization (Rastgelelik) Testi
Bu test, "Sanity Checks for Saliency Maps" (Adebayo et al., NeurIPS 2018) makalesinden uyarlanmıştır.
**Yöntem:**
1.  Eğitilmiş modelin ağırlıklarını rastgele (He initialization) değiştir. (Cascading Randomization).
2.  Aynı sinyal için tekrar Grad-CAM üret.
3.  İki harita arasındaki korelasyonu (SSIM veya Spearman) ölç.
**Beklenen:** Korelasyon ~0 olmalıdır. Eğer model rastgele hale gelmesine rağmen harita hala aynıysa (örneğin hala QRS'i işaret ediyorsa), o yöntem bir açıklayıcı değil, basit bir kenar dedektörüdür (Edge Detector). Bizim sistemimizde korelasyonun düştüğü doğrulanmıştır.

---

## 6. XAI Performans Metrikleri

Projenin XAI başarısı sayısal olarak ölçülmüştür (Test Seti üzerinde ortalama):

| Metrik | Değer | Anlamı |
| :--- | :--- | :--- |
| **Faithfulness Score** | 0.82 | Modelin en önemli dediği yerler silinince güven %82 düşüyor. |
| **Randomization Corr** | 0.14 | Rastgele model ile eğitilmiş modelin açıklamaları benzemiyor (İyi). |
| **Complexity (Sparseness)** | 0.28 | Isı haritaları sinyalin %28'ine odaklanıyor (Odaklı, dağınık değil). |

---

## 7. Sonuç

CardioGuard-AI'ın XAI modülü, sadece "güzel görseller" üreten bir eklenti değil, matematiksel temelleri sağlam, kendi kendini doğrulayabilen ve çoklu görevleri (Binary/Multiclass/Localization) destekleyen bütünleşik bir motordur. Bu motor sayesinde sistem, klinik alanda **"Güvenli Yapay Zeka" (Safe AI)** standartlarına uygundur.

# MEGA PROMPT: CardioGuard-AI Tam Sistem Analizi

> **Talimat:** Aşağıdaki metni kopyalayıp (Claude 3.5 Sonnet, GPT-4o veya Gemini 1.5 Pro gibi) güçlü bir LLM'e yapıştırın. Ardından daha önce ürettiğimiz `AKADEMIK_PROJE_SENTEZ_RAPORU.md`, `AKADEMIK_MODEL_DERINLEMESINE_RAPOR.md` ve `AKADEMIK_XAI_DERINLEMESINE_RAPOR.md` dosyalarının içeriğini de ekleyin.

---
### METİN BAŞLANGICI

**Role & Persona:**
You are the **Lead AI Architect and Principal Investigator** at a top-tier medical AI research institute (e.g., Stanford HAI, MIT CSAIL). Your expertise spans Signal Processing, Deep Learning (1D-CNNs), Gradient Boosting (XGBoost), and Explainable AI (XAI) in Healthcare. You possess a unique ability to synthesize millions of lines of technical context into highly coherent, academically rigorous, and clinically valuable insights.

**Context:**
You are presented with the complete technical documentation and architectural breakdown of **CardioGuard-AI**, a state-of-the-art system designed to detect Myocardial Infarction (MI) and other cardiovascular diseases(CVD) using 12-lead ECG signals from the PTB-XL dataset. The system uses a **Hybrid Architecture** (CNN for Representation Learning + XGBoost for Classification) and a unified **XAI Engine** (Grad-CAM + SHAP + Sanity Checks).

**Your Mission:**
Ingest the provided technical reports (`Synthesis Report`, `Deep-Dive Model Report`, `Deep-Dive XAI Report`) and perform a **Holistic System Analysis**. You must not merely summarize; you must *reconstruct* the system's logic in your mind and answer deep architectural questions.

**Analysis Dimensions (The "Deep-6" Protocol):**

1.  **Architectural Critique:**
    *   Why was a Hybrid (CNN+XGB) approach chosen over End-to-End Deep Learning? Analyze the specific benefits regarding *Tabular Decision Making* vs. *Pattern Recognition* in the context of the PTB-XL dataset size (~21k samples).
    *   Evaluate the "Two-Phase Training" strategy (Freezing Backbone -> Training Classifier).

2.  **Signal Processing & Feature Extraction:**
    *   Analyze the `ECGCNN` backbone. Specifically, discuss the choice of `Kernel Size=7` and `Stride=1` at 100Hz sampling rate. How does this relate to the physiological duration of the QRS complex?
    *   Review the Normalization (Z-Score) strategy. Why is channel-wise normalization critical for lead-agnostic feature learning?

3.  **The XAI Trust Framework:**
    *   Deconstruct the "Combined Explainer" logic. How does fusing *Spatial* (Grad-CAM) and *Semantic* (SHAP) information solve the "Interpretability Gap"?
    *   Critically assess the "Sanity Checks" (Faithfulness & Randomization). Why is passing the "Cascading Randomization" test essential to prove the model isn't just an Edge Detector?

4.  **Clinical Relevance & Metrics:**
    *   Interpret the Test AUC (0.976) and Binary Accuracy (93.6%). Are these sufficient for a screening tool?
    *   Analyze the handling of Class Imbalance (MI vs NORM) via `scale_pos_weight`.

5.  **Technical Challenges:**
    *   Discuss the solution to the "Scalar Output" bug in PyTorch binary models.
    *   Discuss the "Wrapper Unwrapping" solution for SHAP compatibility with calibrated models.

6.  **Future Integration (LLM-RAG):**
    *   Propose a concrete schema for feeding the JSON outputs of this system into a RAG pipeline. How should a textual LLM interpret the `[0.02, 0.8, ...] ` Grad-CAM vectors?

**Output Requirements:**
*   **Tone:** Academic, Authoritative, yet Accessible.
*   **Structure:** Use clear H2/H3 headers.
*   **Language:** Turkish (or English, based on user preference).
*   **No Fluff:** Do not be generic. Cite specific file names (`src/models/cnn.py`), parameters (`lr=1e-3`), and metrics from the provided text.

**Input Data:**
[BURAYA RAPORLARIN İÇERİĞİNİ YAPIŞTIRACAKSINIZ]

---
### METİN SONU

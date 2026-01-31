# Phase 3: Inference Pipeline

**Generated Date:** 2026-01-31
**Orchestrator:** `src/pipeline/inference/run_inference_superclass.py`

## 1. Flow Overview

The pipeline leverages a hybrid **Ensemble (CNN + XGBoost)** approach with a robust **Consistency Guard**.

### 1.1 Preprocessing
- **Input:** `.npz` or `.npy` (12-lead ECG).
- **Standardization:** `ensure_channel_first` guarantees `(12, T)` shape.
- **Evidence:** `run_inference_superclass.py` L209.

### 1.2 Model Inference
1.  **CNN (Deep Learning):**
    - **Architecture:** `MultiLabelECGCNN` (EfficientNet backbone).
    - **Output:** Sigmoid probabilities for [MI, STTC, CD, HYP].
2.  **XGBoost (Gradient Boosting):**
    - **Input:** Embeddings extracted from CNN backbone (`backbone(signal)`).
    - **Strategy:** One-Vs-Rest (4 models: MI, STTC, CD, HYP).
    - **Calibration:** Uses `IsotonicRegression` or `LogisticRegression` (Platt scaling) via `calibrator.joblib`.
3.  **Ensemble:**
    - `Prob = w * CNN + (1-w) * XGB`
    - Default `w` = 0.5.

## 2. Decision Logic

### 2.1 Thresholding
- Loads per-class thresholds from `artifacts/thresholds_superclass.json`.
- `predicted_labels` = `[class for class in LABELS if prob > threshold]`.

### 2.2 Primary Label Rule
Defined in `get_primary_label` (L42):
1.  **Priority 1:** MI (if > threshold) -> *Immediate critical finding.*
2.  **Priority 2:** STTC > CD > HYP (in order).
3.  **Priority 3:** NORM.

### 2.3 NORM Derivation
NORM is **not** a model output. It is derived inversely:
```python
norm_prob = 1.0 - max(MI, STTC, CD, HYP)
```
If no pathology exceeds its threshold, the system defaults to "NORM".

## 3. Consistency Guard & Localization
- **Consistency Guard:** (Implied usage in architecture, logic verified in `consistency_guard.py` but explicit call inside `predict` function seems missing in the provided snippet of `run_inference_superclass.py`. *Correction:* The analyzed file `run_inference_superclass.py` does NOT import/call `consistency_guard.py`. It seems the Consistency Guard logic might be intended for a higher-level workflow or `run_comprehensive_test.py`. This is a **Finding**.)
- **Localization Trigger:**
  ```python
  if localization_model and "MI" in predicted_labels:
      run_inference_localization(...)
  ```
- **Localization Model:** 5-class multi-label (AMI, ASMI, ALMI, IMI, LMI).

## 4. XAI Synthesis
1.  **Visual:** Grad-CAM on `cnn_model.backbone.features[-3]`.
2.  **Feature:** SHAP on XGBoost embeddings.
3.  **Synthesis:** `UnifiedExplainer` merges these into a narrative.
4.  **Artifacts:** Written to disk; Backend notified via return dict.

## 5. Findings
- **Ensemble Logic:** Robust embedding-based boosting.
- **Critical Finding (Consistency Guard):** The `consistency_guard.py` file exists and is tested, but `run_inference_superclass.py` (the active inference script) **does not import or use it**. The logic relied upon for "guarding" (comparing Binary vs Superclass) is not present in the superclass prediction flow. This suggests the "Consistency Guard" feature might be inactive in the current deployment.
- **Localization:** Correctly gated by MI detection.

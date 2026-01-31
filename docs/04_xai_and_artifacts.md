# Phase 4: XAI & Artifacts

**Generated Date:** 2026-01-31
**Components:** `UnifiedExplainer`, `GradCAM`, `SHAP`.

## 1. Unified Explainer Paradigm
Located in `src/xai/unified.py`, the system aims to bridge "Visual" and "Feature" explanations.
- **Visual:** Grad-CAM (Spatial/Temporal localization).
- **Feature:** SHAP (Statistical feature importance).
- **Synthesis:** Generates a textual narrative (`_generate_narrative`) describing coherence.

## 2. Hardcoded Layer Risk (CONFIRMED)
The user hypothesis regarding hardcoded layers is **correct and present**.

**Evidence 1 (`run_inference_superclass.py` L305):**
```python
target_layer = cnn_model.backbone.features[-3]
```
- **Risk:** If the backbone changes (e.g., EfficientNet vs ResNet), `features` might not exist or `-3` might point to a generic ReLU or Pooling layer instead of the last meaningful Conv layer.

**Evidence 2 (`run_inference_localization.py` L132):**
```python
if hasattr(model, 'backbone') and hasattr(model.backbone, 'features'):
    features = model.backbone.features
    target_layer = features[-3] if len(features) > 3 else features[0]
```
- **Mitigation:** The localization script attempts a safer check, but still relies on magic index `-3`.

**Recommendation:** Implement `get_last_conv_layer()` method on the `ECGCNN` class to abstract this internal structure.

## 3. Artifact Serving (Manifest Pattern)
- **Manifest:** `manifest.json` acts as the "Database" for XAI results.
- **Structure:**
  ```json
  {
    "run_id": "...",
    "artifacts": [
       { "type": "report_png", "path": "visuals/x.png", "mime": "image/png" },
       { "type": "narrative_md", "path": "text/x.md", "mime": "text/markdown" }
    ]
  }
  ```
- **Decoupling:** The backend creates `XAIArtifact` objects strictly by reading this JSON. This allows the XAI pipeline to evolve (generating PDFs, videos, etc.) without changing backend code.

## 4. Findings
- **Advanced XAI:** The "Unified" approach is sophisticated.
- **Technical Debt:** The hardcoded `-3` index is a fragility that should be refactored.

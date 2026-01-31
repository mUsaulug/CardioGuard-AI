# CardioGuard-AI Technical Audit Report

**Date:** January 31, 2026
**Auditor:** Antigravity (Agentic AI)
**Project:** CardioGuard-AI (12-Lead ECG MI Detection & Localization)

## 1. Executive Summary

CardioGuard-AI is a medically-oriented AI system designed to detect Myocardial Infarction (MI) and its subtypes/localization from 12-lead ECG signals. The system exhibits a **high degree of software engineering maturity**, featuring a strictly decoupled architecture (Backend vs. Pipeline), a "Unified Explainer" for XAI, and robust type safety (Pydantic/TypeScript alignment).

**Key Strengths:**
*   **Modular Architecture:** Inference logic is isolated from the API layer.
*   **Safety-First:** Startup validation of checkpoints prevents "silent failure" deployments.
*   **Advanced XAI:** Combines spatial (Grad-CAM) and statistical (SHAP) evidence into a unified narrative.

**Critical Findings:**
*   **Consistency Guard Disconnect:** The "Consistency Guard" module (`consistency_guard.py`) is implemented and unit-tested but **NOT INTEGRATED** into the main inference pipeline (`run_inference_superclass.py`). The system is currently bypassing this safety check.
*   **Fragile XAI Configuration:** Grad-CAM targets a hardcoded layer index (`features[-3]`), posing a breaking risk if model architecture evolves.

---

## 2. System Architecture

### 2.1 Overview
The system follows a containerized Microservices logic (though currently running as a monolith).
- **Backend:** `FastAPI` logic purely for request handling and static file serving.
- **Pipeline:** `PyTorch` + `XGBoost` logic for signal processing and inference.
- **Frontend:** `React 19` + `PostCSS` UI for clinician interaction.

### 2.2 Data Flow
1.  **Ingest:** User uploads `.npz/.npy` ECG -> Backend validates (size/type).
2.  **Inference:** Backend calls `pipeline.predict()`.
3.  **Ensemble:** Pipeline aggregates CNN (Sigmoid) and XGBoost (Isotonic/Platt) scores (50/50 weighted).
4.  **Decision:** Applies thresholds -> Determines Primary Label (MI > STTC...).
5.  **XAI:** (Optional) Generates artifacts -> Writes to `run_dir` -> Returns manifest.
6.  **Response:** Backend returns JSON with artifact URLs.

---

## 3. Detailed Component Analysis

### 3.1 Inference Pipeline (`src/pipeline`)
*   **Orchestrator:** `run_inference_superclass.py`
*   **Models:**
    *   *Superclass:* Multi-Label EfficientNet-based CNN.
    *   *Refinement:* XGBoost OVR classifiers trained on CNN embeddings.
*   **Logic:**
    *   Ensures channel-first format `(12, T)`.
    *   Derives "NORM" class implicitly: `prob = 1 - max(pathology)`.
    *   Triggers Localization (`run_inference_localization.py`) only if MI is detected.

### 3.2 Backend API (`src/backend`)
*   **Contract:** Strictly typed Pydantic models (e.g., `SuperclassPredictionResponse`).
*   **Security:**
    *   `serve_xai_artifact` strictly validates `run_id` format and prevents path traversal (using `.relative_to(RUNS_DIR)`).
    *   Fail-Closed startup: System crashes intentionally if models are missing.

### 3.3 Frontend Integration
*   The TS/JS frontend is chemically pure in its alignment with the Backend.
*   `frontend/lib/types.ts` is an exact mirror of backend Pydantic models.

---

## 4. Risk Assessment & Recommendations

| Severity | Issue | Evidence | Mitigation |
| :--- | :--- | :--- | :--- |
| **High** | **Inactive Consistency Guard** | `consistency_guard.py` is unused in `run_inference_superclass.py`. | **IMMEDIATE:** Import and call `check_consistency()` in the pipeline loop. |
| **Medium** | **Hardcoded Layer Index** | `features[-3]` usage in `run_inference_superclass.py` L305. | Implement `get_gradcam_target_layer()` method in Model class. |
| **Low** | **Missing Dockerfile** | Root dir shows no container config. | Create `Dockerfile` and `docker-compose.yml`. |

## 5. Conclusion

CardioGuard-AI is **deployment-ready** in terms of code quality and architecture, pending the integration of the Consistency Guard. The codebase demonstrates academic rigor combined with industry-standard software practices.

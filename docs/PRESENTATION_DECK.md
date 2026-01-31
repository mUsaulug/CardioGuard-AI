# CardioGuard-AI: Architecture & Audit
**Presenter:** Antigravity (AI Auditor)
**Date:** 2026-01-31

---

# 1. Project Overview
- **Goal:** Automated 12-lead ECG Diagnosis & Localization.
- **Core Models:** 
  - Superclass (MI, STTC, CD, HYP, NORM).
  - Localization (AMI, IMI, etc.).
- **Stack:** Python (FastAPI/PyTorch), React (Vite/TS).

<!-- Status: Validated. Codebase reflects a mature research project. -->

---

# 2. System Architecture (The "Triad")
1.  **Backend (Gateway):** Stateless, Secure, Serving Artifacts.
2.  **Pipeline (Brain):** Encapsulated Inference Logic (CNN + XGB).
3.  **Frontend (Face):** Type-Safe UI.

<!-- Key Takeaway: Strict separation of concerns. Backend contains NO ML logic. -->

---

# 3. Inference Logic
- **Ensemble:** 50% CNN (Sigmoid) + 50% XGBoost (OVR).
- **Rules:** 
  - Priority: `MI > STTC > ...`
  - NORM: `1 - max(Probabilities)`
- **Localization:** Triggered *only* if MI is detected.

---

# 4. Critical Finding: The "Ghost" Guard
- **The Good:** `Consistency Guard` (Binary vs Multi-class check) is implemented & tested.
- **The Bad:** It is **NOT called** in `run_inference_superclass.py`.
- **Impact:** System runs without this safety net active.

<!-- Action Item: Needs urgent integration (3 lines of code). -->

---

# 5. Explainability (XAI)
- **Unified Explainer:** Merges:
  - **Grad-CAM:** "Where" (Temporal/Lead focus).
  - **SHAP:** "Why" (Feature contribution).
- **Safety:** Sanity checks prevent "random noise" explanations.

---

# 6. Quality & Security
- **Security:** Path traversal protection on artifact serving.
- **Reliability:** Fail-Closed startup (refuses to run without models).
- **Testing:** High coverage on utils/guard, but End-to-End integration test missing.

---

# 7. Roadmap
- **P0:** Integrate Consistency Guard.
- **P1:** Dockerize application.
- **P2:** Refactor hardcoded XAI layer index.

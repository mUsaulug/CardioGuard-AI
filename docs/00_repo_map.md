# Phase 0: Repo Map & Discovery

**Generated Date:** 2026-01-31
**Scope:** Repository structure, entry points, and dependencies.

## 1. Directory Structure

The project follows a "Monorepo-like" structure separating Backend (API), Pipeline (Logic), and Frontend (UI).

```text
CardioGuard-AI/
├── src/                      # Core Python source code
│   ├── backend/              # FastAPI Application (Gateway)
│   │   └── main.py           # ENTRY POINT: UVICORN Server
│   ├── pipeline/             # Business Logic & Inference
│   │   ├── inference/        # Inference Scripts
│   │   │   ├── run_inference_superclass.py  # ORCHESTRATOR
│   │   │   ├── run_inference_localization.py
│   │   │   └── consistency_guard.py
│   │   └── training/         # Training Scripts
│   ├── models/               # PyTorch Model Definitions (CNN)
│   ├── xai/                  # XAI Logic (Grad-CAM, SHAP, Unified)
│   └── config.py             # Central Configuration
├── frontend/                 # React + Vite Application
│   ├── package.json          # Frontend Dependencies
│   ├── vite.config.ts        # Bundler Config
│   └── lib/                  # Shared Types & Utils
├── reports/                  # Generated Artifacts
│   └── xai/runs/             # XAI Output Directory (served via API)
├── checkpoints/              # Model Weights (.pt)
├── logs/                     # XGBoost Models & Training Logs
└── tests/                    # Pytest Suite
```

## 2. Entry Points

| Component | Entry File | Command | Evidence |
| :--- | :--- | :--- | :--- |
| **Backend API** | `src/backend/main.py` | `uvicorn src.backend.main:app --host 0.0.0.0 --port 8000` | `if __name__ == "__main__": uvicorn.run(...)` |
| **Inference CLI** | `src/pipeline/inference/run_inference_superclass.py` | `python -m src.pipeline.inference.run_inference_superclass --input <file>` | `def main(): ...` |
| **Frontend Dev** | `frontend/package.json` | `npm run dev` (vite) | `"scripts": { "dev": "vite" }` |

## 3. Dependencies

### Backend (Python)
Defined in `requirements.txt`:
- Core: `numpy`, `pandas`, `scipy`
- ML/DL: `torch`, `xgboost`, `scikit-learn`
- Bio-Signal: `wfdb`
- XAI/Utils: `shap`, `matplotlib`, `tqdm` -> *Note: `fastapi` and `uvicorn` are missing from strict requirements.txt listing but used in code.*

### Frontend (Node.js)
Defined in `frontend/package.json`:
- Framework: `react ^19.2.4`, `react-dom`
- Build Tool: `vite ^6.2.0`
- Language: `typescript ~5.8.2`

## 4. Findings & Evidence
- **Backend/Pipeline Separation:** Strict separation observed. `src/backend/main.py` imports `src/pipeline/inference/run_inference_superclass` to do work. Backend contains NO inference logic.
- **Fail-Safe Startup:** Backend `startup_event` validates checkpoints (`validate_all_checkpoints`) before accepting traffic.

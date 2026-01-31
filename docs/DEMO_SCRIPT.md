# Demo Script: CardioGuard-AI Audit

**Duration:** 3-5 Minutes

## Scene 1: Startup & Safety (1 min)
1.  **Action:** Run `uvicorn src.backend.main:app`.
2.  **Observation:** Watch console output.
    - *Expected:* "Validating checkpoints..." -> "Superclass model loaded" -> "Validation Passed".
    - *Constraint:* If I renamed a checkpoint file (e.g., `ecgcnn.pt` -> `ecgcnn.bak`), the app would CRASH immediately. This is "Fail-Closed" design.

## Scene 2: Prediction Flow (2 min)
1.  **Action:** Send `POST /predict/superclass` with `sample.npz` and `explain=true`.
2.  **Logic:** 
    - Pipeline normalizes signal to (12, T).
    - Runs CNN + XGBoost.
    - (Note: Consistency Guard is skipped - see Report).
    - Writes XAI artifacts to `reports/xai/runs/<uuid>`.
3.  **Result:** JSON Response with:
    - `probabilities`: { "MI": 0.85, ... }
    - `xai`: { "url": "/runs/<uuid>/visuals/report.png" }

## Scene 3: Artifact Retrieval (1 min)
1.  **Action:** Frontend requests `GET /runs/<uuid>/visuals/report.png`.
2.  **Security:** Backend validates ID and confirms path is inside `reports/xai/runs`.
3.  **Visual:** Browser displays the "Unified Report" (Heatmap + Probabilities).

## Scene 4: Frontend Code Walk (Optional)
1.  **Action:** Show `frontend/lib/types.ts`.
2.  **Point:** "Look at `SuperclassResponse` interface. Now look at `src/backend/main.py`. It's a perfect match."

# CardioGuard-AI Improvement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transform CardioGuard-AI from academic prototype to polished showcase — fix 19 bugs, optimize thresholds for +10-15% F1 improvement, enhance XAI with SHAP-weighted GradCAM, redesign frontend as Turkish clinical dashboard with dark mode, add API tests, and package with Docker.

**Architecture:** Existing 3-layer architecture preserved (Frontend → FastAPI Gateway → Inference Pipeline). No ML code in backend. All inference through pipeline predict(). XAI artifacts written by pipeline, read by backend.

**Tech Stack:** Python 3.10+, FastAPI, PyTorch, XGBoost, React 19, TypeScript 5.8, Vite 6.2, Tailwind CSS, Recharts, Docker

**Spec:** `docs/superpowers/specs/2026-04-12-cardioguard-improvement-design.md`

---

## File Structure

### Modified Files
| File | Responsibility | Task |
|------|---------------|------|
| `src/pipeline/inference/run_inference_superclass.py` | Main inference orchestrator | 1, 3 |
| `src/xai/unified.py` | Unified XAI explainer | 3 |
| `src/xai/visualize.py` | XAI visualization | 1 |
| `src/backend/main.py` | FastAPI gateway | 1, 6 |
| `requirements.txt` | Python dependencies | 1 |
| `.gitignore` | Git ignore patterns | 1 |
| `artifacts/thresholds_superclass.json` | Threshold config | 2 |
| `frontend/index.html` | HTML entry (remove CDN) | 4 |
| `frontend/package.json` | Node dependencies | 4 |

### New Files
| File | Responsibility | Task |
|------|---------------|------|
| `tests/test_api.py` | API endpoint tests | 5 |
| `Dockerfile` | Container build | 6 |
| `docker-compose.yml` | Container orchestration | 6 |
| `.dockerignore` | Docker ignore patterns | 6 |
| `frontend/tailwind.config.js` | Tailwind config | 4 |
| `frontend/postcss.config.js` | PostCSS config | 4 |
| `frontend/src/styles/globals.css` | Tailwind directives | 4 |
| `frontend/index.tsx` | App entry (rewrite) | 4 |
| `frontend/components/Header.tsx` | Header bar | 4 |
| `frontend/components/UploadPanel.tsx` | File upload + settings | 4 |
| `frontend/components/PredictionResult.tsx` | Results display | 4 |
| `frontend/components/ProbabilityChart.tsx` | Bar chart | 4 |
| `frontend/components/ConsistencyPanel.tsx` | Model agreement | 4 |
| `frontend/components/LocalizationPanel.tsx` | MI regions (rewrite) | 4 |
| `frontend/components/XaiViewer.tsx` | XAI artifacts (rewrite) | 4 |
| `frontend/components/SanityBadge.tsx` | XAI quality badge | 4 |
| `frontend/components/SystemStatus.tsx` | System status (rewrite) | 4 |
| `frontend/components/ThemeProvider.tsx` | Dark/light mode | 4 |
| `docs/threshold_optimization_study.md` | Optimization results | 7 |

---

## Task 1: Backend Bug Fixes

**Files:**
- Modify: `src/pipeline/inference/run_inference_superclass.py`
- Modify: `src/xai/visualize.py`
- Modify: `src/backend/main.py`
- Modify: `requirements.txt`
- Modify: `.gitignore`

- [ ] **Step 1: Add localization_threshold parameter to predict()**

In `src/pipeline/inference/run_inference_superclass.py`, add `localization_threshold` parameter to `predict()` function signature at line 181:

```python
def predict(
    signal: np.ndarray,
    cnn_model: MultiLabelECGCNN,
    xgb_data: Dict[str, Any],
    thresholds: Dict[str, float],
    localization_model: Optional[nn.Module],
    device: torch.device,
    binary_model: Optional[nn.Module] = None,
    ensemble_weight: float = 0.5,
    explain: bool = False,
    sanity_check: bool = False,
    save_plot: Optional[Path] = None,
    run_dir: Optional[Path] = None,
    sample_id: str = "sample",
    localization_threshold: float = 0.5,
) -> Dict[str, Any]:
```

Then at line 309, replace `prob >= 0.5` with `prob >= localization_threshold`:

```python
        detected_regions = [
            region for region, prob in localization_result.items()
            if prob >= localization_threshold
        ]
```

- [ ] **Step 2: Wrap Consistency Guard in try-except**

In `src/pipeline/inference/run_inference_superclass.py`, replace lines 279-290 with:

```python
    consistency_result: Optional[ConsistencyResult] = None
    if binary_model is not None:
        try:
            with torch.no_grad():
                binary_logits = binary_model(signal_tensor)
                binary_mi_prob = float(torch.sigmoid(binary_logits).cpu().numpy().flatten()[0])

            consistency_result = check_consistency(
                superclass_mi_prob=ensemble_probs.get("MI", 0.0),
                binary_mi_prob=binary_mi_prob,
                superclass_threshold=thresholds.get("MI", 0.5),
                binary_threshold=0.5,
            )
        except Exception as e:
            print(f"Warning: Consistency check failed: {e}")
            consistency_result = None
```

- [ ] **Step 3: Remove debug prints**

In `src/pipeline/inference/run_inference_superclass.py`, remove lines 422-423:

Delete:
```python
        print(f"DEBUGGING ERROR: explanation_result is not dict! Type: {type(explanation_result)}")
        print(f"DEBUGGING ERROR: Content: {explanation_result}")
```

- [ ] **Step 4: Fix duplicate plot_gradcam_heatmap**

In `src/xai/visualize.py`, delete the first `plot_gradcam_heatmap` definition (lines 154-171). Keep the second one (line 195+) which has the correct signature `(signal, cam, save_path, title, sampling_rate)`.

- [ ] **Step 5: Update requirements.txt**

Replace contents of `requirements.txt` with:

```
numpy
pandas
torch
scikit-learn
xgboost
wfdb
tabulate
tqdm
shap
matplotlib
scipy
fastapi
uvicorn[standard]
joblib
pydantic
```

- [ ] **Step 6: Expand .gitignore**

Replace contents of `.gitignore` with:

```gitignore
/physionet.org
__pycache__/
*.pyc
*.pyo
.env
.env.*
node_modules/
.venv/
venv/
*.egg-info/
dist/
build/
.DS_Store
*.log
.pytest_cache/
.coverage
htmlcov/
.idea/
```

- [ ] **Step 7: Add CORS environment variable**

In `src/backend/main.py`, add `import os` at the top (after existing imports), then replace the CORS middleware block (lines 279-285) with:

```python
CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "*").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in CORS_ORIGINS],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

- [ ] **Step 8: Run existing tests to verify no regressions**

Run: `pytest tests/ -v --tb=short 2>&1 | head -80`
Expected: All existing tests pass (some may skip if dependencies missing)

- [ ] **Step 9: Commit**

```bash
git add src/pipeline/inference/run_inference_superclass.py src/xai/visualize.py src/backend/main.py requirements.txt .gitignore
git commit -m "fix: backend bug fixes - parameterize thresholds, add exception handling, cleanup"
```

---

## Task 2: Threshold & Ensemble Optimization

**Files:**
- Modify: `artifacts/thresholds_superclass.json`
- Read: `predictions/val_cnn_probs.npz`, `predictions/val_xgb_probs.npz`, `predictions/val_labels.npz`
- Read: `src/pipeline/evaluation/optimize_thresholds.py`

- [ ] **Step 1: Write optimization script that combines threshold + ensemble weight search**

Create `scripts/optimize_all.py`:

```python
"""
Combined threshold + ensemble weight optimization.
Reads validation predictions, finds optimal ensemble weight and per-class thresholds.
"""
import json
import numpy as np
from pathlib import Path
from sklearn.metrics import f1_score, fbeta_score, roc_curve

SUPERCLASS_LABELS = ["MI", "STTC", "CD", "HYP"]

def find_threshold_fbeta(y_true, y_prob, beta=2.0):
    thresholds = np.linspace(0.01, 0.95, 95)
    best_t, best_s = 0.5, 0.0
    for t in thresholds:
        s = fbeta_score(y_true, (y_prob >= t).astype(int), beta=beta, zero_division=0)
        if s > best_s:
            best_s, best_t = s, float(t)
    return best_t, best_s

def find_threshold_youden(y_true, y_prob):
    if len(np.unique(y_true)) < 2:
        return 0.5, 0.0
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    j = tpr - fpr
    idx = np.argmax(j)
    return float(thresholds[idx]), float(j[idx])

def main():
    # Load data
    cnn_data = np.load("predictions/val_cnn_probs.npz")
    xgb_data = np.load("predictions/val_xgb_probs.npz")
    lbl_data = np.load("predictions/val_labels.npz")
    y_true = lbl_data["y_multi"]

    cnn = {cls: cnn_data[cls] for cls in SUPERCLASS_LABELS}
    xgb = {cls: xgb_data[cls] for cls in SUPERCLASS_LABELS}

    # Grid search ensemble weight
    print("=" * 60)
    print("ENSEMBLE WEIGHT GRID SEARCH")
    print("=" * 60)
    best_w, best_macro_f1 = 0.5, 0.0

    for w in np.arange(0.0, 1.05, 0.05):
        ens = {cls: w * cnn[cls] + (1 - w) * xgb[cls] for cls in SUPERCLASS_LABELS}
        # Use Youden thresholds for quick eval
        f1s = []
        for i, cls in enumerate(SUPERCLASS_LABELS):
            t, _ = find_threshold_youden(y_true[:, i], ens[cls])
            f1 = f1_score(y_true[:, i], (ens[cls] >= t).astype(int), zero_division=0)
            f1s.append(f1)
        macro = np.mean(f1s)
        if macro > best_macro_f1:
            best_macro_f1, best_w = macro, round(w, 2)
        print(f"  w={w:.2f}: Macro F1={macro:.4f}")

    print(f"\nBest weight: {best_w} (Macro F1={best_macro_f1:.4f})")

    # Optimize thresholds with best weight
    print("\n" + "=" * 60)
    print(f"THRESHOLD OPTIMIZATION (w={best_w})")
    print("=" * 60)
    ens = {cls: best_w * cnn[cls] + (1 - best_w) * xgb[cls] for cls in SUPERCLASS_LABELS}

    details = {}
    thresholds = {}
    for i, cls in enumerate(SUPERCLASS_LABELS):
        y_t = y_true[:, i]
        y_p = ens[cls]

        if cls == "MI":
            t, score = find_threshold_fbeta(y_t, y_p, beta=2.0)
            method = "F_beta (beta=2.0)"
            # Ensure recall >= 0.9
            y_pred = (y_p >= t).astype(int)
            tp = ((y_pred == 1) & (y_t == 1)).sum()
            fn = ((y_pred == 0) & (y_t == 1)).sum()
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            if recall < 0.9:
                for t2 in np.linspace(0.01, 0.5, 50):
                    y_pred2 = (y_p >= t2).astype(int)
                    tp2 = ((y_pred2 == 1) & (y_t == 1)).sum()
                    fn2 = ((y_pred2 == 0) & (y_t == 1)).sum()
                    r2 = tp2 / (tp2 + fn2) if (tp2 + fn2) > 0 else 0.0
                    if r2 >= 0.9:
                        t = float(t2)
                        method += " + recall_min=0.9"
                        break
        else:
            t, score = find_threshold_youden(y_t, y_p)
            method = "Youden_J"

        y_pred = (y_p >= t).astype(int)
        f1 = f1_score(y_t, y_pred, zero_division=0)
        tp = ((y_pred == 1) & (y_t == 1)).sum()
        fn = ((y_pred == 0) & (y_t == 1)).sum()
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        thresholds[cls] = round(t, 6)
        details[cls] = {
            "threshold": round(t, 6),
            "method": method,
            "score": round(score, 6),
            "f1_at_threshold": round(f1, 6),
            "recall_at_threshold": round(recall, 6),
            "support": int(y_t.sum()),
        }
        print(f"  {cls:4}: t={t:.4f}, F1={f1:.4f}, recall={recall:.4f}, method={method}")

    # Save
    output = {
        "thresholds": thresholds,
        "details": details,
        "ensemble_weight": best_w,
        "mi_beta": 2.0,
        "class_order": SUPERCLASS_LABELS,
    }
    out_path = Path("artifacts/thresholds_superclass.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run optimization**

Run: `python3 scripts/optimize_all.py`
Expected: Prints grid search results and optimized thresholds. Updates `artifacts/thresholds_superclass.json`.

- [ ] **Step 3: Verify results**

Run: `python3 -c "import json; d=json.load(open('artifacts/thresholds_superclass.json')); print(json.dumps(d['thresholds'], indent=2)); print('Weight:', d['ensemble_weight'])"`
Expected: Thresholds different from 0.5 for each class.

- [ ] **Step 4: Commit**

```bash
git add scripts/optimize_all.py artifacts/thresholds_superclass.json
git commit -m "feat: optimize thresholds and ensemble weight on validation set"
```

---

## Task 3: XAI Enhancement

**Files:**
- Modify: `src/xai/unified.py`
- Modify: `src/pipeline/inference/run_inference_superclass.py`

- [ ] **Step 1: Add SHAP-weighted GradCAM and contrastive mode to UnifiedExplainer**

Replace the entire content of `src/xai/unified.py` with enhanced version that includes:
- `_compute_shap_weighted_cam()` from combined.py
- `_compute_contrastive()` for pred vs runner-up
- Real `_analyze_coherence()` replacing placeholder 0.85

Read the current file, then apply the changes to add these three methods while keeping existing functionality.

- [ ] **Step 2: Cache embeddings in predict()**

In `src/pipeline/inference/run_inference_superclass.py`, compute embeddings once early and reuse:

After CNN prediction (around line 219), add:

```python
    # Compute embeddings once (used by XGBoost and SHAP)
    embeddings = None
    if xgb_data["models"] or explain:
        with torch.no_grad():
            embeddings = cnn_model.backbone(signal_tensor).cpu().numpy()
```

Then in the XGBoost section (line 225), replace:
```python
        with torch.no_grad():
            embeddings = cnn_model.backbone(signal_tensor).cpu().numpy()
```
with just using the pre-computed `embeddings` variable.

- [ ] **Step 3: Pass contrastive data to synthesize()**

In `src/pipeline/inference/run_inference_superclass.py`, in the XAI section, update the `unifier.synthesize()` call to pass runner-up class info:

```python
        # Get runner-up class for contrastive
        sorted_classes = sorted(ensemble_probs.items(), key=lambda x: x[1], reverse=True)
        runnerup_cls = sorted_classes[1][0] if len(sorted_classes) > 1 else None
        runnerup_shap = shap_res.get(runnerup_cls) if runnerup_cls else None

        explanation_result = unifier.synthesize(
            gradcam_res,
            shap_res,
            ensemble_probs,
            ensemble_weight,
            primary_label=primary_label,
            runnerup_label=runnerup_cls,
        )
```

- [ ] **Step 4: Commit**

```bash
git add src/xai/unified.py src/pipeline/inference/run_inference_superclass.py
git commit -m "feat: enhance XAI with SHAP-weighted GradCAM, contrastive mode, real coherence"
```

---

## Task 4: Frontend Redesign

**Files:**
- Create/Modify: 15+ frontend files (see File Structure above)

This is the largest task. It involves:
1. Setting up Tailwind CSS properly (removing CDN)
2. Installing react-markdown and recharts
3. Rewriting all components with Turkish UI and dark mode
4. Adding ConsistencyPanel, ProbabilityChart, SanityBadge
5. Implementing dark/light mode toggle

- [ ] **Step 1: Install new dependencies**

Run:
```bash
cd frontend
npm install tailwindcss @tailwindcss/vite react-markdown recharts
```

- [ ] **Step 2: Configure Tailwind with Vite plugin**

Update `frontend/vite.config.ts`:
```typescript
import { defineConfig } from "vite";
import tailwindcss from "@tailwindcss/vite";

export default defineConfig({
  plugins: [tailwindcss()],
  server: {
    port: 5173,
  },
});
```

- [ ] **Step 3: Create globals.css with Tailwind directives**

Create `frontend/src/styles/globals.css`:
```css
@import "tailwindcss";
```

- [ ] **Step 4: Remove CDN from index.html**

In `frontend/index.html`, remove the `<script src="https://cdn.tailwindcss.com"></script>` line and add CSS import.

- [ ] **Step 5: Create ThemeProvider**

Create `frontend/components/ThemeProvider.tsx` with dark/light mode context using React context + localStorage.

- [ ] **Step 6: Rewrite App entry (index.tsx)**

Rewrite `frontend/index.tsx` with new layout: Header, left panel (upload + system status), right panel (results + consistency + localization + XAI).

- [ ] **Step 7: Create Header component**

Create `frontend/components/Header.tsx` with logo, API URL input, dark mode toggle, system status badge.

- [ ] **Step 8: Create UploadPanel**

Create `frontend/components/UploadPanel.tsx` with file upload, ensemble weight slider, XAI checkbox, predict button. All labels in Turkish.

- [ ] **Step 9: Create PredictionResult + ProbabilityChart**

Create `frontend/components/PredictionResult.tsx` with primary diagnosis display and `ProbabilityChart.tsx` with recharts horizontal bar chart showing thresholds.

- [ ] **Step 10: Create ConsistencyPanel**

Create `frontend/components/ConsistencyPanel.tsx` rendering agreement type, triage level, MI probability comparison, and warnings.

- [ ] **Step 11: Rewrite LocalizationPanel**

Rewrite `frontend/components/LocalizationPanel.tsx` with Turkish labels, progress bars, dark mode support.

- [ ] **Step 12: Rewrite XaiViewer with markdown parsing**

Rewrite `frontend/components/XaiViewer.tsx` with tabs (GradCAM / SHAP / Rapor), react-markdown for narrative rendering, image display.

- [ ] **Step 13: Create SanityBadge and SystemStatus**

Create `frontend/components/SanityBadge.tsx` and rewrite `frontend/components/SystemStatus.tsx`.

- [ ] **Step 14: Verify frontend builds and runs**

Run:
```bash
cd frontend
npm run dev
```
Expected: No build errors, dark mode works, all panels render.

- [ ] **Step 15: Commit**

```bash
cd frontend
git add -A
git commit -m "feat: complete frontend redesign - Turkish clinical dashboard with dark mode"
```

---

## Task 5: API Tests

**Files:**
- Create: `tests/test_api.py`

- [ ] **Step 1: Write API test file**

Create `tests/test_api.py` with TestClient tests for all endpoints: `/health`, `/ready`, `/predict/superclass`, `/predict/mi-localization`, artifact serving, path traversal protection.

- [ ] **Step 2: Run tests**

Run: `pytest tests/test_api.py -v`
Expected: All tests pass.

- [ ] **Step 3: Commit**

```bash
git add tests/test_api.py
git commit -m "test: add API endpoint tests"
```

---

## Task 6: Docker

**Files:**
- Create: `Dockerfile`
- Create: `docker-compose.yml`
- Create: `.dockerignore`
- Modify: `src/backend/main.py` (static file serving)

- [ ] **Step 1: Create Dockerfile (multi-stage)**

Create `Dockerfile` with Node.js frontend build stage and Python backend stage. Use CPU-only PyTorch.

- [ ] **Step 2: Create docker-compose.yml**

Create `docker-compose.yml` with single service, port 8000, CORS env var, reports volume.

- [ ] **Step 3: Create .dockerignore**

Create `.dockerignore` excluding .git, __pycache__, node_modules, physionet.org, docs, tests.

- [ ] **Step 4: Add static file serving to backend**

In `src/backend/main.py`, at the end of the file (before `if __name__`), add:

```python
# Serve frontend static files in production
frontend_dist = Path("frontend/dist")
if frontend_dist.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dist), html=True), name="frontend")
```

- [ ] **Step 5: Build and test Docker**

Run: `docker-compose up --build`
Expected: Image builds, container starts, http://localhost:8000 serves frontend.

- [ ] **Step 6: Commit**

```bash
git add Dockerfile docker-compose.yml .dockerignore src/backend/main.py
git commit -m "feat: add Docker support with multi-stage build"
```

---

## Task 7: Documentation Update

**Files:**
- Modify: `CLAUDE.md`
- Modify: `docs/01_architecture.md`
- Create: `docs/threshold_optimization_study.md`

- [ ] **Step 1: Update CLAUDE.md**

Update with optimized thresholds, new frontend components, Docker commands.

- [ ] **Step 2: Fix architecture doc**

In `docs/01_architecture.md`, fix "Consistency Guard entegre degil" → "Consistency Guard TAM ENTEGRE".

- [ ] **Step 3: Write threshold optimization study**

Create `docs/threshold_optimization_study.md` documenting the optimization methodology and results.

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md docs/01_architecture.md docs/threshold_optimization_study.md
git commit -m "docs: update documentation with optimization results and fixes"
```

---

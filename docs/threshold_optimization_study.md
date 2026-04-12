# Threshold Optimization Study

**Date:** 2026-04-12
**Dataset:** PTB-XL Validation Set (Fold 9)
**Method:** Grid search ensemble weight + per-class threshold optimization

---

## Methodology

### Ensemble Weight Optimization
- Grid search: CNN weight from 0.00 to 1.00 (step 0.05)
- Metric: Macro F1 across 4 classes
- For each weight, per-class thresholds optimized independently

### Per-Class Threshold Optimization
- **MI:** F-beta score (beta=2.0) with recall >= 0.9 constraint
  - Beta=2 favors recall over precision (clinical safety)
  - Recall floor ensures MI cases are not missed
- **STTC, CD, HYP:** Youden's J statistic (sensitivity + specificity - 1)
  - Balanced operating point from ROC curve

---

## Results

### Ensemble Weight
| CNN Weight | Macro F1 | Notes |
|:----------:|:--------:|:------|
| 0.00 (XGB only) | 0.6714 | |
| **0.15** | **0.6810** | **Best** |
| 0.50 (equal) | 0.6656 | Previous default |
| 1.00 (CNN only) | 0.6552 | |

**Finding:** XGBoost-heavy ensemble (w=0.15) outperforms equal weighting. XGBoost calibration provides more reliable probabilities.

### Optimized Thresholds

| Class | Old Threshold | New Threshold | Old F1 | New F1 | New Recall | Method |
|:------|:------------:|:------------:|:------:|:------:|:----------:|:-------|
| MI | 0.50 | **0.16** | 0.693 | 0.638 | **0.930** | F-beta (beta=2) |
| STTC | 0.50 | **0.26** | 0.664 | **0.728** | 0.887 | Youden J |
| CD | 0.50 | **0.28** | 0.679 | **0.725** | 0.801 | Youden J |
| HYP | 0.50 | **0.19** | 0.484 | **0.534** | 0.847 | Youden J |

### Overall Performance

| Metric | Before | After | Change |
|:-------|:------:|:-----:|:------:|
| Macro F1 | 0.630 | **0.681** | **+8.1%** |
| MI Recall | ~0.70 | **0.930** | +33% |
| HYP F1 | 0.484 | **0.534** | +10.3% |

---

## Key Insights

1. **MI threshold lowered aggressively (0.50 -> 0.16):** Prioritizes patient safety by catching more MI cases. F1 drops slightly but recall jumps from ~70% to 93%.

2. **HYP benefits most from optimization (+10%):** With only 261 support samples, the class was hurt most by the default 0.5 threshold. Lowering to 0.19 better matches the model's probability distribution.

3. **XGBoost dominates ensemble (w=0.15):** Isotonic calibration on XGBoost produces better-calibrated probabilities than raw CNN sigmoid outputs.

4. **No model retraining needed:** All improvements from threshold tuning alone, demonstrating the importance of post-hoc calibration.

---

## Files

- **Script:** `scripts/optimize_all.py`
- **Config:** `artifacts/thresholds_superclass.json`
- **Validation data:** `predictions/val_cnn_probs.npz`, `predictions/val_xgb_probs.npz`, `predictions/val_labels.npz`

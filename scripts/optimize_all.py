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

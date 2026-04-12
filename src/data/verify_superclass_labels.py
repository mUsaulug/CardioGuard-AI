"""
Verify derived NORM labeling and compute correct co-occurrence.

This script validates that:
1. NORM is properly derived (NORM=1 iff MI=STTC=CD=HYP=0)
2. NORM never co-occurs with any pathology
3. Co-occurrence matrix is correct

Usage:
    python -m src.data.verify_superclass_labels
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import get_default_config
from src.data.loader import load_ptbxl_metadata, load_scp_statements
from src.data.splits import get_standard_split, verify_no_patient_leakage
from src.data.labels_superclass import (
    add_superclass_labels_derived,
    compute_cooccurrence_derived,
    verify_norm_cooccurrence,
    extract_y_multi4,
    derive_norm,
    build_label_report,
    compute_pos_weight_train,
    PATHOLOGY_CLASSES,
)


def main():
    parser = argparse.ArgumentParser(description="Verify Superclass Labels with Derived NORM")
    parser.add_argument("--min-likelihood", type=float, default=0.0)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/superclass_labels"))
    args = parser.parse_args()
    
    config = get_default_config()
    
    print("=" * 70)
    print("SUPERCLASS LABEL VERIFICATION (Derived NORM)")
    print("=" * 70)
    
    # Load data
    print(f"\nLoading data from {config.metadata_path}...")
    df = load_ptbxl_metadata(config.metadata_path)
    scp_df = load_scp_statements(config.scp_statements_path)
    print(f"Loaded {len(df)} records")
    
    # Get splits
    train_idx, val_idx, test_idx = get_standard_split(df)
    
    # Verify no patient leakage
    print("\n" + "-" * 40)
    print("PATIENT LEAKAGE CHECK")
    print("-" * 40)
    try:
        verify_no_patient_leakage(df, train_idx, val_idx, test_idx)
        print("✅ PASS: No patient leakage")
    except ValueError as e:
        print(f"❌ FAIL: {e}")
        return
    
    # Add derived labels
    print("\n" + "-" * 40)
    print("ADDING DERIVED LABELS")
    print("-" * 40)
    df_labeled = add_superclass_labels_derived(df, scp_df, args.min_likelihood)
    print("✅ Labels added with derived NORM")
    
    # Verify NORM co-occurrence
    print("\n" + "-" * 40)
    print("NORM CO-OCCURRENCE CHECK (CRITICAL)")
    print("-" * 40)
    passed, msg = verify_norm_cooccurrence(df_labeled)
    if passed:
        print(f"✅ {msg}")
    else:
        print(f"❌ FAIL: {msg}")
        print("\n⚠️  This is a CRITICAL bug - NORM should never co-occur with pathology!")
        return
    
    # Compute and display co-occurrence matrix
    print("\n" + "-" * 40)
    print("CO-OCCURRENCE MATRIX (Derived NORM)")
    print("-" * 40)
    cooccur = compute_cooccurrence_derived(df_labeled)
    print(cooccur.to_string())
    
    # Verify NORM row/column
    norm_row = cooccur.loc["NORM"]
    norm_off_diag = [norm_row[c] for c in PATHOLOGY_CLASSES]
    print(f"\nNORM off-diagonal values: {norm_off_diag}")
    
    if all(v == 0 for v in norm_off_diag):
        print("✅ NORM never co-occurs with any pathology (all zeros)")
    else:
        print("❌ BUG: NORM co-occurs with pathology!")
        return
    
    # Split statistics
    print("\n" + "-" * 40)
    print("SPLIT STATISTICS")
    print("-" * 40)
    
    train_df = df_labeled.loc[train_idx]
    val_df = df_labeled.loc[val_idx]
    test_df = df_labeled.loc[test_idx]
    
    # Extract y_multi4 for each split
    y_train = extract_y_multi4(train_df, scp_df, args.min_likelihood)
    y_val = extract_y_multi4(val_df, scp_df, args.min_likelihood)
    y_test = extract_y_multi4(test_df, scp_df, args.min_likelihood)
    
    for name, y, split_df in [("TRAIN", y_train, train_df), 
                               ("VAL", y_val, val_df), 
                               ("TEST", y_test, test_df)]:
        report = build_label_report(split_df, y, name)
        print(f"\n{name} ({report['n_samples']} samples):")
        print(f"  Avg labels/sample: {report['multi_label']['avg_labels_per_sample']:.2f}")
        print(f"  Multi-label samples: {report['multi_label']['samples_with_multiple_labels']}")
        print(f"  No pathology (NORM): {report['multi_label']['samples_with_no_pathology']}")
    
    # Compute pos_weight from TRAIN only
    print("\n" + "-" * 40)
    print("POS_WEIGHT (Train fold 1-8 only)")
    print("-" * 40)
    pos_weight = compute_pos_weight_train(y_train)
    print(f"{'Class':8} {'Positive':>10} {'Pos Rate':>10} {'pos_weight':>12}")
    print("-" * 50)
    for j, cls in enumerate(PATHOLOGY_CLASSES):
        pos = int(y_train[:, j].sum())
        rate = pos / len(y_train)
        print(f"{cls:8} {pos:>10} {rate:>10.3f} {pos_weight[j]:>12.2f}")
    
    # NORM stats
    norm_train = derive_norm(y_train)
    norm_count = int(norm_train.sum())
    norm_rate = norm_count / len(y_train)
    print(f"{'NORM':8} {norm_count:>10} {norm_rate:>10.3f} {'(derived)':>12}")
    
    # Save results
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save co-occurrence
    cooccur.to_csv(args.output_dir / "cooccurrence_derived.csv")
    
    # Save pos_weight
    pos_weight_dict = {cls: float(pos_weight[j]) for j, cls in enumerate(PATHOLOGY_CLASSES)}
    with open(args.output_dir / "pos_weight.json", "w") as f:
        json.dump(pos_weight_dict, f, indent=2)
    
    # Save full report
    full_report = {
        "total_records": len(df),
        "min_likelihood": args.min_likelihood,
        "splits": {
            "train": build_label_report(train_df, y_train, "train"),
            "val": build_label_report(val_df, y_val, "val"),
            "test": build_label_report(test_df, y_test, "test"),
        },
        "pos_weight": pos_weight_dict,
        "norm_check": {"passed": passed, "message": msg},
    }
    
    with open(args.output_dir / "superclass_label_report.json", "w") as f:
        json.dump(full_report, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else str(x))
    
    print(f"\n✅ Results saved to {args.output_dir}")
    print("\n" + "=" * 70)
    print("ALL CHECKS PASSED - Ready for training!")
    print("=" * 70)


if __name__ == "__main__":
    main()

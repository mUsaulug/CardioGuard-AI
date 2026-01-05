"""
CardioGuard-AI Data Validation Module

Scripts for verifying data integrity, patient leakage, and label analysis.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.config import get_default_config, DIAGNOSTIC_SUPERCLASSES
from src.data.loader import load_ptbxl_metadata, load_scp_statements
from src.data.labels import add_superclass_labels, add_5class_labels
from src.data.splits import get_standard_split, verify_no_patient_leakage, get_split_statistics


# Multi-label superclass order (excluding NORM - derived)
SUPERCLASS_LABELS = ["MI", "STTC", "CD", "HYP"]


def verify_splits(df: pd.DataFrame, verbose: bool = True) -> Dict:
    """
    Verify patient leakage and get split statistics.
    
    Args:
        df: PTB-XL metadata DataFrame
        verbose: Print results
        
    Returns:
        Dictionary with verification results
    """
    train_idx, val_idx, test_idx = get_standard_split(df)
    
    # Verify no patient leakage
    try:
        verify_no_patient_leakage(df, train_idx, val_idx, test_idx)
        leakage_status = "PASS"
    except ValueError as e:
        leakage_status = f"FAIL: {e}"
    
    # Get statistics
    stats = get_split_statistics(df, train_idx, val_idx, test_idx)
    
    result = {
        "leakage_check": leakage_status,
        "splits": stats,
        "fold_distribution": {
            "train": [1, 2, 3, 4, 5, 6, 7, 8],
            "val": [9],
            "test": [10]
        }
    }
    
    if verbose:
        print("=" * 60)
        print("PATIENT LEAKAGE CHECK")
        print("=" * 60)
        print(f"Status: {leakage_status}")
        print()
        print("SPLIT STATISTICS")
        print("-" * 40)
        print(f"Total samples: {stats['total_samples']}")
        print(f"Total unique patients: {stats['total_unique_patients']}")
        print()
        for split in ["train", "val", "test"]:
            s = stats[split]
            print(f"{split.upper():6}: {s['samples']:5} samples ({s['percentage']:5.1f}%), "
                  f"{s['unique_patients']:5} patients")
    
    return result


def compute_label_cooccurrence(df: pd.DataFrame, min_likelihood: float = 0.0) -> pd.DataFrame:
    """
    Compute label co-occurrence matrix for superclasses.
    
    Args:
        df: DataFrame with diagnostic_superclass column (list of superclasses)
        min_likelihood: Minimum likelihood threshold
        
    Returns:
        Co-occurrence matrix as DataFrame
    """
    # Superclasses including NORM
    all_classes = ["NORM", "MI", "STTC", "CD", "HYP"]
    
    # Initialize co-occurrence matrix
    cooccur = np.zeros((len(all_classes), len(all_classes)), dtype=int)
    
    for superclasses in df["diagnostic_superclass"]:
        if not superclasses:
            continue
        # Count co-occurrences
        for i, c1 in enumerate(all_classes):
            for j, c2 in enumerate(all_classes):
                if c1 in superclasses and c2 in superclasses:
                    cooccur[i, j] += 1
    
    return pd.DataFrame(cooccur, index=all_classes, columns=all_classes)


def analyze_class_distribution(
    df: pd.DataFrame,
    train_idx: np.ndarray,
    scp_df: pd.DataFrame,
    min_likelihood: float = 0.0
) -> Dict:
    """
    Analyze multi-label class distribution for training data.
    
    Args:
        df: Full DataFrame
        train_idx: Training indices (fold 1-8 only)
        scp_df: SCP statements DataFrame
        min_likelihood: Minimum likelihood threshold
        
    Returns:
        Dictionary with class distribution and pos_weight recommendations
    """
    # Add superclass labels
    df_labeled = add_superclass_labels(df.copy(), scp_df, min_likelihood)
    
    # Get training subset
    train_df = df_labeled.loc[train_idx]
    n_train = len(train_df)
    
    # Count per-class positives (multi-label)
    class_counts = {}
    for cls in SUPERCLASS_LABELS:  # MI, STTC, CD, HYP (not NORM)
        positive = train_df["diagnostic_superclass"].apply(lambda x: cls in x).sum()
        class_counts[cls] = {
            "positive": int(positive),
            "negative": int(n_train - positive),
            "positive_rate": positive / n_train,
            "pos_weight": (n_train - positive) / positive if positive > 0 else 1.0
        }
    
    # NORM is derived: NORM = 1 iff no pathology
    norm_positive = train_df["diagnostic_superclass"].apply(
        lambda x: len(x) == 1 and "NORM" in x
    ).sum()
    class_counts["NORM (derived)"] = {
        "positive": int(norm_positive),
        "negative": int(n_train - norm_positive),
        "positive_rate": norm_positive / n_train,
        "note": "NORM is derived, not trained"
    }
    
    return {
        "n_train_samples": n_train,
        "class_distribution": class_counts,
        "multi_label_stats": {
            "avg_labels_per_sample": train_df["diagnostic_superclass"].apply(len).mean(),
            "samples_with_multiple_labels": (
                train_df["diagnostic_superclass"].apply(len) > 1
            ).sum()
        }
    }


def analyze_labels(
    df: pd.DataFrame,
    scp_df: pd.DataFrame,
    min_likelihood: float = 0.0,
    output_dir: Optional[Path] = None,
    verbose: bool = True
) -> Dict:
    """
    Full label analysis: co-occurrence, distribution, and statistics.
    
    Args:
        df: PTB-XL metadata DataFrame
        scp_df: SCP statements DataFrame
        min_likelihood: Minimum likelihood threshold
        output_dir: Optional directory to save results
        verbose: Print results
        
    Returns:
        Dictionary with all analysis results
    """
    # Add superclass labels
    df_labeled = add_superclass_labels(df.copy(), scp_df, min_likelihood)
    
    # Get splits
    train_idx, val_idx, test_idx = get_standard_split(df_labeled)
    
    # Compute co-occurrence
    cooccur = compute_label_cooccurrence(df_labeled, min_likelihood)
    
    # Analyze class distribution (train only for pos_weight)
    class_dist = analyze_class_distribution(df_labeled, train_idx, scp_df, min_likelihood)
    
    results = {
        "cooccurrence_matrix": cooccur.to_dict(),
        "class_distribution": class_dist,
        "min_likelihood": min_likelihood
    }
    
    if verbose:
        print()
        print("=" * 60)
        print("LABEL CO-OCCURRENCE MATRIX")
        print("=" * 60)
        print(cooccur.to_string())
        print()
        print("=" * 60)
        print("CLASS DISTRIBUTION (Train fold 1-8 only)")
        print("=" * 60)
        print(f"Total train samples: {class_dist['n_train_samples']}")
        print(f"Avg labels per sample: {class_dist['multi_label_stats']['avg_labels_per_sample']:.2f}")
        print(f"Samples with multi-label: {class_dist['multi_label_stats']['samples_with_multiple_labels']}")
        print()
        print("Per-class statistics:")
        print("-" * 60)
        print(f"{'Class':12} {'Positive':>10} {'Neg':>10} {'Pos Rate':>10} {'pos_weight':>12}")
        print("-" * 60)
        for cls, stats in class_dist["class_distribution"].items():
            pw = stats.get("pos_weight", "-")
            if isinstance(pw, float):
                pw = f"{pw:.2f}"
            print(f"{cls:12} {stats['positive']:>10} {stats['negative']:>10} "
                  f"{stats['positive_rate']:>10.3f} {pw:>12}")
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save co-occurrence matrix
        cooccur.to_csv(output_dir / "label_cooccurrence.csv")
        
        # Save class distribution as JSON
        # Convert non-serializable values
        dist_json = {
            "n_train_samples": class_dist["n_train_samples"],
            "multi_label_stats": class_dist["multi_label_stats"],
            "class_distribution": {
                k: {kk: (float(vv) if isinstance(vv, (int, float, np.integer, np.floating)) else vv)
                    for kk, vv in v.items()}
                for k, v in class_dist["class_distribution"].items()
            }
        }
        
        def np_encoder(obj):
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj) if isinstance(obj, np.floating) else int(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return str(obj)

        with open(output_dir / "class_distribution.json", "w") as f:
            json.dump(dist_json, f, indent=2, default=np_encoder)
        
        print(f"\nResults saved to {output_dir}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="CardioGuard-AI Data Validation")
    parser.add_argument("--assert-no-leakage", action="store_true",
                        help="Exit with error if patient leakage detected")
    parser.add_argument("--co-occurrence", action="store_true",
                        help="Compute label co-occurrence matrix")
    parser.add_argument("--class-distribution", action="store_true",
                        help="Analyze class distribution and pos_weight")
    parser.add_argument("--all", action="store_true",
                        help="Run all validations")
    parser.add_argument("--min-likelihood", type=float, default=0.0,
                        help="Minimum likelihood threshold for labels")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory to save results")
    
    args = parser.parse_args()
    
    # Default to all if no specific option selected
    if not any([args.assert_no_leakage, args.co_occurrence, args.class_distribution]):
        args.all = True
    
    # Load data
    config = get_default_config()
    print(f"Loading data from {config.metadata_path}...")
    df = load_ptbxl_metadata(config.metadata_path)
    scp_df = load_scp_statements(config.scp_statements_path)
    print(f"Loaded {len(df)} records")
    
    # Run validations
    if args.all or args.assert_no_leakage:
        result = verify_splits(df, verbose=True)
        if args.assert_no_leakage and result["leakage_check"] != "PASS":
            print(f"\n❌ ASSERTION FAILED: {result['leakage_check']}")
            exit(1)
        elif result["leakage_check"] == "PASS":
            print("\n✅ No patient leakage detected")
    
    if args.all or args.co_occurrence or args.class_distribution:
        output_dir = Path(args.output_dir) if args.output_dir else Path("reports/data_validation")
        analyze_labels(df, scp_df, args.min_likelihood, output_dir, verbose=True)
    
    print("\n✅ Data validation complete")


if __name__ == "__main__":
    main()

"""
Compare CNN, XGBoost, and Ensemble models on the test set.

Generates a Markdown report and console output comparing metrics.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.config import get_default_config
from src.data.loader import load_ptbxl_metadata
from src.data.signals import SignalDataset
from src.data.splits import get_standard_split
from src.models.cnn import ECGCNN, ECGCNNConfig
from src.models.metrics import compute_classification_metrics
from src.models.xgb import load_xgb, predict_xgb


def get_cnn_probs(
    model: ECGCNN,
    dataloader: DataLoader,
    device: str
) -> Tuple[np.ndarray, np.ndarray]:
    """Run CNN inference to get probabilities and true labels."""
    model.eval()
    device_obj = torch.device(device)
    model.to(device_obj)

    probs_list = []
    labels_list = []

    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 3:
                inputs, labels, _ = batch
            else:
                inputs, labels = batch
            
            # Ensure float32
            inputs = inputs.to(device_obj).float()
            # Forward pass returns logits (or whatever the head returns)
            # BinaryHead returns logits squeezed to (batch,)
            logits = model(inputs)
            
            # Apply sigmoid for binary classification
            probs = torch.sigmoid(logits).cpu().numpy()
            
            probs_list.append(probs)
            labels_list.append(labels.numpy())

    return np.concatenate(probs_list), np.concatenate(labels_list)


def get_cnn_embeddings(
    model: ECGCNN,
    dataloader: DataLoader,
    device: str
) -> np.ndarray:
    """Extract embeddings from CNN backbone for XGBoost."""
    model.eval()
    device_obj = torch.device(device)
    model.to(device_obj)

    embeddings_list = []

    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 3:
                inputs, _, _ = batch
            else:
                inputs, _ = batch

            inputs = inputs.to(device_obj).float()
            # Use backbone directly
            features = model.backbone(inputs).cpu().numpy()
            embeddings_list.append(features)

    return np.concatenate(embeddings_list)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare CNN, XGB, and Ensemble")
    parser.add_argument("--cnn-path", type=Path, default=Path("checkpoints/ecgcnn.pt"), help="Path to CNN checkpoint")
    parser.add_argument("--xgb-path", type=Path, default=Path("logs/xgb/xgb_model.json"), help="Path to XGB model")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for inference")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=Path, default=Path("reports"), help="Output directory")
    args = parser.parse_args()

    # 1. Setup & Data Loading
    print(f"Loading data... (Device: {args.device})")
    config = get_default_config()
    df = load_ptbxl_metadata(config.metadata_path)
    
    # Load SCP statements and generate labels
    from src.data.loader import load_scp_statements
    from src.data.labels import add_binary_mi_labels
    
    print("Generating labels...")
    scp_df = load_scp_statements(config.scp_statements_path)
    df = add_binary_mi_labels(df, scp_df)
    
    # Filter valid labels and test split
    df = df[df["label_mi_norm"] != -1].copy()
    
    # Get test split
    _, _, test_indices = get_standard_split(df)
    
    # Intersect valid labels with test split
    valid_test_indices = np.intersect1d(test_indices, df.index)
    print(f"Test Set Size: {len(valid_test_indices)}")
    
    # Create Dataset & DataLoader
    test_dataset = SignalDataset(
        df=df.loc[valid_test_indices],
        # CSV filenames include 'records100/', so use data_root as base
        base_path=config.data_root,
        filename_column=config.filename_column,
        label_column="label_mi_norm"
    )

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # 2. CNN Inference
    print("Loading CNN model...")
    cnn_config = ECGCNNConfig()
    cnn_model = ECGCNN(cnn_config)
    
    if args.cnn_path.exists():
        checkpoint = torch.load(args.cnn_path, map_location=args.device)
        # Handle checkpoint dict if present
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
            
        # Remap keys if needed (0. -> backbone., 1. -> head.)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("0."):
                new_state_dict[k.replace("0.", "backbone.", 1)] = v
            elif k.startswith("1."):
                new_state_dict[k.replace("1.", "head.", 1)] = v
            else:
                new_state_dict[k] = v
        state_dict = new_state_dict
            
        cnn_model.load_state_dict(state_dict)
    else:
        print(f"Warning: CNN checkpoint not found at {args.cnn_path}")
        sys.exit(1)
        
    print("Running CNN inference...")
    cnn_probs, y_true = get_cnn_probs(cnn_model, test_loader, args.device)
    
    # 3. XGB Inference
    print("Loading XGB model...")
    if not args.xgb_path.exists():
        print(f"Error: XGB model not found at {args.xgb_path}")
        sys.exit(1)
        
    xgb_model = load_xgb(args.xgb_path)
    
    print("Extracting features for XGB...")
    xgb_features = get_cnn_embeddings(cnn_model, test_loader, args.device)
    
    print("Running XGB inference...")
    xgb_probs, _ = predict_xgb(xgb_model, xgb_features)
    
    # 4. Ensemble
    print("Computing Ensemble...")
    ensemble_probs = (cnn_probs + xgb_probs) / 2
    
    # 5. Metrics
    print("\nCalculating metrics...")
    metrics_cnn = compute_classification_metrics(y_true, np.log(cnn_probs / (1 - cnn_probs + 1e-9))) # Convert to logits for helper?
    # Wait, `compute_classification_metrics` takes logits.
    # cnn_probs are probabilities.
    # I should adapt `compute_classification_metrics` or pass logits.
    # The helper `compute_classification_metrics` does `y_probs = 1 / (1 + np.exp(-y_logits))`.
    # So it EXPECTS LOGITS.
    # I can inverse sigmoid: ln(p / (1-p)) or just modify the helper call.
    # Better: Use sklearn directly here or fix the helper usage.
    # I will basically re-implement metric calc here to be safe and clear.
    
    from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, f1_score
    
    def calc_metrics(y_t, y_p, name):
        y_pred = (y_p >= 0.5).astype(int)
        return {
            "Model": name,
            "AUC": roc_auc_score(y_t, y_p),
            "PR_AUC": average_precision_score(y_t, y_p),
            "F1": f1_score(y_t, y_pred),
            "Accuracy": accuracy_score(y_t, y_pred)
        }

    results = []
    results.append(calc_metrics(y_true, cnn_probs, "CNN"))
    results.append(calc_metrics(y_true, xgb_probs, "XGBoost"))
    results.append(calc_metrics(y_true, ensemble_probs, "Ensemble (Avg)"))
    
    results_df = pd.DataFrame(results)
    
    # 6. Output
    print("\n=== Model Comparison Report ===")
    print(results_df.to_markdown(index=False, floatfmt=".4f"))
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "comparison_report.md"
    results_df.to_markdown(report_path, index=False, floatfmt=".4f")
    print(f"\nReport saved to {report_path}")
    
    # Save CSV too
    results_df.to_csv(args.output_dir / "comparison_report.csv", index=False)


if __name__ == "__main__":
    main()
"""
XAI Demo Script for CardioGuard-AI.

Generates Grad-CAM and SHAP visualizations for sample ECG predictions.
Outputs are saved to reports/xai/ directory.

Usage:
    python -m src.pipeline.run_xai_demo
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config import get_default_config
from src.data.labels import add_binary_mi_labels
from src.data.loader import load_ptbxl_metadata, load_scp_statements
from src.data.signals import SignalDataset
from src.data.splits import get_standard_split
from src.models.cnn import ECGCNN, ECGCNNConfig
from src.models.xgb import load_xgb
from src.xai.gradcam import GradCAM
from src.xai.shap_xgb import explain_xgb, plot_shap_summary, plot_shap_waterfall
from src.xai.visualize import plot_gradcam_heatmap, plot_lead_attention, plot_ecg_with_prediction


def load_cnn_model(checkpoint_path: Path, device: str) -> ECGCNN:
    """Load trained CNN model from checkpoint."""
    config = ECGCNNConfig()
    model = ECGCNN(config)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
    
    # Remap keys if saved with nn.Sequential
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("0."):
            new_state_dict[k.replace("0.", "backbone.", 1)] = v
        elif k.startswith("1."):
            new_state_dict[k.replace("1.", "head.", 1)] = v
        else:
            new_state_dict[k] = v
    
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
    
    return model


def get_sample_data(config, num_samples: int = 5):
    """Get sample test data for demonstration."""
    df = load_ptbxl_metadata(config.metadata_path)
    scp_df = load_scp_statements(config.scp_statements_path)
    
    df = add_binary_mi_labels(df, scp_df)
    df = df[df["label_mi_norm"] != -1].copy()
    
    _, _, test_indices = get_standard_split(df)
    valid_test_indices = np.intersect1d(test_indices, df.index)
    
    # Get balanced samples (some MI, some NORM)
    test_df = df.loc[valid_test_indices]
    mi_samples = test_df[test_df["label_mi_norm"] == 1].head(num_samples // 2 + 1)
    norm_samples = test_df[test_df["label_mi_norm"] == 0].head(num_samples // 2 + 1)
    sample_df = mi_samples._append(norm_samples).head(num_samples)
    
    dataset = SignalDataset(
        df=sample_df,
        base_path=config.data_root,
        filename_column=config.filename_column,
        label_column="label_mi_norm",
    )
    
    return dataset, sample_df


def main():
    parser = argparse.ArgumentParser(description="Generate XAI visualizations")
    parser.add_argument("--cnn-path", type=Path, default=Path("checkpoints/ecgcnn.pt"))
    parser.add_argument("--xgb-path", type=Path, default=Path("logs/xgb/xgb_model.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/xai"))
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    
    # Setup
    config = get_default_config()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Device: {args.device}")
    print(f"Output directory: {args.output_dir}")
    
    # Check model files exist
    if not args.cnn_path.exists():
        print(f"Error: CNN checkpoint not found at {args.cnn_path}")
        print("Please train the CNN model first: python -m src.pipeline.run_experiment")
        sys.exit(1)
    
    if not args.xgb_path.exists():
        print(f"Error: XGBoost model not found at {args.xgb_path}")
        print("Please train XGBoost first: python -m src.pipeline.run_xgb")
        sys.exit(1)
    
    # Load models
    print("\n1. Loading models...")
    cnn_model = load_cnn_model(args.cnn_path, args.device)
    xgb_model = load_xgb(args.xgb_path)
    print("   Models loaded successfully!")
    
    # Setup Grad-CAM
    # Target the last conv layer before pooling
    target_layer = cnn_model.backbone.features[4]  # Second Conv1d layer
    gradcam = GradCAM(cnn_model, target_layer)
    
    # Load sample data
    print(f"\n2. Loading {args.num_samples} sample ECGs...")
    dataset, sample_df = get_sample_data(config, args.num_samples)
    print(f"   Samples loaded: {len(dataset)} ECGs")
    
    # Extract CNN embeddings for XGBoost SHAP
    print("\n3. Extracting CNN embeddings for SHAP...")
    all_embeddings = []
    all_signals = []
    all_labels = []
    all_ecg_ids = []
    
    with torch.no_grad():
        for i in range(len(dataset)):
            signal, label, ecg_id = dataset[i]
            signal_tensor = torch.tensor(signal, dtype=torch.float32).unsqueeze(0)
            
            # Ensure (batch, channels, time)
            if signal_tensor.shape[1] != 12:
                signal_tensor = signal_tensor.permute(0, 2, 1)
            
            signal_tensor = signal_tensor.to(args.device)
            embedding = cnn_model.backbone(signal_tensor).cpu().numpy()
            
            all_embeddings.append(embedding)
            all_signals.append(signal)
            all_labels.append(label)
            all_ecg_ids.append(ecg_id)
    
    embeddings = np.concatenate(all_embeddings, axis=0)
    print(f"   Embeddings shape: {embeddings.shape}")
    
    # Generate SHAP values
    print("\n4. Computing SHAP values...")
    shap_result = explain_xgb(xgb_model, embeddings)
    shap_values = shap_result["shap_values"]
    base_value = shap_result["base_value"]
    print(f"   SHAP values shape: {np.array(shap_values).shape}")
    
    # Generate SHAP summary plot
    print("\n5. Generating SHAP summary plot...")
    feature_names = [f"CNN_Feature_{i}" for i in range(embeddings.shape[1])]
    plot_shap_summary(
        shap_values,
        embeddings,
        feature_names=feature_names,
        save_path=args.output_dir / "shap_summary.png",
        title="XGBoost Feature Importance (CNN Embeddings)",
    )
    
    # Generate visualizations for each sample
    print("\n6. Generating per-sample visualizations...")
    
    for i in range(len(dataset)):
        signal = all_signals[i]
        label = all_labels[i]
        ecg_id = all_ecg_ids[i]
        
        # Ensure signal is (12, T)
        if signal.shape[0] != 12:
            signal = signal.T
        
        signal_tensor = torch.tensor(signal, dtype=torch.float32).unsqueeze(0).to(args.device)
        
        print(f"\n   Sample {i+1} (ECG ID: {ecg_id}, True: {'MI' if label == 1 else 'NORM'}):")
        
        # CNN prediction
        with torch.no_grad():
            logits = cnn_model(signal_tensor)
            prob = torch.sigmoid(logits).item()
        print(f"     CNN Prediction: {'MI' if prob >= 0.5 else 'NORM'} ({prob:.2%})")
        
        # Grad-CAM
        cam = gradcam.generate(signal_tensor)
        cam = cam.squeeze()  # (T,)
        
        # a) ECG with prediction
        plot_ecg_with_prediction(
            signal,
            prediction=prob,
            true_label=label,
            save_path=args.output_dir / f"ecg_sample_{i+1}_id{ecg_id}.png",
        )
        
        # b) Grad-CAM heatmap
        plot_gradcam_heatmap(
            signal,
            cam,
            save_path=args.output_dir / f"gradcam_sample_{i+1}_id{ecg_id}.png",
            title=f"Grad-CAM Attention (ECG ID: {ecg_id})",
        )
        
        # c) Lead attention
        plot_lead_attention(
            cam,
            signal=signal,
            save_path=args.output_dir / f"lead_attention_sample_{i+1}_id{ecg_id}.png",
            title=f"Per-Lead Attention (ECG ID: {ecg_id})",
        )
        
        # d) SHAP waterfall
        plot_shap_waterfall(
            shap_values,
            base_value,
            sample_idx=i,
            features=embeddings,
            feature_names=feature_names,
            save_path=args.output_dir / f"shap_waterfall_sample_{i+1}_id{ecg_id}.png",
            title=f"SHAP Explanation (ECG ID: {ecg_id})",
        )
    
    print(f"\n✓ All visualizations saved to {args.output_dir}")
    print("\nGenerated files:")
    for f in sorted(args.output_dir.glob("*.png")):
        print(f"   - {f.name}")


if __name__ == "__main__":
    main()

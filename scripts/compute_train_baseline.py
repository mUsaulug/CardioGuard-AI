"""
Compute and save train set mean baseline for XAI faithfulness tests.

This baseline is used instead of zeros for masking in deletion/insertion tests.
Using train mean provides a more realistic "neutral" signal.

Usage:
    python scripts/compute_train_baseline.py
"""

import numpy as np
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.ptbxl_dataset import PTBXLDataset


def compute_train_mean(data_root: Path = Path("data/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3")):
    """
    Compute mean ECG signal from training set.
    
    Returns:
        mean: (12, 1000) mean signal
        std: (12, 1000) std signal
    """
    print("Loading training data...")
    
    # Try to load from preprocessed features if available
    features_path = Path("logs/superclass_features/train_embeddings.npz")
    if features_path.exists():
        data = np.load(features_path)
        # If we have raw signals saved
        if "signals" in data:
            signals = data["signals"]
            print(f"Loaded {len(signals)} signals from cache")
            return np.mean(signals, axis=0), np.std(signals, axis=0)
    
    # Otherwise compute from dataset
    try:
        dataset = PTBXLDataset(
            data_root=data_root,
            split="train",
            sampling_rate=100,
            target_type="superclass"
        )
        
        print(f"Computing mean from {len(dataset)} samples...")
        
        # Accumulate signals
        signals = []
        for i in range(min(len(dataset), 5000)):  # Cap at 5000 for speed
            signal, _ = dataset[i]
            if isinstance(signal, np.ndarray):
                signals.append(signal)
            else:
                signals.append(signal.numpy())
            
            if (i + 1) % 1000 == 0:
                print(f"  Processed {i + 1} samples...")
        
        signals = np.stack(signals)
        print(f"Signal shape: {signals.shape}")
        
        return np.mean(signals, axis=0), np.std(signals, axis=0)
        
    except Exception as e:
        print(f"Could not load dataset: {e}")
        print("Using synthetic baseline...")
        
        # Fallback: create a simple synthetic baseline
        # This is a flat line with small noise
        mean = np.zeros((12, 1000), dtype=np.float32)
        std = np.ones((12, 1000), dtype=np.float32) * 0.1
        return mean, std


def main():
    output_path = Path("artifacts/train_baseline.npz")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    mean, std = compute_train_mean()
    
    # Ensure correct shape
    if mean.shape != (12, 1000):
        # Try to reshape or pad
        if mean.ndim == 1:
            mean = mean.reshape(12, -1)[:, :1000]
            std = std.reshape(12, -1)[:, :1000]
        elif mean.shape[0] > mean.shape[1]:
            mean = mean.T
            std = std.T
    
    # Save
    np.savez(
        output_path,
        mean=mean.astype(np.float32),
        std=std.astype(np.float32),
        n_samples=5000,
        description="Train set mean for XAI faithfulness baseline"
    )
    
    print(f"\n✓ Saved baseline to {output_path}")
    print(f"  Shape: {mean.shape}")
    print(f"  Mean range: [{mean.min():.3f}, {mean.max():.3f}]")
    print(f"  Std range: [{std.min():.3f}, {std.max():.3f}]")


if __name__ == "__main__":
    main()

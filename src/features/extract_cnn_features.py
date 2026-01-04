"""Extract CNN encoder features and save to disk."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.models.cnn import CNNEncoder


def extract_cnn_features(
    model: CNNEncoder,
    dataloader: DataLoader,
    device: str,
    output_path: str | Path,
) -> Tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Run the encoder on the dataloader and save features to disk."""

    model.eval()
    device_obj = torch.device(device)
    model.to(device_obj)

    features: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    ids: List[str] = []
    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 4:
                inputs, batch_labels, _, batch_ids = batch
            elif len(batch) == 3:
                inputs, batch_labels, batch_ids = batch
            else:
                raise ValueError("Expected batch with 3 to 4 elements (inputs, labels, [localization], ids).")
            inputs = inputs.to(device_obj)
            embeddings = model(inputs).cpu().numpy()
            features.append(embeddings)
            if batch_labels is None:
                raise ValueError("Feature extraction requires labels to populate the 'y' field.")
            labels.append(batch_labels.cpu().numpy())
            if batch_ids is None:
                raise ValueError("Feature extraction requires ids to populate the 'ids' field.")
            ids.extend([str(item) for item in batch_ids])

    feature_array = np.concatenate(list(features), axis=0)
    if not labels:
        raise ValueError("No labels collected; cannot save 'y' field.")
    if not ids:
        raise ValueError("No ids collected; cannot save 'ids' field.")
    label_array = np.concatenate(labels, axis=0)
    label_array = np.asarray(label_array).reshape(-1)
    ids_array = np.array(ids)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, X=feature_array, y=label_array, ids=ids_array)
    return feature_array, label_array, ids_array


def extract_cnn_feature_splits(
    model: CNNEncoder,
    dataloaders: Dict[str, DataLoader],
    device: str,
    output_dir: str | Path,
) -> Dict[str, Path]:
    """Extract features for train/val/test splits and save .npz files."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths: Dict[str, Path] = {}

    for split, loader in dataloaders.items():
        output_path = output_dir / f"{split}.npz"
        extract_cnn_features(model, loader, device, output_path)
        output_paths[split] = output_path

    return output_paths

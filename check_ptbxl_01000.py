from pathlib import Path
import torch
from torch.utils.data import DataLoader

from src.config import get_default_config
from src.data.signals import SignalDataset, compute_channel_stats_streaming, normalize_with_stats
from src.pipeline.data_pipeline import prepare_splits, ECGDatasetTorch
from src.models.cnn import CNNEncoder, ECGCNNConfig
from src.features.extract_cnn_features import extract_cnn_feature_splits

config = get_default_config()
df, splits, label_column = prepare_splits(config)

train_df = df.loc[splits["train"]]
val_df = df.loc[splits["val"]]
test_df = df.loc[splits["test"]]

mean, std = compute_channel_stats_streaming(
    train_df, base_path=config.data_root, filename_column=config.filename_column,
    batch_size=128, progress=False, expected_channels=12
)

def normalize(signal):
    norm = normalize_with_stats(signal, mean.reshape(-1), std.reshape(-1))
    return norm.transpose(1, 0)

datasets = {
    "train": SignalDataset(train_df, config.data_root, label_column=label_column, transform=normalize, expected_channels=12),
    "val": SignalDataset(val_df, config.data_root, label_column=label_column, transform=normalize, expected_channels=12),
    "test": SignalDataset(test_df, config.data_root, label_column=label_column, transform=normalize, expected_channels=12),
}

torch_datasets = {k: ECGDatasetTorch(v) for k, v in datasets.items()}
loaders = {k: DataLoader(v, batch_size=32, shuffle=(k=="train")) for k, v in torch_datasets.items()}

encoder = CNNEncoder(ECGCNNConfig())
extract_cnn_feature_splits(encoder, loaders, "cpu", Path("features_out"))
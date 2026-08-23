import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from typing import Dict, Tuple, Optional
from pathlib import Path


class StandardScaler:
    def __init__(self, mean: float = 0.0, std: float = 1.0):
        self.mean = mean
        self.std = std

    def fit(self, data: np.ndarray):
        valid_mask = data > 0.0
        if np.any(valid_mask):
            self.mean = float(np.mean(data[valid_mask]))
            self.std = float(np.std(data[valid_mask]))
        else:
            self.mean = float(np.mean(data))
            self.std = float(np.std(data))
        if self.std < 1e-8:
            self.std = 1.0

    def transform(self, data: np.ndarray) -> np.ndarray:
        return (data - self.mean) / self.std

    def inverse_transform(self, data: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
        return (data * self.std) + self.mean

    def to_dict(self) -> Dict[str, float]:
        return {"mean": self.mean, "std": self.std}

    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> "StandardScaler":
        return cls(mean=d["mean"], std=d["std"])


class TrafficDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


def load_raw_traffic_data(h5_path: Path) -> Tuple[np.ndarray, list, pd.DatetimeIndex]:
    with h5py.File(h5_path, "r") as f:
        sensor_ids = [s.decode("utf-8") if isinstance(s, bytes) else str(s) for s in f["df/axis0"][:]]
        timestamps = pd.to_datetime(f["df/axis1"][:])
        speeds = f["df/block0_values"][:]
    return speeds, sensor_ids, timestamps


def add_time_features(speeds: np.ndarray, timestamps: pd.DatetimeIndex) -> np.ndarray:
    T, N = speeds.shape
    time_of_day = np.array((timestamps.hour * 60 + timestamps.minute) / 1440.0, dtype=np.float32)
    time_feature = np.broadcast_to(time_of_day[:, None, None], (T, N, 1))
    speed_feature = speeds[:, :, None]
    return np.concatenate([speed_feature, time_feature], axis=-1)


def generate_sliding_windows(data: np.ndarray, in_steps: int = 12, out_steps: int = 12) -> Tuple[np.ndarray, np.ndarray]:
    T, N, F = data.shape
    num_samples = T - in_steps - out_steps + 1
    x = np.zeros((num_samples, in_steps, N, F), dtype=np.float32)
    y = np.zeros((num_samples, out_steps, N), dtype=np.float32)

    for i in range(num_samples):
        x[i] = data[i : i + in_steps]
        y[i] = data[i + in_steps : i + in_steps + out_steps, :, 0]
    return x, y


def build_traffic_dataloaders(
    data_dir: Path,
    in_steps: int = 12,
    out_steps: int = 12,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
    batch_size: int = 64,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, DataLoader, StandardScaler, list]:
    h5_path = data_dir / "METR-LA.h5"
    if not h5_path.exists():
        h5_path = Path.home() / ".cache/kagglehub/datasets/annnnguyen/metr-la-dataset/versions/4/METR-LA.h5"
    if not h5_path.exists():
        raise FileNotFoundError(f"METR-LA.h5 not found in {data_dir}. Run scripts/setup.py first.")

    speeds, sensor_ids, timestamps = load_raw_traffic_data(h5_path)
    data = add_time_features(speeds, timestamps)

    T = len(data)
    train_len = int(T * train_ratio)
    val_len = int(T * val_ratio)

    train_raw = data[:train_len]
    val_raw = data[train_len : train_len + val_len]
    test_raw = data[train_len + val_len :]

    scaler = StandardScaler()
    scaler.fit(train_raw[:, :, 0])

    train_scaled = train_raw.copy()
    val_scaled = val_raw.copy()
    test_scaled = test_raw.copy()

    train_scaled[:, :, 0] = scaler.transform(train_raw[:, :, 0])
    val_scaled[:, :, 0] = scaler.transform(val_raw[:, :, 0])
    test_scaled[:, :, 0] = scaler.transform(test_raw[:, :, 0])

    x_train, y_train = generate_sliding_windows(train_scaled, in_steps, out_steps)
    x_val, y_val = generate_sliding_windows(val_scaled, in_steps, out_steps)
    x_test, y_test = generate_sliding_windows(test_scaled, in_steps, out_steps)

    train_dataset = TrafficDataset(x_train, y_train)
    val_dataset = TrafficDataset(x_val, y_val)
    test_dataset = TrafficDataset(x_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, scaler, sensor_ids

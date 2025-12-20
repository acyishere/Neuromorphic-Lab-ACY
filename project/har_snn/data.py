"""Utilities for loading and encoding the UCI HAR dataset for SNN training."""
from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

CHANNELS = [
    "body_acc_x",
    "body_acc_y",
    "body_acc_z",
    "body_gyro_x",
    "body_gyro_y",
    "body_gyro_z",
    "total_acc_x",
    "total_acc_y",
    "total_acc_z",
]


@dataclass
class HARSequenceDataset(Dataset):
    """Minimal Dataset wrapper for pre-windowed HAR data."""

    signals: torch.Tensor
    labels: torch.Tensor

    def __post_init__(self) -> None:
        if self.signals.shape[0] != self.labels.shape[0]:
            raise ValueError("Signals and labels must share the batch dimension")

    def __len__(self) -> int:
        return self.signals.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.signals[idx], self.labels[idx]


def _load_signal_file(path: pathlib.Path) -> np.ndarray:
    data = np.loadtxt(path)
    return data.astype(np.float32)


def _stack_channels(channel_arrays: Iterable[np.ndarray]) -> np.ndarray:
    stacked = np.stack(channel_arrays, axis=1)
    return stacked


def load_har_dataset(root: pathlib.Path) -> Tuple[HARSequenceDataset, HARSequenceDataset]:
    """Load the UCI HAR dataset from the extracted Kaggle archive.

    Parameters
    ----------
    root:
        Path to the root of the extracted dataset folder (the one containing
        ``train`` and ``test`` directories from the Kaggle download).

    Returns
    -------
    (train_ds, test_ds)
        Torch Dataset objects wrapping the windowed inertial sequences and labels.
    """

    train_dir = root / "train"
    test_dir = root / "test"
    if not train_dir.exists() or not test_dir.exists():
        raise FileNotFoundError(
            f"Expected 'train' and 'test' directories under {root}. Have you extracted the Kaggle archive?"
        )

    def load_split(split_dir: pathlib.Path) -> Tuple[np.ndarray, np.ndarray]:
        x_signals = []
        for channel in CHANNELS:
            path = split_dir / "Inertial Signals" / f"{channel}_train.txt"
            if not path.exists():
                path = split_dir / "Inertial Signals" / f"{channel}_{split_dir.name}.txt"
            if not path.exists():
                raise FileNotFoundError(f"Missing signal file: {path}")
            x_signals.append(_load_signal_file(path))

        x = _stack_channels(x_signals)
        y_path = split_dir / f"y_{split_dir.name}.txt"
        if not y_path.exists():
            raise FileNotFoundError(f"Missing label file: {y_path}")
        y = np.loadtxt(y_path).astype(np.int64) - 1  # zero-based classes
        return x, y

    x_train, y_train = load_split(train_dir)
    x_test, y_test = load_split(test_dir)

    train_ds = HARSequenceDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    test_ds = HARSequenceDataset(torch.from_numpy(x_test), torch.from_numpy(y_test))
    return train_ds, test_ds

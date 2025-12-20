"""SpiNNaker-oriented SNN pipeline components for UCI HAR data."""

__all__ = [
    "load_har_dataset",
    "HARSequenceDataset",
    "SpikingConvNet",
    "TrainConfig",
    "train_model",
    "evaluate_model",
    "estimate_gating_metrics",
]

from .data import HARSequenceDataset, load_har_dataset
from .model import SpikingConvNet
from .training import TrainConfig, evaluate_model, estimate_gating_metrics, train_model

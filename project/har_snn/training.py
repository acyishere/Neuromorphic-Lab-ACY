"""Training and evaluation helpers for the HAR SNN."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from .model import SpikingConvNet, TrainConfig


@dataclass
class TrainState:
    model: SpikingConvNet
    optimizer: optim.Optimizer
    criterion: nn.Module


def _prepare_loaders(train_ds, test_ds, cfg: TrainConfig) -> Tuple[DataLoader, DataLoader]:
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False)
    return train_loader, test_loader


def train_model(train_ds, test_ds, cfg: TrainConfig) -> TrainState:
    device = torch.device(cfg.device)
    model = SpikingConvNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    train_loader, test_loader = _prepare_loaders(train_ds, test_ds, cfg)

    for epoch in range(cfg.epochs):
        model.train()
        for signals, labels in train_loader:
            signals = signals.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits, gating = model(signals.transpose(1, 2), time_steps=cfg.time_steps)
            loss = criterion(logits, labels)
            # Encourage sparse gating spikes
            gating_penalty = gating.mean()
            loss = loss + 0.01 * gating_penalty
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 5 == 0:
            metrics = evaluate_model(model, test_loader, cfg)
            print(f"Epoch {epoch+1}: val_acc={metrics['accuracy']:.3f} gate_rate={metrics['mean_gate']:.4f}")

    return TrainState(model, optimizer, criterion)


def evaluate_model(model: SpikingConvNet, loader: DataLoader, cfg: TrainConfig) -> Dict[str, float]:
    device = torch.device(cfg.device)
    model.eval()
    correct = 0
    total = 0
    gate_accum = 0.0
    with torch.no_grad():
        for signals, labels in loader:
            signals = signals.to(device)
            labels = labels.to(device)
            logits, gating = model(signals.transpose(1, 2), time_steps=cfg.time_steps)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.numel()
            gate_accum += gating.mean().item() * signals.size(0)
    return {
        "accuracy": correct / max(total, 1),
        "mean_gate": gate_accum / max(total, 1),
    }


def estimate_gating_metrics(gating_rate: torch.Tensor, wake_threshold: float, cooldown: float) -> Dict[str, float]:
    """Compute simple duty-cycle metrics for gating decisions.

    This is a lightweight stand-in for SpiNNaker runtime logic. The gating rate is
    compared to wake/cool-down thresholds to estimate how often a downstream sensor
    would be activated.
    """

    active = gating_rate > wake_threshold
    duty_cycle = active.float().mean().item()
    hysteresis = torch.where(active, gating_rate > (wake_threshold - cooldown), gating_rate < (wake_threshold - cooldown))
    stability = hysteresis.float().mean().item()
    return {
        "duty_cycle": duty_cycle,
        "stability": stability,
    }

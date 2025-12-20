"""Spiking network definition approximating SpiNNaker-friendly building blocks."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch import nn


class SurrogateSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        ctx.save_for_backward(input)
        return (input > 0).float()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor]:  # type: ignore[override]
        (input,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        slope = 10.0
        surrogate = slope * torch.clamp(1 - torch.abs(input), min=0.0)
        return grad_input * surrogate


def spike_fn(x: torch.Tensor) -> torch.Tensor:
    return SurrogateSpike.apply(x)


class LIFLayer(nn.Module):
    def __init__(self, tau: float = 2.0, threshold: float = 1.0, decay_input: float = 0.9):
        super().__init__()
        self.tau = tau
        self.threshold = threshold
        self.decay_input = decay_input

    def forward(self, input: torch.Tensor, mem: torch.Tensor, spike: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mem = mem * self.decay_input + input
        spike = spike_fn(mem - self.threshold)
        mem = mem - spike * self.threshold
        return mem, spike


class SpikingConvNet(nn.Module):
    """Compact SNN with a gating head for downstream activation logic."""

    def __init__(self, num_classes: int = 6, num_channels: int = 9, hidden: int = 32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(num_channels, hidden, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
        )
        self.lif = LIFLayer()
        self.readout = nn.Linear(hidden, num_classes)
        self.gating = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor, time_steps: int = 10):
        # x: [batch, channels, time]
        b, c, t = x.shape
        mem = torch.zeros((b, self.readout.in_features), device=x.device)
        spike = torch.zeros_like(mem)
        class_out = torch.zeros((b, self.readout.out_features), device=x.device)
        gating_sum = torch.zeros((b, 1), device=x.device)
        for _ in range(time_steps):
            emb = self.encoder(x)
            emb = emb.mean(dim=2)
            mem, spike = self.lif(emb, mem, spike)
            class_out += self.readout(spike)
            gating_sum += self.gating(spike)
        class_out /= time_steps
        gating_rate = gating_sum / time_steps
        return class_out, gating_rate


@dataclass
class TrainConfig:
    lr: float = 1e-3
    epochs: int = 20
    batch_size: int = 64
    device: str = "cpu"
    time_steps: int = 10


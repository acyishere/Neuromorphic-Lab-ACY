"""Training entrypoint for the HAR spiking model.

Usage:
    python project/train_har_snn.py --data ./data/UCI_HAR_Dataset
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from har_snn import TrainConfig, evaluate_model, load_har_dataset, train_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an SNN on the UCI HAR dataset")
    parser.add_argument("--data", type=pathlib.Path, required=True, help="Path to extracted UCI HAR dataset")
    parser.add_argument("--device", default="cpu", help="Torch device, e.g., cpu or cuda")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--time-steps", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wake-threshold", type=float, default=0.2)
    parser.add_argument("--cooldown", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_ds, test_ds = load_har_dataset(args.data)
    cfg = TrainConfig(
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
        time_steps=args.time_steps,
    )
    state = train_model(train_ds, test_ds, cfg)
    eval_loader = torch.utils.data.DataLoader(test_ds, batch_size=cfg.batch_size)
    metrics = evaluate_model(state.model, eval_loader, cfg)
    print({"final_accuracy": metrics["accuracy"], "mean_gate_rate": metrics["mean_gate"]})


if __name__ == "__main__":
    main()

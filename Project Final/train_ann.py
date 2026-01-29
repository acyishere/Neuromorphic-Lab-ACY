# train_ann.py
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler


def load_har_csv(path: Path):
    df = pd.read_csv(path)
    if "Activity" not in df.columns:
        raise ValueError(f"{path} must contain column: Activity")
    if "subject" not in df.columns:
        raise ValueError(f"{path} must contain column: subject")

    y = df["Activity"].astype(str).values
    X = df.drop(columns=["Activity", "subject"]).values.astype(np.float32)
    return X, y


class MLP(nn.Module):
    def __init__(self, in_dim: int, h1: int, h2: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, h1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h2, out_dim),
        )

    def forward(self, x):
        return self.net(x)


@torch.no_grad()
def eval_model(model, X, y):
    model.eval()
    logits = model(X)
    pred = torch.argmax(logits, dim=1).cpu().numpy()
    y_np = y.cpu().numpy()
    return accuracy_score(y_np, pred), pred


def export_ann_params(model: nn.Module, out_path: Path, classes, scaler: StandardScaler):
    # Extract weights/biases as numpy float32
    # Architecture: Linear(in->h1), Linear(h1->h2), Linear(h2->out)
    layers = [m for m in model.net if isinstance(m, nn.Linear)]
    if len(layers) != 3:
        raise ValueError("Expected exactly 3 Linear layers (in->h1, h1->h2, h2->out).")

    W1 = layers[0].weight.detach().cpu().numpy().astype(np.float32)
    b1 = layers[0].bias.detach().cpu().numpy().astype(np.float32)
    W2 = layers[1].weight.detach().cpu().numpy().astype(np.float32)
    b2 = layers[1].bias.detach().cpu().numpy().astype(np.float32)
    W3 = layers[2].weight.detach().cpu().numpy().astype(np.float32)
    b3 = layers[2].bias.detach().cpu().numpy().astype(np.float32)

    payload = {
        "W1": W1, "b1": b1,
        "W2": W2, "b2": b2,
        "W3": W3, "b3": b3,
        "classes": np.array(list(classes), dtype=object),
        "scaler_mean": scaler.mean_.astype(np.float32),
        "scaler_scale": scaler.scale_.astype(np.float32),
    }
    np.savez(out_path, **payload)
    print(f"[OK] Saved ANN params to: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", type=str, default="train.csv")
    ap.add_argument("--test_csv", type=str, default="test.csv")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--h1", type=int, default=128)
    ap.add_argument("--h2", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_npz", type=str, default="ann_params_har.npz")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    X_train, y_train_raw = load_har_csv(Path(args.train_csv))
    X_test, y_test_raw = load_har_csv(Path(args.test_csv))

    le = LabelEncoder()
    y_train = le.fit_transform(y_train_raw)
    y_test = le.transform(y_test_raw)
    classes = list(le.classes_)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Xtr = torch.from_numpy(X_train).to(device)
    ytr = torch.from_numpy(y_train).long().to(device)
    Xte = torch.from_numpy(X_test).to(device)
    yte = torch.from_numpy(y_test).long().to(device)

    in_dim = X_train.shape[1]
    out_dim = len(classes)

    model = MLP(in_dim, args.h1, args.h2, out_dim, dropout=args.dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    n = Xtr.shape[0]
    idx = np.arange(n)

    for ep in range(1, args.epochs + 1):
        model.train()
        np.random.shuffle(idx)

        for start in range(0, n, args.batch_size):
            batch_idx = idx[start:start + args.batch_size]
            xb = Xtr[batch_idx]
            yb = ytr[batch_idx]

            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()

        acc, _ = eval_model(model, Xte, yte)
        print(f"Epoch {ep:02d} | Test Acc: {acc:.4f}")

    acc, pred = eval_model(model, Xte, yte)
    print("\n[REPORT]")
    print("Classes:", classes)
    print(classification_report(y_test, pred, target_names=classes, digits=2))

    export_ann_params(model, Path(args.out_npz), classes, scaler)


if __name__ == "__main__":
    main()

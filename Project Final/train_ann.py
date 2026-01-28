# train_ann.py
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, h1: int = 128, h2: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, out_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


def load_har_csv(path: Path):
    df = pd.read_csv(path)
    if "Activity" not in df.columns or "subject" not in df.columns:
        raise ValueError(f"{path} must contain 'Activity' and 'subject' columns.")

    X = df.drop(columns=["Activity", "subject"]).values.astype(np.float32)
    y = df["Activity"].values
    return X, y


def export_ann_params(model: MLP, out_path: Path, le: LabelEncoder, scaler: StandardScaler):
    # Export matrices as separate keys (robust for differing shapes)
    with torch.no_grad():
        W1 = model.fc1.weight.detach().cpu().numpy().astype(np.float32)  # [h1, in]
        b1 = model.fc1.bias.detach().cpu().numpy().astype(np.float32)    # [h1]
        W2 = model.fc2.weight.detach().cpu().numpy().astype(np.float32)  # [h2, h1]
        b2 = model.fc2.bias.detach().cpu().numpy().astype(np.float32)    # [h2]
        W3 = model.fc3.weight.detach().cpu().numpy().astype(np.float32)  # [out, h2]
        b3 = model.fc3.bias.detach().cpu().numpy().astype(np.float32)    # [out]

    np.savez(
        out_path,
        W1=W1, b1=b1,
        W2=W2, b2=b2,
        W3=W3, b3=b3,
        classes=np.array(le.classes_, dtype=object),
        scaler_mean=scaler.mean_.astype(np.float32),
        scaler_scale=scaler.scale_.astype(np.float32),
    )
    print(f"[OK] Saved ANN params to: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", type=str, default="train.csv")
    ap.add_argument("--test_csv", type=str, default="test.csv")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--h1", type=int, default=128)
    ap.add_argument("--h2", type=int, default=64)
    ap.add_argument("--out_npz", type=str, default="ann_params_har.npz")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # Determinism (best-effort)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    X_train, y_train_raw = load_har_csv(Path(args.train_csv))
    X_test, y_test_raw = load_har_csv(Path(args.test_csv))

    le = LabelEncoder()
    y_train = le.fit_transform(y_train_raw).astype(np.int64)
    y_test = le.transform(y_test_raw).astype(np.int64)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    in_dim = X_train.shape[1]
    out_dim = len(le.classes_)

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    test_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=512, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MLP(in_dim=in_dim, out_dim=out_dim, h1=args.h1, h2=args.h2).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()

        # Eval
        model.eval()
        preds = []
        with torch.no_grad():
            for xb, _ in test_dl:
                xb = xb.to(device)
                preds.append(model(xb).argmax(dim=1).cpu().numpy())
        preds = np.concatenate(preds)
        acc = accuracy_score(y_test, preds)
        print(f"Epoch {epoch:02d} | Test Acc: {acc:.4f}")

    print("\n[REPORT]")
    print("Classes:", list(le.classes_))
    print(classification_report(y_test, preds, target_names=le.classes_))

    export_ann_params(model, Path(args.out_npz), le, scaler)


if __name__ == "__main__":
    main()

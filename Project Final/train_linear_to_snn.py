# train_linear_to_snn.py
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def load_har_csv(path: Path):
    df = pd.read_csv(path)
    if "Activity" not in df.columns or "subject" not in df.columns:
        raise ValueError(f"{path} must contain 'Activity' and 'subject' columns.")
    X = df.drop(columns=["Activity", "subject"]).values.astype(np.float32)
    y = df["Activity"].values.astype(str)
    return X, y


def sigmoid(x: np.ndarray):
    # stable sigmoid
    x = np.clip(x, -20, 20)
    return 1.0 / (1.0 + np.exp(-x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", type=str, default="train.csv")
    ap.add_argument("--reduce_dim", type=int, default=48)
    ap.add_argument("--C", type=float, default=1.0, help="Inverse regularization strength for LogisticRegression.")
    ap.add_argument("--max_iter", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="model_snn.npz")
    args = ap.parse_args()

    Xtr, ytr = load_har_csv(Path(args.train_csv))
    classes = np.unique(ytr)
    classes_sorted = np.array(sorted(classes.tolist()), dtype=object)

    print(f"[INFO] Train: {Xtr.shape}, classes={len(classes_sorted)}")
    print(f"[INFO] Class order: {classes_sorted.tolist()}")

    # Scale
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr).astype(np.float32)

    # PCA
    pca = PCA(n_components=int(args.reduce_dim), random_state=int(args.seed))
    Xtr_p = pca.fit_transform(Xtr_s).astype(np.float32)
    var = float(np.sum(pca.explained_variance_ratio_))
    print(f"[INFO] PCA dim={args.reduce_dim} explained_variance={var:.2%}")

    # Linear classifier (multinomial logistic regression)
    clf = LogisticRegression(
        solver="lbfgs",
        C=float(args.C),
        max_iter=int(args.max_iter),
        random_state=int(args.seed),
        n_jobs=None,
    )
    clf.fit(Xtr_p, ytr)

    yhat = clf.predict(Xtr_p)
    acc = accuracy_score(ytr, yhat)
    print(f"[RESULT] Train accuracy={acc:.4f}")

    # Extract float weights
    # W shape: [n_classes, D], b shape: [n_classes]
    W = clf.coef_.astype(np.float32)
    b = clf.intercept_.astype(np.float32)

    # Save PCA-space normalization stats for rate encoding
    mu_p = Xtr_p.mean(axis=0).astype(np.float32)
    sd_p = Xtr_p.std(axis=0).astype(np.float32)
    sd_p[sd_p < 1e-6] = 1.0

    # Quantize to [-15,15] jointly for W and b
    max_abs = float(max(np.max(np.abs(W)), np.max(np.abs(b)), 1e-12))
    scale_q = 15.0 / max_abs
    Wq = np.clip(np.rint(W * scale_q), -15, 15).astype(np.int32)
    bq = np.clip(np.rint(b * scale_q), -15, 15).astype(np.int32)

    nz = int(np.count_nonzero(Wq)) + int(np.count_nonzero(bq))
    print(f"[INFO] Quant scale_q={scale_q:.6f}  nnz(Wq+bq)={nz}")
    print(f"[INFO] Wq range=[{Wq.min()},{Wq.max()}], bq range=[{bq.min()},{bq.max()}]")

    # Persist minimal transform parameters (no sklearn pickles)
    np.savez(
        args.out,
        # label order
        classes=classes_sorted,
        # scaler
        scaler_mean=scaler.mean_.astype(np.float32),
        scaler_scale=scaler.scale_.astype(np.float32),
        # pca
        pca_components=pca.components_.astype(np.float32),  # [D, orig]
        pca_mean=pca.mean_.astype(np.float32),              # mean in scaled space
        # pca-space stats for encoding
        pca_mu=mu_p,
        pca_sd=sd_p,
        # quantized linear model
        Wq=Wq,
        bq=bq,
        scale_q=np.array([scale_q], dtype=np.float32),
    )
    print(f"[OK] Saved: {args.out}")


if __name__ == "__main__":
    main()

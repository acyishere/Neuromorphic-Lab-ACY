# offline_snn_sim.py
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def load_ann_npz(npz_path: Path):
    data = np.load(npz_path, allow_pickle=True)
    W1 = data["W1"].astype(np.float32)
    W2 = data["W2"].astype(np.float32)
    W3 = data["W3"].astype(np.float32)
    b1 = data["b1"].astype(np.float32)
    b2 = data["b2"].astype(np.float32)
    b3 = data["b3"].astype(np.float32)
    classes = list(data["classes"])
    mean = data["scaler_mean"].astype(np.float32)
    scale = data["scaler_scale"].astype(np.float32)
    return (W1, W2, W3), (b1, b2, b3), classes, mean, scale


def load_har_csv(path: Path):
    df = pd.read_csv(path)
    if "Activity" not in df.columns or "subject" not in df.columns:
        raise ValueError(f"{path} must contain 'Activity' and 'subject' columns.")
    X = df.drop(columns=["Activity", "subject"]).values.astype(np.float32)
    y = df["Activity"].values.astype(str)
    return X, y


def standardize(X: np.ndarray, mean: np.ndarray, scale: np.ndarray):
    return ((X - mean) / (scale + 1e-12)).astype(np.float32)


def quantize_weights(W: np.ndarray, w_max: int = 15):
    max_abs = float(np.max(np.abs(W)) + 1e-12)
    s = w_max / max_abs
    Wq = np.clip(np.rint(W * s), -w_max, w_max).astype(np.int32)
    return Wq, s


def quantize_bias(b: np.ndarray, scale: float, b_max: int = 127):
    return np.clip(np.rint(b * scale), -b_max, b_max).astype(np.int32)


def bipolar_rates(x: np.ndarray, r_max_hz: float):
    # ON/OFF encoding
    pos = np.maximum(x, 0.0)
    neg = np.maximum(-x, 0.0)

    maxv = float(np.max(pos + neg) + 1e-12)
    pos01 = pos / maxv
    neg01 = neg / maxv

    rates = np.concatenate([pos01, neg01], axis=0) * r_max_hz
    return rates.astype(np.float32)


def poisson_spikes(rates_hz: np.ndarray, T: int, rng: np.random.Generator):
    p = np.clip(rates_hz / 1000.0, 0.0, 1.0)
    return (rng.random((T, rates_hz.shape[0])) < p[None, :]).astype(np.int8)


def lif_layer(spikes_in: np.ndarray, Wq: np.ndarray, bq: np.ndarray, thr: float, decay: float, bias_gain: float):
    T, N_in = spikes_in.shape
    N_out, N_in2 = Wq.shape
    if N_in != N_in2:
        raise ValueError(f"Shape mismatch: spikes_in {N_in} vs Wq {N_in2}")
    if bq.shape[0] != N_out:
        raise ValueError(f"Shape mismatch: bq {bq.shape[0]} vs Wq out {N_out}")

    v = np.zeros(N_out, dtype=np.float32)
    out = np.zeros((T, N_out), dtype=np.int8)
    b_term = bias_gain * bq.astype(np.float32)

    for t in range(T):
        v = decay * v + (Wq @ spikes_in[t].astype(np.float32)) + b_term
        fired = v >= thr
        out[t, fired] = 1
        v[fired] -= thr
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann_npz", type=str, default="ann_params_har.npz")
    ap.add_argument("--test_csv", type=str, default="test.csv")
    ap.add_argument("--num_samples", type=int, default=200)
    ap.add_argument("--T", type=int, default=200)
    ap.add_argument("--r_max_hz", type=float, default=200.0)
    ap.add_argument("--w_max", type=int, default=15)
    ap.add_argument("--decay", type=float, default=0.92)

    ap.add_argument("--thr1", type=float, default=25.0)
    ap.add_argument("--thr2", type=float, default=25.0)
    ap.add_argument("--thr3", type=float, default=60.0)

    ap.add_argument("--bias1", type=float, default=0.05)
    ap.add_argument("--bias2", type=float, default=0.05)
    ap.add_argument("--bias3", type=float, default=0.02)

    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    (W1, W2, W3), (b1, b2, b3), classes, mean, scale = load_ann_npz(Path(args.ann_npz))
    X_test, y_raw = load_har_csv(Path(args.test_csv))

    class_to_idx = {c: i for i, c in enumerate(classes)}
    y = np.array([class_to_idx[v] for v in y_raw], dtype=np.int64)

    X_test = standardize(X_test, mean, scale)

    # Expand first-layer weights for bipolar input: [W, -W]
    W1_bipolar = np.concatenate([W1, -W1], axis=1)

    W1q, s1 = quantize_weights(W1_bipolar, w_max=args.w_max)
    W2q, s2 = quantize_weights(W2, w_max=args.w_max)
    W3q, s3 = quantize_weights(W3, w_max=args.w_max)

    b1q = quantize_bias(b1, scale=s1)
    b2q = quantize_bias(b2, scale=s2)
    b3q = quantize_bias(b3, scale=s3)

    rng = np.random.default_rng(args.seed)
    tested = min(args.num_samples, X_test.shape[0])
    correct = 0

    for i in range(tested):
        rates = bipolar_rates(X_test[i], r_max_hz=args.r_max_hz)
        s_in = poisson_spikes(rates, T=args.T, rng=rng)

        s_h1 = lif_layer(s_in, W1q, b1q, thr=args.thr1, decay=args.decay, bias_gain=args.bias1)
        s_h2 = lif_layer(s_h1, W2q, b2q, thr=args.thr2, decay=args.decay, bias_gain=args.bias2)
        s_out = lif_layer(s_h2, W3q, b3q, thr=args.thr3, decay=args.decay, bias_gain=args.bias3)

        counts = s_out.sum(axis=0)
        pred = int(np.argmax(counts))
        correct += int(pred == int(y[i]))

        if i < 10:
            print(f"Sample {i:03d} true={classes[y[i]]} pred={classes[pred]} counts={counts.tolist()}")

    acc = correct / tested if tested else 0.0
    print(f"\n[OFFLINE SNN (BIPOLAR)] Tested {tested} samples | Accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()

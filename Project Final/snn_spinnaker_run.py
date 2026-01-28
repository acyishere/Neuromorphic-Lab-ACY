# snn_spinnaker_run.py
# -*- coding: utf-8 -*-
#
# ANN → SNN (SpiNNaker2) pipeline with BIPOLAR (ON/OFF) input encoding.
#
# Key points:
# - Input neurons are doubled: 2*D (pos + neg) to preserve sign of standardized features
# - First-layer weights expanded: W1_bipolar = [W1, -W1]
# - --dry_run works on machines WITHOUT the spinnaker2 Python stack installed
#   (imports are deferred until hardware run)
#
# Usage:
#   Dry run (no hardware, no spinnaker2 install needed):
#     python snn_spinnaker_run.py --dry_run
#
#   Hardware run (must be on lab machine/network with spinnaker2 installed):
#     python snn_spinnaker_run.py --s2_ip 192.168.1.53 --num_samples 20 --duration_ms 200

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd


def load_ann_npz(npz_path: Path):
    data = np.load(npz_path, allow_pickle=True)
    weights = [
        data["W1"].astype(np.float32),
        data["W2"].astype(np.float32),
        data["W3"].astype(np.float32),
    ]
    biases = [
        data["b1"].astype(np.float32),
        data["b2"].astype(np.float32),
        data["b3"].astype(np.float32),
    ]
    classes = list(data["classes"])
    scaler_mean = data["scaler_mean"].astype(np.float32)
    scaler_scale = data["scaler_scale"].astype(np.float32)
    return weights, biases, classes, scaler_mean, scaler_scale


def load_har_csv(path: Path):
    df = pd.read_csv(path)
    if "Activity" not in df.columns or "subject" not in df.columns:
        raise ValueError(f"{path} must contain 'Activity' and 'subject' columns.")
    X = df.drop(columns=["Activity", "subject"]).values.astype(np.float32)
    y = df["Activity"].values
    return X, y


def standardize(X: np.ndarray, mean: np.ndarray, scale: np.ndarray):
    return ((X - mean) / (scale + 1e-12)).astype(np.float32)


def quantize_weights(W: np.ndarray, w_max: int = 15):
    max_abs = float(np.max(np.abs(W)) + 1e-12)
    scale = w_max / max_abs
    Wq = np.clip(np.rint(W * scale), -w_max, w_max).astype(np.int32)
    return Wq, scale


def dense_to_conn_list(Wq: np.ndarray, delay: int = 1):
    """
    Convert dense int weight matrix to connection list.
    Wq shape: [post, pre] = [out, in]
    Each entry: [pre_idx, post_idx, weight, delay]
    """
    out_dim, in_dim = Wq.shape
    conns = []
    for post in range(out_dim):
        row = Wq[post]
        nz = np.nonzero(row)[0]
        for pre in nz:
            conns.append([int(pre), int(post), int(row[pre]), int(delay)])
    return conns


def bipolar_rates(x: np.ndarray, r_max_hz: float):
    """
    BIPOLAR encoding:
      pos = max(x, 0), neg = max(-x, 0)
    Normalize per-sample by max(pos+neg), then map to firing rates.
    Output length = 2*D.
    """
    pos = np.maximum(x, 0.0)
    neg = np.maximum(-x, 0.0)
    maxv = float(np.max(pos + neg) + 1e-12)
    pos01 = pos / maxv
    neg01 = neg / maxv
    rates = np.concatenate([pos01, neg01], axis=0) * r_max_hz
    return rates.astype(np.float32)


def poisson_spike_list(rates_hz: np.ndarray, duration_ms: int, seed: int):
    """
    Generate Poisson spikes with dt=1ms using Bernoulli(rate/1000).
    Returns dict {neuron_id: [times]} compatible with neuron_model="spike_list".
    """
    rng = np.random.default_rng(seed)
    p = np.clip(rates_hz / 1000.0, 0.0, 1.0)
    N = rates_hz.shape[0]

    spikes = {}
    for i in range(N):
        if p[i] <= 0.0:
            spikes[i] = []
            continue
        times = [t for t in range(duration_ms) if rng.random() < p[i]]
        spikes[i] = times
    return spikes


def decode_output(spike_dict: dict, out_dim: int):
    counts = np.zeros(out_dim, dtype=np.int32)
    for k, times in spike_dict.items():
        kk = int(k)
        if 0 <= kk < out_dim:
            counts[kk] = len(times)
    pred = int(np.argmax(counts))
    return pred, counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann_npz", type=str, default="ann_params_har.npz")
    ap.add_argument("--test_csv", type=str, default="test.csv")
    ap.add_argument("--s2_ip", type=str, default=os.environ.get("S2_IP", "192.168.1.53"))
    ap.add_argument("--num_samples", type=int, default=50)
    ap.add_argument("--duration_ms", type=int, default=200)
    ap.add_argument("--r_max_hz", type=float, default=200.0)
    ap.add_argument("--w_max", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry_run", action="store_true", help="Do conversion + stats only (no hardware).")
    args = ap.parse_args()

    weights, _biases, classes, mean, scale = load_ann_npz(Path(args.ann_npz))
    X_test, y_test_raw = load_har_csv(Path(args.test_csv))

    # Map string label -> index (ANN class order)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y_test = np.array([class_to_idx[v] for v in y_test_raw], dtype=np.int64)

    X_test = standardize(X_test, mean, scale)

    # ---- BIPOLAR EXPANSION ----
    W1 = weights[0]  # [h1, D]
    W2 = weights[1]  # [h2, h1]
    W3 = weights[2]  # [out, h2]

    D = W1.shape[1]
    W1_bipolar = np.concatenate([W1, -W1], axis=1)  # [h1, 2D]
    in_dim = 2 * D
    h1 = W1.shape[0]
    h2 = W2.shape[0]
    out_dim = W3.shape[0]

    # Quantize weights
    W1q, s1 = quantize_weights(W1_bipolar, w_max=args.w_max)
    W2q, s2 = quantize_weights(W2, w_max=args.w_max)
    W3q, s3 = quantize_weights(W3, w_max=args.w_max)

    # Build connection lists once
    conns1 = dense_to_conn_list(W1q, delay=1)
    conns2 = dense_to_conn_list(W2q, delay=1)
    conns3 = dense_to_conn_list(W3q, delay=1)

    print("[INFO] BIPOLAR encoding enabled")
    print(f"[INFO] Shapes: input={in_dim} (2*{D}), h1={h1}, h2={h2}, out={out_dim}")
    print(f"[INFO] Quant scales: s1={s1:.3f}, s2={s2:.3f}, s3={s3:.3f}")
    print(f"[INFO] Synapses: L1={len(conns1)}, L2={len(conns2)}, L3={len(conns3)}")

    if args.dry_run:
        print("[DRY RUN] Not connecting to hardware. Conversion looks OK.")
        return

    # Import SpiNNaker stack ONLY when doing a hardware run
    try:
        from spinnaker2 import hardware, snn
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "spinnaker2 is not installed in this environment. "
            "Run with --dry_run on your laptop, or run this script on the lab machine "
            "that has the SpiNNaker2 software stack installed."
        ) from e

    # LIF parameters (baseline; tune later if needed)
    lif_params = {
        "threshold": 10.0,
        "alpha_decay": 0.9,
        "i_offset": 0.0,
        "v_reset": 0.0,
        "reset": "reset_by_subtraction",
    }

    # Connect to SpiNNaker2
    hw = hardware.SpiNNaker2Chip(eth_ip=args.s2_ip)
    print(f"[OK] Connected to SpiNNaker2 at {args.s2_ip}")

    tested = min(args.num_samples, X_test.shape[0])
    correct = 0

    for idx in range(tested):
        x = X_test[idx]
        y = int(y_test[idx])

        rates = bipolar_rates(x, r_max_hz=args.r_max_hz)  # length 2D
        spike_in = poisson_spike_list(rates, duration_ms=args.duration_ms, seed=args.seed + idx)

        # Build network per sample (simple + robust)
        net = snn.Network(f"HAR_SNN_BIPOLAR_{idx}")

        stim = snn.Population(
            size=in_dim,
            neuron_model="spike_list",
            params=spike_in,
            name="Input",
        )
        hid1 = snn.Population(size=h1, neuron_model="lif_curr_exp", params=lif_params, name="H1", record=["spikes"])
        hid2 = snn.Population(size=h2, neuron_model="lif_curr_exp", params=lif_params, name="H2", record=["spikes"])
        out = snn.Population(size=out_dim, neuron_model="lif_curr_exp", params=lif_params, name="OUT", record=["spikes"])

        p1 = snn.Projection(pre=stim, post=hid1, connections=conns1)
        p2 = snn.Projection(pre=hid1, post=hid2, connections=conns2)
        p3 = snn.Projection(pre=hid2, post=out, connections=conns3)

        net.add(stim, hid1, hid2, out, p1, p2, p3)

        hw.run(net, args.duration_ms)

        out_spikes = out.get_spikes()
        pred, counts = decode_output(out_spikes, out_dim=out_dim)
        ok = (pred == y)
        correct += int(ok)

        if idx < 10:
            print(
                f"Sample {idx:04d} | true={classes[y]} pred={classes[pred]} "
                f"| out_counts={counts.tolist()} | ok={ok}"
            )

    acc = correct / tested if tested else 0.0
    print(f"\n[RESULT] Tested {tested} samples | Accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()

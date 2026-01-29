# snn_spinnaker_run.py
# -*- coding: utf-8 -*-

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd


def load_ann_npz(npz_path: Path):
    data = np.load(npz_path, allow_pickle=True)
    W1 = data["W1"].astype(np.float32)
    W2 = data["W2"].astype(np.float32)
    W3 = data["W3"].astype(np.float32)
    classes = list(data["classes"])
    mean = data["scaler_mean"].astype(np.float32)
    scale = data["scaler_scale"].astype(np.float32)
    return (W1, W2, W3), classes, mean, scale


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


def dense_to_conn_list(Wq: np.ndarray, delay: int = 1):
    # Wq: [post, pre]
    out_dim, _ = Wq.shape
    conns = []
    for post in range(out_dim):
        row = Wq[post]
        nz = np.nonzero(row)[0]
        for pre in nz:
            conns.append([int(pre), int(post), int(row[pre]), int(delay)])
    return conns


def bipolar_rates(x: np.ndarray, r_max_hz: float):
    pos = np.maximum(x, 0.0)
    neg = np.maximum(-x, 0.0)
    maxv = float(np.max(pos + neg) + 1e-12)
    rates = np.concatenate([pos / maxv, neg / maxv], axis=0) * r_max_hz
    return rates.astype(np.float32)


def poisson_spike_list(rates_hz: np.ndarray, duration_ms: int, seed: int):
    rng = np.random.default_rng(seed)
    p = np.clip(rates_hz / 1000.0, 0.0, 1.0)
    N = rates_hz.shape[0]
    spikes = {}
    for i in range(N):
        if p[i] <= 0.0:
            spikes[i] = []
        else:
            spikes[i] = [t for t in range(duration_ms) if rng.random() < p[i]]
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
    ap.add_argument("--s2_ip", type=str, default=os.environ.get("S2_IP", "192.168.1.2"))
    ap.add_argument("--num_samples", type=int, default=3)
    ap.add_argument("--duration_ms", type=int, default=200)
    ap.add_argument("--r_max_hz", type=float, default=200.0)
    ap.add_argument("--w_max", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--shards", type=int, default=4, help="Number of diagonal shards (default 4).")
    args = ap.parse_args()

    (W1, W2, W3), classes, mean, scale = load_ann_npz(Path(args.ann_npz))
    X_test, y_raw = load_har_csv(Path(args.test_csv))
    X_test = standardize(X_test, mean, scale)

    class_to_idx = {c: i for i, c in enumerate(classes)}
    y_test = np.array([class_to_idx[v] for v in y_raw], dtype=np.int64)

    # Expected ANN shapes: W1 [128,561], W2 [64,128], W3 [6,64]
    if W1.shape != (128, 561) or W2.shape != (64, 128) or W3.shape != (6, 64):
        raise ValueError(f"Unexpected ANN shapes: W1={W1.shape}, W2={W2.shape}, W3={W3.shape}")

    D = 561
    in_dim = 2 * D  # bipolar ON/OFF
    out_dim = 6

    # Bipolar expansion for W1: [W, -W] -> [128,1122]
    W1_bipolar = np.concatenate([W1, -W1], axis=1)

    # Quantize
    W1q, s1 = quantize_weights(W1_bipolar, w_max=args.w_max)
    W2q, s2 = quantize_weights(W2, w_max=args.w_max)
    W3q, s3 = quantize_weights(W3, w_max=args.w_max)

    # ----- 4-way diagonal sharding (fix SRAM) -----
    shards = int(args.shards)
    if 128 % shards != 0 or 64 % shards != 0:
        raise ValueError("shards must divide H1=128 and H2=64 (valid: 1,2,4,8,16,32,64).")

    h1_sh = 128 // shards   # e.g., 32
    h2_sh = 64 // shards    # e.g., 16

    conns1 = []
    conns2 = []
    conns3 = []

    # Build per-shard connection lists (block-diagonal):
    # Input -> H1_shard_i
    # H1_shard_i -> H2_shard_i
    # H2_shard_i -> OUT (all contribute)
    for i in range(shards):
        # W1 block: rows for H1 shard
        W1_blk = W1q[i*h1_sh:(i+1)*h1_sh, :]          # [h1_sh, in_dim]
        conns1.append(dense_to_conn_list(W1_blk, delay=1))

        # W2 diagonal block: rows for H2 shard, cols for matching H1 shard
        W2_blk = W2q[i*h2_sh:(i+1)*h2_sh, i*h1_sh:(i+1)*h1_sh]  # [h2_sh, h1_sh]
        conns2.append(dense_to_conn_list(W2_blk, delay=1))

        # W3 slice: OUT rows, cols for H2 shard
        W3_blk = W3q[:, i*h2_sh:(i+1)*h2_sh]          # [out_dim, h2_sh]
        conns3.append(dense_to_conn_list(W3_blk, delay=1))

    # Print info
    print("[INFO] BIPOLAR encoding enabled")
    print(f"[INFO] Sharding: {shards} shards | H1={shards}x{h1_sh}, H2={shards}x{h2_sh}")
    print(f"[INFO] Shapes: input={in_dim}, out={out_dim}")
    print(f"[INFO] Quant scales: s1={s1:.3f}, s2={s2:.3f}, s3={s3:.3f}")
    print("[INFO] Synapses per shard:",
          " | ".join([f"L1[{i}]={len(conns1[i])}, L2[{i}]={len(conns2[i])}, L3[{i}]={len(conns3[i])}"
                     for i in range(shards)]))

    if args.dry_run:
        print("[DRY RUN] Not connecting to hardware. Conversion looks OK.")
        return

    # Import stack only for hardware run
    from spinnaker2 import hardware, snn

    lif_params = {
        "threshold": 10.0,
        "alpha_decay": 0.9,
        "i_offset": 0.0,
        "v_reset": 0.0,
        "reset": "reset_by_subtraction",
    }

    hw = hardware.SpiNNaker2Chip(eth_ip=args.s2_ip)
    print(f"[OK] Connected to SpiNNaker2 at {args.s2_ip}")

    tested = min(args.num_samples, X_test.shape[0])
    correct = 0

    for idx in range(tested):
        rates = bipolar_rates(X_test[idx], r_max_hz=args.r_max_hz)
        spike_in = poisson_spike_list(rates, duration_ms=args.duration_ms, seed=args.seed + idx)

        net = snn.Network(f"HAR_SNN_BIPOLAR_SHARDED_{idx}")

        stim = snn.Population(size=in_dim, neuron_model="spike_list", params=spike_in, name="Input")

        # Build shard populations
        h1_pops = []
        h2_pops = []
        for i in range(shards):
            h1_pops.append(
                snn.Population(size=h1_sh, neuron_model="lif_curr_exp", params=lif_params,
                               name=f"H1_{i}", record=["spikes"])
            )
            h2_pops.append(
                snn.Population(size=h2_sh, neuron_model="lif_curr_exp", params=lif_params,
                               name=f"H2_{i}", record=["spikes"])
            )

        out = snn.Population(size=out_dim, neuron_model="lif_curr_exp", params=lif_params,
                             name="OUT", record=["spikes"])

        # Projections per shard
        projs = []
        for i in range(shards):
            p1 = snn.Projection(pre=stim, post=h1_pops[i], connections=conns1[i])
            p2 = snn.Projection(pre=h1_pops[i], post=h2_pops[i], connections=conns2[i])
            p3 = snn.Projection(pre=h2_pops[i], post=out, connections=conns3[i])
            projs.extend([p1, p2, p3])

        net.add(stim, *h1_pops, *h2_pops, out, *projs)

        hw.run(net, args.duration_ms)

        out_spikes = out.get_spikes()
        pred, counts = decode_output(out_spikes, out_dim=out_dim)
        ok = (pred == int(y_test[idx]))
        correct += int(ok)

        if idx < 10:
            print(f"Sample {idx:04d} | true={classes[int(y_test[idx])]} pred={classes[pred]} "
                  f"| out_counts={counts.tolist()} | ok={ok}")

    acc = correct / tested if tested else 0.0
    print(f"\n[RESULT] Tested {tested} samples | Accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()

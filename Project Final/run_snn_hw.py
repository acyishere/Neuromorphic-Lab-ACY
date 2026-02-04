# run_snn_hw.py
# -*- coding: utf-8 -*-

import argparse
import os
import uuid
from pathlib import Path

import numpy as np
import pandas as pd


def load_har_csv(path: Path):
    df = pd.read_csv(path)
    if "Activity" not in df.columns or "subject" not in df.columns:
        raise ValueError(f"{path} must contain 'Activity' and 'subject' columns.")
    X = df.drop(columns=["Activity", "subject"]).values.astype(np.float32)
    y = df["Activity"].values.astype(str)
    return X, y


def sigmoid(x: np.ndarray):
    x = np.clip(x, -20, 20)
    return 1.0 / (1.0 + np.exp(-x))


def transform_with_saved_params(X: np.ndarray, scaler_mean, scaler_scale, pca_mean, pca_components):
    """
    X: [N, 561]
    scaler: Xs = (X - mean)/scale
    pca: Xp = (Xs - pca_mean) @ components.T
    """
    Xs = (X - scaler_mean) / scaler_scale
    Xs = Xs.astype(np.float32)
    Xp = (Xs - pca_mean) @ pca_components.T
    return Xp.astype(np.float32)


def rates_to_spike_times(rates_hz: np.ndarray, duration_ms: int, sys_tick_ms: int = 1):
    """
    Convert per-neuron rates (Hz) to spike time lists (integer timesteps).
    timestep assumed 1ms unless sys_tick differs.
    """
    T = int(duration_ms // sys_tick_ms)
    spikes = {}

    for i, r in enumerate(rates_hz):
        r = float(r)
        if r <= 0.0:
            continue
        # expected inter-spike interval in ms
        isi_ms = 1000.0 / r
        step = max(1, int(round(isi_ms / sys_tick_ms)))
        times = list(range(0, T, step))
        if times:
            spikes[int(i)] = times
    return spikes


def build_dense_conns(Wq: np.ndarray, bq: np.ndarray, bias_index: int, delay: int = 1):
    """
    Wq: [C, D] int in [-15,15]
    bq: [C]
    Connections are [pre, post, w, delay]
    We connect each input j -> each class c using Wq[c,j]
    bias input (bias_index) -> each class c using bq[c]
    """
    C, D = Wq.shape
    conns = []

    # feature inputs
    for c in range(C):
        for j in range(D):
            w = int(Wq[c, j])
            if w != 0:
                conns.append([int(j), int(c), w, int(delay)])

    # bias input
    for c in range(C):
        w = int(bq[c])
        if w != 0:
            conns.append([int(bias_index), int(c), w, int(delay)])

    return conns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="model_snn.npz")
    ap.add_argument("--test_csv", type=str, default="test.csv")
    ap.add_argument("--num_samples", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)

    # board / timing
    ap.add_argument("--stm_ip", type=str, default=os.environ.get("STM_IP", "192.168.1.2"))
    ap.add_argument("--duration_ms", type=int, default=150)
    ap.add_argument("--max_rate", type=int, default=150, help="Max input rate (Hz) after sigmoid mapping.")
    ap.add_argument("--bias_rate", type=int, default=150, help="Bias input rate (Hz).")

    # LIF params
    ap.add_argument("--lif_threshold", type=float, default=12.0)
    ap.add_argument("--alpha_decay", type=float, default=0.9)
    ap.add_argument("--atoms_per_core", type=int, default=64)

    # decoding window
    ap.add_argument("--late_frac", type=float, default=0.0, help="If >0, only count spikes after late_frac*T.")
    
    ap.add_argument("--sample_idx", type=int, default=0,
                help="Which test sample index to run (0-based).")


    args = ap.parse_args()
    rng = np.random.default_rng(int(args.seed))

    # load model
    m = np.load(args.model, allow_pickle=True)
    classes = m["classes"].astype(object)
    scaler_mean = m["scaler_mean"].astype(np.float32)
    scaler_scale = m["scaler_scale"].astype(np.float32)
    pca_components = m["pca_components"].astype(np.float32)
    pca_mean = m["pca_mean"].astype(np.float32)
    pca_mu = m["pca_mu"].astype(np.float32)
    pca_sd = m["pca_sd"].astype(np.float32)
    Wq = m["Wq"].astype(np.int32)  # [C, D]
    bq = m["bq"].astype(np.int32)  # [C]

    C, D = Wq.shape
    bias_index = D  # last input is bias

    print(f"[INFO] Loaded model: C={C}, D={D} (+bias=1), classes={classes.tolist()}")
    print(f"[INFO] Board stm_ip={args.stm_ip}, duration_ms={args.duration_ms}")

    # load test data
    Xte, yte = load_har_csv(Path(args.test_csv))
    Xte_p = transform_with_saved_params(Xte, scaler_mean, scaler_scale, pca_mean, pca_components)

    tested = min(int(args.num_samples), Xte_p.shape[0])
    if tested <= 0:
        raise SystemExit("No samples to test.")

    # Import spinnaker2 only when needed
    from spinnaker2 import hardware, snn

    from spinnaker2 import hardware, snn

    lif_params = {
        "threshold": float(args.lif_threshold),
        "alpha_decay": float(args.alpha_decay),
        "i_offset": 0.0,
        "v_init": 0.0,
        "v_reset": 0.0,
        "reset": "reset_by_subtraction",
    }

    conns = build_dense_conns(Wq=Wq, bq=bq, bias_index=bias_index, delay=1)

    correct = 0
    for i in range(tested):
        # IMPORTANT: create a fresh hardware object per run
        hw = hardware.SpiNNcloud48NodeBoard(stm_ip=str(args.stm_ip))

        x = Xte_p[i]

        z = (x - pca_mu) / pca_sd
        a = sigmoid(z)
        rates = a * float(args.max_rate)

        rates_full = np.concatenate([rates, np.array([float(args.bias_rate)], dtype=np.float32)], axis=0)
        input_spikes = rates_to_spike_times(rates_full, duration_ms=int(args.duration_ms), sys_tick_ms=1)

        net_name = f"HAR_LINEAR_{uuid.uuid4().hex[:8]}"
        net = snn.Network(net_name)

        stim = snn.Population(size=D + 1, neuron_model="spike_list", params=input_spikes, name="stim")
        out = snn.Population(size=C, neuron_model="lif", params=lif_params, name="out", record=["spikes"])

        if hasattr(stim, "set_max_atoms_per_core"):
            stim.set_max_atoms_per_core(int(args.atoms_per_core))
        if hasattr(out, "set_max_atoms_per_core"):
            out.set_max_atoms_per_core(int(args.atoms_per_core))

        proj = snn.Projection(pre=stim, post=out, connections=conns)
        net.add(stim, out, proj)

        hw.run(net, int(args.duration_ms))

        sp = out.get_spikes()
        late_t = int(float(args.duration_ms) * float(args.late_frac))

        counts = np.zeros(C, dtype=np.int32)
        for nid, times in sp.items():
            nid = int(nid)
            counts[nid] = sum(1 for t in times if int(t) >= late_t)

        pred_idx = int(np.argmax(counts))
        pred = str(classes[pred_idx])
        true = str(yte[i])
        correct += int(pred == true)

        print(f"[HW] {i+1:3d}/{tested} true={true:>18s} pred={pred:>18s} counts={counts.tolist()}")


    acc = correct / float(tested)
    print(f"\n[RESULT] HW accuracy (on {tested} samples) = {acc:.4f} ({correct}/{tested})")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
HAR classification on SpiNNaker2 using a compact Hopfield network.

Pipeline
- train.csv -> StandardScaler -> PCA(reduce_dim)
- class prototypes (mean per class) in PCA space
- patterns = bipolar(prototypes)
- Hopfield weights (Hebbian or Storkey)
- Quantize + (SYMMETRIC) sparsify (top-k per neuron) -> very low synapse count
- (Optional) Recurrent Hopfield run on SpiNNaker2 (disabled if relax_iter=0)
- Decode by similarity to prototypes
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd


# -------------------------
# Data utils
# -------------------------
def load_har_csv(path: Path):
    df = pd.read_csv(path)
    if "Activity" not in df.columns or "subject" not in df.columns:
        raise ValueError(f"{path} must contain 'Activity' and 'subject' columns.")
    X = df.drop(columns=["Activity", "subject"]).values.astype(np.float32)
    y = df["Activity"].values.astype(str)
    return X, y


def compute_class_prototypes(X: np.ndarray, y: np.ndarray, classes: list[str]):
    protos = {}
    for cls in classes:
        m = (y == cls)
        if np.any(m):
            protos[cls] = X[m].mean(axis=0)
    return protos


def binarize_bipolar(x: np.ndarray, threshold: float = 0.0):
    return np.where(x >= threshold, 1.0, -1.0).astype(np.float32)


# -------------------------
# Hopfield learning
# -------------------------
def hopfield_weights_hebb(patterns: np.ndarray):
    _, N = patterns.shape
    W = (patterns.T @ patterns) / float(N)
    np.fill_diagonal(W, 0.0)
    return W.astype(np.float32)


def hopfield_weights_storkey(patterns: np.ndarray):
    P, N = patterns.shape
    W = np.zeros((N, N), dtype=np.float32)
    for p in range(P):
        x = patterns[p].astype(np.float32)
        h = W @ x
        xxT = np.outer(x, x)
        xhT = np.outer(x, h)
        hxT = np.outer(h, x)
        dW = (xxT - xhT - hxT) / float(N)
        np.fill_diagonal(dW, 0.0)
        W += dW
    np.fill_diagonal(W, 0.0)
    return W.astype(np.float32)


# -------------------------
# Quantize + symmetric sparsify
# -------------------------
def quantize_weights(W: np.ndarray, w_max: int = 15):
    max_abs = float(np.max(np.abs(W)) + 1e-12)
    s = w_max / max_abs
    Wq = np.clip(np.rint(W * s), -w_max, w_max).astype(np.int32)
    return Wq, float(s)


def sparsify_topk_symmetric(Wq: np.ndarray, k: int):
    N = Wq.shape[0]
    if k <= 0:
        return np.zeros_like(Wq)

    if k >= (N - 1):
        Wdense = Wq.copy()
        np.fill_diagonal(Wdense, 0)
        Wsym = ((Wdense.astype(np.int32) + Wdense.T.astype(np.int32)) // 2).astype(np.int32)
        np.fill_diagonal(Wsym, 0)
        return Wsym

    mask = np.zeros((N, N), dtype=bool)
    for i in range(N):
        row = Wq[i]
        nz = np.nonzero(row)[0]
        nz = nz[nz != i]
        if nz.size == 0:
            continue
        idx = nz[np.argsort(np.abs(row[nz]))[::-1]]
        keep = idx[:k]
        mask[i, keep] = True

    mask = mask | mask.T
    W2 = np.zeros_like(Wq)
    W2[mask] = Wq[mask]

    Wsym = ((W2.astype(np.int32) + W2.T.astype(np.int32)) // 2).astype(np.int32)
    np.fill_diagonal(Wsym, 0)
    return Wsym


def to_conn_list(Wq_post_pre: np.ndarray, delay: int = 1):
    """
    Convert weight matrix [post, pre] to SpiNNaker2 conn list [pre, post, w, d]
    """
    N = Wq_post_pre.shape[0]
    conns = []
    for post in range(N):
        row = Wq_post_pre[post]
        nz = np.nonzero(row)[0]
        for pre in nz:
            w = int(row[pre])
            if w != 0 and pre != post:
                conns.append([int(pre), int(post), w, int(delay)])
    return conns


# -------------------------
# Local (numpy) inference for debugging
# -------------------------
def hopfield_relax_async(state: np.ndarray, W: np.ndarray, max_iter: int = 50, seed: int = 0):
    rng = np.random.default_rng(seed)
    s = state.copy().astype(np.float32)
    N = s.size
    for _ in range(max_iter):
        changed = False
        for i in rng.permutation(N):
            h = float(W[i] @ s)
            new = 1.0 if h >= 0 else -1.0
            if new != s[i]:
                s[i] = new
                changed = True
        if not changed:
            break
    return s


def classify_by_similarity(final_state: np.ndarray, prototypes: dict, threshold: float = 0.0):
    best_cls, best_sim = None, -1e18
    for cls, proto in prototypes.items():
        pb = binarize_bipolar(proto, threshold=threshold)
        sim = float(final_state @ pb)
        if sim > best_sim:
            best_sim = sim
            best_cls = cls
    return best_cls, best_sim


# -------------------------
# SpiNNaker2 run
# -------------------------
def set_atoms(pop, n: int):
    if hasattr(pop, "set_max_atoms_per_core"):
        pop.set_max_atoms_per_core(int(n))


def run_one_on_spinnaker(
    query_bin: np.ndarray,
    Wq2: np.ndarray,
    prototypes: dict,
    s2_ip: str,
    duration_ms: int,
    input_burst_hz: int,
    input_burst_ms: int,
    late_frac: float,
    lif_threshold: float,
    atoms_per_core: int,
    rec_clip: int,
    relax_iter: int,
    run_id: int = 0,
):
    """
    If relax_iter <= 0  => NO recurrent Hopfield projection (relax disabled on hardware).
    Else               => recurrent projection enabled with clipped weights.
    """
    from spinnaker2 import hardware, snn

    N = int(query_bin.size)
    pos_idx = np.where(query_bin > 0)[0].astype(int)

    # Input drive spikes for +1 neurons
    if input_burst_hz <= 0:
        step = 5
    else:
        step = max(1, int(round(1000.0 / float(input_burst_hz))))
    drive_times = list(range(0, int(input_burst_ms), step))
    spikes = {i: (drive_times if i in pos_idx else []) for i in range(N)}

    lif_params = {
        "threshold": float(lif_threshold),
        "alpha_decay": 0.90,
        "i_offset": 0.0,
        "v_reset": 0.0,
        "reset": "reset_by_subtraction",
    }

    hw = hardware.SpiNNcloud48NodeBoard(stm_ip=s2_ip)
    net = snn.Network(f"HOPFIELD_HAR_{run_id}")

    stim = snn.Population(size=N, neuron_model="spike_list", params=spikes, name="STIM")
    hop = snn.Population(size=N, neuron_model="lif", params=lif_params, name="HOP", record=["spikes"])

    set_atoms(stim, atoms_per_core)
    set_atoms(hop, atoms_per_core)

    # one-to-one strong excitation for +1 neurons
    in_w = 15
    in_conns = [[int(i), int(i), int(in_w), 1] for i in pos_idx]

    net.add(stim, hop)
    net.add(snn.Projection(pre=stim, post=hop, connections=in_conns))

    # recurrent projection only if relax_iter > 0
    if int(relax_iter) > 0:
        Wc = np.clip(Wq2, -int(rec_clip), int(rec_clip)).astype(np.int32)
        np.fill_diagonal(Wc, 0)
        rec_conns = to_conn_list(Wc, delay=1)
        net.add(snn.Projection(pre=hop, post=hop, connections=rec_conns))

    hw.run(net, int(duration_ms))

    # Decode from late-window spikes
    sp = hop.get_spikes()
    late_t = int(float(duration_ms) * float(late_frac))

    counts = np.zeros(N, dtype=np.int32)
    for nid, times in sp.items():
        # keep spikes that occur in late window
        counts[int(nid)] = sum(1 for t in times if int(t) >= late_t)

    final_state = np.where(counts > 0, 1.0, -1.0).astype(np.float32)
    pred_cls, sim = classify_by_similarity(final_state, prototypes)
    return pred_cls, sim, int(counts.sum())


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", type=str, default="train.csv")
    ap.add_argument("--test_csv", type=str, default="test.csv")
    ap.add_argument("--num_samples", type=int, default=10)

    # PCA/Hopfield
    ap.add_argument("--reduce_dim", type=int, default=48)
    ap.add_argument("--use_storkey", action="store_true")
    ap.add_argument("--w_max", type=int, default=15)
    ap.add_argument("--topk", type=int, default=24)
    ap.add_argument("--binarize_thr", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)

    # recurrent safety
    ap.add_argument("--rec_clip", type=int, default=12, help="Clip recurrent weights to [-rec_clip, rec_clip].")
    ap.add_argument("--relax_iter", type=int, default=50, help="Dry-run relax iterations; on HW: 0 disables recurrence.")

    # modes
    ap.add_argument("--dry_run", action="store_true", help="Run only numpy path.")
    ap.add_argument("--run_hw", action="store_true", help="Run on SpiNNaker2 hardware.")

    # HW params
    ap.add_argument("--s2_ip", type=str, default=os.environ.get("S2_IP", "192.168.1.2"))
    ap.add_argument("--duration_ms", type=int, default=200)
    ap.add_argument("--input_burst_hz", type=int, default=200)
    ap.add_argument("--input_burst_ms", type=int, default=60)
    ap.add_argument("--late_frac", type=float, default=0.6)
    ap.add_argument("--lif_threshold", type=float, default=10.0)
    ap.add_argument("--atoms_per_core", type=int, default=64)

    args = ap.parse_args()

    if args.dry_run and args.run_hw:
        raise SystemExit("Choose only one: --dry_run OR --run_hw")

    # Load data
    Xtr, ytr = load_har_csv(Path(args.train_csv))
    Xte, yte = load_har_csv(Path(args.test_csv))
    classes = sorted(list(set(ytr)))

    print(f"[INFO] Classes ({len(classes)}): {classes}")
    print(f"[INFO] Train: {Xtr.shape}  Test: {Xte.shape}")

    # PCA + scaling
    try:
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
    except Exception as e:
        raise RuntimeError("scikit-learn is required (StandardScaler + PCA).") from e

    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xte_s = scaler.transform(Xte)

    pca = PCA(n_components=int(args.reduce_dim), random_state=int(args.seed))
    Xtr_p = pca.fit_transform(Xtr_s).astype(np.float32)
    Xte_p = pca.transform(Xte_s).astype(np.float32)

    var = float(np.sum(pca.explained_variance_ratio_))
    print(f"[INFO] PCA dim={args.reduce_dim}  explained_variance={var:.2%}")

    # Prototypes
    protos = compute_class_prototypes(Xtr_p, ytr, classes)
    patterns = np.stack([binarize_bipolar(protos[c], threshold=args.binarize_thr) for c in classes], axis=0)
    P, N = patterns.shape
    print(f"[INFO] Hopfield patterns: P={P}, N={N}")

    # Weights
    if args.use_storkey:
        print("[INFO] Learning: Storkey")
        W = hopfield_weights_storkey(patterns)
    else:
        print("[INFO] Learning: Hebbian")
        W = hopfield_weights_hebb(patterns)

    Wq, scale = quantize_weights(W, w_max=int(args.w_max))
    print(f"[INFO] Quant scale: {scale:.3f}  | Wq nnz(before sparsify)={np.count_nonzero(Wq)}")

    Wq2 = sparsify_topk_symmetric(Wq, k=int(args.topk))
    Wq2 = np.clip(Wq2, -int(args.rec_clip), int(args.rec_clip)).astype(np.int32)
    np.fill_diagonal(Wq2, 0)

    nnz = int(np.count_nonzero(Wq2))
    print(f"[INFO] Sparsify(sym) topk={args.topk} -> nnz={nnz}")

    tested = min(int(args.num_samples), len(Xte_p))

    # -------- dry_run --------
    if args.dry_run:
        print(f"[DRY RUN] relax_iter={int(args.relax_iter)}")

        # Float version for numpy
        W_sparse = (Wq2.astype(np.float32) / float(scale)).astype(np.float32)
        W_sparse = 0.5 * (W_sparse + W_sparse.T)
        np.fill_diagonal(W_sparse, 0.0)

        correct_init = 0
        correct_relax = 0

        for i in range(tested):
            q = binarize_bipolar(Xte_p[i], threshold=args.binarize_thr)

            no_relax_pred, _ = classify_by_similarity(q, protos, threshold=args.binarize_thr)

            if int(args.relax_iter) <= 0:
                fs = q
            else:
                fs = hopfield_relax_async(q, W_sparse, max_iter=int(args.relax_iter), seed=int(args.seed) + i)

            relax_pred, _ = classify_by_similarity(fs, protos, threshold=args.binarize_thr)

            true = yte[i]
            correct_init += int(no_relax_pred == true)
            correct_relax += int(relax_pred == true)

        acc_init = correct_init / tested if tested else 0.0
        acc_relax = correct_relax / tested if tested else 0.0

        print(f"\n[RESULT] dry_run no-relax accuracy={acc_init:.4f} ({correct_init}/{tested})")
        print(f"[RESULT] dry_run relaxed  accuracy={acc_relax:.4f} ({correct_relax}/{tested})")
        return

    # -------- hardware --------
    if args.run_hw:
        print(f"[HW] s2_ip={args.s2_ip} duration_ms={args.duration_ms} relax_iter={int(args.relax_iter)}")
        print(f"[HW] NOTE: relax_iter<=0 => recurrent projection DISABLED (pure input drive + LIF).")

        correct = 0
        total_spikes = 0

        for i in range(tested):
            q = binarize_bipolar(Xte_p[i], threshold=args.binarize_thr)

            pred, sim, spike_sum = run_one_on_spinnaker(
                query_bin=q,
                Wq2=Wq2,
                prototypes=protos,
                s2_ip=str(args.s2_ip),
                duration_ms=int(args.duration_ms),
                input_burst_hz=int(args.input_burst_hz),
                input_burst_ms=int(args.input_burst_ms),
                late_frac=float(args.late_frac),
                lif_threshold=float(args.lif_threshold),
                atoms_per_core=int(args.atoms_per_core),
                rec_clip=int(args.rec_clip),
                relax_iter=int(args.relax_iter),
                run_id=i,
            )

            true = yte[i]
            correct += int(pred == true)
            total_spikes += int(spike_sum)

            # light progress print
            if (i + 1) % max(1, min(25, tested)) == 0 or (i + 1) == tested:
                acc = correct / float(i + 1)
                print(f"[HW] {i+1:4d}/{tested} acc={acc:.4f} last_true={true} pred={pred} sim={sim:.1f} spikes={spike_sum}")

        acc = correct / float(tested) if tested else 0.0
        print(f"\n[RESULT] hw accuracy={acc:.4f} ({correct}/{tested})  total_spikes={total_spikes}")
        return

    raise SystemExit("No mode selected. Use --dry_run or --run_hw.")


if __name__ == "__main__":
    main()

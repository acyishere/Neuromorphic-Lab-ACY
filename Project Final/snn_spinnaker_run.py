# -*- coding: utf-8 -*-
"""
Hopfield Network based HAR Classification on SpiNNaker2
Uses Modern Hopfield Network with stored class prototypes
"""

import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd


def load_har_csv(path: Path):
    df = pd.read_csv(path)
    X = df.drop(columns=["Activity", "subject"]).values.astype(np.float32)
    y = df["Activity"].values.astype(str)
    return X, y


def compute_class_prototypes(X_train: np.ndarray, y_train: np.ndarray, classes: list):
    """
    Compute mean prototype for each class - these become Hopfield memories
    """
    prototypes = {}
    for cls in classes:
        mask = (y_train == cls)
        if np.sum(mask) > 0:
            prototypes[cls] = np.mean(X_train[mask], axis=0)
    return prototypes


def binarize(x: np.ndarray, threshold: float = 0.0):
    """Convert to bipolar {-1, +1} for Hopfield"""
    return np.where(x >= threshold, 1, -1).astype(np.float32)


def hopfield_weights(patterns: np.ndarray):
    """
    Compute Hopfield weight matrix using Hebbian learning
    W_ij = (1/N) * sum_p(xi_p * xj_p)
    """
    N = patterns.shape[1]
    P = patterns.shape[0]
    W = np.zeros((N, N), dtype=np.float32)
    
    for p in range(P):
        xi = patterns[p].reshape(-1, 1)
        W += xi @ xi.T
    
    W /= N
    np.fill_diagonal(W, 0)  # No self-connections
    return W


def hopfield_weights_storkey(patterns: np.ndarray):
    """
    Storkey learning rule - better capacity than Hebbian
    """
    N = patterns.shape[1]
    P = patterns.shape[0]
    W = np.zeros((N, N), dtype=np.float32)
    
    for p in range(P):
        xi = patterns[p]
        h = W @ xi  # Local field before adding pattern
        for i in range(N):
            for j in range(N):
                if i != j:
                    W[i, j] += (xi[i] * xi[j] - xi[i] * h[j] - xi[j] * h[i]) / N
    
    np.fill_diagonal(W, 0)
    return W


def hopfield_energy(state: np.ndarray, W: np.ndarray):
    """Compute Hopfield energy: E = -0.5 * s^T * W * s"""
    return -0.5 * state @ W @ state


def hopfield_update_async(state: np.ndarray, W: np.ndarray, max_iter: int = 100):
    """
    Asynchronous update until convergence
    """
    N = len(state)
    state = state.copy()
    
    for _ in range(max_iter):
        changed = False
        order = np.random.permutation(N)
        for i in order:
            h = W[i] @ state
            new_s = 1 if h >= 0 else -1
            if new_s != state[i]:
                state[i] = new_s
                changed = True
        if not changed:
            break
    return state


def classify_hopfield(query: np.ndarray, prototypes: dict, W: np.ndarray):
    """
    Classify by:
    1. Initialize Hopfield with query
    2. Run until convergence
    3. Compare final state with stored prototypes
    """
    query_bin = binarize(query)
    final_state = hopfield_update_async(query_bin, W)
    
    best_cls = None
    best_sim = -np.inf
    for cls, proto in prototypes.items():
        proto_bin = binarize(proto)
        sim = np.dot(final_state, proto_bin)
        if sim > best_sim:
            best_sim = sim
            best_cls = cls
    
    return best_cls, final_state


def quantize_weights_hopfield(W: np.ndarray, w_max: int = 15):
    """Quantize Hopfield weights for SpiNNaker"""
    max_abs = float(np.max(np.abs(W)) + 1e-12)
    s = w_max / max_abs
    Wq = np.clip(np.rint(W * s), -w_max, w_max).astype(np.int32)
    return Wq, s


def hopfield_to_snn_conns(Wq: np.ndarray, delay: int = 1):
    """Convert Hopfield weight matrix to SNN connection list"""
    N = Wq.shape[0]
    conns = []
    for post in range(N):
        for pre in range(N):
            if Wq[post, pre] != 0:
                conns.append([int(pre), int(post), int(Wq[post, pre]), delay])
    return conns


# ...existing code...

def run_single_sample(
    query_bin: np.ndarray,
    Wq: np.ndarray,
    prototypes: dict,
    classes: list,
    s2_ip: str,
    duration_ms: int,
    sample_idx: int
):
    from spinnaker2 import hardware, snn
    N = len(query_bin)
    
    # Updated LIF parameters
    lif_params = {
        "threshold": 10.0,
        "v_rest": 0.0,
        "v_reset": 0.0,
        "alpha_decay": 0.9,
        "i_offset": 0.0,
        "reset": "reset_to_v_reset"
    }
    
    hw = hardware.SpiNNcloud48NodeBoard(stm_ip=s2_ip)
    net = snn.Network(f"Hopfield_HAR_{sample_idx}")

    pos_neurons = np.where(query_bin > 0)[0]
    
    # Strong input burst to kickstart the pattern
    spike_times = {i: list(range(0, 80, 5)) if i in pos_neurons else [] for i in range(N)}

    stim = snn.Population(size=N, neuron_model="spike_list", params=spike_times)
    
    # Use 'lif' model with proper reset parameter
    hopfield_pop = snn.Population(
        size=N, 
        neuron_model="lif", 
        params=lif_params, 
        record=["spikes"]
    )

    # Connections
    input_conns = [[i, i, 15, 1] for i in pos_neurons]
    # Clip weights to stay within hardware limits and avoid warnings
    hopfield_conns = hopfield_to_snn_conns(np.clip(Wq, -12, 12), delay=1)

    net.add(stim, hopfield_pop)
    net.add(snn.Projection(pre=stim, post=hopfield_pop, connections=input_conns))
    net.add(snn.Projection(pre=hopfield_pop, post=hopfield_pop, connections=hopfield_conns))

    hw.run(net, duration_ms)

    # Decoding: Check activity at the end of the simulation
    spikes = hopfield_pop.get_spikes()
    late_window = int(duration_ms * 0.6)
    spike_counts = np.zeros(N)
    
    for neuron_id, times in spikes.items():
        spike_counts[int(neuron_id)] = len([t for t in times if t >= late_window])
    
    # State determined by spike activity
    final_state = np.where(spike_counts > 1, 1, -1).astype(np.float32)

    # Similarity comparison
    best_cls = max(prototypes.keys(), key=lambda cls: np.dot(final_state, binarize(prototypes[cls])))
    
    return best_cls, final_state, int(np.sum(spike_counts)), {}
# ...existing code...

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", type=str, default="train.csv")
    ap.add_argument("--test_csv", type=str, default="test.csv")
    ap.add_argument("--s2_ip", type=str, default="192.168.1.2")
    ap.add_argument("--num_samples", type=int, default=10)
    ap.add_argument("--duration_ms", type=int, default=300)
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--reduce_dim", type=int, default=48,  # Reduced for better Hopfield capacity
                    help="Reduce input dimension via PCA for Hopfield capacity")
    ap.add_argument("--use_storkey", action="store_true", default=True,
                    help="Use Storkey learning rule (better capacity)")
    args = ap.parse_args()

    # Load data
    X_train, y_train = load_har_csv(Path(args.train_csv))
    X_test, y_test = load_har_csv(Path(args.test_csv))
    
    classes = sorted(list(set(y_train)))
    print(f"[INFO] Classes: {classes}")
    print(f"[INFO] Train: {X_train.shape}, Test: {X_test.shape}")

    # Reduce dimensionality (Hopfield capacity ~ 0.15N for 6 patterns need N >= 40)
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    pca = PCA(n_components=args.reduce_dim)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    print(f"[INFO] Reduced to {args.reduce_dim} dimensions (PCA variance: {pca.explained_variance_ratio_.sum():.2%})")

    # Compute class prototypes
    prototypes = compute_class_prototypes(X_train_pca, y_train, classes)
    
    # Create pattern matrix for Hopfield
    pattern_matrix = np.array([binarize(prototypes[cls]) for cls in classes])
    print(f"[INFO] Stored {len(classes)} patterns in Hopfield network")
    
    # Check pattern orthogonality
    print("[INFO] Pattern similarity matrix:")
    for i, cls_i in enumerate(classes):
        sims = []
        for j, cls_j in enumerate(classes):
            sim = np.dot(pattern_matrix[i], pattern_matrix[j]) / args.reduce_dim
            sims.append(f"{sim:+.2f}")
        print(f"  {cls_i}: {sims}")

    # Compute Hopfield weights
    if args.use_storkey:
        print("[INFO] Using Storkey learning rule")
        W = hopfield_weights_storkey(pattern_matrix)
    else:
        print("[INFO] Using Hebbian learning rule")
        W = hopfield_weights(pattern_matrix)
    
    Wq, scale = quantize_weights_hopfield(W, w_max=15)
    
    nnz = np.count_nonzero(Wq)
    print(f"[INFO] Hopfield weights: {Wq.shape}, nonzero: {nnz}")

    # ========== DRY RUN ==========
    if args.dry_run:
        print("\n[DRY RUN] Testing Hopfield classification (numpy)...")
        correct = 0
        tested = min(args.num_samples, len(X_test_pca))
        
        class_correct = {cls: 0 for cls in classes}
        class_total = {cls: 0 for cls in classes}
        
        for i in range(tested):
            pred_cls, _ = classify_hopfield(X_test_pca[i], prototypes, W)
            true_cls = y_test[i]
            ok = (pred_cls == true_cls)
            correct += int(ok)
            class_total[true_cls] += 1
            if ok:
                class_correct[true_cls] += 1
            print(f"  Sample {i}: true={true_cls}, pred={pred_cls}, ok={ok}")
        
        acc = correct / tested
        print(f"\n[DRY RUN RESULT] Accuracy: {acc:.4f} ({correct}/{tested})")
        print("\nPer-class accuracy:")
        for cls in classes:
            if class_total[cls] > 0:
                print(f"  {cls}: {class_correct[cls]}/{class_total[cls]} = {class_correct[cls]/class_total[cls]:.2%}")
        return

    # ========== SpiNNaker2 ==========
    print(f"\n[INFO] Running on SpiNNaker2...")
    print(f"[INFO] Duration: {args.duration_ms}ms per sample")

    tested = min(args.num_samples, len(X_test_pca))
    correct = 0

    for idx in range(tested):
        query = X_test_pca[idx]
        query_bin = binarize(query)
        true_cls = y_test[idx]
        
        try:
            pred_cls, final_state, total_spikes, sims = run_single_sample(
                query_bin=query_bin,
                Wq=Wq,
                prototypes=prototypes,
                classes=classes,
                s2_ip=args.s2_ip,
                duration_ms=args.duration_ms,
                sample_idx=idx
            )
            
            ok = (pred_cls == true_cls)
            correct += int(ok)
            
            print(f"Sample {idx:04d} | true={true_cls:20s} pred={pred_cls:20s} | spikes={total_spikes:4d} | ok={ok}")
            
        except Exception as e:
            print(f"Sample {idx:04d} | ERROR: {e}")

    acc = correct / tested if tested else 0.0
    print(f"\n[RESULT] Hopfield Network Accuracy: {acc:.4f} ({correct}/{tested})")


if __name__ == "__main__":
    main()

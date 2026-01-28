# main.py
import argparse
import subprocess
import sys


def run(cmd):
    print("\n>>>", " ".join(cmd))
    subprocess.check_call(cmd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true", help="Train ANN and export ann_params_har.npz")
    ap.add_argument("--run_snn", action="store_true", help="Run SNN on SpiNNaker2 using exported params")
    ap.add_argument("--s2_ip", type=str, default="192.168.1.53")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--num_samples", type=int, default=200)
    ap.add_argument("--duration_ms", type=int, default=200)
    args = ap.parse_args()

    if not args.train and not args.run_snn:
        print("Nothing to do. Use --train and/or --run_snn")
        sys.exit(0)

    if args.train:
        run([sys.executable, "train_ann.py", "--epochs", str(args.epochs), "--out_npz", "ann_params_har.npz"])

    if args.run_snn:
        run([
            sys.executable, "snn_spinnaker_run.py",
            "--ann_npz", "ann_params_har.npz",
            "--s2_ip", args.s2_ip,
            "--num_samples", str(args.num_samples),
            "--duration_ms", str(args.duration_ms),
        ])


if __name__ == "__main__":
    main()

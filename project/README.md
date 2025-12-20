# Neuromorphic Lab Project

This directory contains a minimal training stack for building a spiking neural
network (SNN) on the UCI Human Activity Recognition (HAR) dataset, aligned with
the SpiNNaker-oriented roadmap. The code prepares the dataset, defines a compact
SNN with a gating head, and provides a training entrypoint.

## Setup

1. Install dependencies:
   ```bash
   pip install -r project/requirements.txt
   ```
2. Download and extract the Kaggle dataset so it contains `train/` and `test/`
   folders (e.g., `./data/UCI_HAR_Dataset`). Network access is restricted in
   this environment, so the repository does not include the dataset; see
   `project/data/README.md` for manual download instructions.

## Training

Run the training script:
```bash
python project/train_har_snn.py --data ./data/UCI_HAR_Dataset --device cpu --epochs 5
```

The model outputs classification logits and a gating rate used to estimate when
higher-power sensors should be activated.

## Components

- `har_snn/data.py`: dataset loader for the HAR inertial signals.
- `har_snn/model.py`: spiking convolutional model with a gating head and simple
  surrogate gradient.
- `har_snn/training.py`: training/evaluation utilities and gate duty-cycle
  estimation.
- `train_har_snn.py`: CLI entrypoint tying everything together.

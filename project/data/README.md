# UCI HAR Dataset placeholder

The UCI Human Activity Recognition dataset is required for the HAR SNN training
pipeline. Network access is restricted in this environment, so the dataset
cannot be downloaded automatically here.

## How to add the dataset
1. Download the archive locally (e.g., from Kaggle or the UCI mirror):
   - Kaggle: https://www.kaggle.com/datasets/uciml/human-activity-recognition-with-smartphones
   - UCI mirror (zip): https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip
2. Extract the archive so you have the `UCI_HAR_Dataset` folder containing
   `train/` and `test/` subfolders.
3. Place the extracted folder at `project/data/UCI_HAR_Dataset` relative to the
   repository root, or pass the path via `--data` when running
   `project/train_har_snn.py`.

After placing the dataset, the training command from the root of the repository
looks like:

```bash
python project/train_har_snn.py --data project/data/UCI_HAR_Dataset --device cpu --epochs 5
```

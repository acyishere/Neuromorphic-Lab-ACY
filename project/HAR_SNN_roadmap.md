# SpiNNaker SNN Roadmap for HAR Smartphone Dataset

This roadmap tailors the SpiNNaker spiking neural network (SNN) gating approach to the UCI Smartphone Human Activity Recognition dataset from Kaggle (<https://www.kaggle.com/datasets/uciml/human-activity-recognition-with-smartphones>). The dataset contains tri-axial accelerometer and gyroscope time-series (50 Hz) from 30 participants performing six activities, with train/test splits already provided.

## 1. Problem Framing & Constraints
- **Goal:** Detect activity transitions or specific activities and trigger higher-power sensing/processing only when SNN confidence (spike rate) crosses thresholds to save energy on phones/wearables.
- **Targets:** Latency <100 ms on-device, minimal false wake-ups, and reduced duty cycle versus continuous sensing.
- **Deployment:** Lightweight always-on pipeline on-device; SpiNNaker used for low-power spiking inference or prototyping.

## 2. Data Handling
- **Ingest:** Use provided train/test splits; keep subject-wise separation to avoid leakage.
- **Signals:** 3-axis accelerometer + 3-axis gyroscope, 50 Hz, windowed into ~2.56 s segments (128 samples) with 50% overlap in the original dataset.
- **Preprocess:**
  - Normalize per-axis using training statistics.
  - Consider downsampling to 25 Hz for lower spike load if accuracy holds.
  - Derive delta/level-crossing events to reduce spikes versus raw rate coding.
- **Labels:** Six classes (WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING).

## 3. Neuromorphic Encoding
- **Baseline:** Rate coding of each axis value (positive/negative channels) with clipping to limit spike rates.
- **Low-spike option:** Delta/modulus-level-crossing encoding to emit spikes only when magnitude changes exceed thresholds; tune thresholds per axis using training stats.
- **Windowing:** Maintain the dataset’s 128-sample windows; optionally shorten to 64 samples for faster decisions if accuracy allows.

## 4. SNN Architecture
- **Model:** Lightweight convolutional SNN (1–2 temporal conv layers + spiking dense head) sized to fit SpiNNaker core memory.
- **Training:** Surrogate-gradient training in PyTorch; convert via sPyNNaker. Apply firing-rate regularization and weight sparsity to reduce traffic.
- **Gating head:** Add a small spiking neuron population whose firing rate encodes confidence; exposes a gate signal when threshold exceeded.

## 5. Energy-Aware Gating Logic
- **Dual thresholds:** Wake-up threshold (opens gate), and cool-down threshold with decay/hysteresis to avoid rapid toggling.
- **Duty-cycle simulation:** Use training windows to estimate gate open/close frequency; adjust thresholds to keep wake-ups rare during static activities (e.g., SITTING/STANDING).
- **On-device hooks:** Map gate events to enable/disable high-power sensors (e.g., continuous high-rate gyro or GPS) or trigger heavier downstream models only when needed.

## 6. SpiNNaker Deployment
- **Conversion:** Export trained weights to sPyNNaker; verify core allocation and routing tables for the chosen neuron/synapse counts.
- **Profiling:** Measure spike traffic per layer and packet loss under dataset-like loads; compress synapses and tune delays as needed.
- **Logging:** Record gate-trigger times and spike stats for energy accounting and threshold tuning.

## 7. On-Device Integration
- **Pipeline:** Lightweight preprocessor (sensor read → normalization → encoder) → SpiNNaker inference → gate output → sensor manager API.
- **Communication:** If SpiNNaker is off-board, minimize BLE/USB message frequency; batch gate events where possible.
- **Fail-safe:** Default to low-power baseline behavior if gate signal is unavailable.

## 8. Evaluation
- **Metrics:** Accuracy/F1 per activity, gate precision/recall, average gate open percentage, end-to-end latency, estimated energy savings versus continuous sensing.
- **Ablations:** Encoding schemes (rate vs. delta), firing-rate regularization strengths, window lengths, threshold settings.
- **Robustness:** Test noisy motion and transitions (e.g., sit-to-stand) to ensure stable gating and avoid false wake-ups.

## 9. Iteration & Hardening
- Shrink model/neuron counts until accuracy breaks; pick Pareto-optimal point for energy vs. accuracy.
- Add per-user calibration for sensor drift and personalized thresholds; allow OTA updates of thresholds/policies.
- Document reproducible pipeline scripts (data prep → training → conversion → deployment → mobile integration).

## 10. Pilot & Field Validation
- Run week-long pilots on representative phones/watches; log gate activations, battery impact, and user feedback.
- Refine thresholds and encoding parameters based on field logs; prioritize responsiveness to transitions with minimal false positives.

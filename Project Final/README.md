# Human Activity Recognition on SpiNNaker2 Neuromorphic Hardware

## Overview

This project implements Human Activity Recognition (HAR) on the SpiNNaker2 neuromorphic computing platform using Spiking Neural Networks (SNNs). The work explores the conversion of classical Artificial Neural Networks (ANNs) to SNNs and their deployment on neuromorphic hardware, addressing the unique challenges posed by fixed-point arithmetic, memory constraints, and spike-based computation.

The dataset used is based on accelerometer and gyroscope sensor data, with 561-dimensional feature vectors representing various human activities. The primary objective is to demonstrate a functional ANN-to-SNN conversion pipeline that can be executed on SpiNNaker2 hardware.

## Project Structure

```
Project Final/
├── main.py                   # Main execution script
├── train_ann.py             # ANN training and conversion pipeline
├── offline_snn_sim.py       # Offline SNN simulation (validation)
├── snn_spinnaker_run.py     # Hardware deployment script
├── train.csv                # Training dataset
├── test.csv                 # Test dataset
├── CONFIGS.md               # Configuration documentation
└── README.md                # This file
```

## Methodology

### 1. Dimensionality Reduction

- **StandardScaler**: Feature normalization
- **PCA**: Reduction to 48 dimensions (~87.06% explained variance)

### 2. ANN Training

A linear classifier (Logistic Regression) is trained on the preprocessed data, achieving:

- **Training Accuracy**: 95.87%

The trained weights and biases are quantized to integer values compatible with SpiNNaker2:

- **Weight Range**: [-11, 12]
- **Bias Range**: [-13, 15]
- **Non-zero Parameters**: 253

### 3. ANN-to-SNN Conversion

The conversion employs rate coding:

- Input features are encoded as spike rates
- Leaky Integrate-and-Fire (LIF) neurons replace linear units
- Output classification based on spike counts per class
- Quantized weights mapped to SpiNNaker2's fixed-point format

### 4. Hardware Deployment

The converted SNN is deployed on SpiNNaker2 with the following considerations:

- Integer weight constraints ([-15, 15])
- Memory limitations per core
- UDP-based communication protocol
- Session management for stable inference

## Hardware Constraints and Solutions

### Key Challenges

1. **Memory Overflow**: High-dimensional dense connections exceed per-core memory limits
2. **Weight Quantization**: Conversion from floating-point to integer weights
3. **Communication Stability**: Timeout issues during host-board communication
4. **Session Management**: Sequential run stability requires careful handling

### Implemented Solutions

- PCA-based dimensionality reduction (561 → 48)
- Integer quantization with controlled range
- Sparse weight matrices
- Rate-based spike encoding for robustness

## Alternative Approaches Explored

### Hopfield-Inspired SNN Classifier

An initial approach using Hopfield-like dynamics with prototype-based classification was investigated:

- **Methodology**: Class prototypes + Hebbian/Storkey learning
- **Results**: Dry-run accuracy ~53.20% (266/500 samples)
- **Limitation**: Relaxation dynamics did not improve accuracy; interference between attractors
- **Conclusion**: Not methodologically aligned with ANN-to-SNN conversion objective

This approach, while theoretically interesting, was not pursued further due to accuracy limitations and methodological considerations.

## Experimental Results

### Offline Validation

The quantized ANN model maintains high accuracy after conversion, validating the quantization strategy.

### Hardware Inference

The SNN successfully executes on SpiNNaker2:

- **Output Generation**: Meaningful spike counts per class
- **Example Result** (Sample 1):
  - True Label: `STANDING`
  - Predicted: `WALKING_DOWNSTAIRS`
  - Spike Counts: `[0, 0, 23, 0, 39, 0]`

**Note**: Current implementation demonstrates feasibility but requires further calibration for improved classification accuracy and batch inference stability.

## Usage

### Prerequisites

- Python 3.x
- SpiNNaker2 hardware access
- Required Python packages (see requirements section)

### Training and Conversion

```bash
python train_ann.py --reduce_dim 48 --output model_snn.npz
```

### Offline Simulation

```bash
python offline_snn_sim.py --model model_snn.npz --test_csv test.csv --num_samples 100
```

### Hardware Deployment

```bash
python snn_spinnaker_run.py --model model_snn.npz --test_csv test.csv \
    --num_samples 5 --duration_ms 150 --max_rate 150 \
    --bias_rate 150 --lif_threshold 12
```

## Configuration Parameters

Key parameters for SNN inference (see [CONFIGS.md](CONFIGS.md) for details):

- `duration_ms`: Simulation duration per sample
- `max_rate`: Maximum input spike rate (Hz)
- `bias_rate`: Bias neuron spike rate (Hz)
- `lif_threshold`: Membrane threshold for LIF neurons
- `rec_clip`: Recurrent weight clipping value

## Known Issues and Future Work

### Current Limitations

1. **Batch Inference Stability**: Sequential runs may encounter timeout exceptions
2. **Parameter Calibration**: Threshold and rate parameters require fine-tuning
3. **Communication Layer**: UDP-based host-board communication needs robustness improvements

### Future Directions

1. **Latency Coding**: Explore temporal coding schemes beyond rate coding
2. **Energy Analysis**: Proxy metrics based on spike counts
3. **Threshold Optimization**: Automated calibration procedures
4. **Batch Processing**: Stable multi-sample inference pipelines
5. **Alternative Encodings**: Population coding and other spike encoding strategies

## References

This work builds upon established ANN-to-SNN conversion methodologies and adapts them for the SpiNNaker2 neuromorphic platform, addressing hardware-specific constraints while maintaining computational efficiency.

## License

This project is part of academic coursework at the Neuromorphic Computing Lab.

## Contact

For questions or collaboration inquiries, please refer to the course instructors or the project repository.

---

**Note**: This implementation represents a functional proof-of-concept for neuromorphic inference on SpiNNaker2. The system demonstrates successful spike generation and basic classification, forming a foundation for further optimization and research in neuromorphic computing applications.

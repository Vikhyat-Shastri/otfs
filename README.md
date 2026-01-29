# OTFS Modulation Deep Learning System

A physics-compliant Orthogonal Time Frequency Space (OTFS) modulation system with deep learning-based channel estimation and data detection.

## Overview

This project implements a complete OTFS communication system with neural network-based receivers. The system uses the full OTFS modulation chain (ISFFT → Heisenberg → Channel → Wigner → SFFT) and realistic Linear Time-Variant (LTV) channels with multiple propagation paths, Doppler shifts, and delays.

### Key Features

- **Physics-Compliant OTFS Chain**: Full modulation/demodulation transforms
- **Realistic Channel Models**: LTV channels with multipath, Doppler, and delays
- **Deep Learning Receivers**: Neural network-based channel estimation and data detection
- **Multiple Architectures**: ResNet, U-Net, and Attention-enhanced models
- **Classical Baselines**: MMSE and Zero Forcing detectors for comparison

## Project Structure

```
otfs-modulation/
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── otfs/                          # Main package
│   ├── __init__.py
│   ├── modem.py                   # OTFS modulation/demodulation
│   ├── channel.py                 # Channel simulation (LTV, paths)
│   ├── models/                    # Neural network architectures
│   │   ├── attention.py           # CBAM attention modules
│   │   ├── estimators.py          # Channel estimators (ResNet, U-Net, CBAM)
│   │   ├── detectors.py           # Data detectors (CNN, ResNet)
│   │   └── end_to_end.py          # Combined systems
│   ├── classical/                 # Classical baselines
│   │   └── detectors.py           # MMSE, ZF detectors
│   ├── data/                      # Dataset generation
│   │   └── datasets.py            # PyTorch datasets
│   ├── training/                  # Training utilities
│   │   └── trainer.py             # Training loops
│   └── utils/                     # Utilities
│       └── metrics.py             # BER, NMSE calculations
├── experiments/                   # Organized experiment notebooks
│   ├── 01_baseline_symbol_dnn.ipynb
│   ├── 02_resnet_channel_estimation.ipynb
│   ├── 03_cbam_attention_estimator.ipynb
│   ├── 04_two_phase_training.ipynb
│   ├── 05_physics_compliant_system.ipynb
│   └── 06_unet_32x32_production.ipynb
├── results/                       # Documented results
│   ├── figures/                   # Saved plots
│   └── RESULTS.md                 # Experimental findings summary
└── archive/                       # Original notebooks (preserved)
    ├── OTFS_2.ipynb
    ├── OTFS_3.ipynb
    ├── OTFS_4.ipynb
    └── OTFS_DNN.ipynb
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd otfs-modulation
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### Basic Usage

```python
import numpy as np
from otfs.modem import OTFS_Modem
from otfs.channel import generate_channel_params, ltv_channel_sim

# OTFS Parameters
M, N = 4, 4  # Delay bins, Doppler bins
num_symbols = M * N

# Generate data
x_dd_grid = np.random.choice([-1, 1], size=(M, N)).astype(np.complex128)

# Modulate
tx_signal = OTFS_Modem.modulate(x_dd_grid, M, N)

# Channel
paths = generate_channel_params()
rx_signal, noise_power = ltv_channel_sim(tx_signal, paths, snr_db=10, M=M, N=N)

# Demodulate
y_dd_grid = OTFS_Modem.demodulate(rx_signal, M, N)
```

### Training a Channel Estimator

```python
import torch
from torch.utils.data import DataLoader
from otfs.models.estimators import AttentionChannelEstimator
from otfs.data.datasets import ChannelEstimatorDataset
from otfs.training.trainer import train_model

# Setup
M, N = 4, 4
pilot_indices = np.array([0, 2, 5, 7, 8, 10, 13, 15])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Dataset
train_dataset = ChannelEstimatorDataset(10000, M, N, pilot_indices, snr_range=(0, 25))
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

# Model
model = AttentionChannelEstimator()
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train
history = train_model(model, train_loader, criterion, optimizer, device, epochs=100)
```

## Module Descriptions

### Core Modules

- **`otfs.modem`**: OTFS modulation (ISFFT, Heisenberg) and demodulation (Wigner, SFFT)
- **`otfs.channel`**: LTV channel simulation with multiple paths, delays, and Doppler shifts

### Neural Network Models

- **`otfs.models.estimators`**: Channel estimation networks
  - `ChannelDenoisingResNet`: Standard ResNet-based estimator
  - `AttentionChannelEstimator`: CBAM-enhanced ResNet
  - `UNetEstimator`: U-Net for sparse channel estimation

- **`otfs.models.detectors`**: Data detection networks
  - `DetectorCNN`: CNN-based detector
  - `DetectorNet`: Enhanced detector with batch normalization

- **`otfs.models.attention`**: CBAM attention modules
  - `ChannelAttention`: Channel-wise attention
  - `SpatialAttention`: Spatial attention
  - `CBAM`: Combined attention module

### Classical Baselines

- **`otfs.classical.detectors`**: MMSE and Zero Forcing detectors

### Utilities

- **`otfs.utils.metrics`**: BER, NMSE, SER calculations
- **`otfs.training.trainer`**: Generic training loops with checkpointing

## Experiments

The `experiments/` directory contains organized notebooks:

1. **01_baseline_symbol_dnn.ipynb**: Basic TensorFlow/Keras Symbol-DNN
2. **02_resnet_channel_estimation.ipynb**: ResNet channel estimator
3. **03_cbam_attention_estimator.ipynb**: Attention-enhanced estimator
4. **04_two_phase_training.ipynb**: Two-phase training (estimator + detector)
5. **05_physics_compliant_system.ipynb**: Full physics-compliant system
6. **06_unet_32x32_production.ipynb**: Production-scale U-Net (32×32 grid)

## Results

See `results/RESULTS.md` for a comprehensive summary of experimental findings, including:
- BER curves comparison
- NMSE performance
- Effect of attention mechanisms
- Grid size impact
- Pilot overhead analysis

## OTFS System Diagram

```
Delay-Doppler Grid → ISFFT → Time-Frequency → Heisenberg Transform → 
Time Domain Signal → LTV Channel → Received Signal → 
Wigner Transform → Time-Frequency → SFFT → Delay-Doppler Grid
```

## Key Technical Concepts

- **OTFS Modulation**: Delay-Doppler domain representation with ISFFT and Heisenberg transforms
- **LTV Channels**: Linear Time-Variant channels with multipath, Doppler shifts, and delays
- **Channel Estimation**: Neural networks trained to estimate channel state information from pilots
- **Data Detection**: Neural networks for detecting transmitted symbols from received signals
- **Attention Mechanisms**: CBAM (Convolutional Block Attention Module) for improved performance

## License

MIT


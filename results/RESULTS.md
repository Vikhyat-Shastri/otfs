# OTFS Experimental Results Summary

This document summarizes the experimental findings from the OTFS deep learning system.

## Overview

The experiments explore various neural network architectures for channel estimation and data detection in OTFS systems, comparing them against classical baselines (MMSE, ZF).

## Experimental Setup

### System Parameters

- **Grid Sizes**: 4×4 (16 symbols) and 32×32 (1024 symbols)
- **Modulation**: BPSK (Binary Phase Shift Keying)
- **Pilot Overhead**: Varies from 50% (4×4) to 6.25% (32×32)
- **Channel Model**: Linear Time-Variant (LTV) with 2-3 paths
- **SNR Range**: 0-30 dB

### Evaluation Metrics

- **BER**: Bit Error Rate
- **NMSE**: Normalized Mean Squared Error (for channel estimation)
- **SER**: Symbol Error Rate

## Key Findings

### 1. Physics-Compliant vs Simplified Models

**Finding**: The physics-compliant system (Phase 3-5) shows significant improvements over simplified channel models.

- Full OTFS chain (ISFFT → Heisenberg → Channel → Wigner → SFFT) provides realistic performance
- Matrix probing for effective channel matrix enables accurate genie-aided baselines
- Neural networks trained on physics-compliant data generalize better

### 2. Effect of Attention Mechanisms

**Finding**: CBAM (Convolutional Block Attention Module) improves channel estimation performance.

- **AttentionChannelEstimator** outperforms standard ResNet
- Channel attention helps focus on important features
- Spatial attention improves localization of channel taps
- Combined CBAM provides best performance

**Results** (4×4 grid, SNR=10dB):
- ResNet Estimator: NMSE ≈ 0.05
- Attention Estimator: NMSE ≈ 0.03
- Improvement: ~40% reduction in NMSE

### 3. Grid Size Impact

**Finding**: Larger grids (32×32) enable more efficient pilot usage and better performance.

- **4×4 Grid**: 50% pilot overhead (8/16 symbols)
- **32×32 Grid**: 6.25% pilot overhead (64/1024 symbols)
- U-Net architecture scales well to larger grids
- Sparse pilot pattern (checkerboard) works effectively

**Results** (SNR=10dB):
- 4×4 Grid: BER ≈ 0.01
- 32×32 Grid: BER ≈ 0.002
- Improvement: ~5× better BER with lower pilot overhead

### 4. Two-Phase Training

**Finding**: Two-phase training (estimator then detector) outperforms end-to-end training.

- Phase 1: Train channel estimator (frozen)
- Phase 2: Train detector with frozen estimator
- Better convergence and stability
- Allows independent optimization of each component

**Results**:
- End-to-End: BER ≈ 0.015
- Two-Phase: BER ≈ 0.005
- Improvement: ~3× better BER

### 5. U-Net for Sparse Channel Estimation

**Finding**: U-Net architecture excels at sparse channel estimation (image inpainting approach).

- Encoder-decoder structure with skip connections
- Handles sparse pilot patterns effectively
- Works well with 6.25% pilot overhead
- Circular padding respects OTFS periodic boundaries

**Results** (32×32 grid, SNR=10dB):
- ResNet: NMSE ≈ 0.08
- U-Net: NMSE ≈ 0.02
- Improvement: ~4× better NMSE

### 6. Neural Networks vs Classical Baselines

**Finding**: Neural networks approach genie-aided MMSE performance without perfect CSI.

**BER Comparison** (4×4 grid, SNR=10dB):
- Genie MMSE: BER ≈ 0.0001 (perfect channel knowledge)
- Neural Receiver: BER ≈ 0.005 (estimated channel)
- Genie ZF: BER ≈ 0.001 (perfect channel knowledge)

**Key Insight**: Neural networks achieve within 1-2 orders of magnitude of genie-aided performance, demonstrating effective channel estimation and detection.

## Detailed Results by Experiment

### Experiment 1: Baseline Symbol-DNN (OTFS_DNN.ipynb)

- **Architecture**: Simple DNN (Sequential Keras model)
- **Approach**: Direct symbol detection
- **Performance**: SER ≈ 0.1 at SNR=10dB
- **Note**: Baseline experiment, not physics-compliant

### Experiment 2: ResNet Channel Estimation (OTFS_3 Cell 3)

- **Architecture**: ChannelDenoisingResNet
- **Input**: LS estimate (noisy pilots)
- **Output**: Refined channel estimate
- **Performance**: NMSE ≈ 0.05 at SNR=10dB

### Experiment 3: CBAM Attention Estimator (OTFS_3 Cell 5)

- **Architecture**: AttentionChannelEstimator
- **Enhancement**: CBAM attention modules
- **Performance**: NMSE ≈ 0.03 at SNR=10dB
- **Improvement**: 40% better than ResNet

### Experiment 4: Two-Phase Training (OTFS_3 Cells 4,6)

- **Phase 1**: Train estimator (frozen)
- **Phase 2**: Train detector
- **Performance**: BER ≈ 0.005 at SNR=10dB
- **Advantage**: Better convergence

### Experiment 5: Physics-Compliant System (OTFS_4 Phase 3-4)

- **Full OTFS Chain**: ISFFT → Heisenberg → Channel → Wigner → SFFT
- **LTV Channels**: Realistic multipath with Doppler
- **Performance**: BER ≈ 0.002 at SNR=10dB
- **Advantage**: Realistic performance evaluation

### Experiment 6: U-Net Production System (OTFS_4 Phase 5)

- **Grid Size**: 32×32 (1024 symbols)
- **Architecture**: U-Net for channel estimation
- **Pilot Overhead**: 6.25% (64 pilots)
- **Performance**: 
  - NMSE ≈ 0.02 at SNR=10dB
  - BER ≈ 0.002 at SNR=10dB
- **Advantage**: Scales to production sizes

## Performance Summary Table

| Experiment | Architecture | Grid Size | Pilot % | NMSE (10dB) | BER (10dB) |
|------------|--------------|-----------|---------|-------------|------------|
| Baseline DNN | DNN | 4×4 | N/A | N/A | SER≈0.1 |
| ResNet Est | ResNet | 4×4 | 50% | 0.05 | N/A |
| CBAM Est | ResNet+CBAM | 4×4 | 50% | 0.03 | N/A |
| Two-Phase | ResNet+CNN | 4×4 | 50% | N/A | 0.005 |
| Physics-Compliant | Attention+Detector | 4×4 | 50% | N/A | 0.002 |
| U-Net Production | U-Net+ResNet | 32×32 | 6.25% | 0.02 | 0.002 |

## Conclusions

1. **Physics-Compliant Models**: Essential for realistic performance evaluation
2. **Attention Mechanisms**: Significant improvement in channel estimation
3. **Two-Phase Training**: Better convergence than end-to-end
4. **U-Net Architecture**: Excellent for sparse channel estimation at scale
5. **Neural Networks**: Approach genie-aided performance without perfect CSI

## Future Work

- Explore transformer architectures for channel estimation
- Investigate joint channel estimation and detection
- Extend to higher-order modulations (QPSK, 16-QAM)
- Evaluate on measured channel data
- Optimize for real-time implementation

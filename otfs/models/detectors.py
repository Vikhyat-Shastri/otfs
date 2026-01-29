"""
Data Detector Models
=====================

Neural network architectures for data detection:
- DetectorCNN: CNN-based detector
- DetectorNet: Enhanced detector with batch normalization

Extracted from OTFS_3.ipynb and OTFS_4.ipynb
"""

import torch
import torch.nn as nn


class DetectorCNN(nn.Module):
    """
    CNN-based Data Detector
    
    Takes concatenated received signal and channel estimate to detect transmitted data.
    
    Input: (B, 4, M, N) [Real Y, Imag Y, Real H_hat, Imag H_hat]
    Output: (B, 1, M, N) detected data symbols
    
    Extracted from OTFS_3.ipynb
    """
    def __init__(self, in_channels=4, out_channels=1, base_filters=64):
        super(DetectorCNN, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(in_channels, base_filters, 3, 1, 1),
            nn.BatchNorm2d(base_filters),
            nn.ReLU(),
            nn.Conv2d(base_filters, base_filters * 2, 3, 1, 1),
            nn.BatchNorm2d(base_filters * 2),
            nn.ReLU(),
            nn.Conv2d(base_filters * 2, base_filters, 3, 1, 1),
            nn.BatchNorm2d(base_filters),
            nn.ReLU(),
            nn.Conv2d(base_filters, out_channels, 1, 1, 0),
            nn.Tanh()  # For BPSK: outputs in [-1, 1]
        )
        
    def forward(self, x):
        return self.network(x)


class DetectorNet(nn.Module):
    """
    Enhanced Data Detector with Input Batch Normalization
    
    Improved version with input batch normalization for SNR stability.
    Takes received signal and channel estimate separately.
    
    Input: 
        y_grid: (B, 2, M, N) received signal [Real, Imag]
        h_hat_grid: (B, 2, M, N) channel estimate [Real, Imag]
    Output: (B, 1, M, N) detected data symbols
    
    Extracted from OTFS_4.ipynb Phase 4
    """
    def __init__(self):
        super().__init__()
        # NEW: BatchNorm at input to handle SNR scaling variations
        self.input_bn = nn.BatchNorm2d(4)
        
        # Input: 4 Channels (2 for Y_rx, 2 for H_hat)
        self.features = nn.Sequential(
            nn.Conv2d(4, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, 1),  # 1x1 conv to mix features
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        # Output: 1 Channel (Real Data X)
        self.final = nn.Conv2d(64, 1, 1)
        self.tanh = nn.Tanh()  # Enforce [-1, 1] range for BPSK
        
    def forward(self, y_grid, h_hat_grid):
        # Concatenate Input along channel dimension
        x = torch.cat([y_grid, h_hat_grid], dim=1)
        x = self.input_bn(x)  # Normalize inputs
        x = self.features(x)
        out = self.tanh(self.final(x))
        return out

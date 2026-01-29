"""
End-to-End System Models
=========================

Combined systems that integrate channel estimation and data detection:
- FullSystemModel: Two-phase system with frozen estimator
- NeuralReceiver: Complete receiver with estimator + detector

Extracted from OTFS_3.ipynb and OTFS_4.ipynb
"""

import torch
import torch.nn as nn
import os
from .estimators import AttentionChannelEstimator, ChannelDenoisingResNet
from .detectors import DetectorNet, DetectorCNN


class FullSystemModel(nn.Module):
    """
    Full System Model: Estimator + Detector
    
    Two-phase architecture:
    1. Channel Estimator (frozen, pre-trained)
    2. Data Detector (trainable)
    
    Input: (B, 2, M, N) received signal y_grid
    Output: (B, 1, M, N) detected data x_hat
    
    Extracted from OTFS_3.ipynb Cell 4
    """
    def __init__(self, estimator_model, estimator_path=None):
        super(FullSystemModel, self).__init__()
        
        # Load pre-trained estimator
        self.estimator = estimator_model
        if estimator_path and os.path.exists(estimator_path):
            self.estimator.load_state_dict(torch.load(estimator_path, map_location='cpu'))
            print(f"✓ Loaded pre-trained estimator from {estimator_path}")
        
        # Freeze estimator
        for param in self.estimator.parameters():
            param.requires_grad = False
        
        # Detector
        self.detector = DetectorCNN(in_channels=4, out_channels=1)
        
    def forward(self, y_grid):
        # Estimate channel (no gradients)
        with torch.no_grad():
            h_hat = self.estimator(y_grid)
        
        # Detect data
        x_hat = self.detector(torch.cat([y_grid, h_hat], dim=1))
        return x_hat


class NeuralReceiver(nn.Module):
    """
    Neural Receiver: Complete OTFS Receiver
    
    Two-stage architecture:
    1. Attention-based Channel Estimator (frozen)
    2. Data Detector (trainable)
    
    Input: 
        ls_input: (B, 2, M, N) LS estimate for estimator
        y_grid_full: (B, 2, M, N) full received grid for detector
    Output: (B, 1, M, N) detected data
    
    Extracted from OTFS_4.ipynb Phase 4
    """
    def __init__(self, estimator_path=None):
        super().__init__()
        self.estimator = AttentionChannelEstimator()
        
        if estimator_path and os.path.exists(estimator_path):
            self.estimator.load_state_dict(torch.load(estimator_path, map_location='cpu'))
            print("✓ Loaded Pre-trained Estimator.")
        else:
            print("Warning: Estimator weights not found. Training from scratch.")
            
        # Freeze estimator
        for param in self.estimator.parameters():
            param.requires_grad = False
            
        self.detector = DetectorNet()
        
    def forward(self, ls_input, y_grid_full):
        # Estimate channel (no gradients)
        with torch.no_grad():
            h_hat = self.estimator(ls_input)
        
        # Detect data
        return self.detector(y_grid_full, h_hat)

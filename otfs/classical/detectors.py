"""
Classical Detectors for OTFS
=============================

MMSE and Zero Forcing detectors using the effective channel matrix.

Extracted from OTFS_4.ipynb Phase 3-4
"""

import numpy as np


def mmse_detector(y_clean, H_data, noise_power):
    """
    Minimum Mean Square Error (MMSE) Detector
    
    Uses the formula: W_mmse = (H^H H + N0*I)^-1 H^H
    
    Args:
        y_clean: Clean received signal (after pilot cancellation) [num_symbols]
        H_data: Effective channel matrix for data symbols [num_symbols, num_data]
        noise_power: Noise power N0
        
    Returns:
        x_est: Estimated transmitted symbols [num_data]
    """
    H_conj = H_data.conj().T
    gram = H_conj @ H_data
    W_mmse = np.linalg.inv(gram + noise_power * np.eye(H_data.shape[1])) @ H_conj
    x_est = W_mmse @ y_clean
    return x_est


def zf_detector(y_clean, H_data):
    """
    Zero Forcing (ZF) Detector
    
    Uses pseudo-inverse: W_zf = (H^H H)^-1 H^H = pinv(H)
    
    Args:
        y_clean: Clean received signal (after pilot cancellation) [num_symbols]
        H_data: Effective channel matrix for data symbols [num_symbols, num_data]
        
    Returns:
        x_est: Estimated transmitted symbols [num_data]
    """
    W_zf = np.linalg.pinv(H_data)
    x_est = W_zf @ y_clean
    return x_est

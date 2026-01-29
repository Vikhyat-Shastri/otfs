"""
OTFS Performance Metrics
=========================

Evaluation metrics for OTFS systems:
- BER: Bit Error Rate
- NMSE: Normalized Mean Squared Error
- SER: Symbol Error Rate

Extracted from OTFS_3.ipynb and OTFS_4.ipynb
"""

import numpy as np
import torch


def calculate_ber(predicted, true, data_indices=None):
    """
    Calculate Bit Error Rate (BER)
    
    Args:
        predicted: Predicted symbols (can be numpy array or torch tensor)
        true: True transmitted symbols
        data_indices: Optional indices to mask (e.g., exclude pilots)
        
    Returns:
        ber: Bit error rate (0-1)
    """
    if isinstance(predicted, torch.Tensor):
        pred_bits = torch.sign(predicted).cpu().numpy()
    else:
        pred_bits = np.sign(predicted)
    
    if isinstance(true, torch.Tensor):
        true_bits = true.cpu().numpy()
    else:
        true_bits = true
    
    if data_indices is not None:
        pred_bits = pred_bits.flatten()[data_indices]
        true_bits = true_bits.flatten()[data_indices]
    
    errors = np.sum(pred_bits != true_bits)
    total = len(pred_bits)
    
    return errors / total if total > 0 else 0.0


def calculate_nmse(predicted, true):
    """
    Calculate Normalized Mean Squared Error (NMSE)
    
    NMSE = ||predicted - true||^2 / ||true||^2
    
    Args:
        predicted: Predicted channel/data (numpy array or torch tensor)
        true: True channel/data
        
    Returns:
        nmse: Normalized MSE value
    """
    if isinstance(predicted, torch.Tensor):
        pred = predicted.detach().cpu().numpy()
    else:
        pred = predicted
    
    if isinstance(true, torch.Tensor):
        true_val = true.cpu().numpy()
    else:
        true_val = true
    
    # Flatten for calculation
    pred_flat = pred.flatten()
    true_flat = true_val.flatten()
    
    numerator = np.sum(np.abs(pred_flat - true_flat)**2)
    denominator = np.sum(np.abs(true_flat)**2)
    
    if denominator == 0:
        return float('inf')
    
    return numerator / denominator


def calculate_ser(predicted, true, constellation=None):
    """
    Calculate Symbol Error Rate (SER)
    
    Args:
        predicted: Predicted symbols
        true: True transmitted symbols
        constellation: Optional constellation mapping
        
    Returns:
        ser: Symbol error rate (0-1)
    """
    if isinstance(predicted, torch.Tensor):
        pred = predicted.detach().cpu().numpy()
    else:
        pred = predicted
    
    if isinstance(true, torch.Tensor):
        true_val = true.cpu().numpy()
    else:
        true_val = true
    
    # For BPSK, symbols are -1 or 1
    if constellation is None:
        pred_symbols = np.sign(pred.real) if np.iscomplexobj(pred) else np.sign(pred)
        true_symbols = np.sign(true_val.real) if np.iscomplexobj(true_val) else np.sign(true_val)
    else:
        # Map to nearest constellation point
        pred_symbols = np.array([constellation[np.argmin(np.abs(constellation - s))] for s in pred])
        true_symbols = true_val
    
    errors = np.sum(pred_symbols != true_symbols)
    total = len(pred_symbols)
    
    return errors / total if total > 0 else 0.0

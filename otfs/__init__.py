"""
OTFS Modulation Deep Learning System
=====================================

A physics-compliant Orthogonal Time Frequency Space (OTFS) modulation system
with deep learning-based channel estimation and data detection.

Modules:
    modem: OTFS modulation and demodulation transforms
    channel: Channel simulation (LTV channels with Doppler)
    models: Neural network architectures (estimators, detectors, attention)
    data: Dataset generation for training
    classical: Classical baseline detectors (MMSE, ZF)
    training: Training utilities
    utils: Metrics and helper functions
"""

__version__ = "1.0.0"

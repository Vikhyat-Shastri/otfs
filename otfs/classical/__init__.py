"""
Classical Baseline Detectors
==============================

Classical signal processing detectors for OTFS:
- MMSE: Minimum Mean Square Error detector
- ZF: Zero Forcing detector

These serve as baselines for comparison with neural network approaches.
"""

from .detectors import mmse_detector, zf_detector

__all__ = ['mmse_detector', 'zf_detector']

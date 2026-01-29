"""
Neural Network Models for OTFS
================================

Contains:
    attention: CBAM attention modules
    estimators: Channel estimation networks (ResNet, U-Net, Attention)
    detectors: Data detection networks (CNN, ResNet)
    end_to_end: Combined systems
"""

from .attention import ChannelAttention, SpatialAttention, CBAM, AttentionResidualBlock
from .estimators import ChannelDenoisingResNet, AttentionChannelEstimator, UNetEstimator
from .detectors import DetectorCNN, DetectorNet

__all__ = [
    'ChannelAttention', 'SpatialAttention', 'CBAM', 'AttentionResidualBlock',
    'ChannelDenoisingResNet', 'AttentionChannelEstimator', 'UNetEstimator',
    'DetectorCNN', 'DetectorNet'
]

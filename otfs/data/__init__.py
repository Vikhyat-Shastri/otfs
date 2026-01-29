"""
Dataset Generation for OTFS
============================

PyTorch Dataset classes for training:
- OTFSPhysicsDataset: Physics-compliant dataset
- ChannelEstimatorDataset: For channel estimator training
- DetectorDataset: For detector training
- UNetDataset: For U-Net estimator training
"""

from .datasets import (
    OTFSPhysicsDataset,
    ChannelEstimatorDataset,
    DetectorDataset,
    UNetDataset
)

__all__ = [
    'OTFSPhysicsDataset',
    'ChannelEstimatorDataset',
    'DetectorDataset',
    'UNetDataset'
]

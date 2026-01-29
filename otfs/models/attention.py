"""
CBAM Attention Modules
======================

Convolutional Block Attention Module (CBAM) implementation:
- Channel Attention: Focuses on "what" features are important
- Spatial Attention: Focuses on "where" features are important

Extracted from OTFS_3.ipynb Cell 5 and OTFS_4.ipynb Phase 3
"""

import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """
    Channel Attention Module
    
    Uses both average and max pooling to generate channel attention weights.
    Helps the model focus on important feature channels.
    
    Args:
        in_planes: Number of input channels
        ratio: Reduction ratio (default: 16)
    """
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = self.sigmoid(avg_out + max_out)
        return out


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module
    
    Uses average and max pooling across channels to generate spatial attention weights.
    Helps the model focus on important spatial locations.
    
    Args:
        kernel_size: Convolution kernel size (default: 7)
    """
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module
    
    Combines Channel and Spatial attention in sequence.
    First applies channel attention, then spatial attention.
    
    Args:
        planes: Number of feature planes (channels)
    """
    def __init__(self, planes):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(planes)
        self.sa = SpatialAttention()
        
    def forward(self, x):
        # Channel attention first
        x = self.ca(x) * x
        # Then spatial attention
        x = self.sa(x) * x
        return x


class AttentionResidualBlock(nn.Module):
    """
    ResNet Block with CBAM Attention
    
    Standard residual block enhanced with CBAM attention mechanism.
    Used in attention-enhanced channel estimators.
    
    Args:
        channels: Number of input/output channels
    """
    def __init__(self, channels):
        super(AttentionResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.attention = CBAM(channels)
        
    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.attention(out)  # Apply CBAM
        out += residual  # Residual connection
        out = self.relu(out)
        return out

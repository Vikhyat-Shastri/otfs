"""
Channel Estimator Models
========================

Neural network architectures for channel estimation:
- ChannelDenoisingResNet: Standard ResNet-based estimator
- AttentionChannelEstimator: CBAM-enhanced ResNet estimator
- UNetEstimator: U-Net architecture for sparse channel estimation

Extracted from OTFS_3.ipynb and OTFS_4.ipynb
"""

import torch
import torch.nn as nn
from .attention import AttentionResidualBlock, CBAM


class ResidualBlock(nn.Module):
    """A standard ResNet block without attention"""
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual  # The "skip connection"
        out = self.relu(out)
        return out


class ChannelDenoisingResNet(nn.Module):
    """
    MODEL V1: Standard ResNet for Channel Estimation
    
    Input: (B, 2, M, N) noisy/sparse h_hat_ls (LS estimate)
    Output: (B, 2, M, N) refined h_hat_refined
    
    Extracted from OTFS_3.ipynb Cell 3
    """
    def __init__(self, in_channels=2, out_channels=2, base_filters=64, num_blocks=4):
        super(ChannelDenoisingResNet, self).__init__()
        
        self.init_conv = nn.Conv2d(in_channels, base_filters, 3, 1, 1)
        
        blocks = []
        for _ in range(num_blocks):
            blocks.append(ResidualBlock(base_filters))
        self.res_blocks = nn.Sequential(*blocks)
        
        self.out_conv = nn.Conv2d(base_filters, out_channels, 1, 1, 0)

    def forward(self, h_hat_ls_grid):
        x = self.init_conv(h_hat_ls_grid)
        x = self.res_blocks(x)
        h_hat_refined_grid = self.out_conv(x)
        return h_hat_refined_grid


class AttentionChannelEstimator(nn.Module):
    """
    MODEL V2: Attention-Enhanced ResNet for Channel Estimation
    
    Uses CBAM attention mechanism to focus on important channel features.
    
    Input: (B, 2, M, N) noisy/sparse h_hat_ls
    Output: (B, 2, M, N) refined h_hat_refined
    
    Extracted from OTFS_3.ipynb Cell 5 and OTFS_4.ipynb Phase 3
    """
    def __init__(self, in_channels=2, out_channels=2, base_filters=64, num_blocks=4):
        super(AttentionChannelEstimator, self).__init__()
        
        self.init_conv = nn.Conv2d(in_channels, base_filters, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(base_filters)
        self.relu = nn.ReLU()
        
        # Attention-enhanced residual blocks
        blocks = []
        for _ in range(num_blocks):
            blocks.append(AttentionResidualBlock(base_filters))
        self.res_blocks = nn.Sequential(*blocks)
        
        self.out_conv = nn.Conv2d(base_filters, out_channels, 1, 1, 0)

    def forward(self, h_hat_ls_grid):
        x = self.relu(self.bn1(self.init_conv(h_hat_ls_grid)))
        x = self.res_blocks(x)
        h_hat_refined_grid = self.out_conv(x)
        return h_hat_refined_grid


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class UNetEstimator(nn.Module):
    """
    U-Net Architecture for Channel Estimation
    
    Image inpainting approach for sparse channel estimation.
    Uses encoder-decoder with skip connections.
    
    Input: (B, 3, M, N) [Real Y, Imag Y, Pilot Mask]
    Output: (B, 2, M, N) [Real H, Imag H]
    
    Extracted from OTFS_4.ipynb Phase 5
    """
    def __init__(self):
        super().__init__()
        # Input: 3 Channels (Real, Imag, Mask)
        
        # Encoder (Downsampling)
        self.inc = DoubleConv(3, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        
        # Bottleneck
        self.bot = DoubleConv(512, 512)

        # Decoder (Upsampling + Skip Connections)
        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(512, 256)  # 512 because 256(up) + 256(skip)
        
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(256, 128)
        
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(128, 64)
        
        # Final Output
        self.outc = nn.Conv2d(64, 2, kernel_size=1)  # Output: Real, Imag H

    def forward(self, x):
        # Encoding
        x1 = self.inc(x)          # [64, M, N]
        x2 = self.down1(x1)       # [128, M/2, N/2]
        x3 = self.down2(x2)       # [256, M/4, N/4]
        x4 = self.down3(x3)       # [512, M/8, N/8]
        
        # Bottleneck
        xb = self.bot(x4)         # [512, M/8, N/8]
        
        # Decoding with Skips
        x = self.up1(xb)          # [256, M/4, N/4]
        x = torch.cat([x, x3], dim=1)  # Skip connection
        x = self.conv1(x)
        
        x = self.up2(x)           # [128, M/2, N/2]
        x = torch.cat([x, x2], dim=1)
        x = self.conv2(x)
        
        x = self.up3(x)           # [64, M, N]
        x = torch.cat([x, x1], dim=1)
        x = self.conv3(x)
        
        return self.outc(x)

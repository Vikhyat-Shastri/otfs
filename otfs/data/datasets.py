"""
OTFS Dataset Classes
====================

PyTorch Dataset implementations for various training scenarios.

Extracted from OTFS_3.ipynb and OTFS_4.ipynb
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from ..modem import OTFS_Modem
from ..channel import (
    ltv_channel_sim,
    generate_channel_params,
    get_ground_truth_channel_grid
)


class OTFSPhysicsDataset(Dataset):
    """
    Physics-Compliant OTFS Dataset
    
    Generates samples using the full OTFS modulation chain:
    DD Grid → ISFFT → TF → Heisenberg → Time → Channel → Wigner → TF → SFFT → DD
    
    Input: (2, M, N) received signal [Real, Imag]
    Target: (1, M, N) transmitted data [Real]
    
    Extracted from OTFS_4.ipynb Phase 3
    """
    def __init__(self, size, M, N, pilot_indices, data_indices, snr_range=(10, 20)):
        self.size = size
        self.M = M
        self.N = N
        self.pilot_indices = pilot_indices
        self.data_indices = data_indices
        self.num_data = len(data_indices)
        self.num_symbols = M * N
        self.snr_min, self.snr_max = snr_range
        self.samples = []
        
        print(f"Generating {size} physics-simulated samples...")
        
        for _ in range(size):
            # 1. Generate Bits & Grid
            bits = np.random.choice([-1, 1], size=self.num_data)  # BPSK
            x_dd = np.zeros(self.num_symbols, dtype=np.complex128)
            x_dd[self.data_indices] = bits
            x_dd[self.pilot_indices] = 1.0  # Pilot value
            x_dd_grid = x_dd.reshape(M, N)
            
            # 2. Generate Physical Channel Paths
            paths = generate_channel_params()
            
            # 3. Simulate Physics (Mod -> Channel -> Demod)
            snr = np.random.uniform(self.snr_min, self.snr_max)
            tx_sig = OTFS_Modem.modulate(x_dd_grid, M, N)
            rx_sig, _ = ltv_channel_sim(tx_sig, paths, snr, M, N)
            y_dd_grid = OTFS_Modem.demodulate(rx_sig, M, N)
            
            # 4. Prepare Tensors
            # Input: Received Grid Y (Real, Imag)
            y_real = torch.from_numpy(y_dd_grid.real).float()
            y_imag = torch.from_numpy(y_dd_grid.imag).float()
            inp = torch.stack([y_real, y_imag], dim=0)
            
            # Target: Transmitted Grid X (Real only for BPSK)
            target = torch.from_numpy(x_dd_grid.real).float().unsqueeze(0)
            
            self.samples.append((inp, target))
            
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return self.samples[idx]


class ChannelEstimatorDataset(Dataset):
    """
    Dataset for Channel Estimator Training
    
    Generates LS estimates and ground truth channel grids.
    
    Input: (2, M, N) LS estimate [Real, Imag]
    Target: (2, M, N) true channel [Real, Imag]
    
    Extracted from OTFS_3.ipynb Cell 3
    """
    def __init__(self, size, M, N, pilot_indices, snr_range=(0, 25)):
        self.size = size
        self.M = M
        self.N = N
        self.pilot_indices = pilot_indices
        self.snr_min, self.snr_max = snr_range
        
        print(f"Generating {size} channel estimator samples...")
        self.samples = []
        
        for _ in range(size):
            # Generate channel
            paths = generate_channel_params()
            h_true = get_ground_truth_channel_grid(paths, M, N)
            
            # Generate LS estimate (noisy pilots)
            x_pilot = np.zeros((M, N), dtype=np.complex128)
            x_pilot.flat[pilot_indices] = 1.0
            
            snr = np.random.uniform(self.snr_min, self.snr_max)
            tx_sig = OTFS_Modem.modulate(x_pilot, M, N)
            rx_sig, _ = ltv_channel_sim(tx_sig, paths, snr, M, N)
            y_rx = OTFS_Modem.demodulate(rx_sig, M, N)
            
            # LS estimate: received pilots (since pilot=1)
            h_ls = np.zeros_like(y_rx)
            h_ls.flat[pilot_indices] = y_rx.flat[pilot_indices]
            
            # Convert to tensors
            h_ls_tensor = torch.stack([
                torch.from_numpy(h_ls.real).float(),
                torch.from_numpy(h_ls.imag).float()
            ], dim=0)
            
            h_true_tensor = torch.stack([
                torch.from_numpy(h_true.real).float(),
                torch.from_numpy(h_true.imag).float()
            ], dim=0)
            
            self.samples.append((h_ls_tensor, h_true_tensor))
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return self.samples[idx]


class DetectorDataset(Dataset):
    """
    Dataset for Detector Training
    
    Generates received signals and channel estimates for detector training.
    
    Input: 
        ls_input: (2, M, N) LS estimate for estimator
        y_grid: (2, M, N) received signal
    Target: (1, M, N) transmitted data
    
    Extracted from OTFS_4.ipynb Phase 4
    """
    def __init__(self, size, M, N, pilot_indices, data_indices, snr_range=(0, 25)):
        self.size = size
        self.M = M
        self.N = N
        self.pilot_indices = pilot_indices
        self.data_indices = data_indices
        self.num_symbols = M * N
        self.snr_min, self.snr_max = snr_range
        
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        # Generate data
        x_data = np.random.choice([-1, 1], size=self.num_symbols).astype(np.complex128)
        x_data.flat[self.pilot_indices] = 1.0
        x_grid = x_data.reshape(M, N)
        
        # Generate channel
        paths = generate_channel_params()
        snr = np.random.uniform(self.snr_min, self.snr_max)
        
        # Simulate transmission
        tx_sig = OTFS_Modem.modulate(x_grid, M, N)
        rx_sig, _ = ltv_channel_sim(tx_sig, paths, snr, M, N)
        y_grid = OTFS_Modem.demodulate(rx_sig, M, N)
        
        # LS estimate
        h_ls = np.zeros_like(y_grid)
        h_ls.flat[self.pilot_indices] = y_grid.flat[self.pilot_indices]
        
        # Convert to tensors
        ls_tensor = torch.stack([
            torch.from_numpy(h_ls.real).float(),
            torch.from_numpy(h_ls.imag).float()
        ], dim=0)
        
        y_tensor = torch.stack([
            torch.from_numpy(y_grid.real).float(),
            torch.from_numpy(y_grid.imag).float()
        ], dim=0)
        
        target = torch.from_numpy(x_grid.real).float().unsqueeze(0)
        
        return ls_tensor, y_tensor, target


class UNetDataset(Dataset):
    """
    Dataset for U-Net Channel Estimator Training
    
    Uses sparse pilot transmission for channel estimation.
    
    Input: (3, M, N) [Real Y, Imag Y, Pilot Mask]
    Target: (2, M, N) [Real H, Imag H]
    
    Extracted from OTFS_4.ipynb Phase 5
    """
    def __init__(self, size, M, N, pilot_indices, snr_range=(0, 30)):
        self.size = size
        self.M = M
        self.N = N
        self.pilot_indices = pilot_indices
        self.snr_min, self.snr_max = snr_range
        
        # Pre-compute Pilot Mask (Static)
        mask = np.zeros((M, N), dtype=np.float32)
        mask.flat[pilot_indices] = 1.0
        self.pilot_mask = torch.from_numpy(mask)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # 1. Generate Channel
        paths = generate_channel_params()
        h_true = get_ground_truth_channel_grid(paths, self.M, self.N)

        # 2. Generate Noisy Input
        # Send ONLY pilots (sparse transmission for estimation)
        x_tx = np.zeros((self.M, self.N), dtype=np.complex128)
        x_tx.flat[self.pilot_indices] = 1.0  # Pilot power
        
        snr = np.random.uniform(self.snr_min, self.snr_max)
        tx_sig = OTFS_Modem.modulate(x_tx, self.M, self.N)
        rx_sig, _ = ltv_channel_sim(tx_sig, paths, snr, self.M, self.N)
        y_rx = OTFS_Modem.demodulate(rx_sig, self.M, self.N)

        # 3. Prepare Input Tensor [3, M, N]
        # Ch0: Real Y, Ch1: Imag Y, Ch2: Pilot Mask
        y_real = torch.from_numpy(y_rx.real).float()
        y_imag = torch.from_numpy(y_rx.imag).float()
        
        inp = torch.stack([y_real, y_imag, self.pilot_mask], dim=0)

        # 4. Target [2, M, N]
        tgt = torch.stack([
            torch.from_numpy(h_true.real).float(),
            torch.from_numpy(h_true.imag).float()
        ], dim=0)

        return inp, tgt

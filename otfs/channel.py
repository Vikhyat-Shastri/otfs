"""
OTFS Channel Simulation
========================

Implements Linear Time-Variant (LTV) channel simulation with:
- Multiple propagation paths
- Delay (time shifts)
- Doppler shifts (frequency offsets)
- AWGN noise

Extracted from OTFS_4.ipynb Phase 3
"""

import numpy as np
from .modem import OTFS_Modem


def ltv_channel_sim(s_t, paths, snr_db, M=None, N=None):
    """
    Simulates a Linear Time-Variant (LTV) Channel.
    
    The channel consists of multiple paths, each with:
    - gain: Complex gain (magnitude * phase)
    - delay: Integer sample delay
    - doppler: Doppler shift (can be fractional)
    
    Args:
        s_t: Transmitted time signal (1D array)
        paths: List of dicts with keys {'gain': complex, 'delay': int, 'doppler': float}
        snr_db: Signal-to-noise ratio in dB
        M: Number of delay bins (for noiseless case)
        N: Number of Doppler bins (for noiseless case)
        
    Returns:
        r_t: Received time signal
        noise_power: Noise power (for MMSE calculations)
    """
    r_t = np.zeros_like(s_t, dtype=np.complex128)
    L = len(s_t)
    
    for path in paths:
        gain = path['gain']
        delay = path['delay']     # Integer sample delay
        doppler = path['doppler'] # Doppler shift
        
        # 1. Apply Delay (Circular shift for simplicity in block simulation)
        s_delayed = np.roll(s_t, delay)
        
        # 2. Apply Doppler (Phase rotation over time)
        # Doppler vector: e^(j * 2pi * doppler * n / L)
        t_indices = np.arange(L)
        doppler_phasor = np.exp(1j * 2 * np.pi * doppler * t_indices / L)
        
        # Accumulate path
        r_t += gain * s_delayed * doppler_phasor

    # 3. Add AWGN Noise
    if snr_db < 100:  # Add noise unless explicitly noiseless
        sig_power = np.mean(np.abs(s_t)**2)
        snr_linear = 10**(snr_db / 10.0)
        noise_power = sig_power / snr_linear
        noise = np.sqrt(noise_power/2) * (np.random.randn(L) + 1j * np.random.randn(L))
        return r_t + noise, noise_power
    else:
        return r_t, 0.0


def generate_channel_params(num_paths=None):
    """
    Generates realistic sparse channel parameters.
    
    Creates a multi-path channel with:
    - Path 1: Line of Sight (LOS) - strong, low delay, low Doppler
    - Paths 2-3: Reflectors - weaker, higher delay, random Doppler
    
    Gains are strictly Magnitudes (0 to 1) * Phase.
    
    Args:
        num_paths: Number of paths (if None, randomly chosen between 2-3)
        
    Returns:
        paths: List of path dictionaries with keys {'gain', 'delay', 'doppler'}
    """
    if num_paths is None:
        num_paths = np.random.randint(2, 4)  # 2 to 3 paths
    
    paths = []
    
    # Path 1: Line of Sight (Strong, Low Delay, Low Doppler)
    paths.append({
        'gain': 1.0 * np.exp(1j * np.random.uniform(0, 2*np.pi)),
        'delay': 0,
        'doppler': 0.0
    })
    
    # Path 2 & 3: Reflectors (Weaker, Higher Delay, Random Doppler)
    for _ in range(num_paths - 1):
        mag = np.random.uniform(0.1, 0.5)  # Gain magnitude close to 0-1
        phase = np.random.uniform(0, 2*np.pi)
        
        paths.append({
            'gain': mag * np.exp(1j * phase),
            'delay': np.random.randint(1, 3),      # Small integer delay
            'doppler': np.random.randint(-2, 3)    # Integer Doppler bin
        })
        
    return paths


def get_effective_channel_matrix(paths, M, N):
    """
    PROBE METHOD: Sends unit vectors through the channel to build 
    the exact H_eff matrix that relates y_dd = H_eff * x_dd.
    
    This guarantees MMSE/ZF are mathematically perfect (genie-aided).
    
    Args:
        paths: Channel path parameters
        M: Number of delay bins
        N: Number of Doppler bins
        
    Returns:
        H_eff: Effective channel matrix of shape (M*N, M*N)
    """
    num_symbols = M * N
    H_eff = np.zeros((num_symbols, num_symbols), dtype=np.complex128)
    
    # Probe every position in the grid
    for k in range(num_symbols):
        # 1. Create one-hot input grid
        probe_vec = np.zeros(num_symbols, dtype=np.complex128)
        probe_vec[k] = 1.0
        probe_grid = probe_vec.reshape(M, N)
        
        # 2. Modulate
        tx_sig = OTFS_Modem.modulate(probe_grid, M, N)
        
        # 3. Pass through noiseless channel
        rx_sig, _ = ltv_channel_sim(tx_sig, paths, snr_db=100, M=M, N=N)
        
        # 4. Demodulate
        rx_grid = OTFS_Modem.demodulate(rx_sig, M, N)
        
        # 5. Store result as column in H matrix
        H_eff[:, k] = rx_grid.flatten()
        
    return H_eff


def get_ground_truth_channel_grid(paths, M, N):
    """
    Transmits a single Dirac Impulse at (0,0) through a noiseless channel.
    The received grid is the exact Channel Impulse Response (CIR) we want to learn.
    
    Args:
        paths: Channel path parameters
        M: Number of delay bins
        N: Number of Doppler bins
        
    Returns:
        h_true_grid: Ground truth channel grid of shape (M, N)
    """
    sounding_grid = np.zeros((M, N), dtype=np.complex128)
    sounding_grid[0, 0] = 1.0  # Impulse
    
    tx_sig = OTFS_Modem.modulate(sounding_grid, M, N)
    rx_sig = ltv_channel_sim(tx_sig, paths, snr_db=200, M=M, N=N)[0]  # Noiseless
    h_true_grid = OTFS_Modem.demodulate(rx_sig, M, N)
    return h_true_grid

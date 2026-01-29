"""
OTFS Modulation and Demodulation
==================================

Implements the full OTFS transform chain:
- Delay-Doppler Grid → ISFFT → Time-Frequency → Heisenberg Transform → Time Domain
- Time Domain → Wigner Transform → Time-Frequency → SFFT → Delay-Doppler Grid

Extracted from OTFS_4.ipynb Phase 3
"""

import numpy as np


def fft(x, axis):
    """Normalized FFT"""
    return np.fft.fft(x, axis=axis, norm='ortho')


def ifft(x, axis):
    """Normalized IFFT"""
    return np.fft.ifft(x, axis=axis, norm='ortho')


class OTFS_Modem:
    """
    Handles the mathematical transforms for OTFS modulation/demodulation.
    
    The OTFS system operates on a Delay-Doppler (DD) grid of size M×N where:
    - M: Number of delay bins (subcarriers)
    - N: Number of Doppler bins (time slots)
    """
    
    @staticmethod
    def modulate(x_dd_grid, M=None, N=None):
        """
        Delay-Doppler → Time Domain
        
        Process:
        1. ISFFT (Inverse Symplectic Finite Fourier Transform): DD → Time-Frequency (TF)
        2. Heisenberg Transform: TF → Time Domain
        
        Args:
            x_dd_grid: Input DD grid of shape (M, N)
            M: Number of delay bins (if None, inferred from grid)
            N: Number of Doppler bins (if None, inferred from grid)
            
        Returns:
            x_t: Time domain signal (1D array of length M*N)
        """
        if M is None:
            M, N = x_dd_grid.shape
        else:
            assert x_dd_grid.shape == (M, N), f"Grid shape mismatch: {x_dd_grid.shape} != ({M}, {N})"
        
        # 1. ISFFT (Inverse Symplectic Finite Fourier Transform)
        # Convert DD (Delay-Doppler) to TF (Time-Frequency)
        # Convention: IFFT along Delay (axis 0), FFT along Doppler (axis 1)
        x_tf = fft(ifft(x_dd_grid, axis=0), axis=1)
        
        # 2. Heisenberg Transform (TF → Time)
        # Simplified: IFFT along frequency axis (cols) then flatten
        # Note: In standard OFDM/OTFS, this is effectively an IFFT on the freq axis
        x_t = ifft(x_tf, axis=0).flatten(order='F')  # Column-major flatten
        return x_t

    @staticmethod
    def demodulate(r_t, M, N):
        """
        Time Domain → Delay-Doppler
        
        Process:
        1. Wigner Transform: Time → Time-Frequency (TF)
        2. SFFT (Symplectic Finite Fourier Transform): TF → DD
        
        Args:
            r_t: Received time domain signal (1D array of length M*N)
            M: Number of delay bins
            N: Number of Doppler bins
            
        Returns:
            y_dd: Demodulated DD grid of shape (M, N)
        """
        # Reshape to TF grid
        r_tf = fft(r_t.reshape(M, N, order='F'), axis=0)
        
        # 3. SFFT (Symplectic Finite Fourier Transform)
        # TF → DD
        # Inverse of ISFFT: FFT along Delay, IFFT along Doppler
        y_dd = ifft(fft(r_tf, axis=1), axis=0)
        return y_dd

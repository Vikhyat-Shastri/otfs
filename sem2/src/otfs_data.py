"""
OTFS Data Generation with Fractional Doppler
=============================================
Rigorous DD-domain physics: Dirichlet kernel for fractional Doppler leakage,
embedded pilot with guard band (circular-safe), and full H_DD channel matrix.
"""

import numpy as np
from scipy.linalg import circulant as scipy_circulant

# ---------------------------------------------------------------------------
# Grid & pilot configuration
# ---------------------------------------------------------------------------
M = 16                      # delay bins
N = 16                      # Doppler bins
MN = M * N                  # 256 total DD symbols

P_PATHS = 3                 # number of channel paths
DELAY_TAPS = np.array([0, 4, 8])
L_MAX = int(DELAY_TAPS.max())      # 8

# Guard delays: stable set under subtraction by DELAY_TAPS (mod M).
# {0,4,8,12} - 4 = {12,0,4,8} ⊂ guard  ✓
# {0,4,8,12} - 8 = {8,12,0,4} ⊂ guard  ✓
GUARD_DELAYS = np.array([0, 4, 8, 12])
ACTIVE_DELAYS = DELAY_TAPS.copy()  # delays where paths exist

K_P, L_P = 0, 0             # pilot position (Doppler, delay)

PILOT_AMP = np.sqrt(M * N)  # boosted pilot amplitude

# Masks (flat, Doppler-major ordering: idx = k*M + l)
GUARD_MASK = np.zeros(MN, dtype=bool)
for _k in range(N):
    for _l in GUARD_DELAYS:
        GUARD_MASK[_k * M + _l] = True

DATA_MASK = ~GUARD_MASK
NUM_DATA = int(DATA_MASK.sum())       # 192
DATA_INDICES = np.where(DATA_MASK)[0]
GUARD_INDICES = np.where(GUARD_MASK)[0]


# ---------------------------------------------------------------------------
# Dirichlet kernel  D_N(alpha)
# ---------------------------------------------------------------------------
def dirichlet_kernel(alpha, N_val):
    """
    D_N(alpha) = (1/N) sum_{n=0}^{N-1} exp(j 2pi n alpha / N)

    Closed form for non-integer alpha:
        (1/N) exp(j pi alpha (N-1)/N) sin(pi alpha) / sin(pi alpha/N)
    For integer alpha: 1 if alpha mod N == 0, else 0.
    """
    alpha = np.asarray(alpha, dtype=np.float64)
    result = np.zeros_like(alpha, dtype=np.complex128)

    int_mask = np.abs(alpha - np.round(alpha)) < 1e-12
    rounded = np.round(alpha).astype(np.int64)
    result[int_mask] = np.where(rounded[int_mask] % N_val == 0,
                                1.0 + 0j, 0.0 + 0j)

    frac_mask = ~int_mask
    a = alpha[frac_mask]
    phase = np.exp(1j * np.pi * a * (N_val - 1) / N_val)
    num = np.sin(np.pi * a)
    den = np.sin(np.pi * a / N_val)
    result[frac_mask] = (1.0 / N_val) * phase * num / den

    return result


# ---------------------------------------------------------------------------
# Channel realisation
# ---------------------------------------------------------------------------
def generate_channel(batch_size, rng=None):
    """
    Returns
    -------
    h_gains  : (batch, P) complex128  – power-normalised
    delays   : (P,) int
    dopplers : (batch, P) float64     – fractional Doppler indices
    """
    if rng is None:
        rng = np.random.default_rng()

    delays = DELAY_TAPS.copy()

    h_gains = (rng.standard_normal((batch_size, P_PATHS))
               + 1j * rng.standard_normal((batch_size, P_PATHS))) / np.sqrt(2)
    power = np.sum(np.abs(h_gains) ** 2, axis=1, keepdims=True)
    h_gains = h_gains / np.sqrt(power)

    k_int = rng.integers(-3, 4, size=(batch_size, P_PATHS))
    eps = rng.uniform(-0.499, 0.499, size=(batch_size, P_PATHS))
    dopplers = k_int.astype(np.float64) + eps

    return h_gains, delays, dopplers


# ---------------------------------------------------------------------------
# Build DD channel matrix  (Kronecker product formulation)
# ---------------------------------------------------------------------------
def _delay_perm(l_p, M_val):
    """M x M circular-shift permutation for delay l_p."""
    P = np.zeros((M_val, M_val), dtype=np.float64)
    for l_in in range(M_val):
        P[(l_in + l_p) % M_val, l_in] = 1.0
    return P


# Pre-compute permutation matrices for each delay tap
_PERM_CACHE = {int(d): _delay_perm(int(d), M) for d in DELAY_TAPS}
_PERM_GUARD_CACHE = {int(d): _delay_perm(int(d), M) for d in GUARD_DELAYS}


def build_dd_channel_matrix(h_gains, delays, dopplers):
    """
    H_DD = sum_p  h_p * kron(D_p, P_p)
    """
    H = np.zeros((MN, MN), dtype=np.complex128)
    dk_vals = (np.arange(N)[:, None] - np.arange(N)[None, :]).ravel()

    for p in range(len(delays)):
        D = dirichlet_kernel(dopplers[p] - dk_vals, N).reshape(N, N)
        H += h_gains[p] * np.kron(D, _PERM_CACHE[int(delays[p])])

    return H


def build_dd_channel_matrix_batch(h_gains_batch, delays, dopplers_batch):
    """Build H_DD for each sample in the batch."""
    return [build_dd_channel_matrix(h_gains_batch[i], delays, dopplers_batch[i])
            for i in range(h_gains_batch.shape[0])]


# ---------------------------------------------------------------------------
# Transmit frame construction
# ---------------------------------------------------------------------------
def build_tx_frame(batch_size, rng=None):
    """Embed pilot + BPSK data into DD frame."""
    if rng is None:
        rng = np.random.default_rng()

    x = np.zeros((batch_size, MN), dtype=np.float64)
    x[:, DATA_INDICES] = rng.choice([-1.0, 1.0], size=(batch_size, NUM_DATA))
    x[:, K_P * M + L_P] = PILOT_AMP

    return x


# ---------------------------------------------------------------------------
# Channel application + AWGN
# ---------------------------------------------------------------------------
def apply_channel(x, H_list, snr_db, rng=None):
    """y = H_DD @ x + noise.  Returns (y_real, y_imag, sigma2)."""
    if rng is None:
        rng = np.random.default_rng()

    B = x.shape[0]
    snr_lin = 10.0 ** (snr_db / 10.0)
    N0 = 1.0 / snr_lin
    sigma2 = N0 / 2.0
    sigma = np.sqrt(sigma2)

    H_batch = np.stack(H_list, axis=0)
    y_complex = np.einsum("bij,bj->bi", H_batch, x)
    noise = sigma * (rng.standard_normal((B, MN))
                     + 1j * rng.standard_normal((B, MN)))
    y_complex += noise

    return y_complex.real.copy(), y_complex.imag.copy(), sigma2


# ---------------------------------------------------------------------------
# Channel estimation from guard band  (circular-safe)
# ---------------------------------------------------------------------------
def estimate_channel_from_guard(y_real, y_imag, threshold_factor=0.1):
    """
    Extract the spreading function from guard-band pilot response and
    reconstruct H_DD_est via Kronecker products.
    """
    B = y_real.shape[0]
    y_complex = y_real + 1j * y_imag
    y_grid = y_complex.reshape(B, N, M)  # (B, k, l)

    h_est = np.zeros((B, N, len(GUARD_DELAYS)), dtype=np.complex128)
    for i, dl in enumerate(GUARD_DELAYS):
        l_idx = (L_P + dl) % M
        h_est[:, :, i] = y_grid[:, :, l_idx] / PILOT_AMP

    # Threshold: suppress inactive delay taps (noise-only)
    energies = np.sum(np.abs(h_est) ** 2, axis=1)  # (B, len(GUARD_DELAYS))
    max_en = energies.max(axis=1, keepdims=True)     # (B, 1)
    inactive = energies < threshold_factor * max_en   # (B, len(GUARD_DELAYS))
    for i in range(len(GUARD_DELAYS)):
        h_est[inactive[:, i], :, i] = 0.0

    # Reconstruct H_DD_est via Kronecker products
    H_est_list = []
    for b in range(B):
        H = np.zeros((MN, MN), dtype=np.complex128)
        for i, dl in enumerate(GUARD_DELAYS):
            col_vec = np.roll(h_est[b, :, i], -K_P)
            D_dl = scipy_circulant(col_vec)
            H += np.kron(D_dl, _PERM_GUARD_CACHE[int(dl)])
        H_est_list.append(H)

    return H_est_list


# ---------------------------------------------------------------------------
# Full sample generation (convenience)
# ---------------------------------------------------------------------------
def generate_samples(batch_size, snr_db, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    h_gains, delays, dopplers = generate_channel(batch_size, rng)
    H_list = build_dd_channel_matrix_batch(h_gains, delays, dopplers)
    x = build_tx_frame(batch_size, rng)
    y_re, y_im, sigma2 = apply_channel(x, H_list, snr_db, rng)
    H_est_list = estimate_channel_from_guard(y_re, y_im)

    return {
        "x": x, "y_real": y_re, "y_imag": y_im,
        "H_true": H_list, "H_est": H_est_list, "sigma2": sigma2,
        "h_gains": h_gains, "delays": delays, "dopplers": dopplers,
    }


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== OTFS Data Generation Sanity Check ===")
    print(f"Grid: {M}x{N} = {MN} symbols")
    print(f"Delay taps: {DELAY_TAPS}")
    print(f"Guard delays: {GUARD_DELAYS}  ({GUARD_MASK.sum()} positions)")
    print(f"Data positions: {NUM_DATA}")
    print(f"Pilot at (k={K_P}, l={L_P}), amplitude={PILOT_AMP:.2f}")

    rng = np.random.default_rng(42)
    h, dl, dp = generate_channel(1, rng)
    print(f"\nChannel: delays={dl}, dopplers={dp[0].round(3)}")

    H = build_dd_channel_matrix(h[0], dl, dp[0])
    print(f"  H_DD nnz ratio: {(np.abs(H) > 1e-10).sum() / MN**2:.3f}")

    x = build_tx_frame(1, rng)

    # Verify no data contamination in guard
    pilot_idx = K_P * M + L_P
    x_pilot_only = np.zeros(MN); x_pilot_only[pilot_idx] = PILOT_AMP
    x_data_only = x[0].copy(); x_data_only[pilot_idx] = 0
    y_pilot = H @ x_pilot_only
    y_data = H @ x_data_only

    gp = sum(abs(y_pilot[k * M + gl]) ** 2
             for k in range(N) for gl in GUARD_DELAYS)
    gd = sum(abs(y_data[k * M + gl]) ** 2
             for k in range(N) for gl in GUARD_DELAYS)
    print(f"\n  Guard pilot power:  {gp:.2f}")
    print(f"  Guard data leakage: {gd:.6f}")
    print(f"  SIR in guard: {10 * np.log10(gp / max(gd, 1e-30)):.1f} dB")

    y_re, y_im, s2 = apply_channel(x, [H], 20.0, rng)
    H_est = estimate_channel_from_guard(y_re, y_im)
    err = np.linalg.norm(H - H_est[0]) / np.linalg.norm(H)
    print(f"\n  Channel est relative error @20dB: {err:.6f}")
    print("=== Done ===")

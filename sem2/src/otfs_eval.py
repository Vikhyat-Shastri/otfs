"""
OTFS Evaluation – BER vs SNR
=============================
Compares OAMPNet (with pilot-based channel estimation) against
genie-aided LMMSE and ZF classical baselines.
"""

import os
import sys
import time
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from otfs_data import (
    M, N, MN, NUM_DATA, DATA_INDICES, DATA_MASK,
    K_P, L_P, PILOT_AMP,
    generate_channel, build_dd_channel_matrix_batch,
    build_tx_frame, apply_channel, estimate_channel_from_guard,
)
from otfs_system import OAMPNet, DEVICE, MODEL_PATH, T_LAYERS

# ---------------------------------------------------------------------------
# Paths (resolve relative to project root)
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SNR_RANGE = np.arange(0, 22, 2)   # 0, 2, ..., 20 dB
NUM_TEST = 2000                    # frames per SNR point
EVAL_BATCH = 200                   # sub-batch size
PLOT_PATH = os.path.join(_PROJECT_ROOT, "results", "ber_vs_snr.png")

GUARD_IDX = np.where(~DATA_MASK)[0]
_x_guard_template = np.zeros(MN)
_x_guard_template[K_P * M + L_P] = PILOT_AMP
_x_guard_at_gidx = _x_guard_template[GUARD_IDX]   # (96,)


# ===================================================================
# Batched classical baselines (genie-aided – perfect H_DD)
# Uses torch.linalg.solve on CPU (complex64) for speed on Apple Silicon
# ===================================================================
_guard_at_t = torch.from_numpy(_x_guard_at_gidx.astype(np.complex64))
_eye_data = torch.eye(NUM_DATA, dtype=torch.complex64)


def detect_lmmse_batch(H_batch, y_re, y_im, sigma2):
    """Genie LMMSE: x_hat = (H_d^H H_d + sigma2 I)^{-1} H_d^H y'"""
    H = torch.from_numpy(H_batch.astype(np.complex64))
    y = torch.from_numpy((y_re + 1j * y_im).astype(np.complex64))

    y_prime = y - torch.bmm(H[:, :, GUARD_IDX], _guard_at_t.expand(H.shape[0], -1).unsqueeze(2)).squeeze(2)

    H_d = H[:, :, DATA_INDICES]
    H_dH = H_d.conj().transpose(1, 2)
    HdH = torch.bmm(H_dH, H_d)
    A = HdH + float(sigma2) * _eye_data.unsqueeze(0)
    rhs = torch.bmm(H_dH, y_prime.unsqueeze(2)).squeeze(2)
    x_est = torch.linalg.solve(A, rhs)

    return np.sign(x_est.real.numpy())


def detect_zf_batch(H_batch, y_re, y_im):
    """Genie ZF: x_hat = (H_d^H H_d)^{-1} H_d^H y'"""
    H = torch.from_numpy(H_batch.astype(np.complex64))
    y = torch.from_numpy((y_re + 1j * y_im).astype(np.complex64))

    y_prime = y - torch.bmm(H[:, :, GUARD_IDX], _guard_at_t.expand(H.shape[0], -1).unsqueeze(2)).squeeze(2)

    H_d = H[:, :, DATA_INDICES]
    H_dH = H_d.conj().transpose(1, 2)
    HdH = torch.bmm(H_dH, H_d)
    A = HdH + 1e-6 * _eye_data.unsqueeze(0)
    rhs = torch.bmm(H_dH, y_prime.unsqueeze(2)).squeeze(2)
    x_est = torch.linalg.solve(A, rhs)

    return np.sign(x_est.real.numpy())


# ===================================================================
# DL detector  (pilot-based channel estimate)
# ===================================================================
def detect_dl(model, y_re, y_im, H_est_list, sigma2):
    """Run OAMPNet on a batch."""
    B = y_re.shape[0]
    Hre = np.stack([h.real for h in H_est_list]).astype(np.float32)
    Him = np.stack([h.imag for h in H_est_list]).astype(np.float32)

    yr_t = torch.from_numpy(y_re.astype(np.float32)).to(DEVICE)
    yi_t = torch.from_numpy(y_im.astype(np.float32)).to(DEVICE)
    Hr_t = torch.from_numpy(Hre).to(DEVICE)
    Hi_t = torch.from_numpy(Him).to(DEVICE)
    s2_t = torch.full((B, 1), float(sigma2),
                      dtype=torch.float32, device=DEVICE)

    with torch.no_grad():
        x_hat = model(yr_t, yi_t, Hr_t, Hi_t, s2_t)

    x_hat_np = x_hat.cpu().numpy()
    return np.sign(x_hat_np[:, DATA_INDICES])


# ===================================================================
# BER evaluation
# ===================================================================
def evaluate_ber(model):
    model.eval()
    rng = np.random.default_rng(12345)

    ber_dl, ber_lmmse, ber_zf = [], [], []

    for si, snr in enumerate(SNR_RANGE):
        errs_dl = 0; errs_lm = 0; errs_zf = 0; total = 0
        t0 = time.time()

        n_done = 0
        while n_done < NUM_TEST:
            bs = min(EVAL_BATCH, NUM_TEST - n_done)

            h_gains, delays, dopplers = generate_channel(bs, rng)
            H_list = build_dd_channel_matrix_batch(h_gains, delays, dopplers)
            x = build_tx_frame(bs, rng)
            y_re, y_im, sigma2 = apply_channel(x, H_list, float(snr), rng)
            H_est = estimate_channel_from_guard(y_re, y_im)
            x_data = x[:, DATA_INDICES]

            H_np = np.stack(H_list, axis=0)                      # (bs, MN, MN)

            det_dl = detect_dl(model, y_re, y_im, H_est, sigma2)
            errs_dl += np.sum(det_dl != x_data)

            det_lm = detect_lmmse_batch(H_np, y_re, y_im, sigma2)
            errs_lm += np.sum(det_lm != x_data)

            det_zf = detect_zf_batch(H_np, y_re, y_im)
            errs_zf += np.sum(det_zf != x_data)

            total += bs * NUM_DATA
            n_done += bs

        b_dl  = max(errs_dl / total, 1e-7)
        b_lm  = max(errs_lm / total, 1e-7)
        b_zf  = max(errs_zf / total, 1e-7)
        ber_dl.append(b_dl); ber_lmmse.append(b_lm); ber_zf.append(b_zf)

        elapsed = time.time() - t0
        print(f"  [{si+1:2d}/{len(SNR_RANGE)}] SNR={snr:2d} dB | "
              f"DL={b_dl:.5f}  LMMSE={b_lm:.5f}  ZF={b_zf:.5f}  "
              f"({elapsed:.1f}s)")
        sys.stdout.flush()

    return np.array(ber_dl), np.array(ber_lmmse), np.array(ber_zf)


# ===================================================================
# Plotting
# ===================================================================
def plot_ber(ber_dl, ber_lmmse, ber_zf):
    plt.figure(figsize=(10, 7))
    plt.semilogy(SNR_RANGE, ber_dl,    "o-",  label="OAMPNet (pilot est.)",
                 linewidth=2.5, markersize=7, color="tab:blue")
    plt.semilogy(SNR_RANGE, ber_lmmse, "s--", label="LMMSE (genie)",
                 linewidth=2, markersize=7, color="tab:green")
    plt.semilogy(SNR_RANGE, ber_zf,    "d:",  label="ZF (genie)",
                 linewidth=2, markersize=7, color="tab:purple")
    plt.xlabel("SNR (dB)", fontsize=13)
    plt.ylabel("Bit Error Rate", fontsize=13)
    plt.title("OTFS Fractional Doppler: OAMPNet vs Classical Baselines\n"
              f"Grid {M}x{N}, 3-tap channel, BPSK", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.ylim(1e-6, 1)
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=150)
    print(f"\nPlot saved to {PLOT_PATH}")


# ===================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("OTFS BER Evaluation")
    print("=" * 70)
    sys.stdout.flush()

    model = OAMPNet(num_layers=T_LAYERS).to(DEVICE)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(
            torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
        )
        print(f"Loaded model from {MODEL_PATH}")
    else:
        print(f"WARNING: {MODEL_PATH} not found – using untrained model")
    sys.stdout.flush()

    ber_dl, ber_lmmse, ber_zf = evaluate_ber(model)

    print("\n" + "=" * 70)
    print("Final BER Table")
    print("=" * 70)
    print(f"{'SNR':>5s} | {'OAMPNet':>10s} | {'LMMSE':>10s} | {'ZF':>10s}")
    print("-" * 45)
    for i, snr in enumerate(SNR_RANGE):
        print(f"{snr:5d} | {ber_dl[i]:10.6f} | {ber_lmmse[i]:10.6f} | {ber_zf[i]:10.6f}")

    plot_ber(ber_dl, ber_lmmse, ber_zf)

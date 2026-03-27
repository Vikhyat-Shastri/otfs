"""
OTFS OAMPNet Receiver – Training Script  (v2: LMMSE via CG on MPS)
===================================================================
Deep Unfolded OAMP with:
  - LMMSE linear estimator  (CG solver, fully MPS-native)
  - Learnable lambda_t, gamma_t, beta_t per layer
  - Noise-conditioned CNN denoiser
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
import sys
from concurrent.futures import ThreadPoolExecutor

from otfs_data import (
    M, N, MN, NUM_DATA, DATA_INDICES, DATA_MASK,
    generate_channel, build_dd_channel_matrix_batch,
    build_tx_frame, apply_channel, estimate_channel_from_guard,
)

# ---------------------------------------------------------------------------
# Paths (resolve relative to project root)
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
print(f"[Device] {DEVICE}")

# ---------------------------------------------------------------------------
# Hyper-parameters
# ---------------------------------------------------------------------------
T_LAYERS = 8
DENOISER_CH = 32
BATCH_SIZE = 32
NUM_TRAIN = 16000
NUM_EPOCHS = 30
LR = 1e-3
SNR_TRAIN_LO = 0.0
SNR_TRAIN_HI = 20.0
MODEL_PATH = os.path.join(_PROJECT_ROOT, "models", "oampnet.pth")
CG_ITERS = 15

_data_mask_np = DATA_MASK.astype(np.float32)
DATA_MASK_T = torch.from_numpy(_data_mask_np).to(DEVICE)


# ===================================================================
# Batched CG solve  –  runs entirely on MPS, differentiable
# ===================================================================
def cg_solve(A, b, max_iter=CG_ITERS):
    """Solve A @ x = b via CG. A: (B, N, N) SPD, b: (B, N)."""
    x = torch.zeros_like(b)
    r = b.clone()
    p = r.clone()
    rs = (r * r).sum(dim=-1, keepdim=True)

    for _ in range(max_iter):
        Ap = torch.bmm(A, p.unsqueeze(2)).squeeze(2)
        pAp = (p * Ap).sum(dim=-1, keepdim=True)
        alpha = rs / (pAp + 1e-10)
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = (r * r).sum(dim=-1, keepdim=True)
        p = r + (rs_new / (rs + 1e-10)) * p
        rs = rs_new
    return x


# ===================================================================
# Noise-conditioned CNN Denoiser
# ===================================================================
class CNNDenoiser(nn.Module):
    def __init__(self, hidden=DENOISER_CH):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2, hidden, 3, padding=1),
            nn.PReLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.PReLU(),
            nn.Conv2d(hidden, 1, 3, padding=1),
        )

    def forward(self, x_2d, noise_map):
        inp = torch.cat([x_2d, noise_map], dim=1)
        return x_2d + self.net(inp)


# ===================================================================
# Single OAMP Layer
# ===================================================================
class OAMPLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.gamma = nn.Parameter(torch.tensor(1.0))
        self.log_lam = nn.Parameter(torch.tensor(0.0))
        self.log_beta = nn.Parameter(torch.tensor(0.0))
        self.denoiser = CNNDenoiser()

    def forward(self, x_hat, y_re, y_im, Hre, Him, HtH, sigma2_est):
        lam = torch.exp(self.log_lam)
        beta = torch.exp(self.log_beta)

        Hx_re = torch.bmm(Hre, x_hat.unsqueeze(2)).squeeze(2)
        Hx_im = torch.bmm(Him, x_hat.unsqueeze(2)).squeeze(2)
        z_re = y_re - Hx_re
        z_im = y_im - Hx_im

        mf = (torch.bmm(Hre.transpose(1, 2), z_re.unsqueeze(2)).squeeze(2)
              + torch.bmm(Him.transpose(1, 2), z_im.unsqueeze(2)).squeeze(2))

        A = HtH + lam * torch.eye(MN, device=HtH.device).unsqueeze(0)
        step = cg_solve(A, mf)

        r = x_hat + self.gamma * step

        eff_sigma2 = beta * sigma2_est
        noise_map = eff_sigma2.view(-1, 1, 1, 1).expand(-1, 1, N, M)
        r_2d = r.view(-1, 1, N, M)
        x_hat_2d = self.denoiser(r_2d, noise_map)
        x_hat_new = torch.tanh(x_hat_2d.view(-1, MN))

        return x_hat_new


# ===================================================================
# Full OAMPNet
# ===================================================================
class OAMPNet(nn.Module):
    def __init__(self, num_layers=T_LAYERS):
        super().__init__()
        self.layers = nn.ModuleList([OAMPLayer() for _ in range(num_layers)])

    def forward(self, y_re, y_im, Hre, Him, sigma2):
        B = y_re.shape[0]
        x_hat = torch.zeros(B, MN, device=y_re.device)

        HtH = (torch.bmm(Hre.transpose(1, 2), Hre)
               + torch.bmm(Him.transpose(1, 2), Him)).detach()

        for layer in self.layers:
            x_hat = layer(x_hat, y_re, y_im, Hre, Him, HtH, sigma2)
        return x_hat


# ===================================================================
# Batch generator (numpy, called from background thread)
# ===================================================================
def _gen_batch_np(batch_size, snr_lo, snr_hi, rng):
    snr_db = rng.uniform(snr_lo, snr_hi)
    h_gains, delays, dopplers = generate_channel(batch_size, rng)
    H_true = build_dd_channel_matrix_batch(h_gains, delays, dopplers)
    x = build_tx_frame(batch_size, rng)
    y_re, y_im, sigma2 = apply_channel(x, H_true, snr_db, rng)
    H_est = estimate_channel_from_guard(y_re, y_im)
    Hre = np.stack([h.real for h in H_est]).astype(np.float32)
    Him = np.stack([h.imag for h in H_est]).astype(np.float32)
    return (x.astype(np.float32), y_re.astype(np.float32),
            y_im.astype(np.float32), Hre, Him, float(sigma2))


def _to_device(batch_np, device, batch_size):
    x_np, yr_np, yi_np, Hre, Him, sigma2 = batch_np
    x_t  = torch.from_numpy(x_np).to(device)
    yr_t = torch.from_numpy(yr_np).to(device)
    yi_t = torch.from_numpy(yi_np).to(device)
    Hr_t = torch.from_numpy(Hre).to(device)
    Hi_t = torch.from_numpy(Him).to(device)
    s2_t = torch.full((batch_size, 1), sigma2,
                      dtype=torch.float32, device=device)
    return yr_t, yi_t, Hr_t, Hi_t, s2_t, x_t


def make_batch(batch_size, snr_lo, snr_hi, rng):
    """Convenience wrapper (non-prefetched)."""
    np_data = _gen_batch_np(batch_size, snr_lo, snr_hi, rng)
    return _to_device(np_data, DEVICE, batch_size)


# ===================================================================
# Training loop with prefetching
# ===================================================================
def train():
    print("=" * 70)
    print("OAMPNet v2 Training  (LMMSE-CG + noise-conditioned CNN)")
    print("=" * 70)
    model = OAMPNet().to(DEVICE)
    params = sum(p.numel() for p in model.parameters())
    print(f"  Layers: {T_LAYERS}   CG iters: {CG_ITERS}   "
          f"Parameters: {params:,}")
    sys.stdout.flush()

    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS, eta_min=1e-5
    )

    rng = np.random.default_rng(0)
    steps_per_epoch = NUM_TRAIN // BATCH_SIZE
    best_loss = float("inf")

    pool = ThreadPoolExecutor(max_workers=1)

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        future = pool.submit(
            _gen_batch_np, BATCH_SIZE, SNR_TRAIN_LO, SNR_TRAIN_HI, rng
        )

        for step in range(steps_per_epoch):
            batch_np = future.result()
            if step + 1 < steps_per_epoch:
                future = pool.submit(
                    _gen_batch_np, BATCH_SIZE,
                    SNR_TRAIN_LO, SNR_TRAIN_HI, rng
                )

            yr, yi, Hr, Hi, s2, x_true = _to_device(
                batch_np, DEVICE, BATCH_SIZE
            )
            x_hat = model(yr, yi, Hr, Hi, s2)

            diff = (x_hat - x_true) * DATA_MASK_T
            loss = (diff ** 2).sum() / (BATCH_SIZE * NUM_DATA)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        epoch_loss /= steps_per_epoch
        elapsed = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), MODEL_PATH)

        print(f"  Epoch {epoch:3d}/{NUM_EPOCHS}  "
              f"loss={epoch_loss:.6f}  best={best_loss:.6f}  "
              f"lr={lr_now:.1e}  time={elapsed:.1f}s")
        sys.stdout.flush()

    pool.shutdown()
    print(f"\nTraining complete.  Best loss: {best_loss:.6f}")
    print(f"Model saved to {MODEL_PATH}")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE,
                                     weights_only=True))
    return model


if __name__ == "__main__":
    train()

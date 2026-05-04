"""Improved ensemble retraining — same architecture, better training.

Key improvements:
  - AdamW with weight decay for better generalization
  - Cosine annealing LR (smoother convergence than ReduceLROnPlateau)
  - More epochs (50) with patience=8
  - Higher dropout (0.3) for regularization
  - Trains both single model (ctmas_model.pt) and ensemble (ctmas_ensemble_*.pt)

Usage:
    python retrain.py
    python retrain.py --epochs 60
"""

from __future__ import annotations

import argparse
import shutil
import warnings
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")

from device import DEVICE
from models.gnn_model import (
    EDGE_INDEX,
    N_STAGES,
    STAGE_SENSOR_COUNTS,
    SpatioTemporalGNNAutoencoder,
    _split_to_stages,
)

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "Data" / "processed"


def denoising_loss(model, x_clean, edge_index, noise_std):
    """MSE between clean target and reconstruction-of-noisy input."""
    x_noisy = (x_clean + torch.randn_like(x_clean) * noise_std).clamp(0.0, 1.0)
    out = model(x_noisy, edge_index)
    clean_staged = _split_to_stages(x_clean)
    recon = out["recon"]
    node_mse = torch.zeros(N_STAGES, device=x_clean.device)
    for i, c in enumerate(STAGE_SENSOR_COUNTS):
        diff = clean_staged[:, i, :, :c] - recon[:, i, :, :c]
        node_mse[i] = (diff ** 2).mean()
    return (model.node_weights * node_mse).sum()


def train_one(
    seed: int,
    X_train: np.ndarray,
    X_val: np.ndarray,
    save_path: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    patience: int,
    dropout: float,
    noise_std: float,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    g = torch.Generator().manual_seed(seed)

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train, dtype=torch.float32)),
        batch_size=batch_size, shuffle=True, drop_last=True, generator=g,
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_val, dtype=torch.float32)),
        batch_size=batch_size, shuffle=False,
    )

    # Same architecture as the rest of the codebase (hidden=64, latent=32)
    model = SpatioTemporalGNNAutoencoder(dropout=dropout).to(DEVICE)
    edge = EDGE_INDEX.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)

    best_val, best_state, no_imp = float("inf"), None, 0
    for ep in range(1, epochs + 1):
        model.train()
        tloss, tb = 0.0, 0
        for (x,) in train_loader:
            x = x.to(DEVICE)
            opt.zero_grad()
            loss = denoising_loss(model, x, edge, noise_std)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tloss += loss.item()
            tb += 1
        train_loss = tloss / max(1, tb)

        model.eval()
        vloss, vb = 0.0, 0
        with torch.no_grad():
            for (x,) in val_loader:
                x = x.to(DEVICE)
                out = model(x, edge)
                loss, _ = model.reconstruction_loss(out)
                vloss += loss.item()
                vb += 1
        val_loss = vloss / max(1, vb)
        sched.step()

        marker = ""
        if val_loss < best_val - 1e-7:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_imp = 0
            marker = "  *"
        else:
            no_imp += 1
        lr_now = opt.param_groups[0]["lr"]
        print(f"  [seed={seed} ep={ep:02d}] train={train_loss:.6f}  val={val_loss:.6f}  lr={lr_now:.1e}{marker}")
        if no_imp >= patience:
            print(f"  [seed={seed}] early stop after {ep} epochs")
            break

    torch.save(best_state, save_path)
    print(f"  [seed={seed}] saved -> {save_path}  (best val={best_val:.6f})")
    return best_val


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=3, help="Ensemble size")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=8e-4)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--noise-std", type=float, default=0.05)
    args = p.parse_args()

    print(f"[Device] {DEVICE}")
    X_train = np.load(DATA_DIR / "X_train.npy")
    X_val = np.load(DATA_DIR / "X_val.npy")
    np.nan_to_num(X_train, copy=False, nan=0.0)
    np.nan_to_num(X_val, copy=False, nan=0.0)
    print(f"[Data] train={X_train.shape}  val={X_val.shape}")
    print(f"[Cfg]  ensemble={args.n}  dropout={args.dropout}  noise={args.noise_std}  lr={args.lr}")
    print(f"       epochs={args.epochs}  patience={args.patience}  batch_size={args.batch_size}")

    losses = []
    for i in range(args.n):
        print(f"\n=== Member {i+1}/{args.n} (seed={i}) ===")
        bv = train_one(
            i, X_train, X_val, BASE_DIR / f"ctmas_ensemble_{i}.pt",
            args.epochs, args.batch_size, args.lr, args.patience,
            args.dropout, args.noise_std,
        )
        losses.append(bv)

    # Also save member 0 as the single-model checkpoint
    shutil.copy(BASE_DIR / "ctmas_ensemble_0.pt", BASE_DIR / "ctmas_model.pt")
    print(f"\n[Done] member val losses: {[round(x, 6) for x in losses]}")
    print(f"       mean: {float(np.mean(losses)):.6f}")
    print(f"       ctmas_model.pt updated from member 0")


if __name__ == "__main__":
    main()

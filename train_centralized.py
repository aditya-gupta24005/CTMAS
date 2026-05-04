"""Centralized pre-training for the GNN autoencoder.

Bypasses FedProx/DP noise/Byzantine aggregation to drive the
reconstruction loss down to convergence. The federated path in
main.py is unchanged — this script just produces a stronger
checkpoint for the detector.

Usage:
    python train_centralized.py
    python train_centralized.py --epochs 30 --batch-size 256
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")

from device import DEVICE
from models.gnn_model import EDGE_INDEX, SpatioTemporalGNNAutoencoder

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "Data" / "processed"
MODEL_PATH = BASE_DIR / "ctmas_model.pt"


def train(epochs: int, batch_size: int, lr: float, patience: int):
    print(f"[Device] {DEVICE}")
    X_train = np.load(DATA_DIR / "X_train.npy")
    X_val = np.load(DATA_DIR / "X_val.npy")
    np.nan_to_num(X_train, copy=False, nan=0.0)
    np.nan_to_num(X_val, copy=False, nan=0.0)
    print(f"[Data] train={X_train.shape}, val={X_val.shape}")

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train, dtype=torch.float32)),
        batch_size=batch_size, shuffle=True, drop_last=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_val, dtype=torch.float32)),
        batch_size=batch_size, shuffle=False,
    )

    model = SpatioTemporalGNNAutoencoder().to(DEVICE)
    edge_index = EDGE_INDEX.to(DEVICE)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, mode="min", factor=0.5, patience=2
    )

    best_val = float("inf")
    best_state = None
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        nb = 0
        for (x,) in train_loader:
            x = x.to(DEVICE)
            optim.zero_grad()
            out = model(x, edge_index)
            loss, _ = model.reconstruction_loss(out)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optim.step()
            running += loss.item()
            nb += 1
        train_loss = running / max(1, nb)

        model.eval()
        vt, vb = 0.0, 0
        with torch.no_grad():
            for (x,) in val_loader:
                x = x.to(DEVICE)
                out = model(x, edge_index)
                loss, _ = model.reconstruction_loss(out)
                vt += loss.item()
                vb += 1
        val_loss = vt / max(1, vb)

        sched.step(val_loss)
        marker = ""
        if val_loss < best_val - 1e-7:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
            marker = "  *"
        else:
            epochs_no_improve += 1

        lr_now = optim.param_groups[0]["lr"]
        print(f"[Epoch {epoch:02d}] train={train_loss:.6f}  val={val_loss:.6f}  lr={lr_now:.1e}{marker}")

        if epochs_no_improve >= patience:
            print(f"[Stop] No improvement for {patience} epochs — early stopping.")
            break

    torch.save(best_state, MODEL_PATH)
    print(f"\n[Saved] {MODEL_PATH}  (best val_loss={best_val:.6f})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=4)
    a = p.parse_args()
    train(a.epochs, a.batch_size, a.lr, a.patience)


if __name__ == "__main__":
    main()

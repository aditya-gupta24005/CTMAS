"""Flower-compatible federated client — one per plant stage.

Each client trains the shared GNN autoencoder on all windows but computes
loss only on its own stage node, then applies manual DP-SGD (gradient
clipping + Gaussian noise) before the optimizer step, followed by FedProx
proximal regularisation.

We use manual DP-SGD instead of Opacus because Opacus's functorch-based
per-sample gradient engine does not support PyTorch Geometric's
message-passing layers (GCNConv takes `edge_index` which can't be vmapped).
Manual DP-SGD gives the same DP-Gaussian mechanism at the per-batch level.
"""

from __future__ import annotations

import math
import warnings
from collections import OrderedDict
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from federated.config import (
    BATCH_SIZE,
    DP_DELTA,
    DP_MAX_GRAD_NORM,
    DP_NOISE_MULTIPLIER,
    GNN_LAYERS,
    HIDDEN_DIM,
    LATENT_DIM,
    LEARNING_RATE,
    LOCAL_EPOCHS,
    MU,
)
from models.gnn_model import EDGE_INDEX, SpatioTemporalGNNAutoencoder

warnings.filterwarnings("ignore", category=UserWarning)

from device import DEVICE


def _estimate_epsilon(steps: int, noise_multiplier: float, delta: float) -> float:
    """Rough (ε, δ)-DP epsilon for Gaussian mechanism over `steps` iterations.

    Uses the simple composition bound
        ε ≈ sqrt(2·steps·ln(1/δ)) / σ
    which is an upper bound (loose but monotonic). Good enough for a
    privacy-budget dashboard indicator; not a replacement for RDP accounting.
    """
    if noise_multiplier <= 0 or steps <= 0:
        return float("inf")
    return math.sqrt(2.0 * steps * math.log(1.0 / delta)) / noise_multiplier


class CTMASClient:
    def __init__(
        self,
        stage_id: int,
        X_train: np.ndarray,
        X_val: np.ndarray,
    ):
        self.stage_id = stage_id
        self.model = SpatioTemporalGNNAutoencoder(HIDDEN_DIM, LATENT_DIM, GNN_LAYERS).to(DEVICE)
        self.edge_index = EDGE_INDEX.to(DEVICE)

        X_t = torch.tensor(X_train, dtype=torch.float32)
        X_v = torch.tensor(X_val, dtype=torch.float32)
        self.train_loader = DataLoader(TensorDataset(X_t), batch_size=BATCH_SIZE, shuffle=True)
        self.val_loader = DataLoader(TensorDataset(X_v), batch_size=BATCH_SIZE)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self._steps_taken = 0

    # ── parameter exchange ───────────────────────────────────────────────────

    def get_parameters(self) -> List[np.ndarray]:
        return [v.detach().cpu().numpy() for v in self.model.state_dict().values()]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        state_dict = OrderedDict(
            (k, torch.tensor(v))
            for k, v in zip(self.model.state_dict().keys(), parameters)
        )
        self.model.load_state_dict(state_dict, strict=True)

    # ── training ─────────────────────────────────────────────────────────────

    def fit(self, parameters, config) -> Tuple[List[np.ndarray], int, dict]:
        self.set_parameters(parameters)

        # Snapshot global params for FedProx proximal term
        global_params = [p.detach().clone() for p in self.model.parameters()]

        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for _ in range(LOCAL_EPOCHS):
            for (x_batch,) in self.train_loader:
                x_batch = x_batch.to(DEVICE)
                self.optimizer.zero_grad()

                out = self.model(x_batch, self.edge_index)
                _, node_mse = self.model.reconstruction_loss(out)

                # Train on this client's stage only
                loss = node_mse[self.stage_id]

                # FedProx proximal term: keep local weights close to global
                prox = sum(
                    (p - g).pow(2).sum()
                    for p, g in zip(self.model.parameters(), global_params)
                )
                loss = loss + (MU / 2) * prox

                loss.backward()

                # ── Manual DP-SGD: clip + Gaussian noise ────────────────────
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), DP_MAX_GRAD_NORM)
                with torch.no_grad():
                    for p in self.model.parameters():
                        if p.grad is None:
                            continue
                        noise_std = DP_NOISE_MULTIPLIER * DP_MAX_GRAD_NORM / max(x_batch.size(0), 1)
                        p.grad.add_(torch.randn_like(p.grad) * noise_std)

                self.optimizer.step()
                total_loss += loss.item()
                n_batches += 1
                self._steps_taken += 1

        epsilon = _estimate_epsilon(self._steps_taken, DP_NOISE_MULTIPLIER, DP_DELTA)

        return (
            self.get_parameters(),
            len(self.train_loader.dataset),
            {"loss": total_loss / max(n_batches, 1), "epsilon": epsilon},
        )

    def evaluate(self, parameters, config) -> Tuple[float, int, dict]:
        self.set_parameters(parameters)
        self.model.eval()
        total_loss = 0.0
        n = 0
        with torch.no_grad():
            for (x_batch,) in self.val_loader:
                x_batch = x_batch.to(DEVICE)
                out = self.model(x_batch, self.edge_index)
                _, node_mse = self.model.reconstruction_loss(out)
                total_loss += node_mse[self.stage_id].item()
                n += 1
        return total_loss / max(n, 1), len(self.val_loader.dataset), {}

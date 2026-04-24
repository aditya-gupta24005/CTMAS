"""Two-layer anomaly detector.

Layer 1 — Node-level reconstruction error:
  Per-stage MSE threshold = mean + 3*std of validation errors.

Layer 2 — Graph-level EWMA early warning:
  Smoothed global reconstruction error triggers early warning before
  any per-node threshold fires. Catches low-and-slow attacks.

Cross-node correlation:
  If node X is anomalous, check physically downstream nodes for elevated
  (but sub-threshold) error. Classifies as "Propagating Attack" vs "Isolated".
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from models.gnn_model import (
    EDGE_INDEX,
    N_STAGES,
    SpatioTemporalGNNAutoencoder,
    _mask_padded_node_mse,
    _split_to_stages,
)

from device import DEVICE

# Physical downstream adjacency (which stages are downstream of each stage)
# P1→P2, P2→P3, P3→P4, P4→P5, P5→P6, P3→P1, P5→P1
DOWNSTREAM: Dict[int, List[int]] = {
    0: [1],          # P1 → P2
    1: [2],          # P2 → P3
    2: [3, 0],       # P3 → P4, P1 (backwash)
    3: [4],          # P4 → P5
    4: [5, 0],       # P5 → P6, P1 (RO reject)
    5: [],           # P6 → (terminal)
}

STAGE_NAMES = ["P1", "P2", "P3", "P4", "P5", "P6"]


@dataclass
class AnomalyReport:
    window_idx: int
    timestamp_s: int                         # seconds into replay
    node_errors: np.ndarray                  # shape (6,) per-stage MSE
    node_anomaly: np.ndarray                 # bool (6,) per-stage threshold breach
    ewma_score: float
    early_warning: bool
    anomaly_type: str                        # "None", "Isolated", "Propagating"
    anomaly_nodes: List[str]
    severity: int                            # 0-3


class AnomalyDetector:
    def __init__(
        self,
        model: SpatioTemporalGNNAutoencoder,
        ewma_alpha: float = 0.3,
        threshold_std: float = 3.0,
        early_warning_percentile: float = 85.0,
    ):
        self.model = model.to(DEVICE).eval()
        self.edge_index = EDGE_INDEX.to(DEVICE)
        self.ewma_alpha = ewma_alpha
        self.threshold_std = threshold_std
        self.early_warning_percentile = early_warning_percentile

        # Fitted thresholds (set by calibrate())
        self.node_mean: Optional[np.ndarray] = None   # (6,)
        self.node_std: Optional[np.ndarray] = None    # (6,)
        self.node_threshold: Optional[np.ndarray] = None  # (6,)
        self.ewma_threshold: Optional[float] = None

        self._ewma: float = 0.0
        self._first_step = True

    # ── calibration ────────────────────────────────────────────────────────

    def calibrate(self, X_val: np.ndarray, batch_size: int = 256) -> None:
        """Compute per-node thresholds from validation (normal) data."""
        loader = DataLoader(
            TensorDataset(torch.tensor(X_val, dtype=torch.float32)),
            batch_size=batch_size,
        )
        all_node_errors = []
        with torch.no_grad():
            for (x,) in loader:
                x = x.to(DEVICE)
                out = self.model(x, self.edge_index)
                staged = out["input_staged"]
                recon = out["recon"]
                node_mse = _mask_padded_node_mse(staged, recon).cpu().numpy()
                all_node_errors.append(node_mse)

        errors = np.stack(all_node_errors)    # (n_batches, 6)
        self.node_mean = errors.mean(axis=0)
        self.node_std = errors.std(axis=0).clip(min=1e-8)
        self.node_threshold = self.node_mean + self.threshold_std * self.node_std

        global_errors = errors.mean(axis=1)   # (n_batches,)
        self.ewma_threshold = float(np.percentile(global_errors, self.early_warning_percentile))

        print(f"[Detector] Calibrated — node thresholds: {np.round(self.node_threshold, 5)}")
        print(f"[Detector] EWMA early-warning threshold: {self.ewma_threshold:.5f}")

    # ── inference ──────────────────────────────────────────────────────────

    def step(self, x: np.ndarray, window_idx: int) -> AnomalyReport:
        """Process one window (60, 51) and return anomaly report."""
        x_t = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            out = self.model(x_t, self.edge_index)
            node_mse = _mask_padded_node_mse(out["input_staged"], out["recon"]).cpu().numpy()

        global_err = float(node_mse.mean())

        # EWMA update
        if self._first_step:
            self._ewma = global_err
            self._first_step = False
        else:
            self._ewma = self.ewma_alpha * global_err + (1 - self.ewma_alpha) * self._ewma

        early_warning = self._ewma > self.ewma_threshold

        # Per-node threshold breach
        node_anomaly = node_mse > self.node_threshold   # bool (6,)
        anomaly_nodes = [STAGE_NAMES[i] for i in range(N_STAGES) if node_anomaly[i]]

        # Cross-node correlation: propagating vs isolated
        anomaly_type = "None"
        severity = 0

        if early_warning and not node_anomaly.any():
            anomaly_type = "EarlyWarning"
            severity = 1

        if node_anomaly.any():
            anomaly_type = "Isolated"
            severity = 2

            # Check downstream nodes for elevated (sub-threshold) error
            elevated_threshold = self.node_mean + 1.5 * self.node_std
            for i in range(N_STAGES):
                if not node_anomaly[i]:
                    continue
                for ds in DOWNSTREAM.get(i, []):
                    if not node_anomaly[ds] and node_mse[ds] > elevated_threshold[ds]:
                        anomaly_type = "Propagating"
                        severity = 3
                        if STAGE_NAMES[ds] not in anomaly_nodes:
                            anomaly_nodes.append(STAGE_NAMES[ds] + "*")
                        break

        return AnomalyReport(
            window_idx=window_idx,
            timestamp_s=window_idx * 10,   # stride=10s
            node_errors=node_mse,
            node_anomaly=node_anomaly,
            ewma_score=self._ewma,
            early_warning=early_warning,
            anomaly_type=anomaly_type,
            anomaly_nodes=anomaly_nodes,
            severity=severity,
        )

    def batch_evaluate(
        self, X_test: np.ndarray, batch_size: int = 256
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (all_node_errors (N,6), all_global_errors (N,)) for metric computation."""
        return self._batch_per_sample(X_test, batch_size)

    def _batch_per_sample(
        self, X: np.ndarray, batch_size: int
    ) -> tuple[np.ndarray, np.ndarray]:
        loader = DataLoader(
            TensorDataset(torch.tensor(X, dtype=torch.float32)),
            batch_size=batch_size,
        )
        node_err_list = []
        with torch.no_grad():
            for (x,) in loader:
                x = x.to(DEVICE)
                out = self.model(x, self.edge_index)
                staged = out["input_staged"]   # (B, 6, 60, 13)
                recon = out["recon"]
                # Per-sample, per-node MSE
                diff = (staged - recon) ** 2   # (B, 6, 60, 13)
                node_mse_batch = diff.mean(dim=(2, 3))  # (B, 6)
                node_err_list.append(node_mse_batch.cpu().numpy())

        node_errors = np.concatenate(node_err_list, axis=0)    # (N, 6)
        global_errors = node_errors.mean(axis=1)               # (N,)
        return node_errors, global_errors

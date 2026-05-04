"""Per-stage temporal CNN encoder — produces fixed-size embedding per plant stage."""

import torch
import torch.nn as nn


class StageEncoder(nn.Module):
    """1D-CNN encoder for a single plant stage's time-series sensor readings.

    Input:  (batch, seq_len=60, n_sensors_padded=13)
    Output: (batch, hidden_dim)
    No LSTM/GRU — Opacus requires only Conv/Linear layers.
    """

    def __init__(self, n_sensors: int = 13, seq_len: int = 60, hidden_dim: int = 64, dropout: float = 0.0):
        super().__init__()
        self.conv = nn.Sequential(
            # (batch, n_sensors, seq_len)
            nn.Conv1d(n_sensors, 32, kernel_size=5, padding=2),
            nn.GroupNorm(8, 32),            # GroupNorm is DP-safe; BatchNorm is not
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.GroupNorm(8, 64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(10),       # → (batch, 64, 10)  — 60/10=6, MPS-safe
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(64 * 10, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, n_sensors)
        x = x.permute(0, 2, 1)             # → (batch, n_sensors, seq_len)
        x = self.conv(x)                   # → (batch, 64, 10)
        x = x.flatten(1)                   # → (batch, 640)
        x = self.dropout(x)
        return self.fc(x)                  # → (batch, hidden_dim)


class StageDecoder(nn.Module):
    """Reverse: hidden_dim → (batch, seq_len, n_sensors_padded)."""

    def __init__(self, n_sensors: int = 13, seq_len: int = 60, hidden_dim: int = 64):
        super().__init__()
        self.seq_len = seq_len
        self.n_sensors = n_sensors
        self.fc = nn.Linear(hidden_dim, 64 * 10)
        self.deconv = nn.Sequential(
            nn.ConvTranspose1d(64, 32, kernel_size=5, padding=2),
            nn.GroupNorm(8, 32),
            nn.ReLU(),
            nn.ConvTranspose1d(32, n_sensors, kernel_size=5, padding=2),
            nn.Upsample(size=seq_len),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, hidden_dim)
        x = self.fc(x)                     # → (batch, 640)
        x = x.view(x.size(0), 64, 10)     # → (batch, 64, 10)
        x = self.deconv(x)                 # → (batch, n_sensors, seq_len)
        return x.permute(0, 2, 1)          # → (batch, seq_len, n_sensors)

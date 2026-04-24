"""Spatio-Temporal Graph Autoencoder for SWaT plant topology.

Each of the 6 plant stages is a graph node. Node features are the stage's
sensor readings (padded to 13, the max across all stages). The encoder
applies a per-stage 1D-CNN to get temporal embeddings, then two GCN layers
to propagate information across the physical water-flow graph. The decoder
reconstructs each node's original sensor readings.
"""

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

from models.stage_encoder import StageDecoder, StageEncoder

# Physical plant topology (static, defined once)
# P1→P2, P2→P3, P3→P4, P4→P5, P5→P6, P3→P1 (backwash), P5→P1 (RO reject)
EDGE_INDEX = torch.tensor(
    [[0, 1, 2, 3, 4, 2, 4],
     [1, 2, 3, 4, 5, 0, 0]],
    dtype=torch.long,
)

# Sensor count per stage (determines padding mask)
STAGE_SENSOR_COUNTS = [5, 11, 9, 9, 13, 4]
N_STAGES = 6
MAX_SENSORS = 13    # P5
SEQ_LEN = 60


class SpatioTemporalGNNAutoencoder(nn.Module):
    """
    Forward pass (encoder):
      1. Per-stage 1D-CNN encodes (batch, 60, 13) → (batch, hidden_dim)
      2. Stack all 6 stage embeddings → (batch*6, hidden_dim) as PyG batch
      3. Two GCN layers propagate across plant topology → (batch*6, hidden_dim)
      4. Linear bottleneck → latent (batch*6, latent_dim)

    Forward pass (decoder):
      5. GCN layers in reverse topology → (batch*6, hidden_dim)
      6. Per-stage 1D-CNN decoder → (batch*6, 60, 13)

    Returns per-node reconstruction and latent vectors.
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        latent_dim: int = 32,
        gnn_layers: int = 2,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # Shared encoder/decoder CNN for all stages (padded to 13 sensors)
        self.stage_encoder = StageEncoder(MAX_SENSORS, SEQ_LEN, hidden_dim)
        self.stage_decoder = StageDecoder(MAX_SENSORS, SEQ_LEN, hidden_dim)

        # GNN encoder layers
        self.gnn_enc = nn.ModuleList([
            GCNConv(hidden_dim if i == 0 else hidden_dim, hidden_dim)
            for i in range(gnn_layers)
        ])
        self.enc_act = nn.ReLU()

        # Bottleneck
        self.to_latent = nn.Linear(hidden_dim, latent_dim)
        self.from_latent = nn.Linear(latent_dim, hidden_dim)

        # GNN decoder layers
        self.gnn_dec = nn.ModuleList([
            GCNConv(hidden_dim, hidden_dim)
            for _ in range(gnn_layers)
        ])
        self.dec_act = nn.ReLU()

        # Node-degree weights for weighted MSE loss (downstream stages matter more)
        # Degree = in-degree of each node in the directed graph
        in_degrees = torch.zeros(N_STAGES)
        for tgt in EDGE_INDEX[1]:
            in_degrees[tgt] += 1
        in_degrees = in_degrees.clamp(min=1)
        self.register_buffer("node_weights", in_degrees / in_degrees.sum())

    def encode(
        self,
        x: torch.Tensor,           # (batch, n_stages, seq_len, max_sensors)
        edge_index: torch.Tensor,  # (2, E)
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (latent, node_embeddings_pre_bottleneck)."""
        B = batch_size

        # Per-stage temporal encoding
        x_flat = x.view(B * N_STAGES, SEQ_LEN, MAX_SENSORS)   # (B*6, 60, 13)
        h = self.stage_encoder(x_flat)                         # (B*6, hidden)

        # Build PyG-style batch graph for B graphs with 6 nodes each
        edge_index_batched = _batch_edge_index(edge_index, B, N_STAGES)

        # GNN encoding
        for gcn in self.gnn_enc:
            h = self.enc_act(gcn(h, edge_index_batched))       # (B*6, hidden)

        latent = self.to_latent(h)                             # (B*6, latent)
        return latent, h

    def decode(
        self,
        latent: torch.Tensor,      # (B*6, latent)
        edge_index: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        """Returns reconstructed (batch, n_stages, seq_len, max_sensors)."""
        B = batch_size
        h = self.from_latent(latent)                           # (B*6, hidden)

        edge_index_batched = _batch_edge_index(edge_index, B, N_STAGES)

        for gcn in self.gnn_dec:
            h = self.dec_act(gcn(h, edge_index_batched))

        x_recon_flat = self.stage_decoder(h)                   # (B*6, 60, 13)
        return x_recon_flat.view(B, N_STAGES, SEQ_LEN, MAX_SENSORS)

    def forward(
        self,
        x: torch.Tensor,           # (batch, 60, 51) raw window
        edge_index: torch.Tensor | None = None,
    ) -> dict:
        if edge_index is None:
            edge_index = EDGE_INDEX.to(x.device)

        B = x.size(0)
        x_staged = _split_to_stages(x)        # (B, 6, 60, 13) padded

        latent, _ = self.encode(x_staged, edge_index, B)
        x_recon = self.decode(latent, edge_index, B)  # (B, 6, 60, 13)

        return {
            "recon": x_recon,
            "latent": latent.view(B, N_STAGES, self.latent_dim),
            "input_staged": x_staged,
        }

    def reconstruction_loss(self, out: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """Weighted MSE per node, returns (total_loss, per_node_losses (6,))."""
        x = out["input_staged"]      # (B, 6, 60, 13)
        r = out["recon"]             # (B, 6, 60, 13)

        # Per-node MSE: mean over batch, seq_len, sensors
        node_mse = ((x - r) ** 2).mean(dim=(0, 2, 3))   # (6,)

        # Mask padded sensors out of per-node error
        node_mse = _mask_padded_node_mse(x, r)

        weights = self.node_weights.to(x.device)         # (6,)
        total = (weights * node_mse).sum()
        return total, node_mse


# ── helpers ──────────────────────────────────────────────────────────────────

def _split_to_stages(x: torch.Tensor) -> torch.Tensor:
    """Split (batch, 60, 51) into (batch, 6, 60, 13) with zero-padding."""
    stage_slices = [
        list(range(0, 5)),                   # P1 → idx 0-4
        list(range(5, 16)),                  # P2 → idx 5-15
        list(range(16, 25)),                 # P3 → idx 16-24
        list(range(25, 34)),                 # P4 → idx 25-33
        list(range(34, 47)),                 # P5 → idx 34-46
        list(range(47, 51)),                 # P6 → idx 47-50
    ]
    B, T, _ = x.shape
    out = torch.zeros(B, N_STAGES, T, MAX_SENSORS, device=x.device, dtype=x.dtype)
    for i, idxs in enumerate(stage_slices):
        out[:, i, :, :len(idxs)] = x[:, :, idxs]
    return out


def _mask_padded_node_mse(x_in: torch.Tensor, x_out: torch.Tensor) -> torch.Tensor:
    """MSE only on real (non-padded) sensors per stage."""
    counts = STAGE_SENSOR_COUNTS
    node_mse = torch.zeros(N_STAGES, device=x_in.device)
    for i, c in enumerate(counts):
        diff = x_in[:, i, :, :c] - x_out[:, i, :, :c]
        node_mse[i] = (diff ** 2).mean()
    return node_mse


def _batch_edge_index(edge_index: torch.Tensor, batch_size: int, n_nodes: int) -> torch.Tensor:
    """Tile edge_index for B independent graphs in one PyG batch."""
    offsets = torch.arange(batch_size, device=edge_index.device) * n_nodes
    offsets = offsets.repeat_interleave(edge_index.size(1))  # (B * E,)
    src = edge_index[0].repeat(batch_size) + offsets
    dst = edge_index[1].repeat(batch_size) + offsets
    return torch.stack([src, dst])

"""Federated learning hyperparameters."""

FL_ROUNDS = 10
LOCAL_EPOCHS = 3
BATCH_SIZE = 64
LEARNING_RATE = 1e-3

# FedProx proximal penalty weight
MU = 0.01

# Byzantine-robust aggregation: cosine similarity threshold
# Clients below this similarity to the median are down-weighted
BYZANTINE_THRESHOLD = 0.5

# Opacus differential privacy
DP_MAX_GRAD_NORM = 1.0
DP_NOISE_MULTIPLIER = 0.3
DP_DELTA = 1e-5   # target δ for (ε, δ)-DP

# Model architecture (must match gnn_model.py defaults)
HIDDEN_DIM = 64
LATENT_DIM = 32
GNN_LAYERS = 2

# Stage indices for federated data slicing
# Each client owns one stage's sensor columns in the 51-sensor space
STAGE_COL_RANGES = {
    0: list(range(0, 5)),    # P1
    1: list(range(5, 16)),   # P2
    2: list(range(16, 25)),  # P3
    3: list(range(25, 34)),  # P4
    4: list(range(34, 47)),  # P5
    5: list(range(47, 51)),  # P6
}

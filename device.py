"""Centralised device selection — prefers CUDA → MPS (Apple Silicon) → CPU.

Sets PYTORCH_ENABLE_MPS_FALLBACK=1 so any op without an MPS kernel (some
PyG scatter variants, rarely) silently falls back to CPU instead of raising.
"""

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = get_device()

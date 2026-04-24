"""Flower federated server with FedProx + Byzantine-robust aggregation.

Byzantine defense: after each round, compute cosine similarity between each
client's parameter update and the median update. Down-weight outliers (likely
poisoned clients) using a soft weight proportional to their similarity score.
"""

from __future__ import annotations

import pickle
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union

import flwr as fl
import numpy as np
import torch
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import Strategy

from federated.config import BYZANTINE_THRESHOLD, FL_ROUNDS, GNN_LAYERS, HIDDEN_DIM, LATENT_DIM
from models.gnn_model import EDGE_INDEX, SpatioTemporalGNNAutoencoder


class FedProxByzantineStrategy(Strategy):
    """FedProx aggregation with Byzantine-robust cosine-similarity weighting."""

    def __init__(self, initial_parameters: Parameters, fraction_fit: float = 1.0):
        self.fraction_fit = fraction_fit
        self.initial_parameters = initial_parameters
        self.round_metrics: List[dict] = []

    def initialize_parameters(self, client_manager: ClientManager) -> Optional[Parameters]:
        return self.initial_parameters

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        clients = client_manager.sample(
            num_clients=max(1, int(client_manager.num_available() * self.fraction_fit)),
            min_num_clients=1,
        )
        fit_ins = FitIns(parameters, {"round": server_round})
        return [(c, fit_ins) for c in clients]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results:
            return None, {}

        param_arrays = [parameters_to_ndarrays(r.parameters) for _, r in results]
        sample_counts = [r.num_examples for _, r in results]
        metrics_list = [r.metrics for _, r in results]

        # Flatten each client's parameters into a 1D vector for similarity computation
        flat_updates = [
            np.concatenate([p.ravel() for p in arrays])
            for arrays in param_arrays
        ]

        # Median update as reference
        median_update = np.median(flat_updates, axis=0)

        # Cosine similarity of each client vs. median
        def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
            denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
            return float(np.dot(a, b) / denom)

        sims = [cosine_sim(u, median_update) for u in flat_updates]

        # Byzantine weights: zero-out clients below threshold, scale by similarity
        byz_weights = np.array([
            max(0.0, sim - BYZANTINE_THRESHOLD) if sim < BYZANTINE_THRESHOLD else sim
            for sim in sims
        ])
        if byz_weights.sum() < 1e-8:
            byz_weights = np.ones(len(results))

        # Weighted aggregation (FedAvg-style with Byzantine weights * sample count)
        total_weight = sum(byz_weights[i] * sample_counts[i] for i in range(len(results)))
        aggregated = []
        for layer_idx in range(len(param_arrays[0])):
            layer_agg = sum(
                byz_weights[i] * sample_counts[i] * param_arrays[i][layer_idx]
                for i in range(len(results))
            ) / total_weight
            aggregated.append(layer_agg)

        # Log per-round metrics
        avg_epsilon = np.mean([m.get("epsilon", 0.0) for m in metrics_list])
        avg_loss = np.mean([m.get("loss", 0.0) for m in metrics_list])
        self.round_metrics.append({
            "round": server_round,
            "loss": avg_loss,
            "epsilon": avg_epsilon,
            "byz_sims": sims,
        })
        print(
            f"[Round {server_round:02d}] loss={avg_loss:.4f} "
            f"ε={avg_epsilon:.3f} "
            f"byz_sims={[f'{s:.2f}' for s in sims]}"
        )

        return ndarrays_to_parameters(aggregated), {"loss": avg_loss, "epsilon": avg_epsilon}

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        clients = client_manager.sample(
            num_clients=max(1, int(client_manager.num_available() * self.fraction_fit)),
            min_num_clients=1,
        )
        return [(c, EvaluateIns(parameters, {})) for c in clients]

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        if not results:
            return None, {}
        losses = [r.loss * r.num_examples for _, r in results]
        total = sum(r.num_examples for _, r in results)
        return sum(losses) / total, {}

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        return None


def build_initial_parameters() -> Parameters:
    model = SpatioTemporalGNNAutoencoder(HIDDEN_DIM, LATENT_DIM, GNN_LAYERS)
    ndarrays = [v.cpu().numpy() for v in model.state_dict().values()]
    return ndarrays_to_parameters(ndarrays)


def load_model_from_parameters(parameters: Parameters) -> SpatioTemporalGNNAutoencoder:
    model = SpatioTemporalGNNAutoencoder(HIDDEN_DIM, LATENT_DIM, GNN_LAYERS)
    ndarrays = parameters_to_ndarrays(parameters)
    state_dict = OrderedDict(
        {k: torch.tensor(v) for k, v in zip(model.state_dict().keys(), ndarrays)}
    )
    model.load_state_dict(state_dict)
    return model

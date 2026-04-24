"""CTMAS entry point — federated training + evaluation.

Usage:
  python main.py                         # 10 FL rounds on full dataset
  python main.py --subsample 0.1         # 10% of training windows (much faster)
  python main.py --rounds 5              # override round count
  python main.py --eval-only             # skip training, evaluate saved model

After training:
  uvicorn api.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import argparse
import pickle
import warnings
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from sklearn.metrics import auc, f1_score, precision_score, recall_score, roc_curve

from device import DEVICE

warnings.filterwarnings("ignore")
print(f"[Device] Using {DEVICE}")

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "Data" / "processed"
MODEL_PATH = BASE_DIR / "ctmas_model.pt"


def load_data():
    X_train = np.load(DATA_DIR / "X_train.npy")
    X_val = np.load(DATA_DIR / "X_val.npy")
    X_test = np.load(DATA_DIR / "X_test.npy")
    y_test = np.load(DATA_DIR / "y_test.npy")
    with open(DATA_DIR / "metadata.pkl", "rb") as f:
        meta = pickle.load(f)

    # Constant-in-normal sensors yielded NaN under MinMaxScaler; zero is the
    # correct scaled value for a constant signal.
    for arr in (X_train, X_val, X_test):
        np.nan_to_num(arr, copy=False, nan=0.0)

    print(
        f"[Data] train={X_train.shape}, val={X_val.shape}, "
        f"test={X_test.shape} ({int(y_test.sum())} attack / {int((1 - y_test).sum())} normal)"
    )
    return X_train, X_val, X_test, y_test, meta


# ── manual federated loop (no Ray / no Flower simulation) ────────────────────

def _byzantine_weights(client_params: List[List[np.ndarray]], threshold: float) -> tuple[np.ndarray, List[float]]:
    flat = [np.concatenate([p.ravel() for p in arrs]) for arrs in client_params]
    median = np.median(flat, axis=0)
    sims = []
    for v in flat:
        denom = (np.linalg.norm(v) * np.linalg.norm(median)) + 1e-8
        sims.append(float(np.dot(v, median) / denom))
    weights = np.array([
        max(0.0, s - threshold) if s < threshold else s
        for s in sims
    ])
    if weights.sum() < 1e-8:
        weights = np.ones(len(client_params))
    return weights, sims


def _aggregate(
    client_params: List[List[np.ndarray]],
    sample_counts: List[int],
    byz_weights: np.ndarray,
) -> List[np.ndarray]:
    total = sum(byz_weights[i] * sample_counts[i] for i in range(len(client_params)))
    n_layers = len(client_params[0])
    out = []
    for layer_idx in range(n_layers):
        layer = sum(
            byz_weights[i] * sample_counts[i] * client_params[i][layer_idx]
            for i in range(len(client_params))
        ) / total
        out.append(layer)
    return out


def run_federated_training(
    X_train: np.ndarray,
    X_val: np.ndarray,
    fl_rounds: int,
) -> dict:
    from federated.client import CTMASClient
    from federated.config import BYZANTINE_THRESHOLD
    from federated.server import build_initial_parameters
    from flwr.common import parameters_to_ndarrays

    print(f"\n[FL] Starting {fl_rounds} federated rounds with 6 clients (one per stage)...\n")

    clients = [CTMASClient(stage_id=i, X_train=X_train, X_val=X_val) for i in range(6)]
    current_params = parameters_to_ndarrays(build_initial_parameters())

    for rnd in range(1, fl_rounds + 1):
        print(f"\n[Round {rnd:02d}]")

        client_params: List[List[np.ndarray]] = []
        sample_counts: List[int] = []
        losses: List[float] = []
        epsilons: List[float] = []

        for i, client in enumerate(clients):
            new_params, n, metrics = client.fit(current_params, {"round": rnd})
            client_params.append(new_params)
            sample_counts.append(n)
            losses.append(metrics["loss"])
            epsilons.append(metrics["epsilon"])
            print(f"  P{i+1}: loss={metrics['loss']:.4f} ε={metrics['epsilon']:.3f}")

        byz_weights, sims = _byzantine_weights(client_params, BYZANTINE_THRESHOLD)
        current_params = _aggregate(client_params, sample_counts, byz_weights)

        print(
            f"  [Agg] mean_loss={np.mean(losses):.4f} "
            f"ε={np.mean(epsilons):.3f} "
            f"byz_sims={[f'{s:.2f}' for s in sims]}"
        )

    # Push aggregated params into one client's model, save its state_dict
    clients[0].set_parameters(current_params)
    state = clients[0].model.state_dict()
    torch.save(state, MODEL_PATH)
    print(f"\n[FL] Training complete. Model saved to {MODEL_PATH}")
    return state


def evaluate_model(
    X_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    meta: dict,
    model_state: Dict | None = None,
) -> dict:
    from detection.detector import AnomalyDetector
    from intelligence.threat_mapper import ThreatFSM
    from models.gnn_model import SpatioTemporalGNNAutoencoder

    model = SpatioTemporalGNNAutoencoder()
    if model_state:
        model.load_state_dict(model_state)
    elif MODEL_PATH.exists():
        model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    else:
        print("[Eval] No model found — using random weights for demo metrics.")

    detector = AnomalyDetector(model)
    detector.calibrate(X_val)

    print(
        f"\n[Eval] Running on test set: {len(X_test)} windows "
        f"({int(y_test.sum())} attack / {int((1 - y_test).sum())} normal)..."
    )
    node_errors, global_errors = detector._batch_per_sample(X_test, batch_size=256)

    y_true = y_test.astype(int)
    thresh = float(detector.node_threshold.mean())
    y_pred = (global_errors > thresh).astype(int)

    f1 = f1_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    fpr_curve, tpr_curve, _ = roc_curve(y_true, global_errors)
    roc_auc = auc(fpr_curve, tpr_curve)

    # Real FPR at the chosen operating threshold
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fpr_at_thresh = fp / max(1, tn + fp)

    # Real MTTD — per attack event, measure delay to first detection
    test_stride = int(meta.get("test_stride", 10))
    events = meta.get("attack_events_window", [])
    detection_delays_s: List[float] = []
    events_caught = 0
    for (w_start, w_end) in events:
        window_range = y_pred[w_start : w_end + 1]
        hits = np.where(window_range == 1)[0]
        if len(hits) > 0:
            events_caught += 1
            detection_delays_s.append(int(hits[0]) * test_stride)
    mttd_s = float(np.mean(detection_delays_s)) if detection_delays_s else -1.0
    event_recall = events_caught / max(1, len(events))

    print(f"\n{'='*50}")
    print("CTMAS Evaluation Results")
    print(f"{'='*50}")
    print(f"  F1 Score:         {f1:.4f}")
    print(f"  Precision:        {prec:.4f}")
    print(f"  Recall:           {rec:.4f}")
    print(f"  AUC-ROC:          {roc_auc:.4f}")
    print(f"  FPR @ threshold:  {fpr_at_thresh:.4f}")
    print(f"  MTTD:             {mttd_s:.1f}s  (over {events_caught}/{len(events)} events caught)")
    print(f"  Event Recall:     {event_recall:.2%}")
    print(f"  Detection Rate:   {y_pred.mean():.2%}")
    print(f"{'='*50}")

    print("\n[Eval] FSM threat assessment on first 500 test windows:")
    fsm = ThreatFSM()
    state_counts = {"NORMAL": 0, "RECON": 0, "INTRUSION": 0, "IMPACT": 0}
    for i in range(min(500, len(X_test))):
        report = detector.step(X_test[i], window_idx=i)
        assessment = fsm.step(report)
        state_counts[assessment.state] = state_counts.get(assessment.state, 0) + 1

    print("  FSM state distribution over first 500 windows:")
    for state, count in state_counts.items():
        print(f"    {state:12s}: {count} windows ({count/500:.1%})")

    return {"f1": f1, "auc_roc": roc_auc, "mttd_s": mttd_s}


def main():
    parser = argparse.ArgumentParser(description="CTMAS Federated Training & Evaluation")
    parser.add_argument("--eval-only", action="store_true", help="Skip training, evaluate saved model")
    parser.add_argument("--rounds", type=int, default=10, help="FL rounds (default: 10)")
    parser.add_argument("--subsample", type=float, default=1.0,
                        help="Fraction of training windows to use (default: 1.0). "
                             "Use e.g. 0.1 for a fast demo run.")
    args = parser.parse_args()

    X_train, X_val, X_test, y_test, meta = load_data()

    if args.subsample < 1.0:
        rng = np.random.default_rng(42)
        n_keep = int(len(X_train) * args.subsample)
        idx = rng.choice(len(X_train), n_keep, replace=False)
        X_train = X_train[idx]
        print(f"[Data] subsampled train to {X_train.shape}")

    model_state = None
    if not args.eval_only:
        model_state = run_federated_training(X_train, X_val, fl_rounds=args.rounds)

    evaluate_model(X_val, X_test, y_test, meta, model_state)

    print(f"\n[Done] To start the API dashboard:")
    print(f"  uvicorn api.main:app --host 0.0.0.0 --port 8000")
    print(f"  Then open frontend/index.html in your browser.")


if __name__ == "__main__":
    main()

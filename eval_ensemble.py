"""Evaluate the denoising-AE ensemble on the SWaT test set.

Loads all ctmas_ensemble_*.pt members, averages their per-sample
node-MSE, then z-scores against val statistics. Reports the full
metric panel (confusion matrix, F1, AUC-ROC, PR-AUC, MTTD, etc.).
"""

from __future__ import annotations

import pickle
import warnings
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")

from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from device import DEVICE
from models.gnn_model import (
    EDGE_INDEX,
    SpatioTemporalGNNAutoencoder,
    _per_sample_masked_node_mse,
)

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "Data" / "processed"


def compute_node_errors(model, X: np.ndarray, batch_size: int = 256) -> np.ndarray:
    """Returns (N, 6) per-sample masked node MSE."""
    edge = EDGE_INDEX.to(DEVICE)
    loader = DataLoader(
        TensorDataset(torch.tensor(X, dtype=torch.float32)), batch_size=batch_size,
    )
    out_list = []
    model.eval()
    with torch.no_grad():
        for (x,) in loader:
            x = x.to(DEVICE)
            out = model(x, edge)
            out_list.append(
                _per_sample_masked_node_mse(out["input_staged"], out["recon"]).cpu().numpy()
            )
    return np.concatenate(out_list, axis=0)


def main():
    X_val = np.load(DATA_DIR / "X_val.npy")
    X_test = np.load(DATA_DIR / "X_test.npy")
    y_test = np.load(DATA_DIR / "y_test.npy")
    with open(DATA_DIR / "metadata.pkl", "rb") as f:
        meta = pickle.load(f)
    for a in (X_val, X_test):
        np.nan_to_num(a, copy=False, nan=0.0)

    paths = sorted(BASE_DIR.glob("ctmas_ensemble_*.pt"))
    if not paths:
        raise SystemExit("No ctmas_ensemble_*.pt files found. Run train_ensemble.py first.")
    print(f"[Ensemble] loading {len(paths)} members: {[p.name for p in paths]}")

    val_err_stack, test_err_stack = [], []
    for p in paths:
        m = SpatioTemporalGNNAutoencoder().to(DEVICE)
        m.load_state_dict(torch.load(p, map_location=DEVICE))
        val_err_stack.append(compute_node_errors(m, X_val))
        test_err_stack.append(compute_node_errors(m, X_test))

    val_errors = np.mean(val_err_stack, axis=0)
    test_errors = np.mean(test_err_stack, axis=0)

    mean = val_errors.mean(axis=0)
    std = val_errors.std(axis=0).clip(min=1e-8)
    z_val = (val_errors - mean) / std
    z_test = (test_errors - mean) / std
    P2 = 1
    z_val_sum = np.delete(z_val, P2, axis=1).sum(axis=1)
    z_test_sum = np.delete(z_test, P2, axis=1).sum(axis=1)

    threshold = float(np.percentile(z_val_sum, 99.9) * 4.6)
    y_true = y_test.astype(int)
    y_pred = (z_test_sum > threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    print()
    print("=" * 62)
    print(f"  CTMAS Ensemble Eval  (N={len(paths)}, threshold={threshold:.2f})")
    print("=" * 62)
    print(f"  TP={tp:,}    TN={tn:,}    FP={fp:,}    FN={fn:,}")
    print()
    print(classification_report(y_true, y_pred, target_names=["Normal", "Attack"], digits=4))

    auc_roc = roc_auc_score(y_true, z_test_sum)
    ap = average_precision_score(y_true, z_test_sum)
    print("=" * 62)
    print("  THRESHOLD-FREE METRICS")
    print("=" * 62)
    print(f"  AUC-ROC               : {auc_roc:.4f}")
    print(f"  Avg Precision (PR-AUC): {ap:.4f}")

    pa, ra, thr_pr = precision_recall_curve(y_true, z_test_sum)
    f1_arr = 2 * pa * ra / np.clip(pa + ra, 1e-8, None)
    bi = int(np.argmax(f1_arr))
    bt = thr_pr[bi] if bi < len(thr_pr) else thr_pr[-1]
    yb = (z_test_sum > bt).astype(int)
    tn2, fp2, fn2, tp2 = confusion_matrix(y_true, yb).ravel()

    print()
    print(f"  Best F1 (sweep)       : {f1_arr[bi]:.4f}  (thr={bt:.3f})")
    print(f"    Precision           : {pa[bi]:.4f}")
    print(f"    Recall              : {ra[bi]:.4f}")
    print(f"    TP={tp2}  FP={fp2}  FN={fn2}  TN={tn2}")

    fpr_arr, tpr_arr, thr_roc = roc_curve(y_true, z_test_sum)
    print()
    print("  Operating points (recall ⇒ FPR):")
    for tr in [0.50, 0.70, 0.80, 0.90, 0.95]:
        idxs = np.where(tpr_arr >= tr)[0]
        if len(idxs):
            i = int(idxs[0])
            print(f"    Recall>={tr:.0%} -> FPR={fpr_arr[i]:.4f}  (thr={thr_roc[i]:.3f})")

    stride = int(meta.get("test_stride", 10))
    events = meta.get("attack_events_window", [])
    delays, caught = [], 0
    for ws, we in events:
        seg = y_pred[ws : we + 1]
        h = np.where(seg == 1)[0]
        if len(h):
            caught += 1
            delays.append(int(h[0]) * stride)
    mttd = float(np.mean(delays)) if delays else -1.0
    print()
    print(f"  MTTD                  : {mttd:.1f}s  ({caught}/{len(events)} attack events caught)")
    print(f"  Event Recall          : {caught/max(1,len(events)):.1%}")
    print("=" * 62)


if __name__ == "__main__":
    main()

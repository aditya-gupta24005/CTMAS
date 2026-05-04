"""Tree-based anomaly baseline for the current CTMAS data split.

Why this model:
  - The repo's canonical train/val split is normal-only.
  - That makes a supervised binary classifier like RandomForest invalid unless
    we redesign the split to include labeled attack windows in training.
  - IsolationForest is a pragmatic tree-based baseline that fits this regime.

Usage:
    python baseline_isolation_forest.py
    python baseline_isolation_forest.py --threshold-percentile 99.0
    python baseline_isolation_forest.py --n-estimators 500
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "Data" / "processed"


def extract_window_features(X: np.ndarray) -> np.ndarray:
    """Collapse each (60, 51) window into tabular summary features."""
    first = X[:, 0, :]
    last = X[:, -1, :]
    mean = X.mean(axis=1)
    std = X.std(axis=1)
    min_ = X.min(axis=1)
    max_ = X.max(axis=1)
    delta = last - first
    trend = X[:, -5:, :].mean(axis=1) - X[:, :5, :].mean(axis=1)
    change = np.abs(np.diff(X, axis=1)).mean(axis=1)
    return np.concatenate(
        [mean, std, min_, max_, last, delta, trend, change],
        axis=1,
    ).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--max-samples", type=int, default=2048)
    parser.add_argument("--threshold-percentile", type=float, default=99.5)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    X_train = np.load(DATA_DIR / "X_train.npy")
    X_val = np.load(DATA_DIR / "X_val.npy")
    X_test = np.load(DATA_DIR / "X_test.npy")
    y_test = np.load(DATA_DIR / "y_test.npy").astype(int)
    with open(DATA_DIR / "metadata.pkl", "rb") as f:
        meta = pickle.load(f)

    for arr in (X_train, X_val, X_test):
        np.nan_to_num(arr, copy=False, nan=0.0)

    print(
        f"[Data] train={X_train.shape}, val={X_val.shape}, test={X_test.shape} "
        f"({int(y_test.sum())} attack / {int((1 - y_test).sum())} normal)"
    )
    print("[Feat] extracting summary features...")
    train_feat = extract_window_features(X_train)
    val_feat = extract_window_features(X_val)
    test_feat = extract_window_features(X_test)
    print(f"[Feat] train={train_feat.shape}, val={val_feat.shape}, test={test_feat.shape}")

    model = IsolationForest(
        n_estimators=args.n_estimators,
        max_samples=min(args.max_samples, len(train_feat)),
        contamination="auto",
        random_state=args.random_state,
        n_jobs=-1,
    )
    print(
        f"[Model] IsolationForest(n_estimators={args.n_estimators}, "
        f"max_samples={min(args.max_samples, len(train_feat))})"
    )
    model.fit(train_feat)

    val_score = -model.score_samples(val_feat)
    test_score = -model.score_samples(test_feat)
    threshold = float(np.percentile(val_score, args.threshold_percentile))
    y_pred = (test_score > threshold).astype(int)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    auc_roc = roc_auc_score(y_test, test_score)
    ap = average_precision_score(y_test, test_score)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    fpr_at_thresh = fp / max(1, tn + fp)

    pr_prec, pr_rec, pr_thr = precision_recall_curve(y_test, test_score)
    f1_sweep = 2 * pr_prec * pr_rec / np.clip(pr_prec + pr_rec, 1e-8, None)
    best_idx = int(np.argmax(f1_sweep))
    best_f1 = float(f1_sweep[best_idx])
    best_thr = float(pr_thr[min(best_idx, len(pr_thr) - 1)])

    fpr_curve, tpr_curve, _ = roc_curve(y_test, test_score)
    stride = int(meta.get("test_stride", 10))
    events = meta.get("attack_events_window", [])
    delays_s = []
    events_caught = 0
    for ws, we in events:
        seg = y_pred[ws : we + 1]
        hits = np.where(seg == 1)[0]
        if len(hits):
            events_caught += 1
            delays_s.append(int(hits[0]) * stride)
    mttd_s = float(np.mean(delays_s)) if delays_s else -1.0
    event_recall = events_caught / max(1, len(events))

    print()
    print("=" * 62)
    print(f"  IsolationForest Eval  (threshold={threshold:.5f})")
    print("=" * 62)
    print(f"  TP={tp:,}    TN={tn:,}    FP={fp:,}    FN={fn:,}")
    print(f"  Accuracy              : {acc:.4f}")
    print(f"  Precision             : {prec:.4f}")
    print(f"  Recall                : {rec:.4f}")
    print(f"  F1                    : {f1:.4f}")
    print(f"  AUC-ROC               : {auc_roc:.4f}")
    print(f"  Avg Precision (PR-AUC): {ap:.4f}")
    print(f"  FPR @ threshold       : {fpr_at_thresh:.4f}")
    print(f"  MTTD                  : {mttd_s:.1f}s  ({events_caught}/{len(events)} events caught)")
    print(f"  Event Recall          : {event_recall:.1%}")
    print()
    print(f"  Best F1 (sweep)       : {best_f1:.4f}  (thr={best_thr:.5f})")
    print("  Operating points (recall => FPR):")
    for target_recall in [0.50, 0.70, 0.80, 0.90, 0.95]:
        idxs = np.where(tpr_curve >= target_recall)[0]
        if len(idxs):
            idx = int(idxs[0])
            print(f"    Recall>={target_recall:.0%} -> FPR={fpr_curve[idx]:.4f}")
    print("=" * 62)


if __name__ == "__main__":
    main()

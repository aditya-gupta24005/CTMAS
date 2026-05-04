"""Offline diagnostic: does the model have signal?

If AUC-ROC >> 0.5 → model has signal, it's a threshold / calibration problem.
If AUC-ROC ≈ 0.5  → model cannot separate attack from normal, model is the issue.

Usage:
    python offline_eval.py
"""

from __future__ import annotations
import pickle, warnings
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    auc, f1_score, precision_score, recall_score,
    roc_curve, precision_recall_curve, average_precision_score,
)

warnings.filterwarnings("ignore")

from device import DEVICE
from models.gnn_model import SpatioTemporalGNNAutoencoder, EDGE_INDEX
from detection.detector import AnomalyDetector

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "Data" / "processed"
MODEL_PATH = BASE_DIR / "ctmas_model.pt"


def main():
    print(f"[Device] {DEVICE}")

    # ── 1. Load data ─────────────────────────────────────────────────────
    X_val  = np.load(DATA_DIR / "X_val.npy")
    X_test = np.load(DATA_DIR / "X_test.npy")
    y_test = np.load(DATA_DIR / "y_test.npy")
    with open(DATA_DIR / "metadata.pkl", "rb") as f:
        meta = pickle.load(f)

    for arr in (X_val, X_test):
        np.nan_to_num(arr, copy=False, nan=0.0)

    y_true = y_test.astype(int)
    n_attack = int(y_true.sum())
    n_normal = int((1 - y_true).sum())
    print(f"[Data] test={X_test.shape}  ({n_attack} attack / {n_normal} normal)")
    print(f"[Data] val={X_val.shape}  (all normal, used for calibration)")

    # ── 2. Load model ────────────────────────────────────────────────────
    model = SpatioTemporalGNNAutoencoder()
    if MODEL_PATH.exists():
        model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
        print(f"[Model] Loaded {MODEL_PATH}")
    else:
        print("[Model] WARNING: no checkpoint found, using random weights!")

    detector = AnomalyDetector(model)
    detector.calibrate(X_val)

    # ── 3. Compute test errors ───────────────────────────────────────────
    print("\n[Eval] Computing reconstruction errors on test set...")
    node_errors, global_errors = detector._batch_per_sample(X_test, batch_size=256)

    # Z-score transform (excluding P2, same as main.py evaluate_model)
    P2_NODE = 1
    z_scores = (node_errors - detector.node_mean) / detector.node_std  # (N, 6)
    z_no_p2 = np.delete(z_scores, P2_NODE, axis=1)                     # (N, 5)
    score = z_no_p2.sum(axis=1)                                         # (N,)

    # ── 4. Threshold-free signal check (AUC) ─────────────────────────────
    fpr_curve, tpr_curve, roc_thresholds = roc_curve(y_true, score)
    roc_auc = auc(fpr_curve, tpr_curve)

    prec_curve, rec_curve, pr_thresholds = precision_recall_curve(y_true, score)
    ap = average_precision_score(y_true, score)

    print(f"\n{'='*60}")
    print("  SIGNAL CHECK (threshold-independent)")
    print(f"{'='*60}")
    print(f"  AUC-ROC:              {roc_auc:.4f}")
    print(f"  Average Precision:    {ap:.4f}")
    print(f"  (Random baseline AP = {n_attack / (n_attack + n_normal):.4f})")
    if roc_auc > 0.80:
        print(f"  ✅ Model HAS signal (AUC={roc_auc:.3f} >> 0.5)")
        print(f"     → High FN rate is likely a THRESHOLD problem.")
    elif roc_auc > 0.60:
        print(f"  ⚠️  Model has WEAK signal (AUC={roc_auc:.3f})")
        print(f"     → May need retraining AND threshold tuning.")
    else:
        print(f"  ❌ Model has NO signal (AUC={roc_auc:.3f} ≈ 0.5)")
        print(f"     → Model is the issue. Retraining required.")

    # ── 5. Error distribution comparison ─────────────────────────────────
    normal_scores = score[y_true == 0]
    attack_scores = score[y_true == 1]

    print(f"\n{'='*60}")
    print("  ERROR DISTRIBUTION: Normal vs Attack")
    print(f"{'='*60}")
    print(f"  Normal windows:  mean={normal_scores.mean():.4f}  std={normal_scores.std():.4f}  "
          f"median={np.median(normal_scores):.4f}  p95={np.percentile(normal_scores, 95):.4f}")
    print(f"  Attack windows:  mean={attack_scores.mean():.4f}  std={attack_scores.std():.4f}  "
          f"median={np.median(attack_scores):.4f}  p95={np.percentile(attack_scores, 95):.4f}")
    separation = (attack_scores.mean() - normal_scores.mean()) / (normal_scores.std() + 1e-8)
    print(f"  Mean separation:  {separation:.2f} std-devs")

    # ── 6. Best threshold sweep ──────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  THRESHOLD SWEEP (finding best F1)")
    print(f"{'='*60}")

    # Use F1 on PR curve thresholds
    f1_scores = 2 * prec_curve * rec_curve / (prec_curve + rec_curve + 1e-8)
    best_f1_idx = np.argmax(f1_scores)
    best_f1 = f1_scores[best_f1_idx]
    best_thresh = pr_thresholds[min(best_f1_idx, len(pr_thresholds) - 1)]
    best_prec = prec_curve[best_f1_idx]
    best_rec = rec_curve[best_f1_idx]

    print(f"  Best F1:       {best_f1:.4f}  (at threshold={best_thresh:.4f})")
    print(f"  Precision:     {best_prec:.4f}")
    print(f"  Recall:        {best_rec:.4f}")

    # Current threshold performance
    current_thresh = detector.z_threshold
    y_pred_current = (score > current_thresh).astype(int)
    f1_current = f1_score(y_true, y_pred_current, zero_division=0)
    prec_current = precision_score(y_true, y_pred_current, zero_division=0)
    rec_current = recall_score(y_true, y_pred_current, zero_division=0)
    fn_current = int(((y_true == 1) & (y_pred_current == 0)).sum())
    fp_current = int(((y_true == 0) & (y_pred_current == 1)).sum())

    print(f"\n  Current threshold: {current_thresh:.4f}")
    print(f"  Current F1:        {f1_current:.4f}")
    print(f"  Current Precision: {prec_current:.4f}")
    print(f"  Current Recall:    {rec_current:.4f}")
    print(f"  False Negatives:   {fn_current} / {n_attack} attacks missed")
    print(f"  False Positives:   {fp_current} / {n_normal} normals flagged")

    # What if we use the best threshold?
    y_pred_best = (score > best_thresh).astype(int)
    fn_best = int(((y_true == 1) & (y_pred_best == 0)).sum())
    fp_best = int(((y_true == 0) & (y_pred_best == 1)).sum())
    print(f"\n  With optimal threshold ({best_thresh:.4f}):")
    print(f"  False Negatives:   {fn_best} / {n_attack} attacks missed")
    print(f"  False Positives:   {fp_best} / {n_normal} normals flagged")

    # ── 7. Per-node signal check ─────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  PER-NODE AUC (which stages separate attack/normal?)")
    print(f"{'='*60}")
    for i in range(6):
        node_auc = auc(*roc_curve(y_true, z_scores[:, i])[:2])
        normal_mean = z_scores[y_true == 0, i].mean()
        attack_mean = z_scores[y_true == 1, i].mean()
        print(f"  P{i+1}: AUC={node_auc:.4f}  normal_mean_z={normal_mean:.3f}  attack_mean_z={attack_mean:.3f}")

    # ── 8. Verdict ───────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  VERDICT")
    print(f"{'='*60}")
    if roc_auc > 0.80 and best_f1 > f1_current + 0.05:
        print(f"  🔧 THRESHOLD PROBLEM: Model has signal (AUC={roc_auc:.3f}).")
        print(f"     Current threshold is too high ({current_thresh:.3f}).")
        print(f"     Lowering to {best_thresh:.3f} would improve F1 from {f1_current:.3f} → {best_f1:.3f}")
        print(f"     Recommendation: adjust z_threshold_scale in AnomalyDetector.")
    elif roc_auc > 0.80:
        print(f"  ✅ Model is performing near-optimally (AUC={roc_auc:.3f}, best F1={best_f1:.3f}).")
    elif roc_auc > 0.60:
        print(f"  ⚠️  WEAK SIGNAL: AUC={roc_auc:.3f}. Threshold tuning will help partially,")
        print(f"     but the model needs improvement (more training, architecture changes).")
    else:
        print(f"  ❌ NO SIGNAL: AUC={roc_auc:.3f}. The model cannot distinguish attack from normal.")
        print(f"     Retraining or architectural changes are required.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

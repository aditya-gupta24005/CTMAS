"""Supervised tabular baseline built from the CTMAS windowed dataset.

This script uses the labels derived from Data/merged.csv during preprocessing:
  - X_train / X_val are pre-attack normal windows (label 0)
  - X_test / y_test are mixed attack-phase windows with labels

To avoid leakage, we split the mixed attack-phase windows chronologically into
train/val/test segments and only evaluate on the held-out final segment.

Recommended default:
    python baseline_supervised_classifier.py

Alternative models:
    python baseline_supervised_classifier.py --model random_forest
    python baseline_supervised_classifier.py --model extra_trees
    python baseline_supervised_classifier.py --model xgboost
    python baseline_supervised_classifier.py --model lightgbm
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.utils.class_weight import compute_sample_weight

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "Data" / "processed"


def extract_window_features(X: np.ndarray) -> np.ndarray:
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


def build_model(name: str, random_state: int):
    if name == "hist_gb":
        return HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.05,
            max_leaf_nodes=63,
            min_samples_leaf=64,
            l2_regularization=1e-3,
            early_stopping=True,
            validation_fraction=None,
            random_state=random_state,
        )
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=4,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=random_state,
        )
    if name == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced",
            n_jobs=-1,
            random_state=random_state,
        )
    if name == "xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=500,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            min_child_weight=2,
            objective="binary:logistic",
            eval_metric="aucpr",
            tree_method="hist",
            random_state=random_state,
            n_jobs=-1,
        )
    if name == "lightgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=63,
            min_child_samples=40,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            class_weight="balanced",
            objective="binary",
            random_state=random_state,
            n_jobs=-1,
            verbosity=-1,
        )
    raise ValueError(f"Unsupported model: {name}")


def chronological_split(
    X: np.ndarray,
    y: np.ndarray,
    train_frac: float,
    val_frac: float,
    gap_windows: int,
):
    n = len(X)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    train_x = X[:train_end]
    train_y = y[:train_end]
    val_start = min(n, train_end + gap_windows)
    val_x = X[val_start:val_end]
    val_y = y[val_start:val_end]
    test_start = min(n, val_end + gap_windows)
    test_x = X[test_start:]
    test_y = y[test_start:]
    return train_x, train_y, val_x, val_y, test_x, test_y


def event_metrics(y_pred: np.ndarray, events: list[tuple[int, int]], stride_s: int, offset: int):
    delays_s = []
    caught = 0
    for ws, we in events:
        if we < offset:
            continue
        seg_start = max(0, ws - offset)
        seg_end = we - offset
        if seg_start >= len(y_pred):
            continue
        seg = y_pred[seg_start : min(len(y_pred), seg_end + 1)]
        hits = np.where(seg == 1)[0]
        if len(hits):
            caught += 1
            delays_s.append(int(hits[0]) * stride_s)
    mttd_s = float(np.mean(delays_s)) if delays_s else -1.0
    return mttd_s, caught


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["hist_gb", "random_forest", "extra_trees", "xgboost", "lightgbm"],
        default="hist_gb",
    )
    parser.add_argument("--train-frac", type=float, default=0.6)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--gap-windows", type=int, default=6)
    parser.add_argument("--max-extra-normals", type=int, default=20000)
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

    mixed_train_x, mixed_train_y, mixed_val_x, mixed_val_y, mixed_test_x, mixed_test_y = chronological_split(
        X_test, y_test, train_frac=args.train_frac, val_frac=args.val_frac, gap_windows=args.gap_windows
    )

    normal_pool = np.concatenate([X_train, X_val], axis=0)
    rng = np.random.default_rng(args.random_state)
    if len(normal_pool) > args.max_extra_normals:
        keep = rng.choice(len(normal_pool), size=args.max_extra_normals, replace=False)
        normal_pool = normal_pool[np.sort(keep)]

    train_x = np.concatenate([normal_pool, mixed_train_x], axis=0)
    train_y = np.concatenate(
        [np.zeros(len(normal_pool), dtype=np.int8), mixed_train_y.astype(np.int8)],
        axis=0,
    )

    print(f"[Data] extra normal windows={len(normal_pool)}")
    print(
        f"[Data] mixed train={len(mixed_train_x)} ({int(mixed_train_y.sum())} attack), "
        f"val={len(mixed_val_x)} ({int(mixed_val_y.sum())} attack), "
        f"test={len(mixed_test_x)} ({int(mixed_test_y.sum())} attack)"
    )

    print("[Feat] extracting summary features...")
    train_feat = extract_window_features(train_x)
    val_feat = extract_window_features(mixed_val_x)
    test_feat = extract_window_features(mixed_test_x)

    model = build_model(args.model, args.random_state)
    print(f"[Model] {model.__class__.__name__}")

    sample_weight = compute_sample_weight(class_weight="balanced", y=train_y)
    if args.model in {"hist_gb", "xgboost", "lightgbm"}:
        model.fit(train_feat, train_y, sample_weight=sample_weight)
    else:
        model.fit(train_feat, train_y)

    val_score = model.predict_proba(val_feat)[:, 1]
    test_score = model.predict_proba(test_feat)[:, 1]

    val_prec, val_rec, val_thr = precision_recall_curve(mixed_val_y, val_score)
    val_f1 = 2 * val_prec * val_rec / np.clip(val_prec + val_rec, 1e-8, None)
    best_idx = int(np.argmax(val_f1))
    threshold = float(val_thr[min(best_idx, len(val_thr) - 1)])
    y_pred = (test_score >= threshold).astype(int)

    acc = accuracy_score(mixed_test_y, y_pred)
    f1 = f1_score(mixed_test_y, y_pred, zero_division=0)
    prec = precision_score(mixed_test_y, y_pred, zero_division=0)
    rec = recall_score(mixed_test_y, y_pred, zero_division=0)
    auc_roc = roc_auc_score(mixed_test_y, test_score)
    ap = average_precision_score(mixed_test_y, test_score)
    tn, fp, fn, tp = confusion_matrix(mixed_test_y, y_pred).ravel()
    fpr = fp / max(1, tn + fp)

    test_start = int(len(X_test) * (args.train_frac + args.val_frac)) + args.gap_windows
    mttd_s, caught = event_metrics(
        y_pred,
        meta.get("attack_events_window", []),
        int(meta.get("test_stride", 10)),
        test_start,
    )
    n_test_events = sum(1 for _, we in meta.get("attack_events_window", []) if we >= test_start)
    event_recall = caught / max(1, n_test_events)

    print()
    print("=" * 62)
    print(f"  Supervised Baseline Eval ({args.model}, thr={threshold:.5f})")
    print("=" * 62)
    print(f"  TP={tp:,}    TN={tn:,}    FP={fp:,}    FN={fn:,}")
    print(f"  Accuracy              : {acc:.4f}")
    print(f"  Precision             : {prec:.4f}")
    print(f"  Recall                : {rec:.4f}")
    print(f"  F1                    : {f1:.4f}")
    print(f"  AUC-ROC               : {auc_roc:.4f}")
    print(f"  Avg Precision (PR-AUC): {ap:.4f}")
    print(f"  FPR @ threshold       : {fpr:.4f}")
    print(f"  MTTD                  : {mttd_s:.1f}s  ({caught}/{n_test_events} events caught)")
    print(f"  Event Recall          : {event_recall:.1%}")
    print(f"  Val-selected F1       : {float(val_f1[best_idx]):.4f}")
    print("=" * 62)


if __name__ == "__main__":
    main()

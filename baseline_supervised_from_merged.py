"""Build a supervised dataset directly from Data/merged.csv and train a classifier.

Interpretation of merged.csv in this repo:
  - a contiguous normal block
  - followed by a contiguous attack block

That is not a realistic mixed timeline, but it *is* enough to build a proper
binary classification benchmark by windowing each class separately and then
combining class-wise train/val/test splits.

Usage:
    python baseline_supervised_from_merged.py
    python baseline_supervised_from_merged.py --model xgboost
    python baseline_supervised_from_merged.py --threshold-mode val_f1
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump
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
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.class_weight import compute_sample_weight

BASE_DIR = Path(__file__).parent
CSV_PATH = BASE_DIR / "Data" / "merged.csv"
ARTIFACTS_DIR = BASE_DIR / "artifacts" / "supervised_from_merged"

WINDOW_SIZE = 60
STRIDE = 10

STAGE_MAP = {
    "P1": ["FIT101", "LIT101", "MV101", "P101", "P102"],
    "P2": ["AIT201", "AIT202", "AIT203", "FIT201", "MV201", "P201", "P202", "P203", "P204", "P205", "P206"],
    "P3": ["DPIT301", "FIT301", "LIT301", "MV301", "MV302", "MV303", "MV304", "P301", "P302"],
    "P4": ["AIT401", "AIT402", "FIT401", "LIT401", "P401", "P402", "P403", "P404", "UV401"],
    "P5": ["AIT501", "AIT502", "AIT503", "AIT504", "FIT501", "FIT502", "FIT503", "FIT504", "P501", "P502", "PIT501", "PIT502", "PIT503"],
    "P6": ["FIT601", "P601", "P602", "P603"],
}


def create_windows(X: np.ndarray, window_size: int, stride: int) -> np.ndarray:
    n = (len(X) - window_size) // stride + 1
    if n <= 0:
        return np.empty((0, window_size, X.shape[1]), dtype=np.float32)
    out = np.empty((n, window_size, X.shape[1]), dtype=np.float32)
    for i in range(n):
        start = i * stride
        out[i] = X[start : start + window_size]
    return out


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
    return np.concatenate([mean, std, min_, max_, last, delta, trend, change], axis=1).astype(np.float32)


def split_class_windows(
    X: np.ndarray,
    train_frac: float,
    val_frac: float,
    gap_windows: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(X)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    val_start = min(n, train_end + gap_windows)
    test_start = min(n, val_end + gap_windows)
    return X[:train_end], X[val_start:val_end], X[test_start:]


def compute_stage_scores(
    X_ref: np.ndarray,
    X_eval: np.ndarray,
    sensor_cols: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    stage_indices = [
        [sensor_cols.index(col) for col in cols]
        for cols in STAGE_MAP.values()
    ]
    ref_scores = []
    eval_scores = []
    for idxs in stage_indices:
        ref_slice = X_ref[:, :, idxs]
        mean = ref_slice.mean(axis=(0, 1), keepdims=True)
        std = ref_slice.std(axis=(0, 1), keepdims=True).clip(min=1e-6)
        ref_scores.append(np.abs((ref_slice - mean) / std).mean(axis=(1, 2)))
        eval_scores.append(np.abs((X_eval[:, :, idxs] - mean) / std).mean(axis=(1, 2)))
    return np.stack(ref_scores, axis=1), np.stack(eval_scores, axis=1)


def save_frontend_artifacts(
    model_name: str,
    model,
    threshold: float,
    threshold_note: str,
    sensor_cols: list[str],
    X_test: np.ndarray,
    y_test: np.ndarray,
    test_score: np.ndarray,
    stage_scores_test: np.ndarray,
    stage_thresholds: np.ndarray,
    attack_events_window: list[tuple[int, int]],
    test_stride: int,
    replay_source: str,
) -> Path:
    out_dir = ARTIFACTS_DIR / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    dump(model, out_dir / "model.joblib")
    np.save(out_dir / "X_test.npy", X_test)
    np.save(out_dir / "y_test.npy", y_test)
    np.save(out_dir / "test_score.npy", test_score.astype(np.float32))
    np.save(out_dir / "stage_scores_test.npy", stage_scores_test.astype(np.float32))

    metadata = {
        "model_kind": model_name,
        "display_name": model_name.replace("_", " ").title(),
        "window_size": WINDOW_SIZE,
        "test_stride": int(test_stride),
        "n_features": len(sensor_cols),
        "sensor_cols": sensor_cols,
        "stage_map": STAGE_MAP,
        "n_test_windows": int(len(X_test)),
        "n_test_attack_windows": int(y_test.sum()),
        "n_test_normal_windows": int((1 - y_test).sum()),
        "attack_events_window": [(int(s), int(e)) for s, e in attack_events_window],
        "score_label": "Attack Probability",
        "score_threshold": float(threshold),
        "threshold_note": threshold_note,
        "stage_thresholds": stage_thresholds.tolist(),
        "replay_source": replay_source,
    }
    with open(out_dir / "metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)
    return out_dir


def apply_hysteresis(scores: np.ndarray, high: float, low: float) -> np.ndarray:
    out = np.zeros(len(scores), dtype=np.int8)
    active = False
    for i, score in enumerate(scores):
        if not active and score >= high:
            active = True
        elif active and score < low:
            active = False
        out[i] = 1 if active else 0
    return out


def calibrate_hysteresis(
    y_cal: np.ndarray,
    score_cal: np.ndarray,
) -> tuple[float, float, dict[str, float]]:
    neg = score_cal[y_cal == 0]
    pos = score_cal[y_cal == 1]
    thresholds = sorted(
        set(
            [float(x) for x in np.quantile(neg, [0.90, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9995, 0.9999])]
            + [float(x) for x in np.quantile(pos, [0.01, 0.05, 0.1, 0.2, 0.3, 0.5])]
            + [1e-6, 2e-6, 5e-6, 1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2]
        )
    )

    best = None
    for high in thresholds:
        for low in thresholds:
            if low > high:
                continue
            pred = apply_hysteresis(score_cal, high, low)
            tn, fp, fn, tp = confusion_matrix(y_cal, pred).ravel()
            prec = precision_score(y_cal, pred, zero_division=0)
            rec = recall_score(y_cal, pred, zero_division=0)
            f1 = f1_score(y_cal, pred, zero_division=0)
            fpr = fp / max(1, fp + tn)
            objective = f1 - 0.10 * fpr
            candidate = (objective, f1, -fpr, prec, rec, high, low)
            if best is None or candidate > best:
                best = candidate

    _, best_f1, neg_best_fpr, best_prec, best_rec, best_high, best_low = best
    stats = {
        "f1": float(best_f1),
        "precision": float(best_prec),
        "recall": float(best_rec),
        "fpr": float(-neg_best_fpr),
    }
    return float(best_high), float(best_low), stats


def build_model(name: str, random_state: int, scale_pos_weight: float):
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
            n_estimators=700,
            max_depth=7,
            learning_rate=0.04,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=1.0,
            min_child_weight=2,
            gamma=0.0,
            scale_pos_weight=scale_pos_weight,
            objective="binary:logistic",
            eval_metric="aucpr",
            tree_method="hist",
            random_state=random_state,
            n_jobs=-1,
        )
    if name == "lightgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            n_estimators=700,
            learning_rate=0.04,
            num_leaves=63,
            min_child_samples=40,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=1.0,
            scale_pos_weight=scale_pos_weight,
            objective="binary",
            random_state=random_state,
            n_jobs=-1,
            verbosity=-1,
        )
    raise ValueError(f"Unsupported model: {name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["hist_gb", "random_forest", "extra_trees", "xgboost", "lightgbm"],
        default="xgboost",
    )
    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--gap-windows", type=int, default=6)
    parser.add_argument(
        "--threshold-mode",
        choices=["neg_quantile", "val_f1"],
        default="neg_quantile",
        help="How to choose the classification threshold from validation scores.",
    )
    parser.add_argument(
        "--neg-quantile",
        type=float,
        default=0.9995,
        help="Negative-score quantile used when threshold-mode=neg_quantile.",
    )
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(CSV_PATH)
    df.columns = df.columns.str.strip()
    df["Normal/Attack"] = df["Normal/Attack"].str.strip()
    sensor_cols = [c for c in df.columns if c not in ("Timestamp", "Normal/Attack")]

    is_attack = (df["Normal/Attack"] == "Attack").to_numpy(dtype=np.int8)
    attack_idx = np.where(is_attack == 1)[0]
    first_attack_idx = int(attack_idx[0])
    last_attack_idx = int(attack_idx[-1])

    normal_df = df.iloc[:first_attack_idx].reset_index(drop=True)
    attack_df = df.iloc[first_attack_idx : last_attack_idx + 1].reset_index(drop=True)

    scaler = MinMaxScaler()
    scaler.fit(normal_df[sensor_cols])

    normal_raw = scaler.transform(normal_df[sensor_cols]).astype(np.float32)
    attack_raw = scaler.transform(attack_df[sensor_cols]).astype(np.float32)
    np.nan_to_num(normal_raw, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    np.nan_to_num(attack_raw, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    normal_windows = create_windows(normal_raw, WINDOW_SIZE, STRIDE)
    attack_windows = create_windows(attack_raw, WINDOW_SIZE, STRIDE)

    n_train, n_val, n_test = split_class_windows(
        normal_windows, args.train_frac, args.val_frac, args.gap_windows
    )
    a_train, a_val, a_test = split_class_windows(
        attack_windows, args.train_frac, args.val_frac, args.gap_windows
    )

    X_train = np.concatenate([n_train, a_train], axis=0)
    y_train = np.concatenate(
        [np.zeros(len(n_train), dtype=np.int8), np.ones(len(a_train), dtype=np.int8)],
        axis=0,
    )
    X_val = np.concatenate([n_val, a_val], axis=0)
    y_val = np.concatenate(
        [np.zeros(len(n_val), dtype=np.int8), np.ones(len(a_val), dtype=np.int8)],
        axis=0,
    )
    X_test = np.concatenate([n_test, a_test], axis=0)
    y_test = np.concatenate(
        [np.zeros(len(n_test), dtype=np.int8), np.ones(len(a_test), dtype=np.int8)],
        axis=0,
    )

    train_perm = np.random.default_rng(args.random_state).permutation(len(X_train))
    X_train, y_train = X_train[train_perm], y_train[train_perm]

    print(
        f"[Raw] normal_rows={len(normal_df)} attack_rows={len(attack_df)} "
        f"first_attack_idx={first_attack_idx}"
    )
    print(
        f"[Windows] normal={len(normal_windows)} attack={len(attack_windows)} "
        f"(window={WINDOW_SIZE}, stride={STRIDE})"
    )
    print(
        f"[Split] train={len(X_train)} ({int(y_train.sum())} attack), "
        f"val={len(X_val)} ({int(y_val.sum())} attack), "
        f"test={len(X_test)} ({int(y_test.sum())} attack)"
    )

    print("[Feat] extracting summary features...")
    train_feat = extract_window_features(X_train)
    val_feat = extract_window_features(X_val)
    test_feat = extract_window_features(X_test)
    _, val_stage_scores = compute_stage_scores(n_train, X_val, sensor_cols)
    _, test_stage_scores = compute_stage_scores(n_train, X_test, sensor_cols)
    stage_thresholds = np.quantile(val_stage_scores[y_val == 0], 0.9995, axis=0)

    n_pos = int(y_train.sum())
    n_neg = int(len(y_train) - n_pos)
    scale_pos_weight = n_neg / max(1, n_pos)
    model = build_model(args.model, args.random_state, scale_pos_weight)
    print(f"[Model] {model.__class__.__name__}  scale_pos_weight={scale_pos_weight:.2f}")

    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)
    if args.model in {"hist_gb", "xgboost", "lightgbm"}:
        model.fit(train_feat, y_train, sample_weight=sample_weight)
    else:
        model.fit(train_feat, y_train)

    val_score = model.predict_proba(val_feat)[:, 1]
    test_score = model.predict_proba(test_feat)[:, 1]

    val_prec, val_rec, val_thr = precision_recall_curve(y_val, val_score)
    val_f1 = 2 * val_prec * val_rec / np.clip(val_prec + val_rec, 1e-8, None)
    best_idx = int(np.argmax(val_f1))
    val_f1_threshold = float(val_thr[min(best_idx, len(val_thr) - 1)])

    if args.threshold_mode == "val_f1":
        threshold = val_f1_threshold
        threshold_note = "validation F1 maximum"
    else:
        val_neg = val_score[y_val == 0]
        threshold = float(np.quantile(val_neg, args.neg_quantile))
        threshold_note = f"validation negative quantile={args.neg_quantile:.4f}"

    y_pred = (test_score >= threshold).astype(int)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    auc_roc = roc_auc_score(y_test, test_score)
    ap = average_precision_score(y_test, test_score)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    fpr = fp / max(1, tn + fp)

    processed_X_test = np.load(BASE_DIR / "Data" / "processed" / "X_test.npy")
    processed_y_test = np.load(BASE_DIR / "Data" / "processed" / "y_test.npy").astype(np.int8)
    with open(BASE_DIR / "Data" / "processed" / "metadata.pkl", "rb") as f:
        processed_meta = pickle.load(f)
    np.nan_to_num(processed_X_test, copy=False, nan=0.0)
    processed_feat = extract_window_features(processed_X_test)
    _, processed_stage_scores = compute_stage_scores(n_train, processed_X_test, sensor_cols)
    processed_score = model.predict_proba(processed_feat)[:, 1]
    p_auc_roc = roc_auc_score(processed_y_test, processed_score)
    p_ap = average_precision_score(processed_y_test, processed_score)

    cal_end = len(processed_y_test) // 2
    cal_y = processed_y_test[:cal_end]
    cal_score = processed_score[:cal_end]
    replay_high, replay_low, replay_cal_stats = calibrate_hysteresis(cal_y, cal_score)
    processed_pred = apply_hysteresis(processed_score, replay_high, replay_low)
    p_acc = accuracy_score(processed_y_test, processed_pred)
    p_f1 = f1_score(processed_y_test, processed_pred, zero_division=0)
    p_prec = precision_score(processed_y_test, processed_pred, zero_division=0)
    p_rec = recall_score(processed_y_test, processed_pred, zero_division=0)
    p_tn, p_fp, p_fn, p_tp = confusion_matrix(processed_y_test, processed_pred).ravel()
    p_fpr = p_fp / max(1, p_tn + p_fp)

    holdout_y = processed_y_test[cal_end:]
    holdout_score = processed_score[cal_end:]
    holdout_pred = apply_hysteresis(holdout_score, replay_high, replay_low)
    h_acc = accuracy_score(holdout_y, holdout_pred)
    h_f1 = f1_score(holdout_y, holdout_pred, zero_division=0)
    h_prec = precision_score(holdout_y, holdout_pred, zero_division=0)
    h_rec = recall_score(holdout_y, holdout_pred, zero_division=0)
    h_tn, h_fp, h_fn, h_tp = confusion_matrix(holdout_y, holdout_pred).ravel()
    h_fpr = h_fp / max(1, h_tn + h_fp)

    artifact_dir = save_frontend_artifacts(
        args.model,
        model,
        replay_high,
        f"mixed replay hysteresis: high={replay_high:.6f}, low={replay_low:.6f}",
        sensor_cols,
        processed_X_test,
        processed_y_test,
        processed_score,
        processed_stage_scores,
        stage_thresholds,
        processed_meta.get("attack_events_window", []),
        int(processed_meta.get("test_stride", STRIDE)),
        "Data/processed/X_test.npy",
    )
    with open(artifact_dir / "metadata.pkl", "rb") as f:
        artifact_meta = pickle.load(f)
    artifact_meta["score_low_threshold"] = replay_low
    artifact_meta["score_rule"] = "hysteresis"
    artifact_meta["replay_calibration_windows"] = int(cal_end)
    artifact_meta["replay_calibration_stats"] = replay_cal_stats
    with open(artifact_dir / "metadata.pkl", "wb") as f:
        pickle.dump(artifact_meta, f)

    print()
    print("=" * 62)
    print(f"  Supervised From merged.csv ({args.model}, thr={threshold:.5f})")
    print("=" * 62)
    print(f"  Threshold source       : {threshold_note}")
    print(f"  Val-F1 threshold       : {val_f1_threshold:.5f}")
    print(f"  TP={tp:,}    TN={tn:,}    FP={fp:,}    FN={fn:,}")
    print(f"  Accuracy              : {acc:.4f}")
    print(f"  Precision             : {prec:.4f}")
    print(f"  Recall                : {rec:.4f}")
    print(f"  F1                    : {f1:.4f}")
    print(f"  AUC-ROC               : {auc_roc:.4f}")
    print(f"  Avg Precision (PR-AUC): {ap:.4f}")
    print(f"  FPR @ threshold       : {fpr:.4f}")
    print(f"  Val-selected F1       : {float(val_f1[best_idx]):.4f}")
    print(f"  Saved artifacts       : {artifact_dir}")
    print("=" * 62)
    print()
    print("=" * 62)
    print(f"  Mixed Replay Eval ({args.model} on Data/processed/X_test.npy)")
    print("=" * 62)
    print(f"  Replay rule            : hysteresis high={replay_high:.6f} low={replay_low:.6f}")
    print(f"  TP={p_tp:,}    TN={p_tn:,}    FP={p_fp:,}    FN={p_fn:,}")
    print(f"  Accuracy              : {p_acc:.4f}")
    print(f"  Precision             : {p_prec:.4f}")
    print(f"  Recall                : {p_rec:.4f}")
    print(f"  F1                    : {p_f1:.4f}")
    print(f"  AUC-ROC               : {p_auc_roc:.4f}")
    print(f"  Avg Precision (PR-AUC): {p_ap:.4f}")
    print(f"  FPR @ threshold       : {p_fpr:.4f}")
    print("=" * 62)
    print()
    print("=" * 62)
    print("  Mixed Replay Holdout (second half of Data/processed/X_test.npy)")
    print("=" * 62)
    print(f"  Calibration stats      : {replay_cal_stats}")
    print(f"  TP={h_tp:,}    TN={h_tn:,}    FP={h_fp:,}    FN={h_fn:,}")
    print(f"  Accuracy              : {h_acc:.4f}")
    print(f"  Precision             : {h_prec:.4f}")
    print(f"  Recall                : {h_rec:.4f}")
    print(f"  F1                    : {h_f1:.4f}")
    print(f"  FPR @ threshold       : {h_fpr:.4f}")
    print("=" * 62)


if __name__ == "__main__":
    main()

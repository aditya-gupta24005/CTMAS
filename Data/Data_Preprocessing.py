"""
CTMAS - Data Preprocessing Pipeline (v2 — chronological, labeled test set)
SWaT Dataset (merged.csv)

Key fixes over v1:
  * Test set is the CONTIGUOUS attack phase (normal + attack interleaved), not
    attack-rows-only stitched together. This preserves temporal continuity
    and keeps real normal context around each attack for FPR measurement.
  * Per-window labels (y_test) are saved: a window is attack=1 if ANY second
    inside it is labeled "Attack", else 0.
  * Attack-event boundaries are saved so Mean Time To Detect is measurable.
  * Training pool = contiguous pre-attack normal phase, split 85/15 for val.

Run: python Data_Preprocessing.py
"""

import os
import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

HERE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(HERE, "merged.csv")
OUT_DIR = os.path.join(HERE, "processed")
os.makedirs(OUT_DIR, exist_ok=True)

WINDOW_SIZE = 60          # seconds
TRAIN_STRIDE = 10
TEST_STRIDE = 10          # stride=10 keeps test ~manageable; MTTD stays 10s-resolved


# ─────────────────────────────────────────────
# 1. LOAD
# ─────────────────────────────────────────────
print("=" * 60)
print("STEP 1: Load raw data")
print("=" * 60)

df = pd.read_csv(CSV_PATH)
df.columns = df.columns.str.strip()
df["Normal/Attack"] = df["Normal/Attack"].str.strip()
df["Timestamp"] = pd.to_datetime(df["Timestamp"], format="mixed", dayfirst=True)

df = df.sort_values("Timestamp").reset_index(drop=True)
print(f"Raw shape: {df.shape}")
print(f"Timestamp range: {df['Timestamp'].min()} → {df['Timestamp'].max()}")
print(df["Normal/Attack"].value_counts())


# ─────────────────────────────────────────────
# 2. FIX KNOWN DATA ISSUES
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: Fix known data issues")
print("=" * 60)

# All sensor columns with >0 nulls get forward+back filled. SWaT's missing
# values are stretches where the sensor was offline; ffill/bfill holds the
# last known state, which is the correct imputation for piecewise-constant
# actuator signals (valves/pumps) and a reasonable one for analog sensors.
null_cols = df.columns[df.isnull().any()].tolist()
if null_cols:
    print(f"Columns with nulls ({len(null_cols)}): {null_cols}")
    for col in null_cols:
        before = df[col].isnull().sum()
        df[col] = df[col].ffill().bfill()
        print(f"  {col:<10} nulls: {before:>7} → {df[col].isnull().sum()}")
print(f"Total nulls remaining: {df.isnull().sum().sum()}")


# ─────────────────────────────────────────────
# 3. SENSOR COLUMNS & STAGE MAP
# ─────────────────────────────────────────────
SENSOR_COLS = [c for c in df.columns if c not in ("Timestamp", "Normal/Attack")]

STAGE_MAP = {
    "P1": ["FIT101", "LIT101", "MV101", "P101", "P102"],
    "P2": ["AIT201", "AIT202", "AIT203", "FIT201", "MV201",
           "P201", "P202", "P203", "P204", "P205", "P206"],
    "P3": ["DPIT301", "FIT301", "LIT301", "MV301", "MV302",
           "MV303", "MV304", "P301", "P302"],
    "P4": ["AIT401", "AIT402", "FIT401", "LIT401",
           "P401", "P402", "P403", "P404", "UV401"],
    "P5": ["AIT501", "AIT502", "AIT503", "AIT504",
           "FIT501", "FIT502", "FIT503", "FIT504",
           "P501", "P502", "PIT501", "PIT502", "PIT503"],
    "P6": ["FIT601", "P601", "P602", "P603"],
}
CONTINUOUS_COLS = [c for c in SENSOR_COLS if df[c].nunique() > 5]
BINARY_COLS = [c for c in SENSOR_COLS if df[c].nunique() <= 5]

print(f"\nSensors: {len(SENSOR_COLS)} (continuous={len(CONTINUOUS_COLS)}, binary={len(BINARY_COLS)})")


# ─────────────────────────────────────────────
# 4. CHRONOLOGICAL SPLIT
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4: Chronological split (train pool vs test pool)")
print("=" * 60)

is_attack = (df["Normal/Attack"] == "Attack").values
attack_indices = np.where(is_attack)[0]
if len(attack_indices) == 0:
    raise RuntimeError("No 'Attack' rows found — check label column contents.")

first_attack_idx = int(attack_indices[0])
last_attack_idx = int(attack_indices[-1])

train_pool = df.iloc[:first_attack_idx].reset_index(drop=True)
test_pool = df.iloc[first_attack_idx : last_attack_idx + 1].reset_index(drop=True)

# Sanity: train_pool must be 100% normal. If not, drop stray attack rows from it.
train_attack = (train_pool["Normal/Attack"] == "Attack").sum()
if train_attack > 0:
    print(f"WARNING: {train_attack} attack rows found before first_attack_idx — filtering.")
    train_pool = train_pool[train_pool["Normal/Attack"] == "Normal"].reset_index(drop=True)

test_attack_count = (test_pool["Normal/Attack"] == "Attack").sum()
test_normal_count = (test_pool["Normal/Attack"] == "Normal").sum()

print(f"Train pool (normal phase):  {len(train_pool):>8} rows  [{train_pool['Timestamp'].min()} → {train_pool['Timestamp'].max()}]")
print(f"Test  pool (attack phase):  {len(test_pool):>8} rows  [{test_pool['Timestamp'].min()} → {test_pool['Timestamp'].max()}]")
print(f"  └── Attack rows inside test pool: {test_attack_count} ({test_attack_count/len(test_pool)*100:.1f}%)")
print(f"  └── Normal rows inside test pool: {test_normal_count} ({test_normal_count/len(test_pool)*100:.1f}%)")


# ─────────────────────────────────────────────
# 5. NORMALIZE (fit on TRAIN POOL normal only)
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5: Normalization (fit scaler on train pool only)")
print("=" * 60)

scaler = MinMaxScaler()
scaler.fit(train_pool[SENSOR_COLS])

X_train_raw = scaler.transform(train_pool[SENSOR_COLS]).astype(np.float32)
X_test_raw = scaler.transform(test_pool[SENSOR_COLS]).astype(np.float32)

# Some sensors are constant in the train pool → MinMaxScaler produces NaN.
# Replace with 0 (the correct scaled value for a constant signal).
X_train_raw = np.nan_to_num(X_train_raw, nan=0.0, posinf=0.0, neginf=0.0)
X_test_raw = np.nan_to_num(X_test_raw, nan=0.0, posinf=0.0, neginf=0.0)

print(f"X_train_raw: {X_train_raw.shape}, min={X_train_raw.min():.3f}, max={X_train_raw.max():.3f}")
print(f"X_test_raw : {X_test_raw.shape}")

# Labels aligned to test_pool rows (second-level)
labels_test_raw = is_attack[first_attack_idx : last_attack_idx + 1].astype(np.int8)
assert len(labels_test_raw) == len(X_test_raw)


# ─────────────────────────────────────────────
# 6. SLIDING WINDOWS
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 6: Sliding window tokenization")
print("=" * 60)

def create_windows(X, window_size, stride):
    n = (len(X) - window_size) // stride + 1
    if n <= 0:
        return np.empty((0, window_size, X.shape[1]), dtype=X.dtype)
    out = np.empty((n, window_size, X.shape[1]), dtype=X.dtype)
    for i in range(n):
        out[i] = X[i * stride : i * stride + window_size]
    return out


def create_window_labels(labels, window_size, stride):
    """A window is attack (1) if ANY second inside it is attack."""
    n = (len(labels) - window_size) // stride + 1
    if n <= 0:
        return np.empty((0,), dtype=np.int8)
    return np.array(
        [int(labels[i * stride : i * stride + window_size].max()) for i in range(n)],
        dtype=np.int8,
    )


X_train_windows = create_windows(X_train_raw, WINDOW_SIZE, stride=TRAIN_STRIDE)
X_test_windows = create_windows(X_test_raw, WINDOW_SIZE, stride=TEST_STRIDE)
y_test_windows = create_window_labels(labels_test_raw, WINDOW_SIZE, stride=TEST_STRIDE)

assert len(X_test_windows) == len(y_test_windows)

print(f"X_train_windows: {X_train_windows.shape}  (stride={TRAIN_STRIDE}, normal only)")
print(f"X_test_windows : {X_test_windows.shape}  (stride={TEST_STRIDE}, mixed)")
print(f"  Attack windows in test: {int(y_test_windows.sum())} ({y_test_windows.mean()*100:.1f}%)")
print(f"  Normal windows in test: {int((1-y_test_windows).sum())} ({(1-y_test_windows).mean()*100:.1f}%)")


# ─────────────────────────────────────────────
# 7. TRAIN / VAL SPLIT (chronological within train pool)
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 7: Train/val split (chronological, normal only)")
print("=" * 60)

split = int(0.85 * len(X_train_windows))
X_train = X_train_windows[:split]
X_val = X_train_windows[split:]
print(f"X_train: {X_train.shape}")
print(f"X_val  : {X_val.shape}")


# ─────────────────────────────────────────────
# 8. ATTACK EVENT BOUNDARIES (for real MTTD)
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 8: Extract attack event boundaries")
print("=" * 60)

# Contiguous runs of attack=1 in labels_test_raw → one event each.
def contiguous_runs(binary):
    diff = np.diff(np.concatenate([[0], binary, [0]]))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0] - 1
    return list(zip(starts.tolist(), ends.tolist()))


events_row = contiguous_runs(labels_test_raw)   # (row_start, row_end) pairs
events_window = []
for (rs, re) in events_row:
    # Earliest window whose [start, start+W) overlaps [rs, re]
    w_start = max(0, (rs - WINDOW_SIZE + 1) // TEST_STRIDE + 1)
    w_start = max(0, int(np.ceil((rs - WINDOW_SIZE + 1) / TEST_STRIDE)))
    if w_start < 0:
        w_start = 0
    # Latest window whose [start, start+W) overlaps: start <= re, so start <= re
    w_end = re // TEST_STRIDE
    w_start = min(w_start, len(y_test_windows) - 1)
    w_end = min(w_end, len(y_test_windows) - 1)
    if w_end >= w_start:
        events_window.append((int(w_start), int(w_end)))

print(f"Detected {len(events_row)} attack events (row-level)")
print(f"Mapped   {len(events_window)} attack events (window-level, stride={TEST_STRIDE})")
if events_window:
    durations = [e - s + 1 for s, e in events_window]
    print(f"  Event window counts: min={min(durations)}, median={int(np.median(durations))}, max={max(durations)}")


# ─────────────────────────────────────────────
# 9. SAVE
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 9: Save processed data")
print("=" * 60)

np.save(os.path.join(OUT_DIR, "X_train.npy"), X_train)
np.save(os.path.join(OUT_DIR, "X_val.npy"), X_val)
np.save(os.path.join(OUT_DIR, "X_test.npy"), X_test_windows)
np.save(os.path.join(OUT_DIR, "y_test.npy"), y_test_windows)

with open(os.path.join(OUT_DIR, "scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)

metadata = {
    "window_size": WINDOW_SIZE,
    "train_stride": TRAIN_STRIDE,
    "test_stride": TEST_STRIDE,
    "n_features": len(SENSOR_COLS),
    "sensor_cols": SENSOR_COLS,
    "continuous_cols": CONTINUOUS_COLS,
    "binary_cols": BINARY_COLS,
    "stage_map": STAGE_MAP,
    "n_train": int(len(X_train)),
    "n_val": int(len(X_val)),
    "n_test": int(len(X_test_windows)),
    "n_test_attack_windows": int(y_test_windows.sum()),
    "n_test_normal_windows": int((1 - y_test_windows).sum()),
    "attack_events_window": events_window,   # list of (start_window, end_window) inclusive
    "test_pool_start_ts": str(test_pool["Timestamp"].min()),
    "test_pool_end_ts": str(test_pool["Timestamp"].max()),
}
with open(os.path.join(OUT_DIR, "metadata.pkl"), "wb") as f:
    pickle.dump(metadata, f)

print("Saved files:")
for fname in sorted(os.listdir(OUT_DIR)):
    size_mb = os.path.getsize(os.path.join(OUT_DIR, fname)) / 1e6
    print(f"  {fname:<20s} {size_mb:>9.1f} MB")


# ─────────────────────────────────────────────
# 10. SUMMARY
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"""
Train   : {X_train.shape}   (normal only, chronological)
Val     : {X_val.shape}     (normal only, chronological)
Test    : {X_test_windows.shape}  (MIXED normal+attack, chronological)
  ├── attack windows: {int(y_test_windows.sum())}
  └── normal windows: {int((1-y_test_windows).sum())}
Attack events (for MTTD): {len(events_window)}
Scaler  : MinMaxScaler fit on train-pool normal only
Window  : {WINDOW_SIZE}s, train_stride={TRAIN_STRIDE}, test_stride={TEST_STRIDE}

Ready for FL training.
""")

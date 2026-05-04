# CTMAS — Proactive Threat Modeling for Intelligent Cyber-Physical Systems

Supervised anomaly detection and threat intelligence for Industrial Control Systems (ICS), applied to the **SWaT (Secure Water Treatment)** dataset from iTrust, Singapore.

**Primary model: XGBoost** (F1=0.827, AUC-ROC=0.985). A Spatio-Temporal GNN autoencoder is also implemented for research comparison.

---

## What This Project Does

CTMAS detects cyberattacks on water-treatment plant sensors in real time. It does this by:

1. **Classifying 60-second sensor windows as normal or attack** — an XGBoost classifier is trained on statistical features extracted from sliding windows of all 51 sensors. It handles the 96/4 class imbalance via `scale_pos_weight` and a hysteresis threshold to reduce false alarms.
2. **Escalating from anomaly → campaign** — a Finite State Machine classifies ongoing anomalies as RECON → INTRUSION → IMPACT and maps them to MITRE ATT&CK for ICS techniques.
3. **Streaming results to a live dashboard** — a FastAPI WebSocket server replays detections window-by-window, with per-stage scores and FSM state visualised in the browser.
4. **GNN research track (secondary)** — a Spatio-Temporal Graph Neural Network autoencoder also exists for comparison, trained privately via Federated Learning (FedProx + Byzantine-robust aggregation + Opacus differential privacy).

---

## Architecture

### Primary Pipeline (XGBoost)

```
Data/merged.csv  (1.44M rows, 51 sensors, 1-second intervals)
        │
        ▼
 baseline_supervised_from_merged.py
        │  MinMaxScaler (fit on normal rows only)
        │  sliding windows: size=60s, stride=10s
        │  feature extraction: mean, std, min, max, last,
        │                       delta, trend, change-rate
        │                       → 408 features per window
        ▼
 ┌─────────────────────────────────────────────┐
 │   XGBClassifier                              │
 │   700 trees · max_depth=7 · lr=0.04          │
 │   scale_pos_weight=25.4 (handles 96/4 skew)  │
 │   Threshold: hysteresis (high=0.042, low=0.002)│
 └────────────────┬────────────────────────────┘
                  │ attack probability per window
                  ▼
 ┌─────────────────────────────────────────────┐
 │   Threat Intelligence FSM                    │
 │   intelligence/threat_mapper.py              │
 │   States: NORMAL → RECON → INTRUSION → IMPACT│
 │   Maps transitions → MITRE ATT&CK for ICS    │
 └────────────────┬────────────────────────────┘
                  │ JSON events
                  ▼
 ┌─────────────────────────────────────────────┐
 │   FastAPI + WebSocket Server  (api/main.py)  │
 │   GET /health  GET /metadata  WS /ws/stream  │
 └────────────────┬────────────────────────────┘
                  ▼
        frontend/index.html  (live dashboard)
```

### Secondary Pipeline (GNN — research track)

```
Data/processed/X_train.npy  (normal windows only)
        │
        ▼
 ┌─────────────────────────────────────────────┐
 │   Spatio-Temporal GNN Autoencoder            │
 │   models/gnn_model.py + models/stage_encoder │
 │   Per-stage 1D-CNN → GCN × 2 → bottleneck   │
 │   Trained via federated/client.py (FedProx + │
 │   Opacus DP + Byzantine-robust aggregation)  │
 └────────────────┬────────────────────────────┘
                  │ per-stage reconstruction error
                  ▼
 ┌─────────────────────────────────────────────┐
 │   Two-Layer Anomaly Detector                 │
 │   detection/detector.py                      │
 │   Layer 1: per-node MSE threshold (mean+3σ)  │
 │   Layer 2: EWMA early-warning on global MSE  │
 └─────────────────────────────────────────────┘
```

Both pipelines feed the same FSM + API + dashboard. Switch between them via `?model=xgboost` or `?model=neural` on the API endpoints.

---

## Key Design Decisions

| Decision | Why |
|---|---|
| **GNN, not LSTM/GRU** | Opacus (differential privacy) can't wrap recurrent layers; 1D-CNN per stage is Opacus-compatible |
| **FedProx, not FedAvg** | Heterogeneous stage data distributions — proximal term prevents client drift |
| **Byzantine-robust aggregation** | Cosine-similarity filter down-weights any client whose gradient diverges from the median, guarding against compromised plant sites |
| **Differential privacy (ε, δ)-DP** | Per-client Opacus DP-SGD with DP_NOISE_MULTIPLIER=0.3, MAX_GRAD_NORM=1.0, δ=1e-5 |
| **EWMA early warning** | Catches low-and-slow attacks that stay below per-node thresholds |
| **FSM over static lookup** | Campaign state (RECON/INTRUSION/IMPACT) evolves dynamically with EWMA trend + cross-node correlation, not a hard-coded sensor-prefix → T-code table |

---

## Dataset — SWaT

| Property | Value |
|---|---|
| Source | iTrust, Singapore National University |
| Rows | 1,441,719 (1-second intervals) |
| Normal | 1,387,098 rows |
| Attack | 54,621 rows (41 distinct attack scenarios) |
| Class balance | ~96% normal / 4% attack |
| Sensors | 51 (flow, level, pressure, conductivity, actuators) |
| Plant stages | P1 (5 sensors) · P2 (11) · P3 (9) · P4 (9) · P5 (13) · P6 (4) |

The dataset is **not included in this repo** (~427 MB). Ask the project owner for the download link and place it at `Data/merged.csv`.

---

## Repo Structure

```
CTMAS/
├── Data/
│   ├── merged.csv              ← SWaT dataset (not in git)
│   ├── Data_Preprocessing.py   ← generates Data/processed/
│   └── processed/              ← X_train/val/test .npy + scaler.pkl
│
├── models/
│   ├── gnn_model.py            ← SpatioTemporalGNNAutoencoder (core model)
│   └── stage_encoder.py        ← per-stage 1D-CNN encoder/decoder
│
├── federated/
│   ├── client.py               ← FedProx + Opacus DP client
│   ├── server.py               ← Byzantine-robust aggregation
│   └── config.py               ← all FL hyperparameters
│
├── detection/
│   └── detector.py             ← two-layer anomaly detector + AnomalyReport
│
├── intelligence/
│   └── threat_mapper.py        ← FSM + MITRE ATT&CK for ICS mapping
│
├── api/
│   └── main.py                 ← FastAPI WebSocket server
│
├── frontend/
│   └── index.html              ← live replay dashboard
│
├── main.py                     ← federated training entry point
├── train_centralized.py        ← single-machine training (no FL)
├── train_ensemble.py           ← denoising-AE ensemble training
├── eval_ensemble.py            ← ensemble evaluation
├── offline_eval.py             ← full metric report on saved model
│
├── baseline_isolation_forest.py          ← unsupervised baseline
├── baseline_supervised_classifier.py     ← supervised baselines
├── baseline_supervised_from_merged.py    ← XGBoost / RF / LightGBM on merged.csv
│
├── SETUP.md                    ← full environment setup guide
├── requirements.txt
└── ctmas_model.pt              ← saved trained model weights
```

---

## Quick Start

See **SETUP.md** for full environment setup. The short version:

```bash
# 1. Create and activate virtual environment
python3.11 -m venv .venv && source .venv/bin/activate

# 2. Install dependencies (no GPU needed for XGBoost)
pip install -r requirements.txt

# 3. Place merged.csv at Data/merged.csv, then preprocess
cd Data && python Data_Preprocessing.py && cd ..

# 4. Train and evaluate the XGBoost model
python baseline_supervised_from_merged.py --model xgboost

# 5. Run the API + open the dashboard
uvicorn api.main:app --host 0.0.0.0 --port 8000
# → open frontend/index.html in a browser
# → dashboard defaults to XGBoost (?model=xgboost)
```

**GNN track only** — requires GPU + extra setup (see SETUP.md):
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
python main.py --rounds 10   # federated training
python offline_eval.py       # evaluate ctmas_model.pt
```

---

## Model Scripts

### Primary — XGBoost

| Script | What it does |
|---|---|
| `baseline_supervised_from_merged.py` | **Main script.** Trains XGBoost (or RF/LightGBM) directly on `Data/merged.csv`. Saves artifacts to `artifacts/supervised_from_merged/xgboost/` for the dashboard. |
| `baseline_supervised_classifier.py` | Alternative: trains on `Data/processed/` splits instead of merged.csv |
| `baseline_isolation_forest.py` | Unsupervised baseline (no labels needed) |

Run:
```bash
python baseline_supervised_from_merged.py --model xgboost
# Other models: --model random_forest | hist_gb | lightgbm
```

### Secondary — GNN (research track, requires GPU)

| Script | What it does |
|---|---|
| `main.py` | Full federated training (FedProx + DP, 6 clients) |
| `train_centralized.py` | Single-machine GNN training, no FL overhead |
| `train_ensemble.py` | Trains N denoising-AE members for ensemble |
| `eval_ensemble.py` | Evaluates saved ensemble members |
| `offline_eval.py` | Full metric report (F1, AUC-ROC, MTTD, FPR) on `ctmas_model.pt` |

---

## Results

### XGBoost (primary model)

| Eval set | F1 | AUC-ROC | PR-AUC | FPR |
|---|---|---|---|---|
| In-distribution test split | 0.827 | 0.985 | 0.886 | 0.64% |
| Mixed replay (real attack timeline) | 0.772 | 0.917 | 0.824 | 2.85% |
| Replay holdout (unseen calibration) | 0.865 | — | — | 0.04% |

Threshold rule: hysteresis (high=0.042, low=0.002) calibrated on the first half of the mixed replay set.

---

## GNN / Federated Learning Config (`federated/config.py`)

> Only relevant if using the GNN research track.

| Parameter | Value | Meaning |
|---|---|---|
| `FL_ROUNDS` | 10 | Number of federated rounds |
| `LOCAL_EPOCHS` | 3 | Epochs per client per round |
| `MU` | 0.01 | FedProx proximal penalty |
| `BYZANTINE_THRESHOLD` | 0.5 | Cosine similarity cutoff for Byzantine filter |
| `DP_NOISE_MULTIPLIER` | 0.3 | Gaussian noise for differential privacy |
| `DP_MAX_GRAD_NORM` | 1.0 | Gradient clipping norm |
| `HIDDEN_DIM` | 64 | GNN encoder width |
| `LATENT_DIM` | 32 | Bottleneck size |

---

## Physical Plant Graph

The 6 plant stages and their water-flow connections (used by the GNN; also informs which sensors are co-located for feature engineering):

```
P1 ──► P2 ──► P3 ──► P4 ──► P5 ──► P6
 ▲            │             │
 └────────────┘             │   (P3 backwash)
 └───────────────────────────   (P5 RO reject)
```

Each node = one plant stage. Node features = that stage's sensors padded to 13 (max across all stages).

---

## API Endpoints

| Endpoint | Description |
|---|---|
| `GET /health?model=neural\|xgboost` | Liveness check + model metadata |
| `GET /metadata?model=neural\|xgboost` | Full model config, thresholds, stage map |
| `WS /ws/stream?model=neural\|xgboost` | Streams window-by-window replay with anomaly scores |

The dashboard (`frontend/index.html`) connects to the WebSocket and visualises per-stage reconstruction errors, FSM state, and MITRE technique labels in real time.

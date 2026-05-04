# Proactive Threat Modeling for Intelligent Cyber–Physical Systems Using Federated and Privacy-Aware AI

Real-time cyberattack detection and collaborative threat intelligence for Industrial Control Systems (ICS), applied to the **SWaT (Secure Water Treatment)** dataset — 1.44 M sensor readings across a 6-stage water treatment plant.

**Key contributions:** Spatio-Temporal GNN trained via Federated Learning (FedProx + Byzantine-robust aggregation + (ε,δ)-DP), two-layer anomaly detection (per-node MSE + EWMA early warning), and a dynamic MITRE ATT&CK–mapped FSM for collaborative threat intelligence.

---

## What This System Does

Cyber-physical systems like water treatment plants span multiple distributed sub-systems, each with its own sensors and control logic. Centralising all sensor data for a monolithic classifier raises serious privacy and operational concerns — a plant operator at Stage P3 should not expose raw readings to a remote server.

CTMAS addresses this with:

1. **Federated Spatio-Temporal Anomaly Detection** — six clients (one per plant stage) each train the shared GNN autoencoder on their local sensor windows. No raw sensor data leaves the stage. The server aggregates only model updates.
2. **Privacy-Preserving Gradient Updates** — each client applies manual DP-SGD (gradient clipping + Gaussian noise) before sending updates. The server tracks the accumulated (ε, δ)-DP privacy budget across rounds.
3. **Byzantine-Robust Model Aggregation** — a cosine-similarity filter detects and down-weights updates from any client whose gradients diverge from the population median, guarding against poisoned or compromised nodes.
4. **Collaborative Threat Intelligence** — the globally aggregated model learns cross-stage attack propagation patterns. A Finite State Machine (NORMAL→RECON→INTRUSION→IMPACT) maps anomaly signals to MITRE ATT&CK for ICS techniques shared across all nodes.
5. **Live Monitoring Dashboard** — a FastAPI + WebSocket server streams per-stage anomaly scores, FSM state, and MITRE technique labels to a browser dashboard in real time.

---

## Architecture

### Primary Pipeline — Federated GNN

```
Data/processed/X_train.npy  (1-second sensor windows, split by stage)
        │
        ▼ per-stage client (no raw data crosses client boundary)
 ┌─────────────────────────────────────────────────────────────┐
 │  Federated Training (main.py)                               │
 │  6 CTMASClient instances (federated/client.py)              │
 │  ├─ Local loss: stage-node reconstruction MSE               │
 │  ├─ FedProx proximal term (μ=0.01) — prevents client drift  │
 │  ├─ DP-SGD: clip(C=1.0) + N(0, σ·C/B) gradient noise       │
 │  └─ Reports accumulated ε to server each round              │
 │                  │  model parameters only (no raw data)     │
 │  FedProxByzantineStrategy (federated/server.py)             │
 │  ├─ Cosine-similarity Byzantine filter                      │
 │  └─ Weighted FedAvg on filtered updates                     │
 └────────────────┬────────────────────────────────────────────┘
                  │ global model weights
                  ▼
 ┌─────────────────────────────────────────────────────────────┐
 │  SpatioTemporalGNNAutoencoder (models/gnn_model.py)         │
 │  Per-stage 1D-CNN → GCNConv × 2 (physical water-flow graph) │
 │  → bottleneck (latent_dim=32) → GCN decoder → 1D-CNN decoder│
 │  Physical graph: P1↔P2↔P3↔P4↔P5↔P6, P3→P1, P5→P1          │
 └────────────────┬────────────────────────────────────────────┘
                  │ per-node reconstruction error
                  ▼
 ┌─────────────────────────────────────────────────────────────┐
 │  Two-Layer Anomaly Detector (detection/detector.py)          │
 │  Layer 1: per-node MSE vs. calibrated threshold (mean + 3σ)  │
 │  Layer 2: EWMA early warning on global MSE (catches slow     │
 │           low-amplitude attacks below per-node thresholds)   │
 └────────────────┬────────────────────────────────────────────┘
                  │ AnomalyReport (type, nodes, severity, EWMA)
                  ▼
 ┌─────────────────────────────────────────────────────────────┐
 │  Collaborative Threat Intelligence FSM                       │
 │  (intelligence/threat_mapper.py)                            │
 │  States: NORMAL → RECON → INTRUSION → IMPACT                │
 │  Drives: MITRE ATT&CK for ICS dynamic technique mapping     │
 │  (T0842, T0856, T0855, T0831, T0826, T0827)                 │
 └────────────────┬────────────────────────────────────────────┘
                  │ JSON threat assessment events
                  ▼
 ┌─────────────────────────────────────────────────────────────┐
 │  FastAPI + WebSocket Server  (api/main.py)                  │
 │  GET /health   GET /metadata   WS /ws/stream?model=neural   │
 └────────────────┬────────────────────────────────────────────┘
                  ▼
        frontend/index.html  (live monitoring dashboard)
```

### Centralized Supervised Baseline (XGBoost)

For comparison against the federated approach:

```
Data/merged.csv  → statistical features (408 per 60-s window) → XGBClassifier
→ hysteresis threshold → simplified FSM → WS /ws/stream?model=xgboost
```

XGBoost trains on the full merged dataset (no privacy constraints) and serves as the upper bound for a centralised, non-privacy-preserving approach. **F1=0.827, AUC-ROC=0.985** on the in-distribution test split.

---

## Privacy and Federated Learning Design

| Property | Value | Detail |
|---|---|---|
| **DP mechanism** | Gaussian (DP-SGD) | Per-client, per-step: clip + N(0, σ·C/B) noise |
| **Noise multiplier σ** | 0.3 | `DP_NOISE_MULTIPLIER` in `federated/config.py` |
| **Clipping norm C** | 1.0 | `DP_MAX_GRAD_NORM` |
| **δ** | 1e-5 | `DP_DELTA` |
| **ε accounting** | Simple composition bound | `ε ≈ √(2·steps·ln(1/δ)) / σ`, tracked per round |
| **Aggregation** | FedProx + Byzantine | Cosine-sim filter, threshold=0.5 |
| **Data shared** | Model updates only | No raw sensor readings leave any client |
| **FL rounds** | 10 | 3 local epochs per client per round |

The privacy budget (ε, δ) is logged each round and surfaced via the API `/metadata` endpoint. Each plant stage can independently audit its own ε accumulation.

---

## Threat Intelligence — FSM + MITRE ATT&CK for ICS

The FSM is driven dynamically by the anomaly detector's output — not a static sensor-prefix → T-code lookup table.

| FSM State | Trigger | MITRE Techniques |
|---|---|---|
| **NORMAL** | All clear (5 consecutive steps) | — |
| **RECON** | EWMA early warning fires, no threshold breach | T0842 Network Sniffing · T0856 Spoof Reporting Message |
| **INTRUSION** | Per-node threshold breach (isolated anomaly) | T0855 Unauthorized Command · T0831 Manipulation of Control |
| **IMPACT** | Propagating anomaly across multiple nodes | T0826 Loss of Availability · T0827 Loss of Control |

State transitions are guarded by a 5-step cooldown so transient spikes don't cause spurious downgrade. The EWMA trend (10-step window) modulates the impact probability and estimated minutes-to-impact estimates.

---

## Physical Plant Graph (SWaT)

```
P1 ──► P2 ──► P3 ──► P4 ──► P5 ──► P6
 ▲            │             │
 └────────────┘             │   (P3 backwash)
 └───────────────────────────   (P5 RO reject)
```

Each graph node = one plant stage = one federated client in training. The GNN propagates anomaly signals across the physical water-flow topology, enabling detection of downstream effects — attacks on P3 manifest as correlated anomalies at P1 (backwash loop) that a per-stage classifier would miss.

| Stage | Process | Sensors | Client ID |
|---|---|---|---|
| P1 | Raw water intake + chemical dosing | 5 | 0 |
| P2 | Chemical dosing control | 11 | 1 |
| P3 | Sand + UF filtration | 9 | 2 |
| P4 | UV + dechlorination | 9 | 3 |
| P5 | Reverse osmosis (RO) membrane | 13 | 4 |
| P6 | Permeate output | 4 | 5 |

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

The dataset is **not included in this repo** (~427 MB). Place it at `Data/merged.csv`.

**Processed splits** (generated by `Data/Data_Preprocessing.py`):

| File | Shape | Contents |
|---|---|---|
| `X_train.npy` | (117898, 60, 51) | Normal windows, stride=10s |
| `X_val.npy` | (20806, 60, 51) | Validation (normal) |
| `X_test.npy` | (54562, 60, 51) | Attack-period windows |
| `y_test.npy` | (54562,) | Ground-truth labels |

---

## Key Design Decisions

| Decision | Why |
|---|---|
| **GNN, not LSTM/GRU** | Per-sample gradient clipping (DP-SGD) requires gradient isolation per sample; GCNConv + 1D-CNN supports this, recurrent layers do not |
| **Manual DP-SGD, not Opacus** | Opacus's functorch engine cannot vmap PyG's `message_passing` (edge_index dependency); manual clip+noise achieves identical Gaussian DP mechanism |
| **FedProx over FedAvg** | Each stage has a different sensor distribution and anomaly base rate; the proximal term (μ·‖w−w_global‖²) prevents the heterogeneous clients from drifting during local epochs |
| **Byzantine-robust aggregation** | A compromised plant site could poison the global model; cosine-similarity filtering down-weights divergent updates before aggregation |
| **EWMA early warning** | Low-and-slow attacks (gradual sensor manipulation) stay below per-node thresholds for many windows; EWMA catches the cumulative drift before a threshold breach |
| **Dynamic FSM, not static lookup** | Attack campaigns evolve; the FSM state is driven by EWMA trend + cross-node correlation pattern, giving a time-aware threat assessment rather than a fixed sensor-name → T-code table |

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
│   ├── client.py               ← FedProx + DP-SGD per-stage client
│   ├── server.py               ← Byzantine-robust FedProx aggregation server
│   └── config.py               ← all FL + DP hyperparameters
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
│   └── index.html              ← live monitoring dashboard
│
├── main.py                     ← federated training entry point
├── train_centralized.py        ← single-machine GNN training (no FL)
├── train_ensemble.py           ← denoising-AE ensemble training
├── eval_ensemble.py            ← ensemble evaluation
├── offline_eval.py             ← full metric report on saved model
│
├── baseline_isolation_forest.py          ← unsupervised baseline
├── baseline_supervised_classifier.py     ← supervised baseline (processed splits)
├── baseline_supervised_from_merged.py    ← centralised XGBoost/RF/LightGBM
│
├── SETUP.md                    ← full environment setup guide
├── requirements.txt
└── ctmas_model.pt              ← saved federated model weights
```

---

## Quick Start

See **SETUP.md** for full environment setup.

```bash
# 1. Create and activate virtual environment
python3.11 -m venv .venv && source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Place merged.csv at Data/merged.csv, then preprocess
cd Data && python Data_Preprocessing.py && cd ..

# 4. Federated training (primary system)
python main.py --rounds 10        # trains 6 clients with FedProx + DP-SGD
python offline_eval.py            # full metric report on ctmas_model.pt

# 5. Run the API + open the dashboard
uvicorn api.main:app --host 0.0.0.0 --port 8000
# → open frontend/index.html in a browser
# → dashboard connects to the federated GNN model by default (?model=neural)
```

**Centralized baseline only** (no GPU required):
```bash
python baseline_supervised_from_merged.py --model xgboost
# → saves artifacts for dashboard comparison: ?model=xgboost
```

---

## Federated + DP Configuration (`federated/config.py`)

| Parameter | Value | Meaning |
|---|---|---|
| `FL_ROUNDS` | 10 | Federated communication rounds |
| `LOCAL_EPOCHS` | 3 | Local epochs per client per round |
| `MU` | 0.01 | FedProx proximal penalty |
| `BYZANTINE_THRESHOLD` | 0.5 | Cosine similarity cutoff |
| `DP_NOISE_MULTIPLIER` | 0.3 | Gaussian noise σ for DP-SGD |
| `DP_MAX_GRAD_NORM` | 1.0 | Per-sample gradient clipping norm C |
| `DP_DELTA` | 1e-5 | Target δ for (ε,δ)-DP |
| `HIDDEN_DIM` | 64 | GNN encoder width |
| `LATENT_DIM` | 32 | Bottleneck embedding size |

---

## API Endpoints

| Endpoint | Description |
|---|---|
| `GET /health?model=neural\|xgboost` | Liveness check + model metadata |
| `GET /metadata?model=neural\|xgboost` | Full config, thresholds, stage map |
| `WS /ws/stream?model=neural` | Federated GNN stream (default) |
| `WS /ws/stream?model=xgboost` | Centralised baseline stream (comparison) |

The dashboard visualises per-stage reconstruction errors, FSM campaign state, MITRE technique labels, running F1, and ground-truth labels in real time.

---

## Comparison: Federated vs. Centralised

| Property | Federated GNN (primary) | Centralised XGBoost (baseline) |
|---|---|---|
| Raw data shared | No — gradients only | Yes — full merged.csv |
| Privacy guarantee | (ε, δ)-DP per client | None |
| Poisoning resistance | Byzantine cosine filter | None |
| Topology awareness | Physical plant graph (GCN) | Feature engineering only |
| Early warning | EWMA + cross-node propagation | EWMA on score only |
| Threat intelligence | Full FSM + MITRE ATT&CK | Simplified score-ratio FSM |
| F1 (test split) | see `offline_eval.py` output | 0.827 |
| AUC-ROC | see `offline_eval.py` output | 0.985 |

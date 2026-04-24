"""FastAPI WebSocket server — replays X_test through the trained model at 1-second intervals.

Endpoints:
  GET  /health
  GET  /model/status
  WS   /ws/stream     — pushes AnomalyReport + ThreatAssessment as JSON each step
  GET  /metrics       — final evaluation metrics (F1, AUC, MTTD, FPR)
"""

from __future__ import annotations

import asyncio
import json
import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from detection.detector import AnomalyDetector
from intelligence.threat_mapper import ThreatFSM
from models.gnn_model import SpatioTemporalGNNAutoencoder

app = FastAPI(title="CTMAS", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).parent.parent
MODEL_PATH = BASE_DIR / "ctmas_model.pt"
DATA_DIR = BASE_DIR / "Data" / "processed"

_detector: AnomalyDetector | None = None
_X_test: np.ndarray | None = None
_metrics_cache: dict | None = None


def _load_artifacts() -> tuple[AnomalyDetector, np.ndarray]:
    global _detector, _X_test
    if _detector is not None and _X_test is not None:
        return _detector, _X_test

    X_val = np.load(DATA_DIR / "X_val.npy")
    X_test = np.load(DATA_DIR / "X_test.npy")
    np.nan_to_num(X_val, copy=False, nan=0.0)
    np.nan_to_num(X_test, copy=False, nan=0.0)

    model = SpatioTemporalGNNAutoencoder()
    if MODEL_PATH.exists():
        model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    else:
        raise RuntimeError(f"Trained model not found at {MODEL_PATH}. Run main.py first.")

    detector = AnomalyDetector(model)
    detector.calibrate(X_val)

    _detector = detector
    _X_test = X_test
    return detector, X_test


@app.on_event("startup")
async def startup():
    try:
        _load_artifacts()
    except RuntimeError as e:
        print(f"[WARN] Could not load model on startup: {e}")


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL_PATH.exists()}


@app.get("/model/status")
def model_status():
    return {
        "model_path": str(MODEL_PATH),
        "exists": MODEL_PATH.exists(),
        "data_dir": str(DATA_DIR),
    }


@app.get("/metrics")
def get_metrics():
    global _metrics_cache
    if _metrics_cache:
        return _metrics_cache

    from sklearn.metrics import (
        auc,
        f1_score,
        precision_score,
        recall_score,
        roc_curve,
    )

    detector, X_test = _load_artifacts()
    node_errors, global_errors = detector._batch_per_sample(X_test, batch_size=256)

    # X_test is all attack data — ground truth = all 1
    y_true = np.ones(len(X_test), dtype=int)

    # Global threshold (mean + 3std from calibration)
    thresh = float(detector.node_threshold.mean())
    y_pred = (global_errors > thresh).astype(int)

    fpr_curve, tpr_curve, _ = roc_curve(y_true, global_errors)
    roc_auc = auc(fpr_curve, tpr_curve)

    # MTTD: index of first correctly detected window
    detected_at = np.where(y_pred == 1)[0]
    mttd_s = int(detected_at[0]) * 10 if len(detected_at) else -1

    _metrics_cache = {
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "auc_roc": float(roc_auc),
        "mttd_seconds": mttd_s,
        "detection_rate": float(y_pred.mean()),
    }
    return _metrics_cache


@app.websocket("/ws/stream")
async def stream(websocket: WebSocket):
    await websocket.accept()
    try:
        detector, X_test = _load_artifacts()
    except RuntimeError as e:
        await websocket.send_text(json.dumps({"error": str(e)}))
        await websocket.close()
        return

    fsm = ThreatFSM()
    # Reset detector EWMA for clean replay
    detector._ewma = 0.0
    detector._first_step = True

    try:
        for i, window in enumerate(X_test):
            report = detector.step(window, window_idx=i)
            assessment = fsm.step(report)

            payload: dict[str, Any] = {
                "window_idx": i,
                "timestamp_s": report.timestamp_s,
                "node_errors": report.node_errors.tolist(),
                "node_anomaly": report.node_anomaly.tolist(),
                "ewma_score": report.ewma_score,
                "early_warning": report.early_warning,
                "anomaly_type": report.anomaly_type,
                "anomaly_nodes": report.anomaly_nodes,
                "severity": report.severity,
                "threat": {
                    "state": assessment.state,
                    "stage": assessment.stage,
                    "techniques": assessment.techniques,
                    "impact_probability": assessment.impact_probability,
                    "minutes_to_impact": assessment.minutes_to_impact,
                    "description": assessment.description,
                },
            }
            await websocket.send_text(json.dumps(payload))
            await asyncio.sleep(1.0)   # real-time 1s replay
    except WebSocketDisconnect:
        pass

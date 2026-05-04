"""FastAPI WebSocket server for CTMAS model replays.

Supports:
  - neural: existing GNN/ensemble detector over Data/processed/X_test.npy
  - xgboost: saved supervised replay artifacts from baseline_supervised_from_merged.py

Endpoints:
  GET  /health?model=neural|xgboost
  GET  /metadata?model=neural|xgboost
  WS   /ws/stream?model=neural|xgboost
"""

from __future__ import annotations

import asyncio
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch
from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from detection.detector import AnomalyDetector
from intelligence.threat_mapper import ThreatFSM
from models.gnn_model import SpatioTemporalGNNAutoencoder

app = FastAPI(title="CTMAS", version="3.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "Data" / "processed"
SINGLE_MODEL_PATH = BASE_DIR / "ctmas_model.pt"
XGBOOST_DIR = BASE_DIR / "artifacts" / "supervised_from_merged" / "xgboost"
FRONTEND_HTML = BASE_DIR / "frontend" / "index.html"

_CACHE: dict[str, dict[str, Any]] = {}


def _normalize_model_key(model: str | None) -> str:
    key = (model or "neural").strip().lower()
    return "xgboost" if key == "xgboost" else "neural"


def _load_neural_artifacts() -> dict[str, Any]:
    if "neural" in _CACHE:
        return _CACHE["neural"]

    X_val = np.load(DATA_DIR / "X_val.npy")
    X_test = np.load(DATA_DIR / "X_test.npy")
    y_test = np.load(DATA_DIR / "y_test.npy")
    np.nan_to_num(X_val, copy=False, nan=0.0)
    np.nan_to_num(X_test, copy=False, nan=0.0)
    with open(DATA_DIR / "metadata.pkl", "rb") as f:
        meta = pickle.load(f)

    ensemble_paths = sorted(BASE_DIR.glob("ctmas_ensemble_*.pt"))
    if ensemble_paths:
        models = []
        for p in ensemble_paths:
            m = SpatioTemporalGNNAutoencoder()
            m.load_state_dict(torch.load(p, map_location="cpu"))
            models.append(m)
        print(f"[Artifacts] loaded ensemble of {len(models)} members")
        detector = AnomalyDetector(models)
        n_models = len(models)
        display_name = f"GNN Ensemble x{len(models)}"
    elif SINGLE_MODEL_PATH.exists():
        m = SpatioTemporalGNNAutoencoder()
        m.load_state_dict(torch.load(SINGLE_MODEL_PATH, map_location="cpu"))
        print(f"[Artifacts] loaded single model {SINGLE_MODEL_PATH.name}")
        detector = AnomalyDetector(m)
        n_models = 1
        display_name = "GNN Autoencoder"
    else:
        raise RuntimeError(f"No model found. Expected {SINGLE_MODEL_PATH} or ctmas_ensemble_*.pt")

    detector.calibrate(X_val)
    artifacts = {
        "model_kind": "neural",
        "display_name": display_name,
        "ensemble_size": n_models,
        "detector": detector,
        "X_test": X_test,
        "y_test": y_test,
        "meta": meta,
        "score_label": "Detection Score (Z-sum)",
    }
    _CACHE["neural"] = artifacts
    return artifacts


def _load_xgboost_artifacts() -> dict[str, Any]:
    if "xgboost" in _CACHE:
        return _CACHE["xgboost"]
    if not XGBOOST_DIR.exists():
        raise RuntimeError(
            "No XGBoost frontend artifacts found. Run baseline_supervised_from_merged.py --model xgboost first."
        )

    with open(XGBOOST_DIR / "metadata.pkl", "rb") as f:
        meta = pickle.load(f)
    artifacts = {
        "model_kind": "xgboost",
        "display_name": meta.get("display_name", "XGBoost"),
        "ensemble_size": 1,
        "X_test": np.load(XGBOOST_DIR / "X_test.npy"),
        "y_test": np.load(XGBOOST_DIR / "y_test.npy"),
        "test_score": np.load(XGBOOST_DIR / "test_score.npy"),
        "stage_scores": np.load(XGBOOST_DIR / "stage_scores_test.npy"),
        "meta": meta,
        "score_label": meta.get("score_label", "Attack Probability"),
    }
    _CACHE["xgboost"] = artifacts
    print(f"[Artifacts] loaded XGBoost replay bundle from {XGBOOST_DIR}")
    return artifacts


def _load_artifacts(model: str | None) -> dict[str, Any]:
    model_key = _normalize_model_key(model)
    return _load_xgboost_artifacts() if model_key == "xgboost" else _load_neural_artifacts()


def _next_attack_event_idx(current_window: int, events: list[tuple[int, int]]) -> int | None:
    for s, _ in events:
        if s > current_window:
            return int(s)
    return None


def _detection_label(predicted: bool, ground_truth: bool) -> str:
    if predicted and ground_truth:
        return "TP"
    if predicted and not ground_truth:
        return "FP"
    if not predicted and ground_truth:
        return "FN"
    return "TN"


def _build_current_event(i: int, events: list[tuple[int, int]]) -> dict[str, int] | None:
    for ei, (s, e) in enumerate(events):
        if s <= i <= e:
            return {"id": ei, "start": s, "end": e, "duration_windows": e - s + 1}
    return None


def _base_metadata(artifacts: dict[str, Any]) -> dict[str, Any]:
    meta = artifacts["meta"]
    y_test = artifacts["y_test"]
    return {
        "model_kind": artifacts["model_kind"],
        "display_name": artifacts["display_name"],
        "score_label": artifacts.get("score_label"),
        "n_test_windows": int(len(artifacts["X_test"])),
        "test_stride_s": int(meta.get("test_stride", 10)),
        "attack_events": [{"start": int(s), "end": int(e)} for s, e in meta.get("attack_events_window", [])],
        "n_attack_windows": int(y_test.sum()),
        "n_normal_windows": int((1 - y_test).sum()),
        "ensemble_size": int(artifacts.get("ensemble_size", 1)),
    }


@app.on_event("startup")
async def startup():
    try:
        _load_neural_artifacts()
    except RuntimeError as e:
        print(f"[WARN] {e}")
    try:
        _load_xgboost_artifacts()
    except RuntimeError as e:
        print(f"[WARN] {e}")


@app.get("/")
def root():
    return FileResponse(FRONTEND_HTML)


@app.get("/health")
def health(model: str = Query("neural")):
    try:
        artifacts = _load_artifacts(model)
    except RuntimeError as e:
        return {"status": "error", "model_kind": _normalize_model_key(model), "error": str(e)}
    return {
        "status": "ok",
        "model_kind": artifacts["model_kind"],
        "display_name": artifacts["display_name"],
        "ensemble_size": artifacts.get("ensemble_size", 1),
        "model_loaded": True,
    }


@app.get("/metadata")
def get_metadata(model: str = Query("neural")):
    artifacts = _load_artifacts(model)
    payload = _base_metadata(artifacts)
    if artifacts["model_kind"] == "xgboost":
        meta = artifacts["meta"]
        payload.update(
            {
                "node_threshold": meta.get("stage_thresholds"),
                "z_threshold": float(meta.get("score_threshold", 0.5)),
                "ewma_threshold": float(meta.get("score_low_threshold", meta.get("score_threshold", 0.5))),
            }
        )
    else:
        detector = artifacts["detector"]
        payload.update(
            {
                "node_threshold": detector.node_threshold.tolist(),
                "z_threshold": float(detector.z_threshold),
                "ewma_threshold": float(detector.ewma_threshold),
            }
        )
    return payload


async def _recv_loop(websocket: WebSocket, state: dict[str, Any]) -> None:
    try:
        while state["running"]:
            msg = await websocket.receive_text()
            try:
                cmd = json.loads(msg)
            except json.JSONDecodeError:
                continue
            c = cmd.get("cmd")
            if c == "set_speed":
                state["speed"] = max(1, min(100, int(cmd.get("value", 1))))
            elif c == "pause":
                state["paused"] = True
            elif c == "resume":
                state["paused"] = False
            elif c == "jump_to":
                state["jump_to"] = int(cmd.get("value", 0))
            elif c == "jump_next_attack":
                state["jump_to"] = "next_attack"
    except WebSocketDisconnect:
        state["running"] = False


async def _stream_neural(websocket: WebSocket, artifacts: dict[str, Any]) -> None:
    detector = artifacts["detector"]
    X_test = artifacts["X_test"]
    y_test = artifacts["y_test"]
    meta = artifacts["meta"]
    events = [(int(s), int(e)) for s, e in meta.get("attack_events_window", [])]

    fsm = ThreatFSM()
    detector._ewma = 0.0
    detector._first_step = True
    state = {"speed": 10, "paused": False, "jump_to": None, "running": True}
    recv_task = asyncio.create_task(_recv_loop(websocket, state))
    counters = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}

    def _f1() -> float:
        tp, fp, fn = counters["TP"], counters["FP"], counters["FN"]
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        return 2 * prec * rec / max(1e-8, prec + rec)

    i = 0
    try:
        while i < len(X_test) and state["running"]:
            if state["jump_to"] is not None:
                if state["jump_to"] == "next_attack":
                    nxt = _next_attack_event_idx(i, events)
                    if nxt is not None:
                        i = nxt
                        detector._ewma = 0.0
                        detector._first_step = True
                else:
                    i = max(0, min(len(X_test) - 1, int(state["jump_to"])))
                    detector._ewma = 0.0
                    detector._first_step = True
                state["jump_to"] = None

            if state["paused"]:
                await asyncio.sleep(0.1)
                continue

            window = X_test[i]
            report = detector.step(window, window_idx=i)
            assessment = fsm.step(report)
            z = (report.node_errors - detector.node_mean) / detector.node_std
            z_sum = float(np.delete(z, [1, 2]).sum())
            predicted = bool(z_sum > detector.z_threshold)
            ground_truth = bool(y_test[i] == 1)
            label = _detection_label(predicted, ground_truth)
            counters[label] += 1

            payload = {
                "model_kind": "neural",
                "model_name": artifacts["display_name"],
                "score_label": artifacts["score_label"],
                "window_idx": i,
                "timestamp_s": report.timestamp_s,
                "n_test_windows": len(X_test),
                "node_errors": report.node_errors.tolist(),
                "node_anomaly": report.node_anomaly.tolist(),
                "node_threshold": detector.node_threshold.tolist(),
                "ewma_score": report.ewma_score,
                "ewma_threshold": float(detector.ewma_threshold),
                "z_score_sum": z_sum,
                "z_threshold": float(detector.z_threshold),
                "predicted_attack": predicted,
                "early_warning": report.early_warning,
                "anomaly_type": report.anomaly_type,
                "anomaly_nodes": report.anomaly_nodes,
                "severity": report.severity,
                "ground_truth_attack": ground_truth,
                "detection_label": label,
                "current_event": _build_current_event(i, events),
                "counters": dict(counters),
                "running_f1": _f1(),
                "threat": {
                    "state": assessment.state,
                    "stage": assessment.stage,
                    "techniques": assessment.techniques,
                    "impact_probability": assessment.impact_probability,
                    "minutes_to_impact": assessment.minutes_to_impact,
                    "description": assessment.description,
                },
                "speed": state["speed"],
            }
            await websocket.send_text(json.dumps(payload))
            i += 1
            await asyncio.sleep(max(0.005, 1.0 / state["speed"]))
    finally:
        state["running"] = False
        recv_task.cancel()


def _xgb_threat(score: float, threshold: float, predicted: bool) -> dict[str, Any]:
    ratio = score / max(threshold, 1e-8)
    if predicted:
        return {
            "state": "IMPACT" if ratio > 2.0 else "INTRUSION",
            "stage": 3 if ratio > 2.0 else 2,
            "techniques": [],
            "impact_probability": float(min(0.99, 0.55 + 0.2 * min(ratio, 2.0))),
            "minutes_to_impact": 0.0 if ratio > 2.0 else 1.0,
            "description": "Supervised classifier alarm triggered.",
        }
    if ratio > 0.6:
        return {
            "state": "RECON",
            "stage": 1,
            "techniques": [],
            "impact_probability": float(min(0.49, 0.2 + 0.3 * ratio)),
            "minutes_to_impact": None,
            "description": "Elevated supervised attack probability.",
        }
    return {
        "state": "NORMAL",
        "stage": 0,
        "techniques": [],
        "impact_probability": float(max(0.0, score)),
        "minutes_to_impact": None,
        "description": "No elevated supervised attack probability.",
    }


async def _stream_xgboost(websocket: WebSocket, artifacts: dict[str, Any]) -> None:
    X_test = artifacts["X_test"]
    y_test = artifacts["y_test"]
    test_score = artifacts["test_score"]
    stage_scores = artifacts["stage_scores"]
    meta = artifacts["meta"]
    threshold = float(meta.get("score_threshold", 0.5))
    low_threshold = float(meta.get("score_low_threshold", threshold))
    stage_threshold = np.array(meta.get("stage_thresholds", [1.0] * 6), dtype=np.float32)
    events = [(int(s), int(e)) for s, e in meta.get("attack_events_window", [])]
    stage_names = list(meta.get("stage_map", {}).keys()) or ["P1", "P2", "P3", "P4", "P5", "P6"]

    state = {"speed": 10, "paused": False, "jump_to": None, "running": True}
    recv_task = asyncio.create_task(_recv_loop(websocket, state))
    counters = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}
    ewma = 0.0
    first_step = True
    alarm_active = False

    def _f1() -> float:
        tp, fp, fn = counters["TP"], counters["FP"], counters["FN"]
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        return 2 * prec * rec / max(1e-8, prec + rec)

    i = 0
    try:
        while i < len(X_test) and state["running"]:
            if state["jump_to"] is not None:
                if state["jump_to"] == "next_attack":
                    nxt = _next_attack_event_idx(i, events)
                    if nxt is not None:
                        i = nxt
                        ewma = 0.0
                        first_step = True
                        alarm_active = False
                else:
                    i = max(0, min(len(X_test) - 1, int(state["jump_to"])))
                    ewma = 0.0
                    first_step = True
                    alarm_active = False
                state["jump_to"] = None

            if state["paused"]:
                await asyncio.sleep(0.1)
                continue

            score = float(test_score[i])
            stage_vec = stage_scores[i]
            if first_step:
                ewma = score
                first_step = False
            else:
                ewma = 0.3 * score + 0.7 * ewma

            if not alarm_active and score >= threshold:
                alarm_active = True
            elif alarm_active and score < low_threshold:
                alarm_active = False
            predicted = bool(alarm_active)
            ground_truth = bool(y_test[i] == 1)
            label = _detection_label(predicted, ground_truth)
            counters[label] += 1
            node_anomaly = stage_vec > stage_threshold
            anomaly_nodes = [stage_names[idx] for idx, is_anom in enumerate(node_anomaly) if is_anom]
            severity = 3 if predicted else 2 if score >= threshold * 0.6 else 1 if score >= threshold * 0.3 else 0
            assessment = _xgb_threat(score, threshold, predicted)

            payload = {
                "model_kind": "xgboost",
                "model_name": artifacts["display_name"],
                "score_label": artifacts["score_label"],
                "window_idx": i,
                "timestamp_s": i * int(meta.get("test_stride", 10)),
                "n_test_windows": len(X_test),
                "node_errors": stage_vec.tolist(),
                "node_anomaly": node_anomaly.tolist(),
                "node_threshold": stage_threshold.tolist(),
                "ewma_score": float(ewma),
                "ewma_threshold": low_threshold,
                "z_score_sum": score,
                "z_threshold": threshold,
                "predicted_attack": predicted,
                "early_warning": bool(score >= threshold * 0.6),
                "anomaly_type": "XGBoostAlarm" if predicted else "ElevatedRisk" if severity > 0 else "None",
                "anomaly_nodes": anomaly_nodes,
                "severity": severity,
                "ground_truth_attack": ground_truth,
                "detection_label": label,
                "current_event": _build_current_event(i, events),
                "counters": dict(counters),
                "running_f1": _f1(),
                "threat": assessment,
                "speed": state["speed"],
            }
            await websocket.send_text(json.dumps(payload))
            i += 1
            await asyncio.sleep(max(0.005, 1.0 / state["speed"]))
    finally:
        state["running"] = False
        recv_task.cancel()


@app.websocket("/ws/stream")
async def stream(websocket: WebSocket):
    await websocket.accept()
    model_key = _normalize_model_key(websocket.query_params.get("model"))
    try:
        artifacts = _load_artifacts(model_key)
    except RuntimeError as e:
        await websocket.send_text(json.dumps({"error": str(e)}))
        await websocket.close()
        return

    try:
        if model_key == "xgboost":
            await _stream_xgboost(websocket, artifacts)
        else:
            await _stream_neural(websocket, artifacts)
    except WebSocketDisconnect:
        pass

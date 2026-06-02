"""
FastAPI serving for HeteroPriceForecaster.

GET  /health             -> model status
GET  /metrics            -> test metrics from hetero_metrics_clean.json
POST /predict            -> day-ahead price forecast for DK1 or DK2
POST /pipeline/run       -> trigger the full MLOps pipeline as a background task
GET  /pipeline/status    -> return last pipeline run status
GET  /monitoring/report  -> return rolling MAE and drift status
POST /monitor/log-actual -> log the actual observed price for MAE tracking
"""
import sys
import json
import math
import pickle
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_SRC = Path(__file__).parent
_ARTIFACTS_HETERO = _SRC / "artifacts_hetero"
_GRAPH_DIR = _SRC / "data" / "graphs_hetero"
_METRICS_FILE = _ARTIFACTS_HETERO / "hetero_metrics_clean.json"
_SCALER_FILE = _GRAPH_DIR / "hetero_scalers.pkl"
_CKPT_FILE = _ARTIFACTS_HETERO / "best_hetero_model.pt"

app = FastAPI(
    title="Electricity Price Forecasting API",
    description="HeteroPriceForecaster (HeteroSAGE) — DK1 & DK2 zones",
    version="1.0.0",
)

# Track pipeline state between background runs
_pipeline_status: dict = {"status": "idle"}

# ---------------------------------------------------------------------------
# Startup: load model + scalers
# ---------------------------------------------------------------------------
_model = None
_x_override = None
_scalers = None
_hetero_data = None
_num_hours = None
_device = None


@app.on_event("startup")
async def load_model():
    global _model, _x_override, _scalers, _hetero_data, _num_hours, _device

    try:
        import torch
        from hetero_config import DEVICE, GRAPH_DIR, ARTIFACTS_DIR
        from hetero_models import load_hetero_model

        _device = DEVICE

        # Load graph for metadata + num_hours
        data = torch.load(GRAPH_DIR / "hetero_graph.pt", map_location=DEVICE, weights_only=False)
        _num_hours = int(data["hour"].num_hours_per_zone)
        _hetero_data = data

        # Load model
        ckpt = ARTIFACTS_DIR / "best_hetero_model.pt"
        _model, _x_override = load_hetero_model(data, ckpt, DEVICE)
        _model.eval()

        # Load scalers
        scaler_path = GRAPH_DIR / "hetero_scalers.pkl"
        with open(scaler_path, "rb") as f:
            _scalers = pickle.load(f)

        print(f"[API] Model loaded — {_num_hours} hours/zone, device={DEVICE}")

    except Exception as exc:
        print(f"[API] WARNING: model load failed — {exc}")
        print("[API] /predict will return 503 until model is available.")


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
class PredictRequest(BaseModel):
    """
    Day-ahead feature vector. All price lags are >= 24h (known at gate closure)
    and weather is the day-ahead forecast for the target hour.
    """
    zone: str = Field(..., description="DK1 or DK2")
    lag_24h: float = Field(..., description="Price same hour previous day (DKK)")
    lag_48h: float = Field(..., description="Price same hour two days ago (DKK)")
    lag_168h: float = Field(..., description="Price same hour previous week (DKK)")
    rolling_24h_mean: float = Field(..., description="Mean price 24-47h before target (DKK)")
    rolling_24h_std: float = Field(..., description="Std price 24-47h before target (DKK)")
    temperature_c: float
    wind_speed_ms: float
    cloud_cover_pct: float
    humidity_pct: float
    hour_of_day: int = Field(..., ge=0, le=23)
    day_of_week: int = Field(default=0, ge=0, le=6)


class PredictResponse(BaseModel):
    zone: str
    predicted_price_dkk: float


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok", "model": "HeteroSAGE", "zones": ["DK1", "DK2"]}


@app.get("/metrics")
def get_metrics():
    if not _METRICS_FILE.exists():
        raise HTTPException(status_code=404, detail="Metrics file not found")
    with open(_METRICS_FILE) as f:
        return json.load(f)


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if _model is None or _scalers is None or _hetero_data is None:
        raise HTTPException(status_code=503, detail="Model not loaded — check startup logs")

    zone = req.zone.upper()
    if zone not in ("DK1", "DK2"):
        raise HTTPException(status_code=422, detail="zone must be DK1 or DK2")

    try:
        import torch

        # Build 13-feature vector (same order as hetero_graph_builder extract_features + cyclical)
        # [lag_24h, lag_48h, lag_168h, roll24_mean, roll24_std, temp, wind, cloud, humidity,
        #  hour_sin, hour_cos, week_sin, week_cos]
        hour_sin = math.sin(2 * math.pi * req.hour_of_day / 24.0)
        hour_cos = math.cos(2 * math.pi * req.hour_of_day / 24.0)
        week_sin = math.sin(2 * math.pi * req.day_of_week / 7.0)
        week_cos = math.cos(2 * math.pi * req.day_of_week / 7.0)

        raw_features = np.array([[
            req.lag_24h,
            req.lag_48h,
            req.lag_168h,
            req.rolling_24h_mean,
            req.rolling_24h_std,
            req.temperature_c,
            req.wind_speed_ms,
            req.cloud_cover_pct,
            req.humidity_pct,
            hour_sin,
            hour_cos,
            week_sin,
            week_cos,
        ]], dtype=np.float32)  # shape (1, 13)

        feat_scaler = _scalers["feature_scaler"]
        scaled_features = feat_scaler.transform(raw_features)  # (1, 13)

        # Clone x_dict, replace last test node for the requested zone
        # Zone layout: DK1 = [0 : num_hours], DK2 = [num_hours : 2*num_hours]
        x_dict = {k: v.clone() for k, v in _hetero_data.x_dict.items()}

        if _x_override is not None:
            x_dict.update(_x_override)

        zone_offset = 0 if zone == "DK1" else _num_hours
        # Replace the last node in the zone with the request features
        node_idx = zone_offset + _num_hours - 1
        x_dict["hour"][node_idx] = torch.tensor(scaled_features[0], dtype=torch.float32).to(_device)

        ei = {k: v.to(_device) for k, v in _hetero_data.edge_index_dict.items()}
        x_dict = {k: v.to(_device) for k, v in x_dict.items()}

        with torch.no_grad():
            out = _model(x_dict, ei, num_hours=_num_hours).view(-1)

        predicted_scaled = out[node_idx].item()

        # Inverse-transform: the model outputs raw DKK (hetero model is trained on raw y)
        # (If target_scaler was used, uncomment the line below)
        # target_scaler = _scalers["target_scaler"]
        # predicted_dkk = float(target_scaler.inverse_transform([[predicted_scaled]])[0][0])
        predicted_dkk = float(predicted_scaled)

        return PredictResponse(zone=zone, predicted_price_dkk=predicted_dkk)

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Pipeline & monitoring endpoints
# ---------------------------------------------------------------------------

def _run_pipeline_task(force_rebuild: bool):
    """Background task: run the full pipeline and update _pipeline_status."""
    global _pipeline_status
    _pipeline_status["status"] = "running"
    _pipeline_status["started_at"] = __import__("datetime").datetime.utcnow().isoformat()
    try:
        from pipeline import run_pipeline
        result = run_pipeline(force_rebuild_graph=force_rebuild)
        _pipeline_status = {
            "status": result.status,
            "finished_at": __import__("datetime").datetime.utcnow().isoformat(),
            "stages_completed": result.stages_completed,
            "metrics": result.metrics,
            "improved": result.improved,
            "errors": result.errors,
            "duration_seconds": result.duration_seconds,
        }
    except Exception as exc:
        _pipeline_status = {
            "status": "failed",
            "error": str(exc),
            "finished_at": __import__("datetime").datetime.utcnow().isoformat(),
        }


@app.post("/pipeline/run")
async def trigger_pipeline(
    background_tasks: BackgroundTasks,
    force_rebuild: bool = False,
):
    """Trigger the full MLOps pipeline as a background task. Returns immediately."""
    global _pipeline_status
    if _pipeline_status.get("status") == "running":
        return {"status": "already_running", "message": "Pipeline is already in progress"}
    _pipeline_status["status"] = "running"
    background_tasks.add_task(_run_pipeline_task, force_rebuild)
    return {"status": "started", "message": "Pipeline running in background"}


@app.get("/pipeline/status")
def pipeline_status():
    """Return last pipeline run status."""
    try:
        from pipeline import get_pipeline_status
        persisted = get_pipeline_status()
        # Merge with in-memory status (running state is only in memory)
        if _pipeline_status.get("status") == "running":
            return _pipeline_status
        return persisted
    except Exception as exc:
        return {"status": "unknown", "error": str(exc)}


@app.get("/monitoring/report")
def monitoring_report():
    """Return rolling MAE and drift status."""
    try:
        from monitoring import get_monitoring_report
        return get_monitoring_report()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/monitor/log-actual")
def log_actual_price(zone: str, timestamp: str, actual_price_dkk: float):
    """Log the actual observed price for a previous prediction (enables MAE tracking)."""
    try:
        from monitoring import log_prediction
        log_prediction(zone=zone, predicted_dkk=None, features={}, actual_dkk=actual_price_dkk)
        return {"status": "logged", "zone": zone, "timestamp": timestamp, "actual_price_dkk": actual_price_dkk}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

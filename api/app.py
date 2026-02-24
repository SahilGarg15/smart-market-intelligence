"""
api/app.py — Smart Market Intelligence Platform v2

Upgraded FastAPI application.

New in v2 (vs root app.py):
  • Loads best_model.pkl + metadata.json at startup
  • Measures and returns per-request latency
  • Logs every prediction to Neon PostgreSQL via db_logger
  • /drift endpoint — compares training baseline vs recent rolling mean
  • /benchmark endpoint — returns benchmark_results.json
  • /health endpoint — DB ping + model version

Architecture:
  ┌──────────────────────────────────────────────────┐
  │  Client  →  FastAPI  →  best_model.pkl           │
  │                      →  db_logger.log_prediction │
  │                      →  /drift (monitoring)      │
  └──────────────────────────────────────────────────┘
"""

import json
import os
import time
import warnings
from contextlib import asynccontextmanager
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Global application state (loaded once at startup)
# ---------------------------------------------------------------------------
STATE: dict = {
    "model":          None,
    "metadata":       None,
    "training_mean":  None,  # baseline for drift detection
    "feature_cols":   None,
}

MODEL_PATH    = "models/best_model.pkl"
METADATA_PATH = "models/metadata.json"
DAILY_CSV     = "data/daily_sales.csv"


# ---------------------------------------------------------------------------
# Lifespan — startup / shutdown
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model artefacts once at startup; create DB tables."""
    print("[Startup] Loading model artefacts...")

    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(
            f"Model not found at {MODEL_PATH}. "
            "Run the training pipeline first: python run_pipeline.py"
        )

    STATE["model"] = joblib.load(MODEL_PATH)
    print(f"  [OK] Model loaded from {MODEL_PATH}")

    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH) as fh:
            STATE["metadata"] = json.load(fh)
        STATE["feature_cols"] = STATE["metadata"].get("feature_names", [])
        print(f"  [OK] Metadata loaded. Model: {STATE['metadata'].get('model_name')}")
    else:
        STATE["feature_cols"] = []

    # Training baseline for drift monitoring
    if os.path.exists(DAILY_CSV):
        df_train = pd.read_csv(DAILY_CSV)
        STATE["training_mean"] = float(df_train["Sales"].mean())
        print(f"  [OK] Training mean Sales = {STATE['training_mean']:,.2f}")

    # DB — table creation (idempotent)
    try:
        from db.db_models import create_tables
        from db.db_config import ping_db
        create_tables()
        if ping_db():
            print("  [OK] Database connected")
        else:
            print("  [WARN] DB ping failed - logging disabled")
    except Exception as exc:
        print(f"  [WARN] DB setup skipped: {exc}")

    yield  # Application runs
    print("[Shutdown] Cleaning up...")


# ---------------------------------------------------------------------------
# App definition
# ---------------------------------------------------------------------------

app = FastAPI(
    title       = "Smart Market Intelligence Platform v2",
    description = "ML forecasting API with benchmarking, monitoring, and cloud persistence.",
    version     = "2.0.0",
    lifespan    = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class ForecastRequest(BaseModel):
    days: int = Field(default=30, ge=1, le=365, description="Days to forecast ahead")


class ForecastResponse(BaseModel):
    model_version:     str
    forecast_days:     int
    forecast_values:   list[float]
    avg_prediction:    float
    request_latency_ms: float
    generated_at:      str


class DriftResponse(BaseModel):
    training_mean:    float
    rolling_mean_50:  float
    drift_pct:        float
    drift_flag:       bool
    threshold_pct:    float


# ---------------------------------------------------------------------------
# Utility: build a synthetic feature row for N-day-ahead inference
# ---------------------------------------------------------------------------

def _build_inference_features(
    days_ahead: int,
    feature_cols: list,
) -> np.ndarray:
    """
    Build a feature matrix for `days_ahead` future days.

    We cannot know lag/rolling features for future dates, so we use a
    sensible strategy:
      • Calendar features computed exactly.
      • Lag / rolling features approximated from the last known daily CSV.
      • For missing lag features, median imputation (conservative estimate).

    This is clearly documented so that future engineers can replace the
    approximation with a proper autoregressive rollout.
    """
    future_dates = pd.date_range(start=datetime.today(), periods=days_ahead, freq="D")
    rows = []

    # Best-effort: load last known values for lag/rolling features
    last_lags    = {}
    last_rolling = {}
    last_context = {}

    if os.path.exists(DAILY_CSV):
        df_hist = pd.read_csv(DAILY_CSV, parse_dates=["Date"])
        df_hist = df_hist.sort_values("Date")
        if len(df_hist) >= 30:
            last_lags = {
                "Sales_Lag1":  df_hist["Sales"].iloc[-1],
                "Sales_Lag7":  df_hist["Sales"].iloc[-7],
                "Sales_Lag14": df_hist["Sales"].iloc[-14],
            }
            last_rolling = {
                "Sales_MA7":  df_hist["Sales"].iloc[-7:].mean(),
                "Sales_MA30": df_hist["Sales"].iloc[-30:].mean(),
            }
            last_context = {
                "NumTransactions": df_hist["NumTransactions"].median(),
                "NumCustomers":    df_hist["NumCustomers"].median(),
                "TotalQuantity":   df_hist["TotalQuantity"].median(),
            }

    for d in future_dates:
        row = {}
        for col in feature_cols:
            if col == "Year":         row[col] = d.year
            elif col == "Month":      row[col] = d.month
            elif col == "Day":        row[col] = d.day
            elif col == "DayOfWeek":  row[col] = d.dayofweek
            elif col == "Quarter":    row[col] = d.quarter
            elif col == "DayOfYear":  row[col] = d.dayofyear
            elif col == "WeekOfYear": row[col] = d.isocalendar().week
            elif col == "IsWeekend":  row[col] = int(d.dayofweek >= 5)
            elif col in last_lags:    row[col] = last_lags[col]
            elif col in last_rolling: row[col] = last_rolling[col]
            elif col in last_context: row[col] = last_context[col]
            else:                     row[col] = 0.0
        rows.append(row)

    df_future = pd.DataFrame(rows, columns=feature_cols)
    return df_future.values


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    """Liveness check — returns model version and DB status."""
    model_version = "not_loaded"
    if STATE["metadata"]:
        model_version = (
            f"v2-{STATE['metadata'].get('model_name')}-"
            f"{STATE['metadata'].get('training_date', '')[:10]}"
        )
    try:
        from db.db_config import ping_db
        db_ok = ping_db()
    except Exception:
        db_ok = False

    return {
        "status":        "ok",
        "model_version": model_version,
        "db_connected":  db_ok,
        "api_version":   "2.0.0",
    }


@app.post("/predict", response_model=ForecastResponse)
def predict(request: ForecastRequest):
    """
    Generate a multi-day sales forecast.

    Returns:
      - forecast_values: list of daily predictions (£ total sales)
      - avg_prediction:  mean of the forecast (KPI summary)
      - request_latency_ms: end-to-end latency in milliseconds
    """
    t_start = time.time()

    model = STATE["model"]
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Retry after startup.")

    feature_cols = STATE["feature_cols"]
    if not feature_cols:
        raise HTTPException(status_code=503, detail="Feature columns not configured.")

    try:
        X = _build_inference_features(request.days, feature_cols)
        predictions = model.predict(X).tolist()
        # Floor at 0 — negative revenue is not meaningful
        predictions = [max(0.0, round(v, 2)) for v in predictions]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")

    avg_pred   = round(float(np.mean(predictions)), 2)
    latency_ms = round((time.time() - t_start) * 1000, 3)

    # Build model version tag
    meta = STATE["metadata"] or {}
    model_version = (
        f"v2-{meta.get('model_name', 'unknown')}-"
        f"{meta.get('training_date', '')[:10]}"
    )

    # Non-blocking DB log
    try:
        from db.db_logger import log_prediction
        log_prediction(
            forecast_days   = request.days,
            avg_prediction  = avg_pred,
            model_version   = model_version,
            latency_ms      = latency_ms,
            forecast_values = predictions,
        )
    except Exception as exc:
        print(f"[API] DB log failed (non-fatal): {exc}")

    return ForecastResponse(
        model_version      = model_version,
        forecast_days      = request.days,
        forecast_values    = predictions,
        avg_prediction     = avg_pred,
        request_latency_ms = latency_ms,
        generated_at       = datetime.utcnow().isoformat(),
    )


@app.get("/drift", response_model=DriftResponse)
def drift_monitor(
    window:    int = Query(default=50,  ge=5, le=500, description="Rolling window size"),
    threshold: float = Query(default=20.0, ge=1.0, le=100.0, description="Drift threshold %"),
):
    """
    Detect data/concept drift by comparing the training mean Sales to the
    rolling mean of recent API predictions.

    Why this approach?
      The simplest reliable drift signal for an MVP: if the model is being
      queried for periods where average predicted sales are significantly
      different from the training distribution mean, the model may no longer
      be representative. A proper drift monitor would use PSI or KS-test on
      feature distributions — this is the v2 baseline with minimal overhead.

    Drift flag: True when |drift_pct| > threshold
    """
    training_mean = STATE.get("training_mean")
    if training_mean is None:
        raise HTTPException(status_code=503, detail="Training baseline not loaded.")

    try:
        from db.db_logger import get_recent_predictions
        recent = get_recent_predictions(limit=window)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Could not query logs: {exc}")

    if len(recent) < 5:
        return DriftResponse(
            training_mean   = round(training_mean, 2),
            rolling_mean_50 = round(training_mean, 2),
            drift_pct       = 0.0,
            drift_flag      = False,
            threshold_pct   = threshold,
        )

    rolling_mean = float(np.mean([r["avg_prediction"] for r in recent]))
    drift_pct    = ((rolling_mean - training_mean) / training_mean) * 100

    return DriftResponse(
        training_mean   = round(training_mean, 2),
        rolling_mean_50 = round(rolling_mean, 2),
        drift_pct       = round(drift_pct, 2),
        drift_flag      = abs(drift_pct) > threshold,
        threshold_pct   = threshold,
    )


@app.get("/benchmark")
def get_benchmark():
    """Return the stored benchmark comparison results (from benchmark_results.json)."""
    path = "models/benchmark_results.json"
    if not os.path.exists(path):
        raise HTTPException(
            status_code=404,
            detail="Benchmark results not found. Run the training pipeline first.",
        )
    with open(path) as fh:
        return json.load(fh)


@app.get("/metadata")
def get_metadata():
    """Return the model metadata (version, metrics, hyperparameters, features)."""
    if not STATE["metadata"]:
        raise HTTPException(status_code=404, detail="Metadata not loaded.")
    return STATE["metadata"]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.app:app", host="0.0.0.0", port=8000, reload=True)

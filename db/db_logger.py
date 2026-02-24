"""
db/db_logger.py

High-level prediction logging interface — v2

Abstracts the SQLAlchemy session from the API layer so that:
  - API code stays clean (no ORM imports in app.py)
  - The logging logic can be unit-tested independently
  - Switching to a different backend only requires changing this file

The public interface is a single function: log_prediction()
"""

import json
from datetime import datetime

from sqlalchemy.orm import Session

from db.db_config import SessionLocal
from db.db_models import PredictionLog


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def log_prediction(
    forecast_days:     int,
    avg_prediction:    float,
    model_version:     str,
    latency_ms:        float,
    forecast_values:   list | None = None,
    db:                Session | None = None,
) -> PredictionLog | None:
    """
    Persist one prediction event to the database.

    Parameters
    ----------
    forecast_days    : number of days requested by the client
    avg_prediction   : mean of the forecast array (summary metric)
    model_version    : e.g. "v2-RandomForest-2024-01-15"
    latency_ms       : end-to-end request duration in milliseconds
    forecast_values  : optional full forecast list (stored as JSON text)
    db               : optional external Session (useful in FastAPI Depends);
                       if None, a fresh session is created and closed here

    Returns
    -------
    PredictionLog ORM instance on success, None on failure.

    Why log failures silently?
      A DB write failure must NOT crash the prediction endpoint — forecasts
      are the core value, logging is secondary. Errors are printed for ops
      visibility but do not propagate to the client.
    """
    own_session = db is None
    if own_session:
        db = SessionLocal()

    try:
        log = PredictionLog(
            timestamp          = datetime.utcnow(),
            forecast_days      = forecast_days,
            model_version      = model_version,
            avg_prediction     = round(float(avg_prediction), 2),
            request_latency_ms = round(float(latency_ms), 3),
            forecast_json      = json.dumps(forecast_values) if forecast_values else None,
        )
        db.add(log)
        db.commit()
        db.refresh(log)
        return log

    except Exception as exc:
        db.rollback()
        # Non-fatal: print to stderr but let the caller continue
        print(f"[DB Logger] Failed to write prediction log: {exc}")
        return None

    finally:
        if own_session:
            db.close()


# ---------------------------------------------------------------------------
# Analytics helpers
# ---------------------------------------------------------------------------

def get_recent_predictions(limit: int = 100, db: Session | None = None) -> list[dict]:
    """
    Return the most recent `limit` prediction logs as plain dicts.
    Used by the drift-monitoring endpoint.
    """
    own_session = db is None
    if own_session:
        db = SessionLocal()

    try:
        rows = (
            db.query(PredictionLog)
            .order_by(PredictionLog.timestamp.desc())
            .limit(limit)
            .all()
        )
        return [
            {
                "id":               r.id,
                "timestamp":        r.timestamp.isoformat(),
                "forecast_days":    r.forecast_days,
                "model_version":    r.model_version,
                "avg_prediction":   r.avg_prediction,
                "latency_ms":       r.request_latency_ms,
            }
            for r in rows
        ]
    except Exception as exc:
        print(f"[DB Logger] Failed to read prediction logs: {exc}")
        return []
    finally:
        if own_session:
            db.close()

"""
db/db_models.py

SQLAlchemy ORM table definitions for the Smart Market Intelligence Platform v2.

Table: prediction_logs
  Stores every API prediction request for:
    - Audit trail (who asked for what forecast)
    - Latency monitoring (track P95 response times over time)
    - Drift detection (rolling mean of avg_prediction vs. training baseline)
    - Model comparison (filter by model_version to see if v3 beats v2)

Design choices:
  - id: BIGINT auto-increment (enough for millions of requests)
  - timestamp: stored UTC; convert in the application layer if needed
  - avg_prediction: single float summary of the N-day forecast array
  - request_latency_ms: float (milliseconds) — allows sub-ms resolution
  - model_version: VARCHAR(32) — e.g., "v2-RandomForest-2024-01-15"
"""

from datetime import datetime

from sqlalchemy import Column, Integer, BigInteger, Float, String, DateTime, Text
from db.db_config import Base


class PredictionLog(Base):
    """
    One row per incoming /predict API request.

    Columns aligned with the API's return payload so that replaying logs
    is straightforward.
    """
    __tablename__ = "prediction_logs"

    id                = Column(BigInteger, primary_key=True, index=True, autoincrement=True)
    timestamp         = Column(DateTime,  default=datetime.utcnow, nullable=False, index=True)
    forecast_days     = Column(Integer,   nullable=False)
    model_version     = Column(String(64), nullable=False)
    avg_prediction    = Column(Float,     nullable=False)
    request_latency_ms = Column(Float,   nullable=False)
    # Optional: raw forecast array stored as JSON string for future replay
    forecast_json     = Column(Text,     nullable=True)

    def __repr__(self):
        return (
            f"<PredictionLog id={self.id} model={self.model_version} "
            f"days={self.forecast_days} avg=£{self.avg_prediction:,.0f} "
            f"latency={self.request_latency_ms:.1f}ms>"
        )


# ---------------------------------------------------------------------------
# Table creation helper
# ---------------------------------------------------------------------------

def create_tables() -> None:
    """
    Create all tables defined in this module if they don't exist.

    Call once at application startup (see api/app.py lifespan handler).
    Idempotent — safe to call repeatedly.
    """
    from db.db_config import engine, Base
    Base.metadata.create_all(bind=engine)
    print("[DB] Tables created / verified ✓")

"""
db/db_config.py

Database configuration — Neon PostgreSQL via SQLAlchemy

Why Neon?
  Neon is a serverless PostgreSQL provider with a generous free tier and
  branch-based architecture. Each branch is an isolated DB copy — ideal for
  dev/staging/production parity without extra cost.

Connection string format (from Neon dashboard):
  postgresql://user:password@ep-xxxx.us-east-2.aws.neon.tech/neondb?sslmode=require

NEVER hard-code credentials. Load from environment variable DATABASE_URL.
Use python-dotenv to load a local .env file in development.
"""

import os

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, declarative_base

# ---------------------------------------------------------------------------
# Load .env in development (no-op in production if env vars are already set)
# ---------------------------------------------------------------------------
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise EnvironmentError(
        "DATABASE_URL is not set.\n"
        "Create a .env file with:\n"
        "  DATABASE_URL=postgresql://user:pass@host/db?sslmode=require\n"
        "Or export it as an environment variable.\n"
        "Get a free Neon DB at https://neon.tech"
    )

# ---------------------------------------------------------------------------
# SQLAlchemy engine
# ---------------------------------------------------------------------------
# pool_pre_ping=True — verify connections before use; essential for
# serverless Neon which may suspend after 5 min of inactivity (cold starts).
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=10,
)

SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)

Base = declarative_base()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_db():
    """
    FastAPI dependency: yield a database session and ensure it's closed.

    Usage:
        @app.get("/endpoint")
        def handler(db: Session = Depends(get_db)):
            ...
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def ping_db() -> bool:
    """Quick health-check — returns True if the DB is reachable."""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as exc:
        print(f"[DB] Ping failed: {exc}")
        return False

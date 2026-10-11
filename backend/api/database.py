"""
backend/api/database.py
SQLAlchemy engine, session factory, and DB initialisation.
"""
import os
import sys
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from dotenv import load_dotenv

# Load .env from project root
_root = Path(__file__).resolve().parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
load_dotenv(_root / ".env")

DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./incident_alerts.db")

# SQLite needs connect_args for thread safety when used with FastAPI
_connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(
    DATABASE_URL,
    connect_args=_connect_args,
    echo=False,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


def get_db():
    """Dependency that yields a DB session and closes it after use."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Create all tables (called once at startup)."""
    # Import models so SQLAlchemy registers them before create_all
    from backend.api import models  # noqa: F401
    Base.metadata.create_all(bind=engine)

    # Auto-migrate columns for SQLite if missing
    try:
        with engine.connect() as conn:
            # Check incidents table
            result = conn.exec_driver_sql("PRAGMA table_info(incidents);").fetchall()
            cols = [r[1] for r in result]
            if cols and "decision" not in cols:
                conn.exec_driver_sql("ALTER TABLE incidents ADD COLUMN decision FLOAT;")
            # Check alert_logs table
            result_logs = conn.exec_driver_sql("PRAGMA table_info(alert_logs);").fetchall()
            cols_logs = [r[1] for r in result_logs]
            if cols_logs and "decision" not in cols_logs:
                conn.exec_driver_sql("ALTER TABLE alert_logs ADD COLUMN decision FLOAT;")
            conn.commit()
    except Exception:
        pass


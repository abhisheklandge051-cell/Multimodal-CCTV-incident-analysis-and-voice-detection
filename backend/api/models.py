"""
backend/api/models.py
SQLAlchemy ORM models for owners, incidents, and alert logs.
"""
import sys
from datetime import datetime
from pathlib import Path

_root = Path(__file__).resolve().parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from sqlalchemy import (
    Column, Integer, String, Float, Boolean,
    DateTime, ForeignKey, Text, JSON
)
from sqlalchemy.orm import relationship

from backend.api.database import Base


class Owner(Base):
    """A registered owner who receives incident alert notifications."""
    __tablename__ = "owners"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(120), nullable=False)
    phone = Column(String(30), nullable=False, unique=True)   # E.164 format
    channel = Column(String(20), nullable=False, default="sms")  # 'sms' | 'whatsapp'
    active = Column(Boolean, default=True, nullable=False)
    # Per-owner alert tuning
    alert_threshold = Column(Float, default=0.70, nullable=False)   # 0.0–1.0
    alert_cooldown_sec = Column(Integer, default=60, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    alert_logs = relationship("AlertLog", back_populates="owner", cascade="all, delete-orphan")


class Incident(Base):
    """A detected incident event from the multimodal analysis pipeline."""
    __tablename__ = "incidents"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    label = Column(String(50), nullable=False)           # Fused incident class
    confidence = Column(Float, nullable=False)           # Fused confidence 0–1
    decision = Column(Float, nullable=True)              # Decision percentage (e.g. 85.0)
    video_label = Column(String(50), nullable=True)
    video_conf = Column(Float, nullable=True)
    audio_label = Column(String(50), nullable=True)
    audio_conf = Column(Float, nullable=True)
    modality_status = Column(String(30), nullable=True)  # 'V+A FUSED', 'VIDEO ONLY', etc.
    acoustic_alert = Column(String(200), nullable=True)
    object_counts = Column(JSON, nullable=True)          # {label: count}
    snapshot_b64 = Column(Text, nullable=True)           # Base64 JPEG thumbnail
    source = Column(String(200), nullable=True)          # Camera index or video path
    owners_notified = Column(Integer, default=0)         # Count of owners alerted

    alert_logs = relationship("AlertLog", back_populates="incident", cascade="all, delete-orphan")


class AlertLog(Base):
    """Log of each notification attempt sent to an owner for an incident."""
    __tablename__ = "alert_logs"

    id = Column(Integer, primary_key=True, index=True)
    incident_id = Column(Integer, ForeignKey("incidents.id"), nullable=False, index=True)
    owner_id = Column(Integer, ForeignKey("owners.id"), nullable=False, index=True)
    channel = Column(String(20), nullable=False)         # 'sms' | 'whatsapp'
    status = Column(String(30), nullable=False)          # 'sent' | 'failed' | 'skipped'
    error_message = Column(Text, nullable=True)
    twilio_sid = Column(String(60), nullable=True)
    sent_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    incident = relationship("Incident", back_populates="alert_logs")
    owner = relationship("Owner", back_populates="alert_logs")


class AuthorizedUser(Base):
    """An authorized person enrolled to bypass incident detection in CCTV feeds."""
    __tablename__ = "authorized_users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(120), nullable=False)
    role = Column(String(120), nullable=True, default="Authorized Personnel")
    active = Column(Boolean, default=True, nullable=False)
    face_encoding = Column(JSON, nullable=True)     # 128-d vector stored as JSON array of floats
    photo_b64 = Column(Text, nullable=True)         # Thumbnail or photo base64
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


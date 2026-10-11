"""
backend/api/schemas.py
Pydantic v2 request/response schemas.
"""
from __future__ import annotations
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, field_validator, field_serializer


# ─── Owner schemas ────────────────────────────────────────────────────────────

class OwnerBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=120)
    phone: str = Field(..., description="E.164 phone number, e.g. +919876543210 or ntfy topic name")
    channel: str = Field("ntfy", pattern="^(sms|whatsapp|ntfy)$")
    active: bool = True
    alert_threshold: float = Field(0.70, ge=0.0, le=1.0)
    alert_cooldown_sec: int = Field(60, ge=5, le=3600)


class OwnerCreate(OwnerBase):
    pass


class OwnerUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=120)
    phone: Optional[str] = None
    channel: Optional[str] = Field(None, pattern="^(sms|whatsapp|ntfy)$")
    active: Optional[bool] = None
    alert_threshold: Optional[float] = Field(None, ge=0.0, le=1.0)
    alert_cooldown_sec: Optional[int] = Field(None, ge=5, le=3600)


class OwnerOut(OwnerBase):
    id: int
    created_at: datetime

    model_config = {"from_attributes": True}

    @field_serializer("created_at")
    def serialize_created_at(self, dt: datetime, _info) -> str:
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.isoformat()


# ─── Alert log schemas ────────────────────────────────────────────────────────

class AlertLogOut(BaseModel):
    id: int
    incident_id: int
    owner_id: int
    channel: str
    status: str
    error_message: Optional[str] = None
    twilio_sid: Optional[str] = None
    sent_at: datetime
    owner_name: Optional[str] = None

    model_config = {"from_attributes": True}

    @field_serializer("sent_at")
    def serialize_sent_at(self, dt: datetime, _info) -> str:
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.isoformat()


# ─── Incident schemas ─────────────────────────────────────────────────────────

class IncidentOut(BaseModel):
    id: int
    timestamp: datetime
    label: str
    confidence: float
    decision: Optional[float] = None
    video_label: Optional[str] = None
    video_conf: Optional[float] = None
    audio_label: Optional[str] = None
    audio_conf: Optional[float] = None
    modality_status: Optional[str] = None
    acoustic_alert: Optional[str] = None
    object_counts: Optional[Dict[str, int]] = None
    snapshot_b64: Optional[str] = None
    source: Optional[str] = None
    owners_notified: int = 0

    model_config = {"from_attributes": True}

    @field_serializer("timestamp")
    def serialize_timestamp(self, dt: datetime, _info) -> str:
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.isoformat()


class IncidentDetail(IncidentOut):
    alert_logs: List[AlertLogOut] = []


# ─── Camera control schemas ───────────────────────────────────────────────────

class CameraStartRequest(BaseModel):
    source: str = Field("0", description="Camera index (0,1) or video file path")
    video_weight: float = Field(0.55, ge=0.0, le=1.0)
    audio_weight: float = Field(0.45, ge=0.0, le=1.0)
    enable_yolo: bool = True
    enable_video_incident: bool = True
    enable_audio_incident: bool = True
    yolo_conf: float = Field(0.35, ge=0.1, le=0.95)


class CameraStatusOut(BaseModel):
    running: bool
    source: Optional[str] = None
    fps: float = 0.0
    frame_count: int = 0
    uptime_sec: float = 0.0
    pipeline_error: Optional[str] = None   # populated when ML import fails
    auth_bypass: bool = False
    authorized_users: List[str] = []


# ─── WebSocket event schema ───────────────────────────────────────────────────

class IncidentEvent(BaseModel):
    """Payload broadcast over WebSocket to all React clients."""
    type: str = "incident"
    incident_id: int
    timestamp: str
    label: str
    confidence: float
    decision: Optional[float] = None
    video_label: str
    video_conf: float
    audio_label: str
    audio_conf: float
    modality_status: str
    acoustic_alert: Optional[str] = None
    object_counts: Dict[str, int] = {}
    source: Optional[str] = None
    owners_notified: int = 0
    alert_statuses: List[Dict[str, Any]] = []
    snapshot_b64: Optional[str] = None
    auth_bypass: bool = False
    authorized_users: List[str] = []


class AlertSettingsOut(BaseModel):
    default_threshold: float
    default_cooldown_sec: int
    active_owners: int
    total_owners: int


# ─── Authorized User schemas ──────────────────────────────────────────────────

class AuthorizedUserBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=120)
    role: Optional[str] = Field("Authorized Personnel", max_length=120)
    active: bool = True


class AuthorizedUserCreate(AuthorizedUserBase):
    photo_b64: str = Field(..., description="Base64 encoded face photo or data URI")


class AuthorizedUserOut(AuthorizedUserBase):
    id: int
    photo_b64: Optional[str] = None
    created_at: datetime

    model_config = {"from_attributes": True}

    @field_serializer("created_at")
    def serialize_created_at(self, dt: datetime, _info) -> str:
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.isoformat()


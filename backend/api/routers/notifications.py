"""
backend/api/routers/notifications.py
Endpoints for ntfy.sh alert configuration, sending test alerts, and dispatching live incident alerts.
"""
from __future__ import annotations

import base64
import sys
from pathlib import Path
from typing import Optional

_root = Path(__file__).resolve().parent.parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from backend.api import notifier

router = APIRouter(prefix="/api/notifications", tags=["Notifications"])


class TestAlertRequest(BaseModel):
    topic: Optional[str] = Field(None, description="ntfy.sh topic name (defaults to configured topic)")


class SendNotificationRequest(BaseModel):
    topic: Optional[str] = None
    title: str = Field("🚨 Suspicious Activity Detected!", description="Notification title")
    message: str = Field("Movement detected in Camera Feed 1", description="Notification message")
    priority: str = Field("high", description="Priority level (e.g. min, low, default, high, urgent)")
    tags: str = Field("warning,rotating_light", description="Comma-separated ntfy tags/emojis")
    snapshot_b64: Optional[str] = Field(None, description="Base64 encoded JPEG thumbnail or image")
    filename: Optional[str] = Field("incident.jpg", description="Attachment filename")


class UpdateTopicRequest(BaseModel):
    topic: str = Field(..., min_length=1, max_length=120)


@router.get("/settings")
def get_notification_settings():
    """Returns current ntfy.sh configuration and direct subscription URLs."""
    topic = notifier.get_current_topic()
    server = notifier.NTFY_SERVER
    return {
        "server": server,
        "topic": topic,
        "threshold": notifier.NTFY_ALERT_THRESHOLD,
        "threshold_pct": int(notifier.NTFY_ALERT_THRESHOLD * 100),
        "subscribe_url": f"{server}/{topic}",
        "web_app_url": f"{server}/{topic}",
        "android_app_url": "https://play.google.com/store/apps/details?id=io.heckel.ntfy",
        "ios_app_url": "https://apps.apple.com/app/ntfy/id1625396347",
    }


@router.post("/settings/topic")
def update_notification_topic(body: UpdateTopicRequest):
    """Dynamically update the default ntfy.sh topic and sync with .env and database."""
    new_topic = notifier.clean_topic(body.topic)
    notifier.NTFY_TOPIC = new_topic

    # Persist to .env
    try:
        env_path = _root / ".env"
        if env_path.exists():
            content = env_path.read_text(encoding="utf-8")
            import re
            if "NTFY_TOPIC=" in content:
                content = re.sub(r"NTFY_TOPIC=.*", f"NTFY_TOPIC={new_topic}", content)
            else:
                content += f"\nNTFY_TOPIC={new_topic}\n"
            env_path.write_text(content, encoding="utf-8")
    except Exception:
        pass

    # Update active ntfy owners in DB so everything stays in sync
    try:
        from backend.api.database import SessionLocal
        from backend.api.models import Owner
        with SessionLocal() as db:
            ntfy_owners = db.query(Owner).filter(Owner.channel == "ntfy").all()
            for o in ntfy_owners:
                o.phone = new_topic
            db.commit()
    except Exception:
        pass

    return {
        "topic": notifier.NTFY_TOPIC,
        "server": notifier.NTFY_SERVER,
        "subscribe_url": f"{notifier.NTFY_SERVER}/{notifier.NTFY_TOPIC}",
        "message": f"ntfy topic successfully updated to '{new_topic}'",
    }


@router.post("/test-ntfy")
def test_ntfy_alert(body: TestAlertRequest = None):
    """
    Executes the exact test alert implementation from Notification/test_alert.py.
    Sends sample animated GIF / alert to ntfy.sh and returns delivery confirmation.
    """
    target_topic = notifier.clean_topic(body.topic if body and body.topic else None)
    result = notifier.send_test_alert(topic=target_topic)
    if not result.get("success"):
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=result.get("message", f"Failed to deliver alert to ntfy.sh/{target_topic}"),
        )
    return result


@router.post("/send")
def send_custom_notification(body: SendNotificationRequest):
    """
    Send an immediate notification to the ntfy.sh topic.
    Supports optional base64 image attachment.
    """
    img_data = None
    if body.snapshot_b64:
        try:
            # Handle potential data URL prefix
            raw_b64 = body.snapshot_b64
            if "," in raw_b64:
                raw_b64 = raw_b64.split(",", 1)[1]
            img_data = base64.b64decode(raw_b64)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid base64 image data: {e}")

    st, sid = notifier.send_ntfy(
        topic=body.topic,
        title=body.title,
        message=body.message,
        priority=body.priority,
        tags=body.tags,
        data=img_data,
        filename=body.filename if img_data else None,
    )

    if st != "sent":
        raise HTTPException(status_code=502, detail="Failed to deliver notification to ntfy.sh")

    return {
        "success": True,
        "status": st,
        "topic": body.topic or notifier.NTFY_TOPIC,
        "reference": sid,
        "message": "Notification delivered to ntfy.sh user",
    }

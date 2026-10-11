"""
backend/api/notifier.py
ntfy.sh push notification dispatcher + Twilio SMS/WhatsApp fallback.
Reads credentials from environment variables (set via .env).
"""
from __future__ import annotations

import base64
import io
import os
import sys
import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Tuple, Dict, Any

import requests
from dotenv import load_dotenv

_root = Path(__file__).resolve().parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
load_dotenv(_root / ".env")

logger = logging.getLogger("notifier")

# ─── ntfy.sh Configuration ────────────────────────────────────────────────────
NTFY_SERVER: str = os.getenv("NTFY_SERVER", "https://ntfy.sh").rstrip("/")
NTFY_TOPIC: str = os.getenv("NTFY_TOPIC", "incident_alert_shubh")
NTFY_ALERT_THRESHOLD: float = float(os.getenv("NTFY_ALERT_THRESHOLD", "0.80"))

# ─── Twilio Configuration (Optional Fallback) ──────────────────────────────────
TWILIO_ACCOUNT_SID: str = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN: str = os.getenv("TWILIO_AUTH_TOKEN", "")
TWILIO_PHONE_NUMBER: str = os.getenv("TWILIO_PHONE_NUMBER", "")
TWILIO_WHATSAPP_FROM: str = os.getenv("TWILIO_WHATSAPP_FROM", "whatsapp:+14155238886")

_twilio_client = None


def _get_client():
    """Lazy-initialize Twilio client; returns None if credentials are missing."""
    global _twilio_client
    if _twilio_client is not None:
        return _twilio_client
    if not TWILIO_ACCOUNT_SID or not TWILIO_AUTH_TOKEN:
        return None
    try:
        from twilio.rest import Client
        _twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        return _twilio_client
    except ImportError:
        return None


def _build_alert_message(label: str, confidence: float, source: str | None,
                          video_label: str | None, audio_label: str | None,
                          modality_status: str | None,
                          decision: float | None = None) -> str:
    """Build the human-readable alert message body."""
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    decision_val = decision if decision is not None else (confidence * 100 if confidence <= 1.0 else confidence)
    lines = [
        "🚨 INCIDENT ALERT — CCTV Monitor",
        f"Decision   : {label} ({decision_val:.0f}%)",
        f"Confidence : {confidence * 100:.1f}%",
        f"Modality   : {modality_status or 'N/A'}",
    ]
    if video_label:
        lines.append(f"Video      : {video_label}")
    if audio_label:
        lines.append(f"Audio      : {audio_label}")
    if source:
        lines.append(f"Source     : {source}")
    lines.append(f"Time       : {ts}")
    return "\n".join(lines)


# ─── ntfy.sh Notification Functions ──────────────────────────────────────────

def clean_topic(topic: Optional[str] = None) -> str:
    """Clean topic string: strip URL prefixes, slashes, whitespace, and invalid characters."""
    default = NTFY_TOPIC
    t = (topic or default).strip()
    if "ntfy.sh/" in t:
        t = t.split("ntfy.sh/")[-1]
    t = t.strip("/")
    import re
    t = re.sub(r"[^a-zA-Z0-9_\-]", "_", t)
    return t or default


def get_current_topic() -> str:
    """Return the active ntfy topic, checking DB active owner first, then NTFY_TOPIC."""
    try:
        from backend.api.database import SessionLocal
        from backend.api.models import Owner
        with SessionLocal() as db:
            owner = db.query(Owner).filter(Owner.channel == "ntfy", Owner.active == True).first()
            if owner and owner.phone:
                return clean_topic(owner.phone)
    except Exception:
        pass
    return clean_topic(NTFY_TOPIC)


def send_ntfy(
    topic: Optional[str] = None,
    title: str = "🚨 Incident Alert Detected!",
    message: str = "Movement detected in Camera Feed",
    priority: str = "high",
    tags: str = "warning,rotating_light",
    data: Optional[bytes] = None,
    filename: Optional[str] = None,
) -> Tuple[str, Optional[str]]:
    """
    Send an alert notification to ntfy.sh.
    Matches the protocol demonstrated in Notification/test_alert.py.
    Returns (status, topic_or_id).
    """
    target_topic = clean_topic(topic) if topic else get_current_topic()
    url = f"{NTFY_SERVER}/{target_topic}"
    clean_title = " ".join(title.split())
    clean_msg_header = " · ".join([line.strip() for line in message.splitlines() if line.strip()])

    try:
        if data is not None:
            # Send file/media via PUT as in test_alert.py
            # HTTP header standard requires ASCII-compatible header strings
            ascii_title = clean_title.encode("ascii", "ignore").decode("ascii").strip() or "Incident Alert"
            put_headers = {
                "Title": ascii_title,
                "Priority": priority,
                "Tags": tags,
            }
            if clean_msg_header:
                ascii_msg = clean_msg_header.encode("ascii", "ignore").decode("ascii").strip()
                if ascii_msg:
                    put_headers["Message"] = ascii_msg
            if filename:
                put_headers["Filename"] = filename

            response = requests.put(url, data=data, headers=put_headers, timeout=10)
        else:
            # Send standard notification via JSON API (natively supports full UTF-8 emojis)
            # Note: ntfy JSON publishing requires priority to be an integer (1=min, 2=low, 3=default, 4=high, 5=urgent)
            prio_map = {"min": 1, "low": 2, "default": 3, "high": 4, "urgent": 5, "max": 5}
            prio_int = prio_map.get(str(priority).lower(), 4) if isinstance(priority, str) else int(priority or 4)

            tag_list = [t.strip() for t in tags.split(",") if t.strip()]
            payload = {
                "topic": target_topic,
                "title": clean_title,
                "message": message,
                "priority": prio_int,
                "tags": tag_list,
            }
            response = requests.post(NTFY_SERVER, json=payload, timeout=10)

        if response.status_code == 200:
            logger.info(f"ntfy.sh alert sent successfully to topic '{target_topic}' (Status 200)")
            return "sent", f"ntfy:{target_topic}"
        else:
            logger.warning(f"ntfy.sh returned {response.status_code}: {response.text}")
            return "failed", None
    except Exception as exc:
        logger.error(f"Failed to send alert to ntfy.sh ({target_topic}): {exc}")
        return "failed", None


def send_test_alert(topic: Optional[str] = None) -> Dict[str, Any]:
    """
    Execute the exact test alert workflow implemented in Notification/test_alert.py.
    Creates/reads sample animated GIF and uploads it with alert headers to ntfy.sh.
    """
    target_topic = clean_topic(topic)
    url = f"{NTFY_SERVER}/{target_topic}"
    script_dir = _root / "Notification"
    script_dir.mkdir(parents=True, exist_ok=True)
    file_path = script_dir / "sample.gif"

    # Create sample animated GIF if none exists
    if not file_path.exists():
        try:
            import imageio
            import numpy as np
            test_frames = [
                np.zeros((120, 160, 3), dtype=np.uint8) + int(i * 25)
                for i in range(10)
            ]
            imageio.mimsave(str(file_path), test_frames, format="GIF", fps=10)
        except Exception as e:
            logger.warning(f"Could not generate sample.gif with imageio: {e}")

    try:
        if file_path.exists():
            with open(file_path, "rb") as f:
                response = requests.put(
                    url,
                    data=f,
                    headers={
                        "Title": "Suspicious Activity Detected!",
                        "Message": "Movement detected in Camera Feed 1",
                        "Priority": "high",
                        "Tags": "warning,rotating_light",
                        "Filename": "incident.gif",
                    },
                    timeout=10,
                )
        else:
            # Fallback to JSON API publish
            response = requests.post(
                NTFY_SERVER,
                json={
                    "topic": target_topic,
                    "title": "🚨 Suspicious Activity Detected!",
                    "message": "Movement detected in Camera Feed 1",
                    "priority": "high",
                    "tags": ["warning", "rotating_light"],
                },
                timeout=10,
            )

        success = (response.status_code == 200)
        return {
            "success": success,
            "status_code": response.status_code,
            "topic": target_topic,
            "server": NTFY_SERVER,
            "message": "Alert sent successfully to ntfy.sh!" if success else f"ntfy error ({response.status_code}): {response.text}",
        }
    except Exception as exc:
        return {
            "success": False,
            "status_code": 500,
            "topic": target_topic,
            "server": NTFY_SERVER,
            "message": f"Network exception: {exc}",
        }


# ─── Twilio Functions (SMS & WhatsApp fallback) ───────────────────────────────

def send_sms(to_phone: str, body: str) -> tuple[str, str | None]:
    """Send an SMS message via Twilio."""
    client = _get_client()
    if client is None:
        return "failed", None
    if not TWILIO_PHONE_NUMBER:
        return "failed", None
    try:
        msg = client.messages.create(body=body, from_=TWILIO_PHONE_NUMBER, to=to_phone)
        logger.info(f"SMS sent to {to_phone}: SID={msg.sid}")
        return "sent", msg.sid
    except Exception as exc:
        logger.error(f"SMS send failed to {to_phone}: {exc}")
        return "failed", None


def send_whatsapp(to_phone: str, body: str) -> tuple[str, str | None]:
    """Send a WhatsApp message via Twilio Messaging API."""
    client = _get_client()
    if client is None:
        return "failed", None
    to_wa = f"whatsapp:{to_phone}" if not to_phone.startswith("whatsapp:") else to_phone
    try:
        msg = client.messages.create(body=body, from_=TWILIO_WHATSAPP_FROM, to=to_wa)
        logger.info(f"WhatsApp sent to {to_wa}: SID={msg.sid}")
        return "sent", msg.sid
    except Exception as exc:
        logger.error(f"WhatsApp send failed to {to_wa}: {exc}")
        return "failed", None


# ─── Dispatch Orchestrator ───────────────────────────────────────────────────

def dispatch_incident_alerts(
    incident_id: int,
    label: str,
    confidence: float,
    source: str | None,
    video_label: str | None,
    audio_label: str | None,
    modality_status: str | None,
    owners: list,               # list of Owner ORM objects
    db,                         # SQLAlchemy session
    snapshot_b64: Optional[str] = None,
    decision: Optional[float] = None,
) -> list[dict]:
    """
    Send alerts to all eligible active owners and deliver to ntfy.sh if decision is >= 80%.
    Persists AlertLog records in DB and returns list of result dicts for WebSocket broadcast.
    """
    from backend.api.models import AlertLog, Owner

    decision_val = decision if decision is not None else (confidence * 100 if confidence <= 1.0 else confidence)
    body = _build_alert_message(label, confidence, source, video_label, audio_label, modality_status, decision=decision_val)
    results = []

    # Prepare snapshot bytes if available
    img_data = None
    if snapshot_b64:
        try:
            img_data = base64.b64decode(snapshot_b64)
        except Exception:
            img_data = None

    # Determine tags based on incident class
    tag_map = {
        "Abuse": "warning,punch",
        "CarAccident": "warning,car,rotating_light",
        "Explosion": "bomb,fire,rotating_light",
        "Fighting": "crossed_swords,warning",
        "Riot": "fire,people_holding_hands,warning",
        "Shooting": "gun,rotating_light,sos",
    }
    incident_tags = tag_map.get(label, "warning,rotating_light")

    # ── Automatic delivery to ntfy.sh if decision is >= 80% ───────────────────
    delivered_to_ntfy = False
    active_topic = get_current_topic()
    if decision_val >= 80.0:
        priority = "urgent" if decision_val >= 85.0 else "high"
        ntfy_title = f"🚨 DECISION: {label.upper()} ({decision_val:.0f}%)"
        ntfy_msg = f"Decision: {label} ({decision_val:.0f}%)\nModality: {modality_status or 'N/A'}\nSource: {source or 'Camera Feed 1'}"

        status, sid = send_ntfy(
            topic=active_topic,
            title=ntfy_title,
            message=ntfy_msg,
            priority=priority,
            tags=incident_tags,
            data=img_data,
            filename="incident_snapshot.jpg" if img_data else None,
        )

        # Log system-level ntfy alert in DB
        # Find or create a system ntfy owner or attach to first active owner if exists
        first_owner_id = owners[0].id if owners else None
        if not first_owner_id:
            # Check if an ntfy owner exists
            existing_ntfy_owner = db.query(Owner).filter(Owner.channel == "ntfy").first()
            if existing_ntfy_owner:
                first_owner_id = existing_ntfy_owner.id
            else:
                # Create a default system ntfy owner record for tracking
                system_owner = Owner(
                    name="ntfy.sh Subscriber",
                    phone=active_topic,
                    channel="ntfy",
                    active=True,
                    alert_threshold=0.80,
                    alert_cooldown_sec=30,
                )
                db.add(system_owner)
                db.flush()
                first_owner_id = system_owner.id

        log = AlertLog(
            incident_id=incident_id,
            owner_id=first_owner_id,
            channel="ntfy",
            status=status,
            twilio_sid=sid or f"ntfy:{active_topic}",
        )
        db.add(log)
        results.append({
            "owner": "ntfy.sh User",
            "phone": active_topic,
            "channel": "ntfy",
            "status": status,
            "decision": decision_val,
            "detail": f"Decision {decision_val:.0f}% >= 80%"
        })
        delivered_to_ntfy = True

    # ── Owner-specific dispatch (SMS, WhatsApp, custom ntfy topic) ───────────
    for owner in owners:
        if not owner.active:
            continue

        # If already delivered to active ntfy topic and owner is that ntfy subscriber, skip duplicate
        if delivered_to_ntfy and owner.channel == "ntfy" and (clean_topic(owner.phone) == clean_topic(active_topic) or owner.name == "ntfy.sh Subscriber"):
            continue

        if confidence < owner.alert_threshold:
            log = AlertLog(
                incident_id=incident_id,
                owner_id=owner.id,
                channel=owner.channel,
                status="skipped",
                error_message=f"Confidence {confidence:.2f} < threshold {owner.alert_threshold:.2f}",
            )
            db.add(log)
            results.append({"owner": owner.name, "channel": owner.channel, "status": "skipped"})
            continue

        if owner.channel == "ntfy":
            topic = owner.phone if owner.phone and not owner.phone.startswith("+") else NTFY_TOPIC
            status, sid = send_ntfy(
                topic=topic,
                title=f"🚨 {label.upper()} DETECTED",
                message=body,
                priority="high",
                tags=incident_tags,
                data=img_data,
                filename="incident_snapshot.jpg" if img_data else None,
            )
        elif owner.channel == "whatsapp":
            status, sid = send_whatsapp(owner.phone, body)
        else:
            status, sid = send_sms(owner.phone, body)

        log = AlertLog(
            incident_id=incident_id,
            owner_id=owner.id,
            channel=owner.channel,
            status=status,
            twilio_sid=sid,
        )
        db.add(log)
        results.append({"owner": owner.name, "phone": owner.phone, "channel": owner.channel, "status": status})

    db.commit()
    return results

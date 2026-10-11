"""
backend/api/routers/incidents.py
Endpoints for querying incident history and alert logs.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

_root = Path(__file__).resolve().parent.parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session, joinedload

from backend.api.database import get_db
from backend.api.models import Incident, AlertLog, Owner
from backend.api.schemas import IncidentOut, IncidentDetail, AlertLogOut

router = APIRouter(prefix="/api/incidents", tags=["Incidents"])


@router.get("", response_model=List[IncidentOut])
@router.get("/", response_model=List[IncidentOut], include_in_schema=False)
def list_incidents(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    label: Optional[str] = Query(None, description="Filter by incident class"),
    db: Session = Depends(get_db),
):
    q = db.query(Incident).order_by(Incident.timestamp.desc())
    if label:
        q = q.filter(Incident.label == label)
    return q.offset(skip).limit(limit).all()


@router.get("/stats")
def incident_stats(db: Session = Depends(get_db)):
    """Return per-class incident counts and today's totals."""
    from sqlalchemy import func
    from datetime import date, datetime, timezone

    today_start = datetime.combine(date.today(), datetime.min.time())
    rows = db.query(Incident.label, func.count(Incident.id)).group_by(Incident.label).all()
    today_rows = (
        db.query(Incident.label, func.count(Incident.id))
        .filter(Incident.timestamp >= today_start)
        .group_by(Incident.label)
        .all()
    )
    total = db.query(func.count(Incident.id)).scalar()
    alerts_sent = db.query(func.count(AlertLog.id)).filter(AlertLog.status == "sent").scalar()

    return {
        "total_incidents": total,
        "alerts_sent": alerts_sent,
        "by_class": {r[0]: r[1] for r in rows},
        "today_by_class": {r[0]: r[1] for r in today_rows},
    }


@router.get("/{incident_id}", response_model=IncidentDetail)
def get_incident(incident_id: int, db: Session = Depends(get_db)):
    incident = (
        db.query(Incident)
        .options(joinedload(Incident.alert_logs).joinedload(AlertLog.owner))
        .filter(Incident.id == incident_id)
        .first()
    )
    if not incident:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Incident not found.")

    # Enrich alert logs with owner name
    detail = IncidentDetail.model_validate(incident)
    enriched_logs = []
    for log in incident.alert_logs:
        log_out = AlertLogOut.model_validate(log)
        log_out.owner_name = log.owner.name if log.owner else None
        enriched_logs.append(log_out)
    detail.alert_logs = enriched_logs
    return detail

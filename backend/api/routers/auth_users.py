"""
backend/api/routers/auth_users.py
Authorized Personnel API — enroll, list, toggle, delete, and snapshot from live feed.
"""
from __future__ import annotations

import base64
import logging
from typing import List

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.api.database import get_db
from backend.api.models import AuthorizedUser
from backend.api.schemas import AuthorizedUserCreate, AuthorizedUserOut
from backend.api import analysis_bridge
from backend.video.auth_manager import get_auth_manager

logger = logging.getLogger("auth_users_router")
router = APIRouter(prefix="/api/auth-users", tags=["Authorized Users"])


@router.get("", response_model=List[AuthorizedUserOut])
def list_authorized_users(db: Session = Depends(get_db)):
    """List all authorized personnel registered in the system."""
    return db.query(AuthorizedUser).order_by(AuthorizedUser.id.desc()).all()


@router.post("", response_model=AuthorizedUserOut, status_code=status.HTTP_201_CREATED)
def create_authorized_user(
    body: AuthorizedUserCreate,
    db: Session = Depends(get_db)
):
    """
    Enroll a new authorized user.
    Extracts face embedding from the provided photo_b64.
    If no face is detected, returns a 400 error.
    """
    auth_mgr = get_auth_manager()

    success, msg, encoding, thumb_b64 = auth_mgr.enroll_user(
        name=body.name,
        image_input=body.photo_b64,
        role=body.role or "Authorized Personnel"
    )

    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=msg
        )

    # Use thumbnail if generated, else original
    stored_photo = thumb_b64 or body.photo_b64

    user = AuthorizedUser(
        name=body.name.strip(),
        role=body.role.strip() if body.role else "Authorized Personnel",
        active=body.active,
        face_encoding=encoding,
        photo_b64=stored_photo,
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    logger.info(f"Created authorized user record #{user.id}: {user.name}")
    return user


@router.delete("/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_authorized_user(user_id: int, db: Session = Depends(get_db)):
    """Delete an authorized user and remove their face embeddings from memory and cache."""
    user = db.query(AuthorizedUser).filter(AuthorizedUser.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="Authorized user not found.")

    auth_mgr = get_auth_manager()
    auth_mgr.delete_user(user.name)

    db.delete(user)
    db.commit()
    logger.info(f"Deleted authorized user #{user_id}: {user.name}")
    return None


@router.patch("/{user_id}/toggle", response_model=AuthorizedUserOut)
def toggle_authorized_user(user_id: int, db: Session = Depends(get_db)):
    """Toggle active/inactive status of an authorized user."""
    user = db.query(AuthorizedUser).filter(AuthorizedUser.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="Authorized user not found.")

    user.active = not user.active
    db.commit()
    db.refresh(user)

    # Sync state with auth manager
    auth_mgr = get_auth_manager()
    if not user.active:
        auth_mgr.delete_user(user.name)
    else:
        auth_mgr.sync_with_db(db)

    return user


@router.get("/snapshot-live")
def snapshot_live_camera():
    """
    Captures the current live frame from the camera stream
    to instantly enroll an authorized person without manual file uploads.
    """
    jpg_bytes = analysis_bridge.get_latest_frame_jpg()
    if not jpg_bytes:
        raise HTTPException(
            status_code=400,
            detail="Camera stream is not running. Please start the camera first or upload an image file."
        )

    b64 = "data:image/jpeg;base64," + base64.b64encode(jpg_bytes).decode("utf-8")
    return {"photo_b64": b64}

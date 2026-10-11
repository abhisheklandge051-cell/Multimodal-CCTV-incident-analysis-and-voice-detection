"""
backend/api/routers/owners.py
CRUD endpoints for registered owners.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List

_root = Path(__file__).resolve().parent.parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.api.database import get_db
from backend.api.models import Owner
from backend.api.schemas import OwnerCreate, OwnerUpdate, OwnerOut

router = APIRouter(prefix="/api/owners", tags=["Owners"])


@router.get("", response_model=List[OwnerOut])
@router.get("/", response_model=List[OwnerOut], include_in_schema=False)
def list_owners(db: Session = Depends(get_db)):
    return db.query(Owner).order_by(Owner.created_at.desc()).all()


@router.post("", response_model=OwnerOut, status_code=status.HTTP_201_CREATED)
@router.post("/", response_model=OwnerOut, status_code=status.HTTP_201_CREATED, include_in_schema=False)
def create_owner(body: OwnerCreate, db: Session = Depends(get_db)):
    existing = db.query(Owner).filter(Owner.phone == body.phone).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Owner with phone '{body.phone}' already registered."
        )
    owner = Owner(**body.model_dump())
    db.add(owner)
    db.commit()
    db.refresh(owner)
    return owner


@router.get("/{owner_id}", response_model=OwnerOut)
def get_owner(owner_id: int, db: Session = Depends(get_db)):
    owner = db.query(Owner).filter(Owner.id == owner_id).first()
    if not owner:
        raise HTTPException(status_code=404, detail="Owner not found.")
    return owner


@router.patch("/{owner_id}", response_model=OwnerOut)
def update_owner(owner_id: int, body: OwnerUpdate, db: Session = Depends(get_db)):
    owner = db.query(Owner).filter(Owner.id == owner_id).first()
    if not owner:
        raise HTTPException(status_code=404, detail="Owner not found.")
    update_data = body.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(owner, field, value)
    db.commit()
    db.refresh(owner)
    return owner


@router.delete("/{owner_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_owner(owner_id: int, db: Session = Depends(get_db)):
    owner = db.query(Owner).filter(Owner.id == owner_id).first()
    if not owner:
        raise HTTPException(status_code=404, detail="Owner not found.")
    db.delete(owner)
    db.commit()

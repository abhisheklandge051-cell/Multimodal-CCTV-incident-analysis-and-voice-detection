"""
backend/api/main.py
FastAPI application entry point.

Run with:
    uvicorn backend.api.main:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

# ─── Load .env & ensure project root in sys.path ──────────────────────────────
_root = Path(__file__).resolve().parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
load_dotenv(_root / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger("main")

# ─── DB initialisation ────────────────────────────────────────────────────────
from backend.api.database import init_db  # noqa: E402

@asynccontextmanager
async def lifespan(app: FastAPI):
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.winmm.timeBeginPeriod(1)
            logger.info("High-resolution Windows multimedia timer (1ms) enabled.")
        except Exception as e:
            logger.warning(f"Could not enable high-resolution timer: {e}")
    logger.info("Initialising database tables...")
    init_db()
    logger.info("Database ready. Server starting.")
    yield
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.winmm.timeEndPeriod(1)
        except Exception:
            pass
    logger.info("Server shutting down.")


# ─── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Multimodal CCTV Incident Alert API",
    description=(
        "Real-time multimodal incident analysis (Video + Audio + YOLOv8) "
        "with SMS / WhatsApp owner notifications via Twilio."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ─── CORS ─────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Mount Templates & Static Files ───────────────────────────────────────────
# Make sure the 'templates' folder exists in the same directory where you run uvicorn
templates = Jinja2Templates(directory="templates")

# (Optional) Uncomment this if you add a 'static' folder for local videos or CSS
# app.mount("/static", StaticFiles(directory="static"), name="static")


# ─── Routers ──────────────────────────────────────────────────────────────────
from backend.api.routers import owners, incidents, camera, notifications, auth_users  # noqa: E402

app.include_router(owners.router)
app.include_router(incidents.router)
app.include_router(camera.router)
app.include_router(notifications.router)
app.include_router(auth_users.router)

# ─── WebSocket ────────────────────────────────────────────────────────────────
from backend.api.websocket_manager import manager  # noqa: E402

@app.websocket("/ws/incidents")
async def ws_incidents(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        await websocket.send_json({"type": "connected", "message": "Real-time incident feed active."})
        while True:
            # Keep connection alive; client can send pings
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_json({"type": "pong"})
    except WebSocketDisconnect:
        manager.disconnect(websocket)


# ─── System Routes ────────────────────────────────────────────────────────────
@app.get("/health", tags=["System"])
async def health():
    return {"status": "ok", "version": "1.0.0"}

@app.get("/api-info", tags=["System"])
async def api_info():
    """Returns the JSON metadata that used to be on the root route."""
    return {
        "message": "Multimodal CCTV Incident Alert API",
        "docs": "/docs",
        "ws_endpoint": "ws://localhost:8000/ws/incidents",
    }

# ─── Dashboard Frontend (The new root) ────────────────────────────────────────
@app.get("/", response_class=HTMLResponse, tags=["Dashboard"])
async def serve_dashboard(request: Request):
    """Serves the HTML/JS frontend dashboard."""
    # Use explicit keyword arguments (request, name, and context)
    return templates.TemplateResponse(
        request=request, 
        name="index.html", 
        context={"request": request}
    )
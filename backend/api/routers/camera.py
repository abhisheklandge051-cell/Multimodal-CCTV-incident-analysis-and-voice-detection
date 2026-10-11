
"""
backend/api/routers/camera.py
Camera / video-source control — start, stop, status, and MJPEG live stream.
"""
from __future__ import annotations

import asyncio
import os
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from backend.api import analysis_bridge
from backend.api.database import get_db, SessionLocal
from backend.api.schemas import CameraStartRequest, CameraStatusOut

router = APIRouter(prefix="/api/camera", tags=["Camera"])

# ─── Model path resolution ────────────────────────────────────────────────────
_root = Path(__file__).resolve().parent.parent.parent.parent  # project root

def _resolve(path: str) -> str:
    """Return absolute path; try project root first."""
    p = Path(path)
    if p.is_absolute() and p.exists():
        return str(p)
    candidate = _root / path
    if candidate.exists():
        return str(candidate)
    return str(p)  # Let it fail naturally with an informative error


@router.post("/start", response_model=CameraStatusOut)
async def start_camera(body: CameraStartRequest, request: Request):
    if analysis_bridge.state.running:
        raise HTTPException(status_code=409, detail="Analysis already running.")

    loop = asyncio.get_running_loop()

    video_model = _resolve(os.getenv("VIDEO_MODEL_PATH", "best_fast_gru_resnet_model.pth"))
    audio_model = _resolve(os.getenv("AUDIO_MODEL_PATH", "audio_incident_head.keras"))
    yolo_model = _resolve(os.getenv("YOLO_MODEL_PATH", "yolov8n.pt"))

    started = analysis_bridge.start_analysis(
        source=body.source,
        model_path=video_model,
        audio_model=audio_model,
        yolo_model=yolo_model,
        yolo_conf=body.yolo_conf,
        video_weight=body.video_weight,
        audio_weight=body.audio_weight,
        enable_yolo=body.enable_yolo,
        enable_video=body.enable_video_incident,
        enable_audio=body.enable_audio_incident,
        loop=loop,
        db_session_factory=SessionLocal,
    )

    if not started:
        raise HTTPException(status_code=409, detail="Failed to start analysis thread.")

    return CameraStatusOut(**analysis_bridge.get_state())


@router.post("/stop", response_model=CameraStatusOut)
async def stop_camera():
    if not analysis_bridge.state.running:
        raise HTTPException(status_code=409, detail="No analysis is currently running.")
    analysis_bridge.stop_analysis()
    return CameraStatusOut(**analysis_bridge.get_state())


@router.get("/status", response_model=CameraStatusOut)
async def camera_status():
    return CameraStatusOut(**analysis_bridge.get_state())



@router.get("/stream")
def mjpeg_stream():
    """
    MJPEG live stream of the latest annotated frame.
    Consumed directly by the React <img> tag.
    Synchronous generator executed in Starlette threadpool:
    Waits on _frame_condition with kernel-level instant wakeup (< 0.05 ms).
    Delivers native 30 FPS directly to the browser with zero polling lag or Windows timer quantization.
    """
    def _generate():
        last_version = -1
        try:
            while True:
                with analysis_bridge._frame_condition:
                    while analysis_bridge._frame_version == last_version:
                        if not analysis_bridge._stream_running and not analysis_bridge.state.running:
                            return
                        analysis_bridge._frame_condition.wait(timeout=0.1)
                    jpg = analysis_bridge._latest_encoded_frame
                    last_version = analysis_bridge._frame_version

                if jpg is not None:
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n"
                        b"Content-Length: " + str(len(jpg)).encode("ascii") + b"\r\n\r\n"
                        + jpg
                        + b"\r\n"
                    )
        except (GeneratorExit, ConnectionResetError, BrokenPipeError):
            pass

    return StreamingResponse(
        _generate(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
            "X-Accel-Buffering": "no",
            "Connection": "close",
        }
    )

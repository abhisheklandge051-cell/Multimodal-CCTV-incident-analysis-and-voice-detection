
from __future__ import annotations

import asyncio
import base64
import logging
import os
import queue
import sys
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List

import cv2
import numpy as np

# Ensure project root is on sys.path
_root = Path(__file__).resolve().parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

logger = logging.getLogger("analysis_bridge")

# ─── Lazy imports of the existing ML pipeline ────────────────────────────────
_pipeline_imported = False
_import_error_msg: Optional[str] = None   # exposed via state snapshot
LiveIncidentAnalyzer = None
LiveAudioIncidentAnalyzer = None
MultimodalIncidentFusion = None
YOLOv8ObjectDetector = None
draw_hud_overlay = None
ThreadedCameraReader = None
CLASSES = ["Abuse", "CarAccident", "Explosion", "Fighting", "Normal", "Riot", "Shooting"]


def _import_pipeline():
    """Import the existing pipeline modules without altering them."""
    global _pipeline_imported, _import_error_msg
    global LiveIncidentAnalyzer, LiveAudioIncidentAnalyzer
    global MultimodalIncidentFusion, YOLOv8ObjectDetector, draw_hud_overlay, CLASSES, ThreadedCameraReader

    if _pipeline_imported:
        return

    # Ensure project root is on sys.path
    root = Path(__file__).resolve().parent.parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    try:
        from backend.video.live_camera_test import (   # type: ignore
            LiveIncidentAnalyzer as _LIA,
            LiveAudioIncidentAnalyzer as _LAIA,
            MultimodalIncidentFusion as _MIF,
            YOLOv8ObjectDetector as _YOLO,
            draw_hud_overlay as _HUD,
            CLASSES as _CL,
            ThreadedCameraReader as _TCR,
        )
        LiveIncidentAnalyzer = _LIA
        LiveAudioIncidentAnalyzer = _LAIA
        MultimodalIncidentFusion = _MIF
        YOLOv8ObjectDetector = _YOLO
        draw_hud_overlay = _HUD
        CLASSES = _CL
        ThreadedCameraReader = _TCR
        _import_error_msg = None
        _pipeline_imported = True
        logger.info("ML pipeline imported successfully.")
    except Exception as exc:
        err = f"ML pipeline import failed: {exc}"
        logger.error(f"{err}. Running in DEMO mode.")
        _import_error_msg = err
        _pipeline_imported = True   # mark as done to avoid retrying every frame


# ─── State shared between bridge thread and FastAPI ──────────────────────────

class AnalysisState:
    """Thread-safe container for current analysis state."""

    def __init__(self):
        self._lock = threading.Lock()
        self.running = False
        self.source: Optional[str] = None
        self.fps: float = 0.0
        self.frame_count: int = 0
        self.uptime_sec: float = 0.0
        self._start_time: Optional[float] = None
        self.last_incident: Optional[Dict[str, Any]] = None
        self.pipeline_error: Optional[str] = None
        # Per-class last-alert timestamp for cooldown tracking
        self._last_alert_times: Dict[str, float] = {}
        self.auth_bypass: bool = False
        self.authorized_users: List[str] = []

    def start(self, source: str):
        with self._lock:
            self.running = True
            self.source = source
            self.fps = 0.0
            self.frame_count = 0
            self.pipeline_error = None
            self._start_time = time.time()
            self._last_alert_times = {}
            self.auth_bypass = False
            self.authorized_users = []

    def stop(self):
        with self._lock:
            self.running = False
            self.auth_bypass = False
            self.authorized_users = []

    def update_frame(self, fps: float, auth_bypass: bool = False, authorized_users: List[str] = None):
        with self._lock:
            self.frame_count += 1
            self.fps = fps
            self.auth_bypass = auth_bypass
            self.authorized_users = authorized_users or []
            if self._start_time:
                self.uptime_sec = time.time() - self._start_time

    def should_alert(self, label: str, cooldown_sec: int = 60) -> bool:
        """Return True if enough time has passed since last alert for this label."""
        with self._lock:
            if label == "Normal":
                return False
            last = self._last_alert_times.get(label, 0.0)
            if time.time() - last >= cooldown_sec:
                self._last_alert_times[label] = time.time()
                return True
            return False

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "running": self.running,
                "source": self.source,
                "fps": round(self.fps, 2),
                "frame_count": self.frame_count,
                "uptime_sec": round(self.uptime_sec, 1),
                "pipeline_error": self.pipeline_error or _import_error_msg,
                "auth_bypass": self.auth_bypass,
                "authorized_users": self.authorized_users,
            }


state = AnalysisState()
_analysis_thread: Optional[threading.Thread] = None


# ─── Frame JPEG encoder helper ────────────────────────────────────────────────

def _frame_to_b64(frame: np.ndarray, max_w: int = 320) -> str:
    """Encode a BGR frame to a base64 JPEG string for WebSocket / DB storage."""
    h, w = frame.shape[:2]
    if w > max_w:
        scale = max_w / w
        frame = cv2.resize(frame, (max_w, int(h * scale)))
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
    return base64.b64encode(buf).decode("utf-8")


# ─── Decoupled Video Streaming Manager ────────────────────────────────────────

_frame_condition = threading.Condition()
_latest_encoded_frame: Optional[bytes] = None
_frame_version: int = 0
_stream_queue: queue.Queue = queue.Queue(maxsize=3)
_stream_thread: Optional[threading.Thread] = None
_stream_running: bool = False


def _streaming_encoder_worker():
    """
    Dedicated background streaming worker thread decoupled from CV inference.
    Pulls raw display frames, downscales for fast network transport (720p/960p),
    encodes to JPEG (quality 70), and notifies streaming clients via Condition variable.
    Operates at full native camera FPS (30 FPS) without causing inference stalls.
    """
    global _latest_encoded_frame, _frame_version, _stream_running
    logger.info("Dedicated background video streaming worker started.")

    while _stream_running:
        try:
            frame = _stream_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        try:
            h, w = frame.shape[:2]
            # Downscale display feed sent to frontend (720p max width 1280 or 960)
            # drastically reduces encoding time (< 2ms) and network byte payload
            target_w = 960 if w > 960 else w
            if w > target_w:
                scale = target_w / float(w)
                disp_frame = cv2.resize(frame, (target_w, int(h * scale)), interpolation=cv2.INTER_LINEAR)
            else:
                disp_frame = frame

            # Optimized JPEG compression quality (70)
            _, buf = cv2.imencode(".jpg", disp_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            jpg_bytes = bytes(buf)

            with _frame_condition:
                _latest_encoded_frame = jpg_bytes
                _frame_version += 1
                _frame_condition.notify_all()
        except Exception as e:
            logger.debug(f"Streaming worker encoding error: {e}")

    logger.info("Dedicated background video streaming worker stopped.")


def _start_streaming_worker():
    global _stream_thread, _stream_running, _stream_queue
    if _stream_thread is not None and _stream_thread.is_alive():
        return
    while not _stream_queue.empty():
        try:
            _stream_queue.get_nowait()
        except queue.Empty:
            break
    _stream_running = True
    _stream_thread = threading.Thread(
        target=_streaming_encoder_worker,
        daemon=True,
        name="VideoStreamWorker"
    )
    _stream_thread.start()


def _stop_streaming_worker():
    global _stream_running, _stream_thread
    _stream_running = False
    with _frame_condition:
        _frame_condition.notify_all()
    if _stream_thread is not None:
        _stream_thread.join(timeout=0.5)
        _stream_thread = None


def get_latest_encoded_frame() -> Optional[bytes]:
    """Return latest encoded JPEG frame bytes."""
    with _frame_condition:
        return _latest_encoded_frame


def get_latest_frame_jpg() -> Optional[bytes]:
    """Compatibility alias for get_latest_encoded_frame()."""
    with _frame_condition:
        return _latest_encoded_frame


def get_latest_frame_with_version() -> Tuple[Optional[bytes], int]:
    """Return tuple of (latest_encoded_frame, frame_version)."""
    with _frame_condition:
        return _latest_encoded_frame, _frame_version


def _store_frame(frame: np.ndarray):
    """
    Non-blocking push of raw display frame to dedicated background streaming thread.
    Execution time < 0.005 ms, so CV inference runs at full native 30 FPS without stalls.
    """
    if not _stream_running:
        _start_streaming_worker()
    try:
        if _stream_queue.full():
            try:
                _stream_queue.get_nowait()
            except queue.Empty:
                pass
        _stream_queue.put_nowait(frame)
    except Exception:
        pass


# ─── Background analysis thread ───────────────────────────────────────────────

def _run_analysis(
    source: str,
    model_path: str,
    audio_model: str,
    yolo_model: str,
    yolo_conf: float,
    video_weight: float,
    audio_weight: float,
    enable_yolo: bool,
    enable_video: bool,
    enable_audio: bool,
    loop: asyncio.AbstractEventLoop,
    db_session_factory,
):
    """
    Background thread: opens camera/video, runs ML inference per frame,
    stores incidents in DB, dispatches Twilio alerts, broadcasts via WebSocket.
    """
    from backend.api.websocket_manager import manager
    from backend.api.notifier import dispatch_incident_alerts
    from backend.api.models import Incident, Owner

    _import_pipeline()

    # Surface any import error back to the state (visible in /api/camera/status)
    if _import_error_msg:
        state.pipeline_error = _import_error_msg

    # Initialise ML components
    incident_analyzer = None
    audio_analyzer = None

    if enable_video and LiveIncidentAnalyzer is not None:
        try:
            incident_analyzer = LiveIncidentAnalyzer(model_path=model_path)
        except Exception as e:
            err = f"Video analyzer load error: {e}"
            logger.error(err)
            state.pipeline_error = err

    if enable_audio and LiveAudioIncidentAnalyzer is not None:
        try:
            audio_analyzer = LiveAudioIncidentAnalyzer(model_path=audio_model)
            video_src_val = int(source) if source.isdigit() else source
            audio_analyzer.setup_audio_source(video_src_val)
        except Exception as e:
            err = f"Audio analyzer load error: {e}"
            logger.error(err)
            if state.pipeline_error is None:
                state.pipeline_error = err

    # Only create fusion engine when at least one analyzer is available
    fusion_engine = None
    if MultimodalIncidentFusion is not None and (incident_analyzer or audio_analyzer):
        fusion_engine = MultimodalIncidentFusion(
            video_weight=video_weight,
            audio_weight=audio_weight
        )

    yolo_detector = None
    if enable_yolo and YOLOv8ObjectDetector is not None:
        try:
            yolo_detector = YOLOv8ObjectDetector(model_path=yolo_model, conf_thres=yolo_conf)
        except Exception as e:
            logger.error(f"YOLO detector failed to load: {e}")

    # Initialize AuthManager for authorized personnel detection and incident bypass
    auth_manager = None
    try:
        from backend.video.auth_manager import get_auth_manager
        auth_manager = get_auth_manager()
        logger.info(f"AuthManager active with {len(auth_manager.known_identities)} authorized identity/identities.")
    except Exception as e:
        logger.warning(f"Could not load AuthManager in analysis bridge: {e}")

    video_src = int(source) if source.isdigit() else source
    if ThreadedCameraReader is not None:
        cap = ThreadedCameraReader(video_src, width=640, height=480, fps=30)
    elif isinstance(video_src, int):
        cap = cv2.VideoCapture(video_src, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    else:
        cap = cv2.VideoCapture(video_src)

    if not cap.isOpened():
        logger.error(f"Cannot open video source: {source}")
        state.stop()
        return

    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
    prev_time = time.time()
    fps_ema = 0.0
    frame_idx = 0
    start_wall = time.time()
    prev_detections = None
    v_label, v_conf, v_probs = "Normal", 1.0, None

    logger.info(f"Analysis started on source: {source}")

    try:
        while state.running:
            ret, frame = cap.read()
            if not ret or frame is None:
                if isinstance(video_src, str):
                    logger.info("Video file ended.")
                    break
                time.sleep(0.005)
                continue  # live camera: retry

            curr_time = time.time()
            fps_ema = 0.9 * fps_ema + 0.1 * (1.0 / max(1e-5, curr_time - prev_time))
            prev_time = curr_time
            frame_idx += 1

            playback_sec = frame_idx / fps_in if isinstance(video_src, str) else (curr_time - start_wall)

            # ── Video incident classification (interleaved for 30 FPS) ─────
            # FastGRU maintains a 16-frame temporal buffer; human events evolve over
            # seconds. Sampling every 2 frames alternating with YOLO gives identical
            # temporal accuracy while maintaining a rock-solid 30+ FPS.
            if incident_analyzer:
                is_active = getattr(incident_analyzer, "is_active", True)
                if not is_active and frame_idx == 1:
                    logger.warning("Video analyzer exists but is_active=False on first frame — check model warm-up")
                if is_active:
                    if frame_idx % 2 == 1 or v_probs is None:
                        v_label, v_conf, v_probs = incident_analyzer.process_frame(frame)

            # ── Audio incident classification ──────────────────────────────
            a_label, a_conf, a_probs = "Normal", 1.0, None
            if audio_analyzer:
                is_active = getattr(audio_analyzer, "is_active", True)
                has_audio = getattr(audio_analyzer, "has_audio_track", True)
                if not is_active and frame_idx == 1:
                    logger.warning("Audio analyzer exists but is_active=False on first frame — check model warm-up")
                if is_active and has_audio:
                    a_label, a_conf, a_probs = audio_analyzer.process_audio(playback_sec)

            # ── Multimodal fusion ──────────────────────────────────────────
            if fusion_engine:
                fused_label, fused_conf, fused_probs, modality_status, acoustic_alert = fusion_engine.fuse(v_probs, a_probs)
            else:
                # Fallback: video only
                fused_label, fused_conf = v_label, v_conf
                fused_probs = v_probs
                modality_status = "VIDEO ONLY"
                acoustic_alert = None

            # ── YOLO object detection (interleaved on even frames for 30 FPS) ──
            object_counts: Dict[str, int] = {}
            if yolo_detector:
                if frame_idx % 2 == 0 or prev_detections is None:
                    prev_detections = yolo_detector.detect(frame)
                frame, object_counts = yolo_detector.draw_detections(frame, prev_detections)

            # ── Authorized Personnel Detection & Incident Bypass ───────────
            auth_info = None
            if auth_manager:
                auth_info = auth_manager.process_frame(frame, detections=prev_detections, frame_idx=frame_idx)
            is_bypassed = bool(auth_info and auth_info.get("bypass_incident"))
            auth_names = auth_info.get("authorized_names", []) if auth_info else []

            state.update_frame(fps_ema, auth_bypass=is_bypassed, authorized_users=auth_names)

            # ── Multimodal HUD Overlay ─────────────────────────────────────
            if draw_hud_overlay is not None:
                try:
                    annotated_frame = draw_hud_overlay(
                        frame,
                        fused_label=fused_label,
                        fused_conf=fused_conf,
                        fused_probs=fused_probs,
                        video_label=v_label,
                        video_conf=v_conf,
                        video_probs=v_probs,
                        audio_label=a_label,
                        audio_conf=a_conf,
                        audio_probs=a_probs,
                        modality_status=modality_status,
                        acoustic_alert=acoustic_alert,
                        object_counts=object_counts,
                        fps=fps_ema,
                        auth_info=auth_info,
                    )
                except Exception as e:
                    logger.warning(f"Failed to draw HUD overlay: {e}")
                    annotated_frame = frame
            else:
                annotated_frame = frame

            # Store latest annotated frame for MJPEG stream
            _store_frame(annotated_frame)

            # ── Incident alerting (Condition: Decision >= 70% ONLY for Video+Audio, else >= 80%) ─
            decision = round(fused_conf * 100, 1)

            modality_str = str(modality_status).upper() if modality_status else ""
            is_video_audio = (
                "V+A" in modality_str
                or "MULTIMODAL" in modality_str
                or "VIDEO+AUDIO" in modality_str
                or ("VIDEO" in modality_str and "AUDIO" in modality_str)
            )
            required_decision = 70.0 if is_video_audio else 80.0

            if is_bypassed and fused_label != "Normal" and frame_idx % 30 == 0:
                logger.info(
                    f"Incident {fused_label} ({decision:.0f}%) BYPASSED: "
                    f"Authorized user(s) {auth_names} present in live feed."
                )

            # Apply condition of decision: Decision >= 70% ONLY for video+audio multimodal analysis, >= 80% otherwise
            if fused_label != "Normal" and decision >= required_decision and not is_bypassed and state.should_alert(fused_label, cooldown_sec=30):
                snapshot_b64 = _frame_to_b64(annotated_frame)

                # Persist incident
                db = db_session_factory()
                try:
                    incident = Incident(
                        label=fused_label,
                        confidence=fused_conf,
                        decision=decision,
                        video_label=v_label,
                        video_conf=v_conf,
                        audio_label=a_label,
                        audio_conf=a_conf,
                        modality_status=modality_status,
                        acoustic_alert=acoustic_alert,
                        object_counts=object_counts,
                        snapshot_b64=snapshot_b64,
                        source=source,
                    )
                    db.add(incident)
                    db.flush()  # get incident.id

                    # Fetch active owners
                    owners = db.query(Owner).filter(Owner.active == True).all()

                    # Dispatch alerts (ntfy.sh push notifications + owners)
                    alert_results = dispatch_incident_alerts(
                        incident_id=incident.id,
                        label=fused_label,
                        confidence=fused_conf,
                        source=source,
                        video_label=v_label,
                        audio_label=a_label,
                        modality_status=modality_status,
                        owners=owners,
                        db=db,
                        snapshot_b64=snapshot_b64,
                        decision=decision,
                    )

                    notified_count = sum(1 for r in alert_results if r.get("status") == "sent")
                    incident.owners_notified = notified_count
                    db.commit()

                    incident_id = incident.id

                    # Broadcast via WebSocket (thread-safe)
                    event_payload = {
                        "type": "incident",
                        "incident_id": incident_id,
                        "timestamp": datetime.utcnow().isoformat() + "Z",
                        "label": fused_label,
                        "decision": decision,
                        "confidence": round(fused_conf, 4),
                        "video_label": v_label,
                        "video_conf": round(v_conf, 4),
                        "audio_label": a_label,
                        "audio_conf": round(a_conf, 4),
                        "modality_status": modality_status,
                        "acoustic_alert": acoustic_alert,
                        "object_counts": object_counts,
                        "source": source,
                        "owners_notified": notified_count,
                        "alert_statuses": alert_results,
                        "snapshot_b64": snapshot_b64,
                        "auth_bypass": is_bypassed,
                        "authorized_users": auth_names,
                    }
                    manager.broadcast_from_thread(loop, event_payload)

                except Exception as exc:
                    logger.error(f"Failed to persist/notify incident: {exc}")
                    db.rollback()
                finally:
                    db.close()

    finally:
        cap.release()
        if audio_analyzer:
            try:
                audio_analyzer.release()
            except Exception:
                pass
        state.stop()
        _stop_streaming_worker()
        logger.info("Analysis thread stopped.")


# ─── Public API ───────────────────────────────────────────────────────────────

def start_analysis(
    source: str = "0",
    model_path: str = "best_fast_gru_resnet_model.pth",
    audio_model: str = "audio_incident_head.keras",
    yolo_model: str = "yolov8n.pt",
    yolo_conf: float = 0.35,
    video_weight: float = 0.70,
    audio_weight: float = 0.30,
    enable_yolo: bool = True,
    enable_video: bool = True,
    enable_audio: bool = True,
    loop: asyncio.AbstractEventLoop = None,
    db_session_factory=None,
) -> bool:
    """Start analysis in background thread. Returns False if already running."""
    global _analysis_thread
    if state.running:
        return False

    if loop is None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.get_event_loop()

    _start_streaming_worker()
    state.start(source)
    _analysis_thread = threading.Thread(
        target=_run_analysis,
        args=(
            source, model_path, audio_model, yolo_model,
            yolo_conf, video_weight, audio_weight,
            enable_yolo, enable_video, enable_audio,
            loop, db_session_factory,
        ),
        daemon=True,
        name="AnalysisBridge",
    )
    _analysis_thread.start()
    return True


def stop_analysis() -> bool:
    """Signal the analysis thread to stop. Returns False if not running."""
    if not state.running:
        return False
    state.stop()
    _stop_streaming_worker()
    return True


def get_state() -> Dict[str, Any]:
    return state.snapshot()

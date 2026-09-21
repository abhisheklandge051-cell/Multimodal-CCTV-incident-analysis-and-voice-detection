import os
import sys
import io
import time
import queue
import logging
import threading
from collections import deque
from typing import Optional, Tuple, List, Dict, Any

import cv2
import numpy as np
import requests
import imageio
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from ultralytics import YOLO

# Project root and module resolution
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

for p in [PROJECT_ROOT, SCRIPT_DIR, os.path.join(PROJECT_ROOT, "Backend", "Video")]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from backend.video.loading_processing import LABEL_MAP, REVERSE_LABEL_MAP
    from backend.video.fast_gru_resnet_training import FastGRUResNetDetector
    from backend.video.auth_manager import AuthManager, TrackStatus
except ImportError:
    from loading_processing import LABEL_MAP, REVERSE_LABEL_MAP
    from fast_gru_resnet_training import FastGRUResNetDetector
    from auth_manager import AuthManager, TrackStatus


def resolve_model_path(model_path: str) -> str:
    """
    Resolves relative model paths cleanly against:
    1. Exact/Absolute path if it exists
    2. Current Working Directory
    3. PROJECT_ROOT directory
    4. SCRIPT_DIR directory
    """
    if not model_path:
        return model_path
    if os.path.isabs(model_path) and os.path.exists(model_path):
        return model_path
    if os.path.exists(model_path):
        return os.path.abspath(model_path)

    root_candidate = os.path.join(PROJECT_ROOT, model_path)
    if os.path.exists(root_candidate):
        return root_candidate

    script_candidate = os.path.join(SCRIPT_DIR, model_path)
    if os.path.exists(script_candidate):
        return script_candidate

    return os.path.join(PROJECT_ROOT, model_path)


def _send_ntfy_alert_worker(
    frames_snapshot: List[np.ndarray],
    incident_label: str,
    confidence: float,
    ntfy_url: str = "https://ntfy.sh/incident_alert_shubh",
    duration: float = 0.1,
    sample_stride: int = 3,
    target_size: Tuple[int, int] = (320, 240),
    custom_title: Optional[str] = None,
    custom_message: Optional[str] = None,
    priority: str = "urgent"
):
    """
    Asynchronous background worker that downsamples the rolling buffer (every 3rd frame),
    resizes each sampled frame to 320x240, converts to RGB, and rapidly encodes an in-memory GIF
    (subrectangles=True, duration=0.1) under 400 KB for low-latency network dispatch to ntfy.sh.

    Runs entirely in a separate daemon thread to ensure zero frame drops on live inference.
    """
    try:
        if not frames_snapshot:
            logger.warning("Alert worker received an empty frame buffer snapshot; skipping dispatch.")
            return

        start_time = time.time()

        # 1. Downsample frame list by taking every 3rd frame (~20-25 frames from 75)
        sampled_frames = frames_snapshot[::sample_stride] if len(frames_snapshot) > 25 else frames_snapshot
        logger.info(
            f"Generating low-latency incident GIF for '{incident_label}' "
            f"({len(sampled_frames)} subsampled frames from {len(frames_snapshot)} buffered)..."
        )

        # 2. Resize each sampled frame to 320x240, convert BGR to RGB, and ensure contiguous uint8
        rgb_frames: List[np.ndarray] = []
        for f in sampled_frames:
            if not isinstance(f, np.ndarray):
                continue
            # Ensure uint8 values in range [0, 255]
            raw_bgr = f if f.dtype == np.uint8 else np.clip(f, 0, 255).astype(np.uint8)

            # Resize sampled frame to 320x240
            if target_size is not None:
                resized = cv2.resize(raw_bgr, target_size, interpolation=cv2.INTER_AREA)
            else:
                resized = raw_bgr

            # Convert color: cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            if len(resized.shape) == 3 and resized.shape[2] == 3:
                rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            elif len(resized.shape) == 2:
                rgb_frame = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
            else:
                rgb_frame = resized

            # Verify data type is contiguous uint8
            rgb_frame = np.ascontiguousarray(rgb_frame, dtype=np.uint8)
            rgb_frames.append(rgb_frame)

        if not rgb_frames:
            logger.warning("No valid frames could be converted to RGB for GIF; skipping dispatch.")
            return

        # 3. Fast In-Memory GIF Compilation using duration=0.1 and subrectangles=True
        gif_buffer = io.BytesIO()
        imageio.mimsave(
            gif_buffer,
            rgb_frames,
            format="GIF",
            duration=duration,
            subrectangles=True
        )
        gif_bytes = gif_buffer.getvalue()
        gif_size_kb = len(gif_bytes) / 1024.0

        elapsed_gif = time.time() - start_time
        logger.info(f"Animated GIF compiled in {elapsed_gif:.2f}s ({gif_size_kb:.1f} KB). Dispatching to {ntfy_url}...")

        title_str = custom_title if custom_title else f"Automated Alert: {incident_label}"
        msg_str = (
            custom_message
            if custom_message
            else f"Abnormal incident '{incident_label}' detected with {confidence * 100:.1f}% confidence."
        )
        headers = {
            "Title": title_str.encode("utf-8"),
            "Priority": priority,
            "Filename": "incident.gif",
            "Message": msg_str.encode("utf-8")
        }

        # HTTP PUT request with in-memory GIF binary data
        response = requests.put(
            ntfy_url,
            data=gif_bytes,
            headers=headers,
            timeout=15
        )

        if response.status_code == 200:
            logger.info(f"Alert for '{incident_label}' dispatched successfully to ntfy.sh (HTTP 200).")
        else:
            logger.warning(f"ntfy.sh alert dispatch returned HTTP {response.status_code}: {response.text}")

    except Exception as exc:
        logger.error(f"Failed to generate or dispatch incident alert: {exc}", exc_info=True)


def _incident_analyzer_worker(
    analyzer: Any,
    frame_queue: queue.Queue,
    incident_state: Dict[str, Any],
    state_lock: threading.Lock,
    stop_event: threading.Event
):
    """
    Dedicated background worker thread executing ResNet-50 feature extraction
    and FastGRU temporal classification asynchronously on NVIDIA CUDA GPU.
    Continuously updates incident_state dictionary without blocking the main camera display loop.
    """
    logger.info(f"Background LiveIncidentAnalyzer worker thread started on device: {analyzer.device}")
    while not stop_event.is_set():
        try:
            # Wait for next frame with short timeout so thread can exit cleanly on stop_event
            frame = frame_queue.get(timeout=0.05)
        except queue.Empty:
            continue

        try:
            label, conf, probs = analyzer.process_frame(frame)
            with state_lock:
                incident_state["label"] = label
                incident_state["conf"] = conf
                incident_state["probs"] = probs
                incident_state["timestamp"] = time.time()
        except Exception as e:
            logger.error(f"Error in background incident analyzer worker: {e}", exc_info=True)
        finally:
            frame_queue.task_done()

    logger.info("Background LiveIncidentAnalyzer worker thread stopped.")


os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8"
os.environ["AV_LOG_FORCE_NOCOLOR"] = "1"
if hasattr(cv2, "setLogLevel"):
    cv2.setLogLevel(0)
if hasattr(cv2, "utils") and hasattr(cv2.utils, "logging"):
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")
logger = logging.getLogger("Live_Camera_Test")

# Color palette for incident classes (BGR for OpenCV)
INCIDENT_COLORS = {
    "Abuse": (0, 0, 255),         # Red
    "CarAccident": (0, 140, 255),  # Orange-Red
    "Explosion": (0, 69, 255),    # Red-Orange
    "Fighting": (0, 165, 255),    # Orange
    "Normal": (0, 220, 0),        # Bright Green
    "Riot": (0, 255, 255),        # Yellow
    "Shooting": (0, 0, 200)       # Dark Red
}
DEFAULT_COLOR = (255, 255, 255)


class YOLOv8ObjectDetector:
    """
    YOLOv8 / YOLO object detection module for real-time video frames.
    """
    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        conf_thres: float = 0.35,
        device: Optional[str] = None
    ):
        self.conf_thres = conf_thres
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        resolved_model_path = resolve_model_path(model_path)
        logger.info(f"Initializing YOLOv8 object detector ('{resolved_model_path}') on {self.device}...")
        try:
            self.model = YOLO(resolved_model_path)
            self.model.to(self.device)
            is_cuda = (self.device == "cuda" or (isinstance(self.device, torch.device) and self.device.type == "cuda"))
            if is_cuda and hasattr(self.model, "model") and self.model.model is not None:
                try:
                    self.model.model.half()
                except Exception as ex:
                    logger.debug(f"Could not convert YOLO model to FP16 half: {ex}")
            logger.info("YOLOv8 object detector loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load YOLO model from '{resolved_model_path}': {e}")
            raise

    @torch.inference_mode()
    def detect(self, frame_bgr: np.ndarray, track: bool = True) -> List[Dict]:
        """
        Runs object detection and real-time multi-object tracking (ByteTrack) on a BGR frame.
        Uses FP16 half-precision on CUDA for maximum inference throughput (30+ FPS).
        Returns a list of dicts with keys: 'box', 'class_id', 'label', 'conf', 'track_id'.
        """
        if track:
            try:
                results = self.model.track(
                    source=frame_bgr,
                    conf=self.conf_thres,
                    persist=True,
                    verbose=False,
                    device=self.device
                )
            except Exception as e:
                logger.debug(f"YOLO tracking fallback to predict: {e}")
                results = self.model.predict(
                    source=frame_bgr,
                    conf=self.conf_thres,
                    verbose=False,
                    device=self.device
                )
        else:
            results = self.model.predict(
                source=frame_bgr,
                conf=self.conf_thres,
                verbose=False,
                device=self.device
            )

        detections = []
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            names = results[0].names

            for box in boxes:
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                label = names.get(cls_id, f"obj_{cls_id}")
                track_id = int(box.id[0].cpu().numpy()) if (box.id is not None and len(box.id) > 0) else None

                detections.append({
                    "box": (int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])),
                    "class_id": cls_id,
                    "label": label,
                    "conf": conf,
                    "track_id": track_id
                })

        return detections

    def draw_detections(
        self,
        frame: np.ndarray,
        detections: List[Dict],
        auth_manager: Optional[Any] = None,
        is_emergency: bool = False
    ) -> Tuple[np.ndarray, Dict[str, int]]:
        """
        Draws bounding boxes and caption badges for detected objects on the frame.
        When auth_manager is provided:
        - Normal safe state: Renders Green bounding box + "AUTH SAFE: <NAME>".
        - Emergency state: Renders Amber/Red bounding box + "EMERGENCY: <NAME>".
        - Unverified state: Renders Red bounding box + "MONITORED: UNKNOWN [ID:<tid>]".
        - Pending state: Renders Orange bounding box + "PENDING: VERIFYING... [ID:<tid>]".
        Returns modified frame and object counts map.
        """
        counts: Dict[str, int] = {}
        for det in detections:
            x1, y1, x2, y2 = det["box"]
            label = det["label"]
            conf = det["conf"]
            track_id = det.get("track_id")
            counts[label] = counts.get(label, 0) + 1

            # Check if this person has an authorization status in AuthManager
            auth_state = None
            is_person = (label.lower() in ("person", "human"))
            if auth_manager and is_person and track_id is not None:
                auth_state = auth_manager.authorized_track_ids.get(track_id)

            if auth_state:
                status = auth_state.get("status", "PENDING")
                identity = auth_state.get("identity", "Unknown")
                if status == "AUTHORIZED":
                    if is_emergency:
                        color = (0, 69, 255)  # Amber/Red Emergency
                        caption = f"EMERGENCY: {identity.upper()}"
                    else:
                        color = (0, 220, 0)  # Bright Green
                        caption = f"AUTH SAFE: {identity.upper()}"
                elif status == "UNAUTHORIZED":
                    color = (0, 0, 255)  # Red Alert
                    caption = f"MONITORED: UNKNOWN [ID:{track_id}]"
                else:
                    color = (0, 165, 255)  # Amber / Orange for Pending
                    caption = f"PENDING: VERIFYING... [ID:{track_id}]"
            else:
                color = self.get_class_color(label)
                caption = f"{label} {conf * 100:.0f}%" + (f" [ID:{track_id}]" if track_id is not None else "")

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Prepare text badge
            (tw, th), baseline = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)

            badge_y1 = max(0, y1 - th - baseline - 4)
            badge_y2 = y1 if y1 - th - baseline - 4 >= 0 else y1 + th + baseline + 4

            cv2.rectangle(frame, (x1, badge_y1), (x1 + tw + 6, badge_y2), color, -1)
            text_color = (0, 0, 0) if (sum(color) > 400 and color != (0, 69, 255)) else (255, 255, 255)
            cv2.putText(
                frame,
                caption,
                (x1 + 3, badge_y1 + th + 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                text_color,
                1,
                cv2.LINE_AA
            )

        return frame, counts

    @staticmethod
    def get_class_color(label: str) -> Tuple[int, int, int]:
        """
        Generates dynamic, distinct BGR colors per class.
        Highlights security sensitive classes in red.
        """
        lower_label = label.lower()
        if any(w in lower_label for w in ["knife", "gun", "weapon", "fire", "rifle"]):
            return (0, 0, 255)  # Bright Red Alert
        if lower_label in ["person", "human"]:
            return (255, 140, 0)  # Vivid Orange/Blue
        if lower_label in ["car", "truck", "bus", "motorcycle", "bicycle"]:
            return (0, 255, 255)  # Yellow

        h = hash(label) & 0xFFFFFF
        r = max(100, (h & 0xFF0000) >> 16)
        g = max(100, (h & 0x00FF00) >> 8)
        b = max(100, h & 0x0000FF)
        return (b, g, r)  # BGR


class LiveIncidentAnalyzer:
    """
    Temporal Incident Classifier using ResNet-50 backbone + Fast GRU.
    Optimized for high-FPS live inference with temporal striding, FP16 AMP, and torch.inference_mode().
    """
    def __init__(
        self,
        model_path: str = "best_fast_gru_resnet_model.pth",
        clip_len: int = 16,
        target_short_side: int = 256,
        ema_alpha: float = 0.3,
        frame_stride: int = 1,
        device: Optional[Any] = None
    ):
        self.clip_len = clip_len
        self.target_short_side = target_short_side
        self.ema_alpha = ema_alpha
        self.frame_stride = max(1, frame_stride)
        self.frame_count = 0

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        self.use_amp = (self.device.type == "cuda")
        self.is_active = False

        resolved_model_path = resolve_model_path(model_path)
        if not os.path.exists(resolved_model_path):
            logger.warning(f"Incident classification model '{model_path}' not found at '{resolved_model_path}'! Incident classification will be disabled.")
            return

        logger.info(f"Loading ResNet-50 backbone on CUDA device: {self.device} (AMP FP16={self.use_amp}, stride={self.frame_stride})")
        self.backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.backbone.fc = nn.Identity()
        self.backbone.to(self.device)
        self.backbone.eval()
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        num_classes = len(LABEL_MAP)
        logger.info(f"Loading FastGRUResNetDetector on CUDA device: {self.device} from '{resolved_model_path}'...")
        self.model = FastGRUResNetDetector(
            feature_dim=2048,
            hidden_dim=256,
            num_classes=num_classes,
            dropout_rate=0.4
        ).to(self.device)

        state_dict = torch.load(resolved_model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        self.feature_buffer = deque(maxlen=clip_len)
        self.ema_probs: Optional[np.ndarray] = None
        self.last_prediction: Tuple[str, float, Optional[np.ndarray]] = ("Normal", 0.0, None)
        self.is_active = True

    @torch.inference_mode()
    def extract_frame_feature(self, frame_bgr: np.ndarray) -> np.ndarray:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w, _ = frame_rgb.shape

        if h < w:
            new_h = self.target_short_side
            new_w = int(w * (self.target_short_side / h))
        else:
            new_w = self.target_short_side
            new_h = int(h * (self.target_short_side / w))

        frame_resized = cv2.resize(frame_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        t = torch.from_numpy(frame_resized).to(self.device, non_blocking=True).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        t = F.center_crop(t, (224, 224))
        t = self.normalize(t)

        with torch.inference_mode():
            if self.use_amp:
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    feature = self.backbone(t).squeeze(0).float().cpu().numpy()
            else:
                feature = self.backbone(t).squeeze(0).cpu().numpy()
        return feature

    @torch.inference_mode()
    def predict_clip(self) -> Tuple[str, float, np.ndarray]:
        if len(self.feature_buffer) < self.clip_len:
            feats = list(self.feature_buffer)
            while len(feats) < self.clip_len:
                feats.append(feats[-1] if len(feats) > 0 else np.zeros((2048,), dtype=np.float32))
            clip_arr = np.array(feats, dtype=np.float32)
        else:
            clip_arr = np.array(self.feature_buffer, dtype=np.float32)

        clip_tensor = torch.from_numpy(clip_arr).unsqueeze(0).to(self.device, non_blocking=True)
        with torch.inference_mode():
            if self.use_amp:
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    logits = self.model(clip_tensor)
                    probs = torch.softmax(logits.float(), dim=1).squeeze(0).cpu().numpy()
            else:
                logits = self.model(clip_tensor)
                probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        if self.ema_probs is None:
            self.ema_probs = probs
        else:
            self.ema_probs = self.ema_alpha * probs + (1.0 - self.ema_alpha) * self.ema_probs

        top_idx = int(np.argmax(self.ema_probs))
        top_label = REVERSE_LABEL_MAP.get(top_idx, "Unknown")
        confidence = float(self.ema_probs[top_idx])

        return top_label, confidence, self.ema_probs

    def process_frame(self, frame_bgr: np.ndarray) -> Tuple[str, float, Optional[np.ndarray]]:
        if not self.is_active:
            return "N/A", 0.0, None

        self.frame_count += 1
        # Temporal striding: if frame_stride > 1, reuse latest prediction on skipped frames
        if (self.frame_stride > 1) and (self.frame_count % self.frame_stride != 0) and (self.last_prediction[2] is not None):
            return self.last_prediction

        feat = self.extract_frame_feature(frame_bgr)
        self.feature_buffer.append(feat)
        top_label, confidence, probs = self.predict_clip()
        self.last_prediction = (top_label, confidence, probs)
        return self.last_prediction


def draw_hud_overlay(
    frame: np.ndarray,
    incident_label: str,
    incident_conf: float,
    incident_probs: Optional[np.ndarray],
    object_counts: Dict[str, int],
    fps: float,
    alert_status: Optional[str] = None
) -> np.ndarray:
    """
    Renders a modern semi-transparent HUD overlay with incident monitoring banner,
    YOLOv8 detected object summary, incident confidence breakdown panel, and alert transmission badges.
    """
    h, w, _ = frame.shape
    overlay = frame.copy()

    # 1. Top banner backdrop
    banner_height = 85
    cv2.rectangle(overlay, (0, 0), (w, banner_height), (18, 18, 18), -1)
    cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

    # Incident Badge / Status
    if incident_probs is not None:
        color = INCIDENT_COLORS.get(incident_label, DEFAULT_COLOR)
        badge_text = f"{incident_label.upper()} ({incident_conf * 100:.1f}%)"
        cv2.putText(frame, "INCIDENT MONITORING", (15, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)
        cv2.putText(frame, badge_text, (15, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2, cv2.LINE_AA)
    else:
        cv2.putText(frame, "REAL-TIME OBJECT & INCIDENT MONITOR", (15, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)
        cv2.putText(frame, "YOLOv8 ACTIVE", (15, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 0), 2, cv2.LINE_AA)

    # Object summary text in banner
    if object_counts:
        counts_str = ", ".join([f"{count} {label}{'s' if count > 1 and not label.endswith('s') else ''}" for label, count in object_counts.items()])
        obj_text = f"Objects: {counts_str}"
    else:
        obj_text = "Objects: None detected"

    cv2.putText(frame, obj_text[:65], (15, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)

    # FPS counter top right
    fps_str = f"FPS: {fps:.1f}"
    cv2.putText(frame, fps_str, (w - 125, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

    # Alert Transmission / Cooldown status pill
    if alert_status:
        if "CRITICAL" in alert_status or "SOS" in alert_status or "EMERGENCY" in alert_status:
            badge_bg = (0, 69, 255)      # Amber-Red Emergency
            badge_fg = (255, 255, 255)
        elif "SENT" in alert_status or "TRIGGERED" in alert_status:
            badge_bg = (0, 0, 255)       # Red alert
            badge_fg = (255, 255, 255)
        elif "SAFE" in alert_status or "AUTH" in alert_status or "BYPASS" in alert_status:
            badge_bg = (0, 200, 0)       # Green for authorized safe
            badge_fg = (0, 0, 0)
        else:
            badge_bg = (0, 140, 255)     # Orange cooldown / confirming
            badge_fg = (255, 255, 255)

        (stw, sth), baseline = cv2.getTextSize(alert_status, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        bx1 = w - 140 - stw - 20
        by1 = 14
        bx2 = bx1 + stw + 14
        by2 = by1 + sth + 10
        if bx1 > 280:
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), badge_bg, -1)
            cv2.putText(frame, alert_status, (bx1 + 7, by1 + sth + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, badge_fg, 1, cv2.LINE_AA)

    # 2. Side Panel for Incident Probability Breakdown (if incident analyzer active)
    if incident_probs is not None:
        panel_w = 260
        panel_h = len(LABEL_MAP) * 22 + 35
        panel_x1 = max(10, w - panel_w - 15)
        panel_y1 = banner_height + 15
        panel_x2 = w - 15
        panel_y2 = panel_y1 + panel_h

        side_overlay = frame.copy()
        cv2.rectangle(side_overlay, (panel_x1, panel_y1), (panel_x2, panel_y2), (15, 15, 15), -1)
        cv2.addWeighted(side_overlay, 0.7, frame, 0.3, 0, frame)
        cv2.rectangle(frame, (panel_x1, panel_y1), (panel_x2, panel_y2), (60, 60, 60), 1)

        cv2.putText(frame, "INCIDENT BREAKDOWN", (panel_x1 + 10, panel_y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

        y_offset = panel_y1 + 40
        for idx in range(len(LABEL_MAP)):
            c_name = REVERSE_LABEL_MAP[idx]
            p_val = float(incident_probs[idx])
            c_color = INCIDENT_COLORS.get(c_name, DEFAULT_COLOR)

            cv2.putText(frame, f"{c_name[:11]:<11}", (panel_x1 + 10, y_offset + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1, cv2.LINE_AA)

            bar_x = panel_x1 + 95
            bar_max_w = 100
            bar_w = int(bar_max_w * p_val)

            cv2.rectangle(frame, (bar_x, y_offset + 3), (bar_x + bar_max_w, y_offset + 14), (40, 40, 40), -1)
            if bar_w > 0:
                cv2.rectangle(frame, (bar_x, y_offset + 3), (bar_x + bar_w, y_offset + 14), c_color, -1)

            cv2.putText(frame, f"{p_val * 100:.0f}%", (bar_x + bar_max_w + 8, y_offset + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)

            y_offset += 22

    return frame


def run_live_camera_test(
    source: str = "0",
    model_path: str = "best_fast_gru_resnet_model.pth",
    yolo_model: str = "yolov8n.pt",
    yolo_conf: float = 0.35,
    enable_yolo: bool = True,
    enable_incident: bool = True,
    save_output: Optional[str] = None,
    ntfy_url: str = "https://ntfy.sh/incident_alert_shubh",
    alert_threshold: float = 0.85,
    alert_cooldown: float = 15.0,
    min_consecutive_frames: int = 5,
    enable_alerts: bool = True,
    enable_auth: bool = True,
    faces_dir: str = "authorized_faces",
    whitelist_cache: str = "whitelist_embeddings.pkl",
    auth_tolerance: float = 0.48,
    auth_max_lost_frames: int = 60,
    incident_stride: int = 2,
    enable_flip: bool = True
):
    """
    Main loop for live camera / video stream object detection, tracking, authorized personnel verification,
    and incident classification with rolling buffer and automated asynchronous ntfy alerts.
    """
    # 1. Initialize temporal incident classifier with frame striding & FP16 acceleration
    incident_analyzer = None
    if enable_incident:
        resolved_incident_model = resolve_model_path(model_path)
        incident_analyzer = LiveIncidentAnalyzer(
            model_path=resolved_incident_model,
            frame_stride=incident_stride
        )

    # 2. Initialize YOLOv8 object detector
    yolo_detector = None
    if enable_yolo:
        try:
            resolved_yolo = resolve_model_path(yolo_model)
            if not os.path.exists(resolved_yolo):
                fallback_yolo = resolve_model_path("yolo11n.pt")
                if os.path.exists(fallback_yolo):
                    resolved_yolo = fallback_yolo
            yolo_detector = YOLOv8ObjectDetector(model_path=resolved_yolo, conf_thres=yolo_conf)
        except Exception as e:
            logger.error(f"Could not initialize YOLO object detector: {e}")

    # 3. Initialize Authorized Personnel / Owner Whitelist Manager
    auth_manager = None
    if enable_auth:
        try:
            resolved_faces = resolve_model_path(faces_dir)
            resolved_cache = resolve_model_path(whitelist_cache)
            auth_manager = AuthManager(
                faces_dir=resolved_faces,
                cache_file=resolved_cache,
                tolerance=auth_tolerance,
                max_lost_frames=auth_max_lost_frames
            )
            logger.info(
                f"AuthManager initialized. Whitelist identities: "
                f"{sorted(set(auth_manager.known_identities))} (tolerance={auth_tolerance})"
            )
        except Exception as e:
            logger.warning(f"Could not initialize AuthManager: {e}. Whitelist filtering disabled.")

    video_source = int(source) if source.isdigit() else source

    logger.info(f"Opening video source: '{source}'...")
    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        logger.error(f"Failed to open video source: '{source}'")
        return

    writer = None
    if save_output:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(save_output, fourcc, fps_in, (w, h))
        logger.info(f"Saving output video to: '{save_output}'")

    # In-Memory Rolling Frame Buffer (maxlen=75 frames, ~3s of video at 25-30 FPS)
    frame_buffer: deque = deque(maxlen=75)

    # 4. Decouple Camera & Incident Analysis with Asynchronous Worker Thread
    analyzer_queue: Optional[queue.Queue] = None
    analyzer_thread: Optional[threading.Thread] = None
    analyzer_stop_event = threading.Event()
    incident_lock = threading.Lock()
    latest_incident_state: Dict[str, Any] = {
        "label": "Normal",
        "conf": 0.0,
        "probs": None,
        "timestamp": time.time()
    }

    if incident_analyzer and incident_analyzer.is_active:
        analyzer_queue = queue.Queue(maxsize=1)
        analyzer_thread = threading.Thread(
            target=_incident_analyzer_worker,
            args=(
                incident_analyzer,
                analyzer_queue,
                latest_incident_state,
                incident_lock,
                analyzer_stop_event
            ),
            daemon=True
        )
        analyzer_thread.start()

    # Alert cooldown, status, and temporal persistence tracking
    last_alert_time: float = 0.0
    last_alert_label: str = ""
    last_alert_type: Optional[str] = None
    consecutive_anomaly_count: int = 0
    consecutive_anomaly_label: Optional[str] = None
    window_name = "Real-Time YOLOv8 Object Detection & Incident Monitor"

    print("\n" + "=" * 80)
    print("LIVE REAL-TIME OBJECT DETECTION & INCIDENT MONITORING")
    print(f"Alert notifications: {'ENABLED' if enable_alerts else 'DISABLED'} -> {ntfy_url}")
    print(f"Alert cooldown: {alert_cooldown:.1f}s | Confidence threshold: {alert_threshold * 100:.0f}%")
    print(f"Buffer capacity: {frame_buffer.maxlen} frames | Min consecutive anomaly frames: {min_consecutive_frames}")
    print(f"Incident worker: ASYNC BACKGROUND THREAD (RTX 2050 CUDA) | Mirror flip: {'ON' if enable_flip else 'OFF'}")
    if auth_manager:
        num_identities = len(set(auth_manager.known_identities))
        print(f"Authorized Whitelist: ENABLED ({len(auth_manager.known_encodings)} face embeddings, {num_identities} identities, tol={auth_tolerance})")
    else:
        print("Authorized Whitelist: DISABLED")
    print("Press 'q' or 'ESC' on the video window to exit cleanly.")
    print("=" * 80 + "\n")

    prev_time = time.time()
    fps = 0.0
    frame_idx: int = 0

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                logger.info("Video stream ended or frame unreadable.")
                break

            # 1. Horizontal Mirror / Camera Flip for natural preview
            if enable_flip:
                frame = cv2.flip(frame, 1)

            frame_idx += 1

            # Buffer raw, unnormalized OpenCV BGR uint8 frame before overlays
            frame_buffer.append(frame.copy())

            curr_time = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(1e-5, curr_time - prev_time))
            prev_time = curr_time

            # 2. YOLOv8 Real-Time Object Detection & Multi-Object Tracking (ByteTrack)
            object_counts: Dict[str, int] = {}
            detections: List[Dict] = []
            active_person_track_ids: List[int] = []
            authorized_person_names: List[str] = []

            if yolo_detector:
                detections = yolo_detector.detect(frame, track=True)

                if auth_manager:
                    for det in detections:
                        label = det.get("label", "").lower()
                        track_id = det.get("track_id")
                        if label in ("person", "human") and track_id is not None:
                            active_person_track_ids.append(track_id)
                            st = auth_manager.update_track(
                                track_id=track_id,
                                frame_bgr=frame,
                                bbox=det["box"],
                                frame_idx=frame_idx
                            )
                            if st and st.get("status") == "AUTHORIZED":
                                identity = st.get("identity") or "Authorized"
                                authorized_person_names.append(identity)

                    if frame_idx % 15 == 0:
                        auth_manager.cleanup_lost_tracks(current_frame_idx=frame_idx)

            # 3. Asynchronous Incident Analysis Handoff (Decoupled CUDA Worker)
            if analyzer_queue is not None:
                try:
                    analyzer_queue.put_nowait(frame.copy())
                except queue.Full:
                    try:
                        analyzer_queue.get_nowait()
                    except queue.Empty:
                        pass
                    try:
                        analyzer_queue.put_nowait(frame.copy())
                    except queue.Full:
                        pass

            # Instant non-blocking read of latest incident prediction from background worker
            with incident_lock:
                inc_label = latest_incident_state["label"]
                inc_conf = latest_incident_state["conf"]
                inc_probs = latest_incident_state["probs"]

            # 4. Incident Trigger & Alert Dispatch (Owner Protection & Threat Logic)
            time_since_alert = curr_time - last_alert_time
            alert_status: Optional[str] = None
            buffer_is_full = len(frame_buffer) >= (frame_buffer.maxlen or 75)
            has_authorized_user = len(authorized_person_names) > 0
            auth_names_str = ", ".join(sorted(set(authorized_person_names))) if has_authorized_user else ""
            is_emergency: bool = False

            # Default status during cooldown or buffering
            if last_alert_time > 0:
                if time_since_alert < 3.0:
                    alert_status = f"ALERT SENT: {last_alert_label.upper()}"
                elif time_since_alert < alert_cooldown:
                    remaining_cooldown = int(alert_cooldown - time_since_alert)
                    alert_status = f"COOLDOWN: {remaining_cooldown}s"
            elif not buffer_is_full:
                alert_status = f"BUFFERING: {len(frame_buffer)}/{frame_buffer.maxlen or 75}"

            if enable_alerts and incident_analyzer and incident_analyzer.is_active:
                BENIGN_LABELS = {"Normal", "Normal_Videos_event", "Normal_Videos", "None", "N/A", "Unknown", None}
                is_anomaly = (
                    inc_label not in BENIGN_LABELS
                    and inc_conf >= alert_threshold
                )

                if is_anomaly:
                    if has_authorized_user:
                        # CASE 1: Incident detected involving AUTHORIZED user -> SOS EMERGENCY!
                        is_emergency = True

                        if inc_label == consecutive_anomaly_label and last_alert_type == "SOS":
                            consecutive_anomaly_count += 1
                        else:
                            consecutive_anomaly_label = inc_label
                            consecutive_anomaly_count = 1
                            last_alert_type = "SOS"

                        # HUD status during confirmation / emergency
                        if (
                            buffer_is_full
                            and time_since_alert >= alert_cooldown
                            and 0 < consecutive_anomaly_count < min_consecutive_frames
                        ):
                            alert_status = f"CRITICAL: {auth_names_str.upper()} ({consecutive_anomaly_count}/{min_consecutive_frames})"
                        elif buffer_is_full and time_since_alert >= alert_cooldown:
                            alert_status = f"CRITICAL SOS: {auth_names_str.upper()} ({inc_label.upper()})"

                        # Trigger SOS alert
                        if (
                            buffer_is_full
                            and consecutive_anomaly_count >= min_consecutive_frames
                            and time_since_alert >= alert_cooldown
                        ):
                            last_alert_time = curr_time
                            last_alert_label = f"SOS: {auth_names_str}"
                            consecutive_anomaly_count = 0
                            consecutive_anomaly_label = None
                            logger.warning(
                                f"🚨 CRITICAL SOS ALERT TRIGGERED: Incident '{inc_label}' (Conf: {inc_conf * 100:.1f}%) "
                                f"involving authorized user '{auth_names_str}' confirmed across {min_consecutive_frames}+ frames! "
                                f"Dispatching urgent SOS alert to {ntfy_url}..."
                            )
                            sos_title = f"CRITICAL: Incident detected involving authorized user {auth_names_str}!"
                            sos_msg = (
                                f"EMERGENCY: Abnormal incident '{inc_label}' detected involving authorized user {auth_names_str} "
                                f"with {inc_conf * 100:.1f}% confidence."
                            )
                            buffer_snapshot = list(frame_buffer)
                            alert_thread = threading.Thread(
                                target=_send_ntfy_alert_worker,
                                args=(buffer_snapshot, inc_label, inc_conf, ntfy_url),
                                kwargs={"custom_title": sos_title, "custom_message": sos_msg, "priority": "urgent"},
                                daemon=True
                            )
                            alert_thread.start()

                    else:
                        # CASE 2: Incident detected with UNAUTHORIZED / PENDING / NO authorized user -> Standard threat alert
                        is_emergency = False

                        if inc_label == consecutive_anomaly_label and last_alert_type == "THREAT":
                            consecutive_anomaly_count += 1
                        else:
                            consecutive_anomaly_label = inc_label
                            consecutive_anomaly_count = 1
                            last_alert_type = "THREAT"

                        if (
                            buffer_is_full
                            and time_since_alert >= alert_cooldown
                            and 0 < consecutive_anomaly_count < min_consecutive_frames
                        ):
                            alert_status = f"CONFIRMING: {consecutive_anomaly_label.upper()} ({consecutive_anomaly_count}/{min_consecutive_frames})"

                        # Trigger Standard threat alert
                        if (
                            buffer_is_full
                            and consecutive_anomaly_count >= min_consecutive_frames
                            and time_since_alert >= alert_cooldown
                        ):
                            last_alert_time = curr_time
                            last_alert_label = inc_label
                            consecutive_anomaly_count = 0
                            consecutive_anomaly_label = None
                            logger.warning(
                                f"INCIDENT TRIGGERED: '{inc_label}' (Conf: {inc_conf * 100:.1f}%) "
                                f"confirmed across {min_consecutive_frames}+ consecutive frames! "
                                f"Dispatching threat alert to {ntfy_url} via background thread..."
                            )
                            buffer_snapshot = list(frame_buffer)
                            alert_thread = threading.Thread(
                                target=_send_ntfy_alert_worker,
                                args=(buffer_snapshot, inc_label, inc_conf, ntfy_url),
                                daemon=True
                            )
                            alert_thread.start()

                else:
                    # CASE 3: NO anomaly detected
                    consecutive_anomaly_count = 0
                    consecutive_anomaly_label = None
                    last_alert_type = None

                    # If authorized user is present and safe:
                    if has_authorized_user and (not alert_status or ("BUFFERING" not in alert_status and "COOLDOWN" not in alert_status and "ALERT SENT" not in alert_status)):
                        alert_status = f"AUTH SAFE: {auth_names_str.upper()}"

            # 5. Render Detections & Tracking with Authorization & Emergency Overlay
            if yolo_detector:
                frame, object_counts = yolo_detector.draw_detections(
                    frame,
                    detections,
                    auth_manager=auth_manager,
                    is_emergency=is_emergency
                )

            # 6. Draw HUD Overlay
            annotated_frame = draw_hud_overlay(
                frame,
                incident_label=inc_label,
                incident_conf=inc_conf,
                incident_probs=inc_probs,
                object_counts=object_counts,
                fps=fps,
                alert_status=alert_status
            )

            if writer:
                writer.write(annotated_frame)

            cv2.imshow(window_name, annotated_frame)

            # Check key presses: ESC, 'q', 'Q'
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q'), ord('Q')):
                logger.info("User requested exit via keyboard key.")
                break

            # Handle user closing the window via the OS window close (X) button
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                logger.info("OpenCV window was closed by user.")
                break

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received (Ctrl+C). Exiting...")
    finally:
        # Clean teardown and resource release
        if analyzer_stop_event is not None:
            analyzer_stop_event.set()
        if analyzer_thread is not None and analyzer_thread.is_alive():
            analyzer_thread.join(timeout=1.0)
        if cap is not None and cap.isOpened():
            cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
        # Pump event loop briefly to ensure OS closes GUI window completely
        for _ in range(5):
            cv2.waitKey(1)
        logger.info("Live test ended cleanly.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Live Video / Camera YOLOv8 Object Detection & Incident Analysis with ntfy Alerts")
    parser.add_argument("--source", type=str, default="0", help="Camera index (0, 1) or path to video file")
    parser.add_argument("--model_path", type=str, default="best_fast_gru_resnet_model.pth", help="Incident classification model weights path")
    parser.add_argument("--yolo_model", type=str, default="yolov8n.pt", help="YOLOv8 model path or name (e.g., yolov8n.pt, yolo11n.pt)")
    parser.add_argument("--yolo_conf", type=float, default=0.35, help="YOLOv8 object detection confidence threshold")
    parser.add_argument("--disable_yolo", action="store_true", help="Disable YOLO object detection")
    parser.add_argument("--disable_incident", action="store_true", help="Disable temporal incident classification")
    parser.add_argument("--disable_alerts", action="store_true", help="Disable automated ntfy incident alerts")
    parser.add_argument("--ntfy_url", type=str, default="https://ntfy.sh/incident_alert_shubh", help="ntfy.sh topic URL for incident alerts")
    parser.add_argument("--alert_threshold", type=float, default=0.85, help="Confidence threshold to trigger incident alert (default: 0.85)")
    parser.add_argument("--alert_cooldown", type=float, default=15.0, help="Cooldown in seconds between alert notifications")
    parser.add_argument("--min_consecutive_frames", type=int, default=5, help="Minimum consecutive frames required to confirm an incident (default: 5)")
    parser.add_argument("--disable_auth", action="store_true", help="Disable authorized personnel whitelist filtering")
    parser.add_argument("--faces_dir", type=str, default="authorized_faces", help="Path to authorized personnel faces directory")
    parser.add_argument("--whitelist_cache", type=str, default="whitelist_embeddings.pkl", help="Path to cached embeddings pickle file")
    parser.add_argument("--auth_tolerance", type=float, default=0.48, help="Face distance tolerance for authorization (default: 0.48)")
    parser.add_argument("--auth_max_lost_frames", type=int, default=60, help="Max lost frames before purging track from auth cache")
    parser.add_argument("--incident_stride", type=int, default=2, help="Temporal frame stride for ResNet feature extraction (default: 2)")
    parser.add_argument("--disable_flip", action="store_true", help="Disable horizontal mirror camera flip")
    parser.add_argument("--save_output", type=str, default=None, help="Optional output video file path")
    args = parser.parse_args()

    run_live_camera_test(
        source=args.source,
        model_path=args.model_path,
        yolo_model=args.yolo_model,
        yolo_conf=args.yolo_conf,
        enable_yolo=not args.disable_yolo,
        enable_incident=not args.disable_incident,
        save_output=args.save_output,
        ntfy_url=args.ntfy_url,
        alert_threshold=args.alert_threshold,
        alert_cooldown=args.alert_cooldown,
        min_consecutive_frames=args.min_consecutive_frames,
        enable_alerts=not args.disable_alerts,
        enable_auth=not args.disable_auth,
        faces_dir=args.faces_dir,
        whitelist_cache=args.whitelist_cache,
        auth_tolerance=args.auth_tolerance,
        auth_max_lost_frames=args.auth_max_lost_frames,
        incident_stride=args.incident_stride,
        enable_flip=not args.disable_flip
    )

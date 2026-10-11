import os
import sys
import time
import threading
import logging
import subprocess
from collections import deque
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Union, Any

# Suppress TensorFlow C++ logs and oneDNN warnings before importing tf
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8"
os.environ["AV_LOG_FORCE_NOCOLOR"] = "1"

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from ultralytics import YOLO

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
try:
    from backend.video.loading_processing import LABEL_MAP, REVERSE_LABEL_MAP
    from backend.video.fast_gru_resnet_training import FastGRUResNetDetector
except ImportError:
    from loading_processing import LABEL_MAP, REVERSE_LABEL_MAP
    from fast_gru_resnet_training import FastGRUResNetDetector

# Audio pipeline imports
try:
    import tensorflow as tf
    from backend.audio.YAMNET_training import pool_audio_features, CLASSES
    from backend.audio.audio_feature_extract import YAMNetFeatureExtractor
    from backend.audio.audio_dataload import FFMPEG_EXE
except ImportError:
    try:
        import tensorflow as tf
        from YAMNET_training import pool_audio_features, CLASSES
        from audio_feature_extract import YAMNetFeatureExtractor
        from audio_dataload import FFMPEG_EXE
    except Exception:
        tf = None
        pool_audio_features = None
        CLASSES = [
            "Abuse", "CarAccident", "Explosion", "Fighting", "Normal", "Riot", "Shooting"
        ]
        YAMNetFeatureExtractor = None
        FFMPEG_EXE = "ffmpeg"

# Optional soundfile and sounddevice for audio processing and live mic
try:
    import soundfile as sf
except ImportError:
    sf = None

try:
    import sounddevice as sd
except ImportError:
    sd = None

try:
    from backend.video.auth_manager import get_auth_manager, AuthManager
except ImportError:
    try:
        from auth_manager import get_auth_manager, AuthManager
    except Exception:
        get_auth_manager = None
        AuthManager = None

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


def resolve_model_path(
    model_path: Union[str, Path],
    default_candidates: Optional[List[str]] = None
) -> Path:
    """
    Robustly resolves model weights and checkpoints across standard project paths:
    1. Exact path or relative to CWD
    2. Project root directory
    3. Script directory (backend/video)
    4. Checkpoints directories (root checkpoints and backend/checkpoints)
    """
    p = Path(model_path)
    if p.exists():
        return p.resolve()

    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent

    search_dirs = [
        Path.cwd(),
        project_root,
        script_dir,
        project_root / "backend" / "checkpoints",
        script_dir.parent / "checkpoints",
        project_root / "checkpoints",
    ]

    for d in search_dirs:
        cand = d / model_path
        if cand.exists():
            return cand.resolve()
        cand_name = d / p.name
        if cand_name.exists():
            return cand_name.resolve()

    if default_candidates:
        for c in default_candidates:
            c_p = Path(c)
            if c_p.exists():
                return c_p.resolve()
            for d in search_dirs:
                cand = d / c
                if cand.exists():
                    return cand.resolve()
                cand_name = d / c_p.name
                if cand_name.exists():
                    return cand_name.resolve()

    return p


class ThreadedCameraReader:
    """
    Dedicated background camera capture thread.
    Reads frames from OpenCV VideoCapture asynchronously at native camera FPS,
    storing only the latest frame.
    read() returns immediately (< 0.05 ms) without waiting for camera hardware exposure,
    completely eliminating camera capture stalls and frame drops in inference loops.
    """
    def __init__(self, source: Union[int, str], width: int = 640, height: int = 480, fps: int = 30):
        self.source = source
        self.width = width
        self.height = height
        self.target_fps = fps
        self.cap = None
        self._open()

        self.ret = False
        self.frame = None
        self.running = False
        self.lock = threading.Lock()
        self.thread = None

        if self.cap is not None and self.cap.isOpened():
            self.running = True
            self.ret, self.frame = self.cap.read()
            self.thread = threading.Thread(target=self._capture_worker, daemon=True, name="ThreadedCameraReader")
            self.thread.start()

    def _open(self):
        src_val = int(self.source) if str(self.source).isdigit() else self.source
        if isinstance(src_val, int):
            self.cap = cv2.VideoCapture(src_val, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(src_val)
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        else:
            self.cap = cv2.VideoCapture(src_val)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def _capture_worker(self):
        while self.running and self.cap is not None and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret or frame is None:
                if isinstance(self.source, str) and not str(self.source).isdigit():
                    self.ret = False
                    break
                time.sleep(0.005)
                continue
            with self.lock:
                self.ret = ret
                self.frame = frame

    def isOpened(self) -> bool:
        return self.cap is not None and self.cap.isOpened()

    def get(self, prop_id: int) -> float:
        if self.cap is not None:
            return self.cap.get(prop_id)
        return 0.0

    def set(self, prop_id: int, value: float) -> bool:
        if self.cap is not None:
            return self.cap.set(prop_id, value)
        return False

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        with self.lock:
            if self.frame is not None:
                return self.ret, self.frame.copy()
            return self.ret, None

    def release(self):
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=0.5)
            self.thread = None
        if self.cap is not None:
            self.cap.release()
            self.cap = None


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

        resolved_path = resolve_model_path(
            model_path,
            default_candidates=["yolov8n.pt", "backend/video/yolov8n.pt", "yolo11n.pt"]
        )
        logger.info(f"Initializing YOLO object detector ('{resolved_path}') on {self.device}...")
        try:
            self.model = YOLO(str(resolved_path))
            if self.device == "cuda":
                try:
                    self.model.to("cuda")
                except Exception as cuda_err:
                    logger.warning(f"CUDA initialization failed for YOLO ({cuda_err}); falling back to CPU.")
                    self.device = "cpu"
                    self.model.to("cpu")
            else:
                self.model.to("cpu")
            logger.info(f"YOLO object detector loaded successfully on {self.device}.")
        except Exception as e:
            logger.error(f"Failed to load YOLO model from '{resolved_path}': {e}")
            raise

    @torch.no_grad()
    def detect(self, frame_bgr: np.ndarray) -> List[Dict]:
        """
        Runs object detection and multi-object tracking on a BGR frame.
        Returns a list of dicts with keys: 'box', 'class_id', 'label', 'conf', 'track_id'.
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return []

        try:
            try:
                results = self.model.track(
                    source=frame_bgr,
                    persist=True,
                    conf=self.conf_thres,
                    imgsz=480,
                    verbose=False,
                    device=self.device
                )
            except Exception:
                results = self.model.predict(
                    source=frame_bgr,
                    conf=self.conf_thres,
                    imgsz=480,
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
                    track_id = int(box.id[0].cpu().numpy()) if (hasattr(box, "id") and box.id is not None) else None

                    detections.append({
                        "box": (int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])),
                        "class_id": cls_id,
                        "label": label,
                        "conf": conf,
                        "track_id": track_id
                    })

            return detections
        except Exception as e:
            logger.warning(f"YOLO object detection inference error: {e}")
            return []

    def draw_detections(self, frame: np.ndarray, detections: List[Dict]) -> Tuple[np.ndarray, Dict[str, int]]:
        """
        Draws bounding boxes and caption badges for detected objects on the frame.
        Returns modified frame and object counts map.
        """
        if frame is None or frame.size == 0:
            return frame, {}

        h, w = frame.shape[:2]
        counts: Dict[str, int] = {}

        for det in detections:
            x1, y1, x2, y2 = det["box"]
            x1, y1 = max(0, min(w - 1, x1)), max(0, min(h - 1, y1))
            x2, y2 = max(0, min(w, x2)), max(0, min(h, y2))
            label = det["label"]
            conf = det["conf"]
            counts[label] = counts.get(label, 0) + 1

            color = self.get_class_color(label)

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Prepare text badge
            caption = f"{label} {conf * 100:.0f}%"
            (tw, th), baseline = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)

            badge_y1 = max(0, y1 - th - baseline - 4)
            badge_y2 = y1 if y1 - th - baseline - 4 >= 0 else min(h, y1 + th + baseline + 4)

            cv2.rectangle(frame, (x1, badge_y1), (min(w, x1 + tw + 6), badge_y2), color, -1)
            text_color = (0, 0, 0) if sum(color) > 400 else (255, 255, 255)
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
    """
    def __init__(
        self,
        model_path: str = "best_fast_gru_resnet_model.pth",
        clip_len: int = 16,
        target_short_side: int = 256,
        ema_alpha: float = 0.3
    ):
        self.clip_len = clip_len
        self.target_short_side = target_short_side
        self.ema_alpha = ema_alpha
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type == "cuda":
            try:
                _t = torch.zeros(1, device=self.device)
                del _t
            except Exception as cuda_err:
                logger.warning(f"CUDA validation failed ({cuda_err}); falling back to CPU.")
                self.device = torch.device("cpu")

        self.is_active = False

        resolved_path = resolve_model_path(
            model_path,
            default_candidates=[
                "best_fast_gru_resnet_model.pth",
                "backend/checkpoints/best_fast_gru_resnet_model.pth",
                "checkpoints/best_fast_gru_resnet_model.pth"
            ]
        )

        if not resolved_path.exists():
            logger.warning(
                f"Incident classification model '{model_path}' not found at '{resolved_path}'! "
                f"Searched paths: [., project root, backend/video, checkpoints]. Video classification disabled."
            )
            return

        logger.info(f"Loading ResNet-50 backbone on device: {self.device}")
        try:
            self.backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
            self.backbone.fc = nn.Identity()
            try:
                self.backbone.to(self.device)
            except Exception as e:
                logger.warning(f"Failed to move ResNet backbone to {self.device}: {e}; falling back to CPU.")
                self.device = torch.device("cpu")
                self.backbone.to(self.device)
            self.backbone.eval()
            self.normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )

            num_classes = len(LABEL_MAP)
            logger.info(f"Loading FastGRUResNetDetector from '{resolved_path}'...")
            try:
                self.model = FastGRUResNetDetector(
                    feature_dim=2048,
                    hidden_dim=256,
                    num_classes=num_classes,
                    dropout_rate=0.4
                ).to(self.device)
            except Exception as e:
                logger.warning(f"Failed to move FastGRU to {self.device}: {e}; falling back to CPU.")
                self.device = torch.device("cpu")
                self.model = FastGRUResNetDetector(
                    feature_dim=2048,
                    hidden_dim=256,
                    num_classes=num_classes,
                    dropout_rate=0.4
                ).to(self.device)

            try:
                state_dict = torch.load(resolved_path, map_location=self.device, weights_only=False)
            except TypeError:
                state_dict = torch.load(resolved_path, map_location=self.device)

            if isinstance(state_dict, dict):
                if "model_state_dict" in state_dict:
                    state_dict = state_dict["model_state_dict"]
                elif "state_dict" in state_dict:
                    state_dict = state_dict["state_dict"]
                elif "model" in state_dict:
                    state_dict = state_dict["model"]

            # Remove possible DataParallel 'module.' prefix
            cleaned_state_dict = {}
            for k, v in state_dict.items():
                clean_k = k[7:] if k.startswith("module.") else k
                cleaned_state_dict[clean_k] = v

            try:
                self.model.load_state_dict(cleaned_state_dict, strict=True)
            except Exception as e:
                logger.warning(f"Strict state_dict load failed ({e}); attempting load with strict=False...")
                self.model.load_state_dict(cleaned_state_dict, strict=False)

            self.model.eval()

            self.feature_buffer = deque(maxlen=clip_len)
            self.ema_probs: Optional[np.ndarray] = None
            self.is_active = True
            logger.info("Video incident classification model loaded successfully.")

        except Exception as e:
            logger.error(f"Failed to load video incident model from '{resolved_path}': {e}")
            self.is_active = False

    @torch.no_grad()
    def extract_frame_feature(self, frame_bgr: np.ndarray) -> np.ndarray:
        if frame_bgr is None or frame_bgr.size == 0:
            return np.zeros((2048,), dtype=np.float32)

        if frame_bgr.ndim == 2:
            frame_bgr = cv2.cvtColor(frame_bgr, cv2.COLOR_GRAY2BGR)

        h, w = frame_bgr.shape[:2]
        if h <= 0 or w <= 0:
            return np.zeros((2048,), dtype=np.float32)

        # Fast Center Crop then resize directly to 224x224 (10x faster than full resize then crop)
        crop_sz = min(h, w)
        sy = (h - crop_sz) // 2
        sx = (w - crop_sz) // 2
        crop = frame_bgr[sy:sy + crop_sz, sx:sx + crop_sz]
        frame_resized = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_LINEAR)
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

        t = F.to_tensor(frame_rgb)
        t = self.normalize(t).unsqueeze(0).to(self.device, non_blocking=True)

        feature = self.backbone(t).squeeze(0).cpu().numpy()
        return feature

    @torch.no_grad()
    def predict_clip(self) -> Tuple[str, float, np.ndarray]:
        if len(self.feature_buffer) < self.clip_len:
            feats = list(self.feature_buffer)
            while len(feats) < self.clip_len:
                feats.append(feats[-1] if len(feats) > 0 else np.zeros((2048,), dtype=np.float32))
            clip_arr = np.array(feats, dtype=np.float32)
        else:
            clip_arr = np.array(self.feature_buffer, dtype=np.float32)

        clip_tensor = torch.from_numpy(clip_arr).unsqueeze(0).to(self.device)
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

        try:
            feat = self.extract_frame_feature(frame_bgr)
            self.feature_buffer.append(feat)
            top_label, confidence, probs = self.predict_clip()
            return top_label, confidence, probs
        except Exception as e:
            logger.warning(f"Video frame analysis error: {e}")
            if self.ema_probs is not None:
                top_idx = int(np.argmax(self.ema_probs))
                return REVERSE_LABEL_MAP.get(top_idx, "Normal"), float(self.ema_probs[top_idx]), self.ema_probs
            return "Normal", 1.0, None

class LiveAudioIncidentAnalyzer:
    """
    Real-time YAMNet Audio Incident Classifier using audio_incident_head.keras.
    Captures or extracts audio streams from video files (via FFmpeg in-memory pipe),
    from separate audio files, or from live microphone (via sounddevice if available).
    Applies temporal pooling (mean_max) and evaluates through audio_incident_head.keras.
    """
    def __init__(
        self,
        model_path: str = "audio_incident_head.keras",
        pooling: str = "mean_max",
        window_duration_sec: float = 2.0,
        update_interval_sec: float = 0.20,
        ema_alpha: float = 0.35
    ):
        self.pooling = pooling
        self.window_duration_sec = window_duration_sec
        self.window_samples = int(window_duration_sec * 16000)
        self.min_samples = 15360  # Minimum 0.96s required by YAMNet
        self.update_interval_sec = update_interval_sec
        self.ema_alpha = ema_alpha

        self.model = None
        self.extractor = None
        self.is_active = False

        self.audio_data: Optional[np.ndarray] = None
        self.has_audio_track = False
        self.is_live_mic = False
        self.mic_buffer = deque(maxlen=48000)  # 3 seconds buffer at 16 kHz
        self.mic_stream = None

        self.latest_label = "Normal"
        self.latest_conf = 0.0
        self.latest_probs = np.zeros(len(CLASSES), dtype=np.float32)
        normal_idx = CLASSES.index("Normal") if "Normal" in CLASSES else 0
        self.latest_probs[normal_idx] = 1.0
        self.ema_probs: Optional[np.ndarray] = None
        self.last_infer_time = 0.0

        # Asynchronous background inference worker so video frame loop is NEVER blocked
        self._audio_lock = threading.Lock()
        self._worker_thread: Optional[threading.Thread] = None
        self._worker_running: bool = False
        self._latest_playback_sec: float = 0.0

        self._load_models(model_path)

    def _load_models(self, model_path: str):
        resolved_path = resolve_model_path(
            model_path,
            default_candidates=[
                "audio_incident_head.keras",
                "backend/checkpoints/audio_incident_head.keras",
                "checkpoints/audio_incident_head.keras"
            ]
        )

        if not resolved_path.exists():
            logger.warning(
                f"Audio model '{model_path}' not found at '{resolved_path}'! "
                f"Searched paths: [., project root, backend/video, checkpoints]. Audio classification disabled."
            )
            return

        if tf is None:
            logger.warning("TensorFlow not installed. Audio incident classification disabled.")
            return

        logger.info(f"Loading trained audio classification model from '{resolved_path}'...")
        try:
            self.model = tf.keras.models.load_model(str(resolved_path))
        except Exception as e:
            logger.error(f"Failed to load audio classification head: {e}")
            return

        logger.info("Initializing YAMNet feature extractor...")
        try:
            if YAMNetFeatureExtractor is not None:
                self.extractor = YAMNetFeatureExtractor(feature_type="embeddings")
            else:
                logger.warning("YAMNetFeatureExtractor not available. Audio classification disabled.")
                return
        except Exception as e:
            logger.error(f"Failed to load YAMNet feature extractor: {e}")
            return

        self.is_active = True
        logger.info("Audio incident analyzer initialized successfully.")

    def setup_audio_source(self, video_source: Union[str, int], audio_source: Optional[str] = None):
        """
        Prepares audio stream: extracts audio from video file, loads separate audio file,
        or starts microphone capture.
        """
        if not self.is_active:
            return

        # 1. User supplied separate audio file
        if audio_source and os.path.exists(audio_source):
            logger.info(f"Loading external audio source from: '{audio_source}'...")
            try:
                if sf is not None:
                    data, sr = sf.read(audio_source)
                    if data.ndim > 1:
                        data = np.mean(data, axis=1)
                    if sr != 16000:
                        import librosa
                        data = librosa.resample(data.astype(np.float32), orig_sr=sr, target_sr=16000)
                    self.audio_data = data.astype(np.float32)
                    self.has_audio_track = True
                    logger.info(f"Loaded {len(self.audio_data)/16000:.1f}s audio from '{audio_source}'.")
                    return
            except Exception as e:
                logger.warning(f"Could not load audio file '{audio_source}': {e}")

        # 2. Source is a video file (extract audio track)
        if isinstance(video_source, str) and not video_source.isdigit() and os.path.exists(video_source):
            logger.info(f"Extracting synchronized audio track from video file: '{video_source}'...")
            try:
                cmd = [
                    FFMPEG_EXE, "-i", str(video_source),
                    "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
                    "-f", "s16le", "-loglevel", "error", "-"
                ]
                proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                if proc.returncode == 0 and len(proc.stdout) > 0:
                    self.audio_data = np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0
                    self.has_audio_track = True
                    duration_s = len(self.audio_data) / 16000.0
                    logger.info(f"Audio track extracted successfully ({duration_s:.1f}s @ 16kHz mono).")
                    return
                else:
                    logger.info("Video file has no audio stream or audio track is silent.")
            except Exception as e:
                logger.warning(f"Could not extract audio from video: {e}")

            # Fallback to microphone if video file has no audio track
            if not self.has_audio_track and sd is not None:
                logger.info("Video has no embedded audio track. Initializing live microphone stream for multimodal audio validation...")
                try:
                    def audio_callback(indata, frames, time_info, status):
                        self.mic_buffer.extend(indata[:, 0])

                    self.mic_stream = sd.InputStream(
                        samplerate=16000,
                        channels=1,
                        dtype="float32",
                        callback=audio_callback
                    )
                    self.mic_stream.start()
                    self.is_live_mic = True
                    self.has_audio_track = True
                    logger.info("Live microphone fallback active for video stream.")
                    return
                except Exception as e:
                    logger.warning(f"Could not initialize live mic fallback: {e}")

        # 3. Live camera / webcam: Try starting live microphone capture
        if isinstance(video_source, int) or (isinstance(video_source, str) and video_source.isdigit()):
            if sd is not None:
                try:
                    logger.info("Starting real-time microphone stream (16kHz mono)...")
                    def audio_callback(indata, frames, time_info, status):
                        self.mic_buffer.extend(indata[:, 0])

                    self.mic_stream = sd.InputStream(
                        samplerate=16000,
                        channels=1,
                        dtype="float32",
                        callback=audio_callback
                    )
                    self.mic_stream.start()
                    self.is_live_mic = True
                    self.has_audio_track = True
                    logger.info("Live microphone audio stream active.")
                    return
                except Exception as e:
                    logger.warning(f"Could not initialize microphone input stream: {e}")
            else:
                logger.info("Live mic capture requires 'sounddevice' (pip install sounddevice). Video-only analysis active.")

    def start_worker(self):
        """Starts asynchronous background worker for non-blocking audio inference."""
        if self._worker_thread is not None and self._worker_thread.is_alive():
            return
        self._worker_running = True
        self._worker_thread = threading.Thread(
            target=self._audio_worker_loop,
            daemon=True,
            name="AudioInferWorker"
        )
        self._worker_thread.start()
        logger.info("Asynchronous background audio inference worker started.")

    def _audio_worker_loop(self):
        """Background thread executing YAMNet + Keras inference without stalling video FPS."""
        while self._worker_running and self.is_active:
            try:
                wave = None
                if self.is_live_mic:
                    if len(self.mic_buffer) >= self.min_samples:
                        with self._audio_lock:
                            wave = np.array(list(self.mic_buffer)[-self.window_samples:], dtype=np.float32)
                elif self.audio_data is not None and len(self.audio_data) > 0:
                    playback_sec = self._latest_playback_sec
                    end_sample = int(playback_sec * 16000)
                    start_sample = max(0, end_sample - self.window_samples)
                    if end_sample > 0:
                        if start_sample >= len(self.audio_data):
                            end_sample = len(self.audio_data)
                            start_sample = max(0, end_sample - self.window_samples)
                        wave = self.audio_data[start_sample:end_sample]

                if wave is not None and len(wave) > 0:
                    wave = np.nan_to_num(wave, nan=0.0, posinf=0.0, neginf=0.0)
                    if len(wave) < self.min_samples:
                        pad_len = self.min_samples - len(wave)
                        wave = np.pad(wave, (0, pad_len), mode="constant")

                    feats = self.extractor.extract_features(wave)
                    pooled = pool_audio_features(feats, pooling=self.pooling)
                    try:
                        probs = self.model(pooled[None, :], training=False).numpy()[0].astype(np.float32)
                    except Exception:
                        probs = self.model.predict(pooled[None, :], verbose=0)[0].astype(np.float32)

                    with self._audio_lock:
                        if self.ema_probs is None:
                            self.ema_probs = probs
                        else:
                            self.ema_probs = self.ema_alpha * probs + (1.0 - self.ema_alpha) * self.ema_probs

                        top_idx = int(np.argmax(self.ema_probs))
                        self.latest_label = CLASSES[top_idx]
                        self.latest_conf = float(self.ema_probs[top_idx])
                        self.latest_probs = self.ema_probs.copy()
                        self.last_infer_time = time.time()
            except Exception as e:
                logger.debug(f"Audio inference worker exception: {e}")

            time.sleep(self.update_interval_sec)

    def process_audio(self, playback_sec: float, force: bool = False) -> Tuple[str, float, Optional[np.ndarray]]:
        """
        Instantaneous non-blocking read of the latest audio incident prediction.
        Execution time < 0.001 ms, ensuring camera stream operates at full hardware FPS.
        """
        if not self.is_active or not self.has_audio_track:
            return "N/A", 0.0, None

        self._latest_playback_sec = playback_sec
        if not self._worker_running:
            self.start_worker()

        with self._audio_lock:
            if self.ema_probs is not None:
                return self.latest_label, self.latest_conf, self.latest_probs
            return "Buffering...", 0.0, None

    def release(self):
        self._worker_running = False
        if self._worker_thread is not None:
            self._worker_thread.join(timeout=1.0)
            self._worker_thread = None
        if self.mic_stream is not None:
            try:
                self.mic_stream.stop()
                self.mic_stream.close()
            except Exception:
                pass
            self.mic_stream = None


class MultimodalIncidentFusion:
    """
    Multimodal Incident Fusion Engine for CCTV Monitoring.
    Maintains independent analysis for Video and Audio modalities while providing
    a unified multimodal probability distribution and status based on selected phenomenon/mode.
    Neither modality depends on or blocks the other.
    """
    def __init__(
        self,
        video_weight: float = 0.55,
        audio_weight: float = 0.45,
        acoustic_boost_threshold: float = 0.60
    ):
        self.video_weight = video_weight
        self.audio_weight = audio_weight
        self.acoustic_boost_threshold = acoustic_boost_threshold
        self.classes = CLASSES

    def fuse(
        self,
        video_probs: Optional[np.ndarray],
        audio_probs: Optional[np.ndarray],
        mode: str = "multimodal"
    ) -> Tuple[str, float, np.ndarray, str, Optional[str]]:
        """
        Fuses video and audio incident probabilities based on phenomenon mode.
        Modes:
            - 'multimodal': Independent parallel analysis with Late Fusion and acoustic triggers
            - 'video_only': Pure visual incident analysis (audio ignored)
            - 'audio_only': Pure acoustic incident analysis (video ignored)
            - 'custom': User-configured weighting and modality selection
        Returns:
            fused_label (str): Top predicted class
            fused_conf (float): Confidence score [0.0, 1.0]
            fused_probs (np.ndarray): 7-class fused probability distribution
            modality_status (str): "V+A ACTIVE", "VIDEO ONLY", "AUDIO ONLY", etc.
            acoustic_alert (Optional[str]): Alert banner text if acoustic trigger fired
        """
        mode_clean = (mode or "multimodal").lower().strip().replace("-", "_")
        acoustic_alert = None

        # Mode: Video Only Phenomenon
        if mode_clean in ("video_only", "video", "v", "2"):
            if video_probs is not None:
                top_idx = int(np.argmax(video_probs))
                top_idx = min(top_idx, len(self.classes) - 1)
                return self.classes[top_idx], float(video_probs[top_idx]), video_probs, "VIDEO ONLY", None
            uniform = np.ones(len(self.classes), dtype=np.float32) / len(self.classes)
            return "Normal", 1.0, uniform, "VIDEO STANDBY", None

        # Mode: Audio Only Phenomenon
        if mode_clean in ("audio_only", "audio", "a", "3"):
            if audio_probs is not None:
                top_idx = int(np.argmax(audio_probs))
                top_idx = min(top_idx, len(self.classes) - 1)
                return self.classes[top_idx], float(audio_probs[top_idx]), audio_probs, "AUDIO ONLY", None
            uniform = np.ones(len(self.classes), dtype=np.float32) / len(self.classes)
            return "Normal", 1.0, uniform, "AUDIO STANDBY", None

        # Mode: Multimodal or Custom
        # Case 1: Both modalities active (Audio + Video run independently and combine)
        if video_probs is not None and audio_probs is not None:
            w_v = self.video_weight
            w_a = self.audio_weight

            top_audio_idx = int(np.argmax(audio_probs))
            top_audio_class = self.classes[min(top_audio_idx, len(self.classes) - 1)]
            top_audio_conf = float(audio_probs[top_audio_idx])

            # Acoustic Trigger: Impulsive acoustic events (Shooting, Explosion, Crash)
            if top_audio_class in ("Shooting", "Explosion", "CarAccident") and top_audio_conf >= self.acoustic_boost_threshold:
                w_a = 0.70
                w_v = 0.30
                acoustic_alert = f"ACOUSTIC TRIGGER: {top_audio_class.upper()} ({top_audio_conf * 100:.0f}%)"

            # Weighted sum and re-normalization
            raw_fused = (w_v * video_probs) + (w_a * audio_probs)
            norm_sum = np.sum(raw_fused)
            fused_probs = raw_fused / norm_sum if norm_sum > 0 else raw_fused

            top_idx = int(np.argmax(fused_probs))
            top_idx = min(top_idx, len(self.classes) - 1)
            fused_label = self.classes[top_idx]
            fused_conf = float(fused_probs[top_idx])
            modality_status = "V+A ACTIVE"

            return fused_label, fused_conf, fused_probs, modality_status, acoustic_alert

        # Case 2: Only Video active
        elif video_probs is not None:
            top_idx = int(np.argmax(video_probs))
            top_idx = min(top_idx, len(self.classes) - 1)
            return self.classes[top_idx], float(video_probs[top_idx]), video_probs, "VIDEO ONLY", None

        # Case 3: Only Audio active
        elif audio_probs is not None:
            top_idx = int(np.argmax(audio_probs))
            top_idx = min(top_idx, len(self.classes) - 1)
            return self.classes[top_idx], float(audio_probs[top_idx]), audio_probs, "AUDIO ONLY", None

        # Fallback (Standby)
        uniform = np.ones(len(self.classes), dtype=np.float32) / len(self.classes)
        return "Normal", 1.0, uniform, "STANDBY", None


def draw_hud_overlay(
    frame: np.ndarray,
    fused_label: str,
    fused_conf: float,
    fused_probs: Optional[np.ndarray],
    video_label: str,
    video_conf: float,
    video_probs: Optional[np.ndarray],
    audio_label: str,
    audio_conf: float,
    audio_probs: Optional[np.ndarray],
    modality_status: str,
    acoustic_alert: Optional[str],
    object_counts: Dict[str, int],
    fps: float,
    mode: str = "multimodal",
    auth_info: Optional[Dict[str, Any]] = None
) -> np.ndarray:
    """
    Renders a modern semi-transparent HUD overlay with separate independent
    badges for Video and Audio modalities, active phenomenon tag, combined status,
    authorized personnel detection overlay, incident bypass indicators,
    YOLOv8 object count summaries, and confidence breakdown panel.
    """
    if frame is None or frame.size == 0:
        return frame

    h, w = frame.shape[:2]

    # Render authorized face bounding boxes
    if auth_info and auth_info.get("auth_detections"):
        for adet in auth_info["auth_detections"]:
            bx1, by1, bx2, by2 = adet["box"]
            a_ident = adet.get("identity", "Authorized")
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
            cv2.putText(frame, f"AUTHORIZED: {a_ident.upper()}", (bx1, max(18, by1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 0), 1, cv2.LINE_AA)

    # 1. Top Banner Backdrop (ROI blend - zero full-frame allocations)
    banner_height = 96
    roi_top = frame[0:banner_height, 0:w]
    bg_top = np.full_like(roi_top, (16, 16, 16), dtype=np.uint8)
    cv2.addWeighted(bg_top, 0.82, roi_top, 0.18, 0, roi_top)

    # Header and Phenomenon / Modality Status Tag
    cv2.putText(frame, "MULTIMODAL CCTV INCIDENT MONITOR", (15, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1, cv2.LINE_AA)

    mode_display = mode.upper().replace("_", " ")
    mode_tag = f"[{mode_display}] [{modality_status}]"
    tag_color = (0, 255, 128) if "MULTI" in mode_display else ((0, 215, 255) if "VIDEO" in mode_display else (0, 255, 255))
    cv2.putText(frame, mode_tag, (270, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, tag_color, 1, cv2.LINE_AA)

    # Authorized bypass indicator badge in banner
    if auth_info and auth_info.get("bypass_incident"):
        auth_names_str = ", ".join(auth_info.get("authorized_names", ["Personnel"]))
        auth_tag = f"[AUTH BYPASS: {auth_names_str.upper()}]"
        cv2.putText(frame, auth_tag, (440, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 0), 1, cv2.LINE_AA)

    # FPS counter top right
    fps_str = f"FPS: {fps:.1f}"
    cv2.putText(frame, fps_str, (max(450, w - 120), 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)

    # Independent Modality Badges: Video on left, Audio beside it
    v_color = INCIDENT_COLORS.get(video_label, DEFAULT_COLOR) if video_probs is not None else (140, 140, 140)
    v_text = f"VIDEO: {video_label.upper()} ({video_conf * 100:.0f}%)" if video_probs is not None else f"VIDEO: {video_label}"
    cv2.putText(frame, v_text, (15, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.65, v_color, 2, cv2.LINE_AA)

    a_color = INCIDENT_COLORS.get(audio_label, DEFAULT_COLOR) if audio_probs is not None else (140, 140, 140)
    a_text = f"AUDIO: {audio_label.upper()} ({audio_conf * 100:.0f}%)" if audio_probs is not None else f"AUDIO: {audio_label}"
    audio_x = max(330, int(w * 0.38))
    cv2.putText(frame, a_text, (audio_x, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.65, a_color, 2, cv2.LINE_AA)

    # Sub-status: Combined Fused Incident & Acoustic Trigger
    f_color = INCIDENT_COLORS.get(fused_label, DEFAULT_COLOR)
    fused_text = f"Decision: {fused_label.upper()} ({fused_conf * 100:.0f}%)"
    cv2.putText(frame, fused_text, (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.42, f_color, 1, cv2.LINE_AA)

    if acoustic_alert:
        cv2.putText(frame, f" | ! {acoustic_alert} !", (220, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 255), 1, cv2.LINE_AA)

    # Object summary text in banner
    if object_counts:
        counts_str = ", ".join([f"{count} {label}{'s' if count > 1 and not label.endswith('s') else ''}" for label, count in object_counts.items()])
        obj_text = f"Objects: {counts_str}"
    else:
        obj_text = "Objects: None detected (YOLO active)"
    cv2.putText(frame, obj_text[:80], (15, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (170, 170, 170), 1, cv2.LINE_AA)

    # Acoustic Alert Pill Badge if active
    if acoustic_alert:
        alert_w = min(320, max(150, w - 200))
        alert_x1 = max(15, w - alert_w - 140)
        cv2.rectangle(frame, (alert_x1, 10), (alert_x1 + alert_w, 42), (0, 0, 180), -1)
        cv2.rectangle(frame, (alert_x1, 10), (alert_x1 + alert_w, 42), (0, 0, 255), 2)
        cv2.putText(frame, f"! {acoustic_alert} !", (alert_x1 + 10, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    # 2. Side Panel for Incident Confidence Breakdown
    if fused_probs is not None:
        panel_w = 285
        panel_h = min(h - banner_height - 45, len(CLASSES) * 24 + 36)
        panel_x1 = max(10, w - panel_w - 15)
        panel_y1 = banner_height + 12
        panel_x2 = min(w - 10, panel_x1 + panel_w)
        panel_y2 = min(h - 30, panel_y1 + panel_h)

        if panel_x2 > panel_x1 + 100 and panel_y2 > panel_y1 + 40:
            roi_side = frame[panel_y1:panel_y2, panel_x1:panel_x2]
            bg_side = np.full_like(roi_side, (14, 14, 14), dtype=np.uint8)
            cv2.addWeighted(bg_side, 0.78, roi_side, 0.22, 0, roi_side)
            cv2.rectangle(frame, (panel_x1, panel_y1), (panel_x2, panel_y2), (55, 55, 55), 1)

            breakdown_title = f"{mode.upper().replace('_', ' ')} BREAKDOWN"
            cv2.putText(frame, breakdown_title[:24], (panel_x1 + 10, panel_y1 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (210, 210, 210), 1, cv2.LINE_AA)

            y_offset = panel_y1 + 38
            for idx, c_name in enumerate(CLASSES):
                if y_offset + 16 > panel_y2:
                    break

                p_fused = float(fused_probs[idx]) if idx < len(fused_probs) else 0.0
                p_vid = float(video_probs[idx]) if (video_probs is not None and idx < len(video_probs)) else 0.0
                p_aud = float(audio_probs[idx]) if (audio_probs is not None and idx < len(audio_probs)) else 0.0
                c_color = INCIDENT_COLORS.get(c_name, DEFAULT_COLOR)

                # Class Name
                cv2.putText(frame, f"{c_name[:10]:<10}", (panel_x1 + 10, y_offset + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (225, 225, 225), 1, cv2.LINE_AA)

                # Fused Progress Bar
                bar_x = panel_x1 + 82
                bar_max_w = 80
                bar_w = int(bar_max_w * max(0.0, min(1.0, p_fused)))

                cv2.rectangle(frame, (bar_x, y_offset + 3), (bar_x + bar_max_w, y_offset + 13), (35, 35, 35), -1)
                if bar_w > 0:
                    cv2.rectangle(frame, (bar_x, y_offset + 3), (bar_x + bar_w, y_offset + 13), c_color, -1)

                # Fused %
                cv2.putText(frame, f"{p_fused * 100:.0f}%", (bar_x + bar_max_w + 5, y_offset + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (220, 220, 220), 1, cv2.LINE_AA)

                # Video vs Audio indicators
                va_str = f"V:{int(p_vid*100)}% A:{int(p_aud*100)}%"
                cv2.putText(frame, va_str, (bar_x + bar_max_w + 36, y_offset + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (160, 160, 160), 1, cv2.LINE_AA)

                y_offset += 24

    # 3. Bottom Hotkey Navigation Bar
    footer_h = 24
    roi_footer = frame[h - footer_h:h, 0:w]
    bg_footer = np.full_like(roi_footer, (10, 10, 10), dtype=np.uint8)
    cv2.addWeighted(bg_footer, 0.85, roi_footer, 0.15, 0, roi_footer)

    nav_text = "Mode: [1: Multimodal] [2: Video Only] [3: Audio Only] [4: Custom] | [Y: YOLO] [Q: Exit]"
    cv2.putText(frame, nav_text, (15, h - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (190, 190, 190), 1, cv2.LINE_AA)

    return frame


def run_live_camera_test(
    source: str = "0",
    mode: str = "multimodal",
    model_path: str = "best_fast_gru_resnet_model.pth",
    audio_model: str = "audio_incident_head.keras",
    audio_source: Optional[str] = None,
    yolo_model: str = "yolov8n.pt",
    yolo_conf: float = 0.35,
    video_weight: float = 0.55,
    audio_weight: float = 0.45,
    enable_yolo: bool = True,
    enable_video_incident: bool = True,
    enable_audio_incident: bool = True,
    save_output: Optional[str] = None
):
    """
    Main loop for live camera / video stream multimodal object detection & incident analysis.
    Supports selectable phenomenon modes:
      - 'multimodal': Evaluates both video (ResNet-50 + Fast GRU) and audio (YAMNet) in parallel
      - 'video_only': Pure video temporal classification (audio analysis bypassed)
      - 'audio_only': Pure audio classification (video temporal classification bypassed)
      - 'custom': User-configured modalities and weights
    Allows real-time keyboard mode switching (1: Multi, 2: Video, 3: Audio, 4: Custom, Y: YOLO).
    """
    # Normalize initial mode
    current_mode = (mode or "multimodal").lower().strip()
    if current_mode in ("video_only", "video", "v", "2"):
        current_mode = "video_only"
        enable_video_incident = True
        enable_audio_incident = False
    elif current_mode in ("audio_only", "audio", "a", "3"):
        current_mode = "audio_only"
        enable_video_incident = False
        enable_audio_incident = True
    elif current_mode in ("multimodal", "multi", "m", "1"):
        current_mode = "multimodal"
        enable_video_incident = True
        enable_audio_incident = True
    elif current_mode in ("custom", "c", "4"):
        current_mode = "custom"

    # 1. Initialize Video Incident Analyzer (loaded if video or multimodal is possible)
    incident_analyzer = None
    if enable_video_incident or current_mode in ("multimodal", "video_only", "custom"):
        try:
            incident_analyzer = LiveIncidentAnalyzer(model_path=model_path)
        except Exception as e:
            logger.error(f"Could not initialize video incident analyzer: {e}")

    # 2. Initialize Audio Incident Analyzer (loaded if audio or multimodal is possible)
    audio_analyzer = None
    if enable_audio_incident or current_mode in ("multimodal", "audio_only", "custom"):
        try:
            audio_analyzer = LiveAudioIncidentAnalyzer(model_path=audio_model)
            video_src_val = int(source) if source.isdigit() else source
            audio_analyzer.setup_audio_source(video_src_val, audio_source=audio_source)
        except Exception as e:
            logger.error(f"Could not initialize audio incident analyzer: {e}")

    # 3. Initialize Multimodal Fusion Engine
    fusion_engine = MultimodalIncidentFusion(
        video_weight=video_weight,
        audio_weight=audio_weight
    )

    # 4. Initialize YOLO Object Detector
    yolo_detector = None
    yolo_active = enable_yolo
    if enable_yolo:
        try:
            resolved_yolo = resolve_model_path(
                yolo_model,
                default_candidates=["yolov8n.pt", "backend/video/yolov8n.pt", "yolo11n.pt"]
            )
            yolo_detector = YOLOv8ObjectDetector(model_path=str(resolved_yolo), conf_thres=yolo_conf)
        except Exception as e:
            logger.warning(f"Could not initialize YOLO object detector: {e}. Running without YOLO.")
            yolo_detector = None
            yolo_active = False

    # 5. Initialize AuthManager for authorized personnel detection and detection bypass
    auth_manager = None
    if get_auth_manager is not None:
        try:
            auth_manager = get_auth_manager()
            logger.info(f"AuthManager initialized: {len(auth_manager.known_identities)} authorized identity/identities loaded.")
        except Exception as e:
            logger.warning(f"Could not initialize AuthManager: {e}")

    video_source = int(source) if source.isdigit() else source
    logger.info(f"Opening video source: '{source}'...")
    cap = ThreadedCameraReader(video_source, width=640, height=480, fps=30)

    if not cap.isOpened():
        logger.error(f"Failed to open video source: '{source}'")
        return

    raw_fps = cap.get(cv2.CAP_PROP_FPS)
    if raw_fps is None or np.isnan(raw_fps) or raw_fps <= 1.0 or raw_fps > 240.0:
        fps_in = 30.0
    else:
        fps_in = float(raw_fps)

    writer = None
    if save_output:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(save_output, fourcc, fps_in, (w, h))
        logger.info(f"Saving output video to: '{save_output}'")

    print("\n" + "=" * 80)
    print("LIVE REAL-TIME MULTIMODAL CCTV INCIDENT ANALYSIS & OBJECT MONITORING")
    print(f"Active Phenomenon Mode: {current_mode.upper()}")
    print("Interactive Hotkeys during playback:")
    print("  [1 / M] -> Switch to MULTIMODAL  (Video ResNet50-GRU + Audio YAMNet)")
    print("  [2 / V] -> Switch to VIDEO ONLY  (Visual temporal analysis only)")
    print("  [3 / A] -> Switch to AUDIO ONLY  (Acoustic YAMNet analysis only)")
    print("  [4 / C] -> Switch to CUSTOM      (Configured modalities & weights)")
    print("  [Y]     -> Toggle YOLOv8 Object Detection on / off")
    print("  [Q/ESC] -> Exit Cleanly")
    print("=" * 80 + "\n")

    prev_time = time.time()
    start_wall_time = time.time()
    fps = 0.0
    frame_idx = 0
    last_decision_alerts: Dict[str, float] = {}
    prev_detections = None
    v_label, v_conf, v_probs = "Normal", 1.0, None

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                logger.info("Video stream ended or frame unreadable.")
                break

            curr_time = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(1e-5, curr_time - prev_time))
            prev_time = curr_time
            frame_idx += 1

            # Playback timestamp in seconds (safeguarded against NaN)
            if isinstance(video_source, str) and not video_source.isdigit():
                playback_sec = frame_idx / max(1.0, fps_in)
            else:
                playback_sec = max(0.0, curr_time - start_wall_time)

            # 1. Video Incident Classification (interleaved for 30 FPS)
            if current_mode in ("multimodal", "video_only", "custom") and incident_analyzer and incident_analyzer.is_active:
                if frame_idx % 2 == 1 or v_probs is None:
                    v_label, v_conf, v_probs = incident_analyzer.process_frame(frame)
            elif current_mode == "audio_only":
                v_label = "Disabled"
                v_conf = 0.0

            # 2. Audio Incident Classification (active in multimodal, audio_only, or custom)
            a_label, a_conf, a_probs = "Normal", 1.0, None
            if current_mode in ("multimodal", "audio_only", "custom") and audio_analyzer and audio_analyzer.is_active and audio_analyzer.has_audio_track:
                a_label, a_conf, a_probs = audio_analyzer.process_audio(playback_sec)
            elif current_mode == "video_only":
                a_label = "Disabled"
                a_conf = 0.0

            # 3. Multimodal Late Fusion
            fused_label, fused_conf, fused_probs, modality_status, acoustic_alert = fusion_engine.fuse(
                video_probs=v_probs,
                audio_probs=a_probs,
                mode=current_mode
            )

            # 4. YOLO Real-Time Object Detection (alternate frames for maximum FPS)
            object_counts: Dict[str, int] = {}
            if yolo_active and yolo_detector:
                if frame_idx % 2 == 0 or prev_detections is None:
                    prev_detections = yolo_detector.detect(frame)
                frame, object_counts = yolo_detector.draw_detections(frame, prev_detections)

            # 4b. Authorized Personnel Detection & Incident Detection Bypass
            auth_info = None
            if auth_manager:
                auth_info = auth_manager.process_frame(frame, detections=prev_detections, frame_idx=frame_idx)

            # 5. Draw Enhanced Multimodal HUD Overlay with active mode and auth info
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
                fps=fps,
                mode=current_mode,
                auth_info=auth_info
            )

            if writer:
                writer.write(annotated_frame)

            # Automated ntfy alert dispatch: Decision >= 70% ONLY for Video+Audio (multimodal), else >= 80%
            decision_score = round(fused_conf * 100, 1)
            is_bypassed = bool(auth_info and auth_info.get("bypass_incident"))
            if is_bypassed and fused_label != "Normal" and frame_idx % 30 == 0:
                logger.info(
                    f"Incident {fused_label} ({decision_score:.0f}%) BYPASSED: "
                    f"Authorized person {auth_info.get('authorized_names')} detected."
                )

            modality_str = str(modality_status).upper() if modality_status else ""
            is_video_audio = (
                "V+A" in modality_str
                or "MULTIMODAL" in modality_str
                or "VIDEO+AUDIO" in modality_str
                or ("VIDEO" in modality_str and "AUDIO" in modality_str)
            )
            req_threshold = 70.0 if is_video_audio else 80.0

            if fused_label != "Normal" and decision_score >= req_threshold and not is_bypassed:
                now_t = time.time()
                if now_t - last_decision_alerts.get(fused_label, 0) >= 30.0:
                    last_decision_alerts[fused_label] = now_t
                    try:
                        from backend.api.notifier import send_ntfy, get_current_topic
                        target_topic = get_current_topic()
                        _, snap_buf = cv2.imencode(".jpg", annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
                        prefix_label = "V+A DECISION" if is_video_audio else "DECISION"
                        threading.Thread(
                            target=send_ntfy,
                            kwargs={
                                "topic": target_topic,
                                "title": f"🚨 {prefix_label}: {fused_label.upper()} ({decision_score:.0f}%)",
                                "message": f"Decision: {fused_label} ({decision_score:.0f}%) detected.\nModality: {modality_status}\nSource: {source}",
                                "priority": "urgent" if decision_score >= (75 if is_video_audio else 85) else "high",
                                "tags": "warning,rotating_light",
                                "data": bytes(snap_buf),
                                "filename": "decision_incident.jpg",
                            },
                            daemon=True
                        ).start()
                        logger.info(f"Dispatched ntfy alert to topic '{target_topic}': {prefix_label} {fused_label} ({decision_score:.0f}%) >= {req_threshold:.0f}%")
                    except Exception as e:
                        logger.warning(f"Failed to dispatch live decision alert: {e}")

            cv2.imshow("Real-Time Multimodal CCTV Incident Monitor", annotated_frame)

            # Keypress handling: runtime phenomenon switching
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q'), ord('Q')):
                logger.info("User requested exit.")
                break
            elif key in (ord('1'), ord('m'), ord('M')):
                current_mode = "multimodal"
                logger.info("Switched to Phenomenon: MULTIMODAL (Video + Audio Late Fusion)")
            elif key in (ord('2'), ord('v'), ord('V')):
                current_mode = "video_only"
                logger.info("Switched to Phenomenon: VIDEO ONLY (Visual Analysis)")
            elif key in (ord('3'), ord('a'), ord('A')):
                current_mode = "audio_only"
                logger.info("Switched to Phenomenon: AUDIO ONLY (Acoustic Analysis)")
            elif key in (ord('4'), ord('c'), ord('C')):
                current_mode = "custom"
                logger.info("Switched to Phenomenon: CUSTOM")
            elif key in (ord('y'), ord('Y')):
                yolo_active = not yolo_active
                logger.info(f"YOLO Object Detection toggled: {'ENABLED' if yolo_active else 'DISABLED'}")

    finally:
        cap.release()
        if audio_analyzer:
            audio_analyzer.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        logger.info("Live test ended cleanly.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Live Video / Camera Multimodal Incident Analysis (Video + Audio + YOLOv8)")
    parser.add_argument("--source", type=str, default="0", help="Camera index (0, 1) or path to video file")
    parser.add_argument(
        "--mode", "--phenomenon",
        type=str,
        default=None,
        choices=["multimodal", "video_only", "audio_only", "custom", "1", "2", "3", "4"],
        help="Select incident analysis phenomenon: 'multimodal' (Video+Audio fusion), 'video_only' (Video temporal only), 'audio_only' (Audio acoustic only), or 'custom' (default: multimodal)"
    )
    parser.add_argument("--interactive", action="store_true", help="Show interactive CLI menu to choose phenomenon")
    parser.add_argument("--model_path", type=str, default="best_fast_gru_resnet_model.pth", help="Video incident classification model weights path")
    parser.add_argument("--audio_model", type=str, default="audio_incident_head.keras", help="Audio incident classification head path")
    parser.add_argument("--audio_source", type=str, default=None, help="Optional separate audio file path to sync with video")
    parser.add_argument("--video_weight", type=float, default=0.55, help="Weight for video modality in multimodal fusion (default: 0.55)")
    parser.add_argument("--audio_weight", type=float, default=0.45, help="Weight for audio modality in multimodal fusion (default: 0.45)")
    parser.add_argument("--yolo_model", type=str, default="yolov8n.pt", help="YOLOv8 model path or name (e.g., yolov8n.pt, yolo11n.pt)")
    parser.add_argument("--yolo_conf", type=float, default=0.35, help="YOLOv8 object detection confidence threshold")
    parser.add_argument("--disable_yolo", action="store_true", help="Disable YOLO object detection")
    parser.add_argument("--disable_video", action="store_true", help="Disable video incident classification")
    parser.add_argument("--disable_audio", action="store_true", help="Disable audio incident classification")
    parser.add_argument("--save_output", type=str, default=None, help="Optional output video file path")
    args = parser.parse_args()

    selected_mode = args.mode
    # If no mode is specified and run interactively from terminal, offer clean menu
    if (selected_mode is None and sys.stdin.isatty()) or args.interactive:
        print("\n" + "=" * 70)
        print("  SELECT INCIDENT ANALYSIS PHENOMENON / MODE:")
        print("  [1] Multimodal   - Video ResNet-50 + Audio YAMNet + Late Fusion (Default)")
        print("  [2] Video Only   - Temporal Video Incident Detection Only")
        print("  [3] Audio Only   - Acoustic Stream YAMNet Detection Only")
        print("  [4] Custom       - Custom weights and modality configuration")
        print("=" * 70)
        try:
            choice = input("Enter choice (1-4) [default: 1 (Multimodal)]: ").strip()
            mode_map = {
                "1": "multimodal",
                "2": "video_only",
                "3": "audio_only",
                "4": "custom",
                "multimodal": "multimodal",
                "video": "video_only",
                "video_only": "video_only",
                "audio": "audio_only",
                "audio_only": "audio_only",
                "custom": "custom"
            }
            selected_mode = mode_map.get(choice.lower(), "multimodal") if choice else "multimodal"
        except (EOFError, KeyboardInterrupt):
            selected_mode = "multimodal"
    elif selected_mode is None:
        selected_mode = "multimodal"

    run_live_camera_test(
        source=args.source,
        mode=selected_mode,
        model_path=args.model_path,
        audio_model=args.audio_model,
        audio_source=args.audio_source,
        video_weight=args.video_weight,
        audio_weight=args.audio_weight,
        yolo_model=args.yolo_model,
        yolo_conf=args.yolo_conf,
        enable_yolo=not args.disable_yolo,
        enable_video_incident=not args.disable_video,
        enable_audio_incident=not args.disable_audio,
        save_output=args.save_output
    )


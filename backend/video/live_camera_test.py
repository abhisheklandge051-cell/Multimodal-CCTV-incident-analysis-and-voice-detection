import os
import sys
import time
import logging
from collections import deque
from typing import Optional, Tuple, List, Dict

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

        logger.info(f"Initializing YOLOv8 object detector ('{model_path}') on {self.device}...")
        try:
            self.model = YOLO(model_path)
            self.model.to(self.device)
            logger.info("YOLOv8 object detector loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load YOLO model from '{model_path}': {e}")
            raise

    @torch.no_grad()
    def detect(self, frame_bgr: np.ndarray) -> List[Dict]:
        """
        Runs object detection on a BGR frame.
        Returns a list of dicts with keys: 'box', 'class_id', 'label', 'conf'.
        """
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

                detections.append({
                    "box": (int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])),
                    "class_id": cls_id,
                    "label": label,
                    "conf": conf
                })

        return detections

    def draw_detections(self, frame: np.ndarray, detections: List[Dict]) -> Tuple[np.ndarray, Dict[str, int]]:
        """
        Draws bounding boxes and caption badges for detected objects on the frame.
        Returns modified frame and object counts map.
        """
        counts: Dict[str, int] = {}
        for det in detections:
            x1, y1, x2, y2 = det["box"]
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
            badge_y2 = y1 if y1 - th - baseline - 4 >= 0 else y1 + th + baseline + 4

            cv2.rectangle(frame, (x1, badge_y1), (x1 + tw + 6, badge_y2), color, -1)
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
        self.is_active = False

        if not os.path.exists(model_path):
            logger.warning(f"Incident classification model '{model_path}' not found! Incident classification will be disabled.")
            return

        logger.info(f"Loading ResNet-50 backbone on device: {self.device}")
        self.backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.backbone.fc = nn.Identity()
        self.backbone.to(self.device)
        self.backbone.eval()
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        num_classes = len(LABEL_MAP)
        logger.info(f"Loading FastGRUResNetDetector from '{model_path}'...")
        self.model = FastGRUResNetDetector(
            feature_dim=2048,
            hidden_dim=256,
            num_classes=num_classes,
            dropout_rate=0.4
        ).to(self.device)

        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        self.feature_buffer = deque(maxlen=clip_len)
        self.ema_probs: Optional[np.ndarray] = None
        self.is_active = True

    @torch.no_grad()
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
        t = F.to_tensor(frame_resized)
        t = F.center_crop(t, (224, 224))
        t = self.normalize(t).unsqueeze(0).to(self.device)

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

        feat = self.extract_frame_feature(frame_bgr)
        self.feature_buffer.append(feat)
        top_label, confidence, probs = self.predict_clip()
        return top_label, confidence, probs


def draw_hud_overlay(
    frame: np.ndarray,
    incident_label: str,
    incident_conf: float,
    incident_probs: Optional[np.ndarray],
    object_counts: Dict[str, int],
    fps: float
) -> np.ndarray:
    """
    Renders a modern semi-transparent HUD overlay with incident monitoring banner,
    YOLOv8 detected object summary, and incident confidence breakdown panel.
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
    save_output: Optional[str] = None
):
    """
    Main loop for live camera / video stream object detection & incident classification.
    """
    incident_analyzer = None
    if enable_incident:
        incident_analyzer = LiveIncidentAnalyzer(model_path=model_path)

    yolo_detector = None
    if enable_yolo:
        try:
            # Fallback to local yolo11n.pt if yolov8n.pt not found and yolo11n.pt exists locally
            if yolo_model == "yolov8n.pt" and not os.path.exists("yolov8n.pt") and os.path.exists("yolo11n.pt"):
                yolo_model = "yolo11n.pt"
            yolo_detector = YOLOv8ObjectDetector(model_path=yolo_model, conf_thres=yolo_conf)
        except Exception as e:
            logger.error(f"Could not initialize YOLO object detector: {e}")

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

    print("\n" + "=" * 80)
    print("LIVE REAL-TIME OBJECT DETECTION & INCIDENT MONITORING")
    print("Press 'q' or 'ESC' on the video window to exit.")
    print("=" * 80 + "\n")

    prev_time = time.time()
    fps = 0.0

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                logger.info("Video stream ended or frame unreadable.")
                break

            curr_time = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(1e-5, curr_time - prev_time))
            prev_time = curr_time

            # 1. Incident Classification
            inc_label, inc_conf, inc_probs = "N/A", 0.0, None
            if incident_analyzer and incident_analyzer.is_active:
                inc_label, inc_conf, inc_probs = incident_analyzer.process_frame(frame)

            # 2. YOLOv8 Real-Time Object Detection
            object_counts: Dict[str, int] = {}
            if yolo_detector:
                detections = yolo_detector.detect(frame)
                frame, object_counts = yolo_detector.draw_detections(frame, detections)

            # 3. Draw HUD Overlay
            annotated_frame = draw_hud_overlay(
                frame,
                incident_label=inc_label,
                incident_conf=inc_conf,
                incident_probs=inc_probs,
                object_counts=object_counts,
                fps=fps
            )

            if writer:
                writer.write(annotated_frame)

            cv2.imshow("Real-Time YOLOv8 Object Detection & Incident Monitor", annotated_frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q'), ord('Q')):
                logger.info("User requested exit.")
                break

    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        logger.info("Live test ended cleanly.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Live Video / Camera YOLOv8 Object Detection & Incident Analysis")
    parser.add_argument("--source", type=str, default="0", help="Camera index (0, 1) or path to video file")
    parser.add_argument("--model_path", type=str, default="best_fast_gru_resnet_model.pth", help="Incident classification model weights path")
    parser.add_argument("--yolo_model", type=str, default="yolov8n.pt", help="YOLOv8 model path or name (e.g., yolov8n.pt, yolo11n.pt)")
    parser.add_argument("--yolo_conf", type=float, default=0.35, help="YOLOv8 object detection confidence threshold")
    parser.add_argument("--disable_yolo", action="store_true", help="Disable YOLO object detection")
    parser.add_argument("--disable_incident", action="store_true", help="Disable temporal incident classification")
    parser.add_argument("--save_output", type=str, default=None, help="Optional output video file path")
    args = parser.parse_args()

    run_live_camera_test(
        source=args.source,
        model_path=args.model_path,
        yolo_model=args.yolo_model,
        yolo_conf=args.yolo_conf,
        enable_yolo=not args.disable_yolo,
        enable_incident=not args.disable_incident,
        save_output=args.save_output
    )

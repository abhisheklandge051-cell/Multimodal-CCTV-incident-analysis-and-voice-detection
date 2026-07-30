"""
Real-Time YOLO Object Detection & Bounding Box Overlay Engine
============================================================
Provides real-time object detection, bounding box visualization, and multi-object 
tracking using Ultralytics YOLOv8 integrated with ResNet incident classification.

Features:
- Real-time object detection (Persons, Vehicles, Bags, Hazards).
- Bounding box overlay with customizable colors & confidence tags.
- Hybrid fusion: YOLO object detection + ResNet spatio-temporal incident classification.
- RTSP / Webcam / Video File stream ingestion with live annotated output.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any

import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("video.detector")

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False


class YOLOObjectDetector:
    """
    YOLOv8 Object Detector for real-time hazard detection and bounding box overlays.
    """

    DEFAULT_COLOR_MAP = {
        "person": (0, 255, 127),      # Spring Green
        "car": (255, 191, 0),         # Deep Sky Blue
        "bus": (255, 128, 0),         # Orange
        "truck": (255, 128, 0),
        "backpack": (255, 0, 255),    # Magenta
        "handbag": (255, 0, 255),
        "suitcase": (255, 0, 255),
        "knife": (0, 0, 255),          # Red (Hazard)
        "fire": (0, 0, 255),
        "gun": (0, 0, 255),
    }

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        conf_threshold: float = 0.35,
        device: Optional[str] = None
    ) -> None:
        self.conf_threshold = conf_threshold
        self.device = device or ("cuda" if (HAS_TORCH and torch.cuda.is_available()) else "cpu")
        self.model = None

        if HAS_YOLO:
            try:
                logger.info(f"⚡ Loading YOLOv8 Model '{model_name}' on device '{self.device}'...")
                self.model = YOLO(model_name)
                logger.info("YOLOv8 Object Detector initialized successfully.")
            except Exception as e:
                logger.error(f"Failed to load YOLO model '{model_name}': {e}")
        else:
            logger.warning("ultralytics YOLO package is not installed. Object detection operating in fallback mode.")

    def detect(
        self,
        frame: np.ndarray,
        conf: Optional[float] = None,
        classes: Optional[List[int]] = None
    ) -> List[Dict[str, Any]]:
        """
        Runs object detection on a single video frame.
        
        Returns:
            List of detection dicts:
            [
              {
                "class_id": int,
                "class_name": str,
                "confidence": float,
                "bbox": [x1, y1, x2, y2]
              },
              ...
            ]
        """
        if self.model is None or frame is None:
            return []

        conf_val = conf if conf is not None else self.conf_threshold

        try:
            results = self.model.predict(
                source=frame,
                conf=conf_val,
                classes=classes,
                device=self.device,
                verbose=False
            )

            detections: List[Dict[str, Any]] = []

            for result in results:
                boxes = result.boxes
                if boxes is None:
                    continue

                for box in boxes:
                    xyxy = box.xyxy[0].cpu().numpy().astype(int).tolist()
                    confidence = float(box.conf[0].cpu().numpy())
                    cls_id = int(box.cls[0].cpu().numpy())
                    cls_name = self.model.names[cls_id] if (self.model.names and cls_id in self.model.names) else f"class_{cls_id}"

                    detections.append({
                        "class_id": cls_id,
                        "class_name": cls_name,
                        "confidence": confidence,
                        "bbox": xyxy
                    })

            return detections
        except Exception as e:
            logger.error(f"Error during YOLO detection: {e}")
            return []

    def draw_bounding_boxes(
        self,
        frame: np.ndarray,
        detections: List[Dict[str, Any]],
        draw_labels: bool = True,
        incident_label: Optional[str] = None
    ) -> np.ndarray:
        """
        Overlays bounding boxes, category labels, and confidence tags on frame.
        """
        if not HAS_CV2 or frame is None:
            return frame

        annotated = frame.copy()
        h, w = annotated.shape[:2]

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            cls_name = det["class_name"]
            conf = det["confidence"]

            # Select color based on category
            color = self.DEFAULT_COLOR_MAP.get(cls_name.lower(), (0, 255, 0))

            # Draw bounding box rectangle
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            if draw_labels:
                label_text = f"{cls_name} {conf * 100:.0f}%"
                (text_w, text_h), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

                # Draw filled label background
                cv2.rectangle(
                    annotated,
                    (x1, max(0, y1 - text_h - 6)),
                    (x1 + text_w + 6, y1),
                    color,
                    -1
                )
                # Text label
                cv2.putText(
                    annotated,
                    label_text,
                    (x1 + 3, max(text_h + 2, y1 - 3)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0) if sum(color) > 400 else (255, 255, 255),
                    1,
                    cv2.LINE_AA
                )

        # Overlay Incident Banner if present
        if incident_label:
            banner_bg = (0, 0, 200) if "normal" not in incident_label.lower() else (0, 150, 0)
            cv2.rectangle(annotated, (0, 0), (w, 40), banner_bg, -1)
            cv2.putText(
                annotated,
                f"INCIDENT DETECTED: {incident_label.upper()}",
                (15, 27),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

        return annotated


from collections import deque


class DualStreamProcessor:
    """
    Dual Independent Video Analytics Engine:
    
    Camera Stream
       ├──► Branch 1: YOLOv8 (Real-time Spatial Object Detection: Person, Gun, Fire, Vehicle)
       └──► Branch 2: ResNet + Transformer (Spatio-Temporal Incident Classification: Fight, Explosion, Robbery)
    """

    def __init__(
        self,
        yolo_model_name: str = "yolov8n.pt",
        resnet_model_path: Optional[Union[str, Path]] = None,
        buffer_size: int = 16,
        conf_threshold: float = 0.35,
        device: Optional[str] = None
    ) -> None:
        self.device = device or ("cuda" if (HAS_TORCH and torch.cuda.is_available()) else "cpu")
        
        # Branch 1: YOLO Object Detector
        logger.info("Initializing Branch 1: Real-Time YOLO Object Detector...")
        self.yolo_detector = YOLOObjectDetector(model_name=yolo_model_name, conf_threshold=conf_threshold, device=self.device)
        
        # Branch 2: ResNet + Transformer Incident Classifier
        logger.info("Initializing Branch 2: ResNet + Transformer Incident Classifier...")
        self.buffer_size = buffer_size
        self.frame_buffer = deque(maxlen=buffer_size)
        self.resnet_model = None
        self.categories = [
            "Abuse", "Arrest", "Arson", "Assault", "Burglary", "Explosion",
            "Fighting", "NormalVideos", "RoadAccidents", "Robbery",
            "Shooting", "Shoplifting", "Stealing", "Vandalism"
        ]

        if HAS_TORCH:
            try:
                from app.video.modletraining import IncidentVideoClassifier
                self.resnet_model = IncidentVideoClassifier(num_classes=len(self.categories), pretrained=True).to(self.device)
                
                # Check for trained checkpoint
                if resnet_model_path is None:
                    possible_paths = [
                        Path("models/best_incident_video_classifier.pt"),
                        Path("app/video/models/best_incident_video_classifier.pt"),
                        Path(__file__).resolve().parent / "models" / "best_incident_video_classifier.pt"
                    ]
                    for p in possible_paths:
                        if p.exists():
                            resnet_model_path = p
                            break

                if resnet_model_path and Path(resnet_model_path).exists():
                    checkpoint = torch.load(str(resnet_model_path), map_location=self.device)
                    if "model_state_dict" in checkpoint:
                        self.resnet_model.load_state_dict(checkpoint["model_state_dict"])
                    if "categories" in checkpoint:
                        self.categories = checkpoint["categories"]
                    logger.info(f"Loaded trained ResNet + Transformer weights from '{resnet_model_path}'")
                
                self.resnet_model.eval()
            except Exception as e:
                logger.warning(f"ResNet classifier initialization warning ({e}). Running in default mode.")

    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Processes a single incoming camera frame independently through both branches.
        """
        if frame is None:
            return {"yolo_objects": [], "resnet_incident": None, "annotated_frame": frame}

        # --- BRANCH 1: YOLO Real-Time Object Detection ---
        yolo_detections = self.yolo_detector.detect(frame)

        # Append frame to sliding window buffer for Branch 2
        self.frame_buffer.append(frame)

        # --- BRANCH 2: ResNet + Transformer Incident Classification ---
        resnet_result = {"category": "NormalVideos", "confidence": 1.0}

        if len(self.frame_buffer) == self.buffer_size and self.resnet_model is not None and HAS_TORCH:
            try:
                from app.video.input import preprocess_clip_frames
                frames_list = list(self.frame_buffer)
                clip_tensor = preprocess_clip_frames(
                    frames_list,
                    target_size=(224, 224),
                    training=False,
                    augment=False,
                    normalize=True,
                    layout="TCHW"
                )
                # Batch dimension (1, T, C, H, W)
                input_batch = clip_tensor.unsqueeze(0).to(self.device)

                with torch.inference_mode():
                    logits = self.resnet_model(input_batch)
                    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                    top_idx = int(np.argmax(probs))
                    top_cat = self.categories[top_idx] if top_idx < len(self.categories) else "Unknown"
                    top_conf = float(probs[top_idx])

                    resnet_result = {
                        "category": top_cat,
                        "confidence": top_conf,
                        "all_probabilities": {cat: float(probs[i]) for i, cat in enumerate(self.categories)}
                    }
            except Exception as e:
                logger.error(f"Error in ResNet + Transformer branch: {e}")

        # Combine annotations onto frame
        incident_banner = f"{resnet_result['category']} ({resnet_result['confidence']*100:.0f}%)" if resnet_result else None
        annotated_frame = self.yolo_detector.draw_bounding_boxes(
            frame,
            yolo_detections,
            draw_labels=True,
            incident_label=incident_banner
        )

        return {
            "yolo_objects": yolo_detections,
            "resnet_incident": resnet_result,
            "annotated_frame": annotated_frame
        }


def load_detector(model_name: str = "yolov8n.pt", conf_threshold: float = 0.35) -> YOLOObjectDetector:
    """Convenience factory to load YOLOObjectDetector."""
    return YOLOObjectDetector(model_name=model_name, conf_threshold=conf_threshold)


if __name__ == "__main__":
    print("=" * 60)
    print(" Multimodal Incident Analysis - Dual Parallel Stream Processor Test ")
    print("=" * 60)

    processor = DualStreamProcessor()

    # Create dummy frame sequence to test dual processing pipeline
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.circle(dummy_frame, (320, 240), 50, (255, 255, 255), -1)

    start = time.time()
    for _ in range(16):
        res = processor.process_frame(dummy_frame)
    elapsed = time.time() - start

    print(f"Dual Processor Device: {processor.device}")
    print(f"16-Frame Sequence Processing Time: {elapsed:.3f} s")
    print(f"YOLO Detections Output           : {len(res['yolo_objects'])} objects")
    print(f"ResNet Incident Classification   : {res['resnet_incident']['category']} ({res['resnet_incident']['confidence']*100:.1f}%)")
    print(f"Annotated Image Shape            : {res['annotated_frame'].shape}")
    print("=" * 60)

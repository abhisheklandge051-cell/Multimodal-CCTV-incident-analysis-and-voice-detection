"""
auth_manager.py - Authorized Personnel / Owner Whitelist Filter

Provides low-latency face verification and track-state caching to prevent
suspicious anomaly triggers and false alarms for verified authorized personnel.
Integrates alongside YOLOv8 multi-object tracking (ByteTrack / BoT-SORT).
"""

import os
import time
import pickle
import logging
from enum import Enum
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union

import cv2
import numpy as np

try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    face_recognition = None
    FACE_RECOGNITION_AVAILABLE = False

logger = logging.getLogger("AuthManager")


class TrackStatus(str, Enum):
    """Lifecycle states for tracked targets."""
    PENDING = "PENDING"
    AUTHORIZED = "AUTHORIZED"
    UNAUTHORIZED = "UNAUTHORIZED"


class AuthManager:
    """
    Manages authorized face embeddings, track state caching, and anomaly suppression.

    Key responsibilities:
    1. Precomputes and caches 128-d face encodings from an `authorized_faces/` directory.
    2. Caches verification states per track ID so expensive face recognition executes
       only while status == PENDING.
    3. Once verified, locks the track to AUTHORIZED, bypassing downstream anomaly logic.
    4. Purges expired tracks after `max_lost_frames` to prevent memory leaks.
    5. Renders color-coded HUD badges (Green for AUTHORIZED, Red for MONITORED).
    """

    def __init__(
        self,
        faces_dir: Union[str, Path] = "authorized_faces",
        cache_file: Union[str, Path] = "whitelist_embeddings.pkl",
        tolerance: float = 0.48,
        max_lost_frames: int = 60,
        unmatched_threshold: int = 5,
        min_crop_size: int = 40,
        reverify_interval: int = 30
    ):
        """
        Initialize the AuthManager.

        Args:
            faces_dir: Directory containing authorized face images.
            cache_file: Path to pickle file for cached face embeddings.
            tolerance: Euclidean distance threshold for a match (default: 0.48).
            max_lost_frames: Number of frames before purging an unseen track from cache.
            unmatched_threshold: Consecutive failed face checks before marking UNAUTHORIZED.
            min_crop_size: Minimum pixel width/height to attempt face recognition.
            reverify_interval: Frame interval to re-check unverified targets if face becomes visible.
        """
        self.faces_dir = Path(faces_dir)
        self.cache_file = Path(cache_file)
        self.tolerance = tolerance
        self.max_lost_frames = max_lost_frames
        self.unmatched_threshold = unmatched_threshold
        self.min_crop_size = min_crop_size
        self.reverify_interval = reverify_interval

        # Known whitelist database
        self.known_encodings: List[np.ndarray] = []
        self.known_identities: List[str] = []

        # Active track state cache: track_id -> dict
        self.authorized_track_ids: Dict[int, Dict[str, Any]] = {}

        if not FACE_RECOGNITION_AVAILABLE:
            logger.error("`face_recognition` library is not available! Whitelist verification will be disabled.")
        else:
            self.load_or_generate_whitelist()

    def load_or_generate_whitelist(self, force_recompute: bool = False) -> int:
        """
        Loads embeddings from `whitelist_embeddings.pkl` if present.
        Otherwise, parses the `authorized_faces/` folder and generates the cache.

        Supports two folder layouts:
        1. Single image per user: `authorized_faces/<person_name>.jpg`
        2. Multiple images per user: `authorized_faces/<person_name>/<photo1>.jpg`

        Returns:
            Number of successfully loaded embeddings.
        """
        if not FACE_RECOGNITION_AVAILABLE:
            return 0

        # 1. Attempt loading from disk cache
        if self.cache_file.exists() and not force_recompute:
            try:
                with open(self.cache_file, "rb") as f:
                    data = pickle.load(f)
                self.known_encodings = data.get("encodings", [])
                self.known_identities = data.get("identities", [])
                logger.info(
                    f"Loaded {len(self.known_encodings)} face embedding(s) for "
                    f"{len(set(self.known_identities))} identity/identities from '{self.cache_file}'."
                )
                return len(self.known_encodings)
            except Exception as e:
                logger.warning(f"Failed to read cache file '{self.cache_file}': {e}. Recomputing...")

        # 2. Compute embeddings from authorized_faces directory
        if not self.faces_dir.exists():
            self.faces_dir.mkdir(parents=True, exist_ok=True)
            logger.warning(
                f"Created empty authorized faces directory at '{self.faces_dir}'. "
                f"Add photos of authorized personnel (.jpg, .png) to populate whitelist."
            )
            return 0

        valid_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        self.known_encodings = []
        self.known_identities = []

        logger.info(f"Scanning '{self.faces_dir}' for authorized personnel photos...")
        for root, _, files in os.walk(self.faces_dir):
            for filename in files:
                ext = Path(filename).suffix.lower()
                if ext not in valid_extensions:
                    continue

                filepath = Path(root) / filename
                # Identity inferred from parent subfolder or file stem
                if Path(root) != self.faces_dir:
                    identity = Path(root).name
                else:
                    identity = Path(filename).stem

                try:
                    # Load image in BGR and convert to RGB
                    img_bgr = cv2.imread(str(filepath))
                    if img_bgr is None:
                        logger.warning(f"Could not decode image at '{filepath}'")
                        continue

                    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                    boxes = face_recognition.face_locations(img_rgb, model="hog")
                    if not boxes:
                        logger.warning(f"No face detected in '{filepath}'. Skipping.")
                        continue

                    encodings = face_recognition.face_encodings(img_rgb, known_face_locations=boxes)
                    for enc in encodings:
                        self.known_encodings.append(enc)
                        self.known_identities.append(identity)
                        logger.info(f"Registered authorized embedding for '{identity}' from '{filename}'.")

                except Exception as ex:
                    logger.error(f"Error processing face image '{filepath}': {ex}")

        # 3. Save to cache file
        if self.known_encodings:
            try:
                cache_data = {
                    "encodings": self.known_encodings,
                    "identities": self.known_identities,
                    "generated_at": time.time()
                }
                with open(self.cache_file, "wb") as f:
                    pickle.dump(cache_data, f)
                logger.info(f"Saved {len(self.known_encodings)} embeddings to cache '{self.cache_file}'.")
            except Exception as e:
                logger.error(f"Failed to write whitelist cache file: {e}")

        return len(self.known_encodings)

    def verify_crop(self, person_crop: np.ndarray) -> Tuple[bool, str, float]:
        """
        Executes face detection and 128-d encoding matching on a cropped bounding box.
        Downscales the crop by 50% (fx=0.5, fy=0.5) to minimize dlib CPU latency.

        Args:
            person_crop: BGR numpy image representing the detected person.

        Returns:
            Tuple of:
            - is_authorized: bool (True if distance <= tolerance)
            - identity: str (Matched name, or "Unknown" / "No_Face")
            - best_distance: float (Euclidean distance to closest whitelist match)
        """
        if not FACE_RECOGNITION_AVAILABLE or len(self.known_encodings) == 0:
            return False, "Unknown", 1.0

        if person_crop is None or person_crop.size == 0:
            return False, "No_Face", 1.0

        h, w = person_crop.shape[:2]
        if h < self.min_crop_size or w < self.min_crop_size:
            return False, "No_Face", 1.0

        try:
            # Downscale crop by 50% (fx=0.5, fy=0.5) to drastically reduce dlib CPU latency
            small_crop = cv2.resize(person_crop, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
            rgb_crop = cv2.cvtColor(small_crop, cv2.COLOR_BGR2RGB)

            # Detect face within the downscaled crop (HOG model is fast and CPU-efficient)
            face_locations = face_recognition.face_locations(rgb_crop, model="hog")
            if not face_locations:
                return False, "No_Face", 1.0

            # Extract 128-d face encodings
            crop_encodings = face_recognition.face_encodings(rgb_crop, known_face_locations=face_locations)
            if not crop_encodings:
                return False, "No_Face", 1.0

            # Compare each detected face against whitelist
            for face_enc in crop_encodings:
                distances = face_recognition.face_distance(self.known_encodings, face_enc)
                best_idx = int(np.argmin(distances))
                min_dist = float(distances[best_idx])

                if min_dist <= self.tolerance:
                    matched_identity = self.known_identities[best_idx]
                    return True, matched_identity, min_dist

            # Faces found, but none matched within tolerance
            best_unmatched_dist = float(np.min(distances)) if len(distances) > 0 else 1.0
            return False, "Unknown", best_unmatched_dist

        except Exception as e:
            logger.debug(f"Exception during face verification on crop: {e}")
            return False, "Error", 1.0

    def update_track(
        self,
        track_id: int,
        frame_bgr: np.ndarray,
        bbox: Tuple[int, int, int, int],
        frame_idx: int
    ) -> Dict[str, Any]:
        """
        Updates the authorization state of a track ID using cached state and lazy face verification.
        Runs face recognition ONLY ONCE when a new track ID appears, and is completely bypassed
        once the track ID is verified as AUTHORIZED.

        Args:
            track_id: Unique tracking ID assigned by ByteTrack / BoT-SORT.
            frame_bgr: Full BGR frame.
            bbox: (x1, y1, x2, y2) bounding box of the detected target.
            frame_idx: Current frame sequence index.

        Returns:
            Dictionary containing the updated track state.
        """
        # 1. Initialize track state if new
        if track_id not in self.authorized_track_ids:
            self.authorized_track_ids[track_id] = {
                "status": TrackStatus.PENDING.value,
                "identity": "Unknown",
                "last_seen_frame": frame_idx,
                "consecutive_unmatched_frames": 0,
                "best_distance": 1.0,
                "attempted": False,
                "last_check_frame": 0
            }

        state = self.authorized_track_ids[track_id]
        state["last_seen_frame"] = frame_idx

        # 2. If already AUTHORIZED or already attempted, bypass expensive face recognition entirely
        if state["status"] == TrackStatus.AUTHORIZED.value:
            return state

        if state.get("attempted", False):
            return state

        # 3. Periodic gate: only check if never attempted or if reverify_interval has passed
        if state.get("last_check_frame", 0) > 0 and (frame_idx - state["last_check_frame"]) < self.reverify_interval:
            return state

        state["last_check_frame"] = frame_idx
        x1, y1, x2, y2 = bbox
        h, w = frame_bgr.shape[:2]
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        person_crop = frame_bgr[y1:y2, x1:x2]

        is_auth, identity, dist = self.verify_crop(person_crop)

        if is_auth:
            state["status"] = TrackStatus.AUTHORIZED.value
            state["identity"] = identity
            state["best_distance"] = dist
            state["attempted"] = True
            logger.info(
                f"Track ID #{track_id} locked to AUTHORIZED identity '{identity}' "
                f"(Distance: {dist:.3f} <= {self.tolerance})"
            )
        else:
            if identity != "No_Face":
                # Definitively checked face and not authorized -> Lock to UNAUTHORIZED
                state["status"] = TrackStatus.UNAUTHORIZED.value
                state["identity"] = "Unknown"
                state["best_distance"] = dist
                state["attempted"] = True
            else:
                # Face not visible yet (back turned) -> keep as PENDING, will retry after reverify_interval
                state["status"] = TrackStatus.PENDING.value

        return state

    def cleanup_lost_tracks(self, current_frame_idx: int) -> List[int]:
        """
        Purges tracks from cache that have not been observed for more than `max_lost_frames`.
        Prevents unbounded memory growth during continuous 24/7 surveillance.

        Args:
            current_frame_idx: Current frame sequence counter.

        Returns:
            List of track IDs that were removed.
        """
        purged_ids = []
        for track_id, state in list(self.authorized_track_ids.items()):
            if (current_frame_idx - state["last_seen_frame"]) > self.max_lost_frames:
                purged_ids.append(track_id)
                del self.authorized_track_ids[track_id]

        if purged_ids:
            logger.debug(f"Purged {len(purged_ids)} inactive track(s) from auth cache: {purged_ids}")

        return purged_ids

    def should_bypass_anomaly(self, active_track_ids: List[int]) -> bool:
        """
        Determines whether downstream anomaly detection and alerts should be bypassed.

        Rule:
        If ALL active person tracks in the scene are verified AUTHORIZED, anomaly alerts are suppressed.
        If any active person is UNAUTHORIZED or PENDING, downstream monitoring remains active.

        Args:
            active_track_ids: List of active track IDs in the current frame.

        Returns:
            True if all tracked persons are verified authorized, False otherwise.
        """
        if not active_track_ids:
            return False

        for tid in active_track_ids:
            state = self.authorized_track_ids.get(tid)
            if not state or state["status"] != TrackStatus.AUTHORIZED.value:
                return False

        return True

    def draw_auth_overlay(
        self,
        frame: np.ndarray,
        detections: List[Dict[str, Any]]
    ) -> np.ndarray:
        """
        Renders color-coded bounding boxes and badges:
        - Green box + "AUTHORIZED: <NAME>" for verified authorized personnel.
        - Red box + "MONITORED: UNKNOWN" for unverified/pending targets.

        Args:
            frame: OpenCV BGR frame.
            detections: List of detection dicts with keys: 'box', 'track_id', 'label'.

        Returns:
            Annotated OpenCV BGR frame.
        """
        for det in detections:
            track_id = det.get("track_id")
            x1, y1, x2, y2 = det["box"]

            state = self.authorized_track_ids.get(track_id) if track_id is not None else None
            status = state["status"] if state else TrackStatus.PENDING.value
            identity = state["identity"] if state else "Unknown"

            if status == TrackStatus.AUTHORIZED.value:
                color = (0, 220, 0)  # Bright Green
                badge_text = f"AUTHORIZED: {identity.upper()}"
            elif status == TrackStatus.UNAUTHORIZED.value:
                color = (0, 0, 255)  # Red
                badge_text = f"MONITORED: UNKNOWN [ID:{track_id}]"
            else:
                color = (0, 165, 255)  # Orange / Amber for Pending
                badge_text = f"PENDING: VERIFYING... [ID:{track_id}]"

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Draw upper badge
            (tw, th), baseline = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            badge_y1 = max(0, y1 - th - baseline - 4)
            badge_y2 = y1 if y1 - th - baseline - 4 >= 0 else y1 + th + baseline + 4
            cv2.rectangle(frame, (x1, badge_y1), (x1 + tw + 6, badge_y2), color, -1)

            text_color = (0, 0, 0) if sum(color) > 400 else (255, 255, 255)
            cv2.putText(
                frame,
                badge_text,
                (x1 + 3, badge_y1 + th + 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                text_color,
                1,
                cv2.LINE_AA
            )

        return frame

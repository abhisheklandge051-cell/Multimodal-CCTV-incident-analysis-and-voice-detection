"""
auth_manager.py - Authorized Personnel / Owner Whitelist Filter

Provides low-latency face verification and track-state caching to prevent
suspicious anomaly triggers and false alarms for verified authorized personnel.
Integrates alongside YOLOv8 multi-object tracking (ByteTrack / BoT-SORT).
"""

import os
import sys
import time
import pickle
import logging
import io
import base64
import re
import threading
from enum import Enum
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union

import cv2
import numpy as np

# Project root resolution
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

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
        # Resolve paths relative to project root if relative
        self.faces_dir = Path(faces_dir)
        if not self.faces_dir.is_absolute():
            self.faces_dir = _PROJECT_ROOT / self.faces_dir

        self.cache_file = Path(cache_file)
        if not self.cache_file.is_absolute():
            self.cache_file = _PROJECT_ROOT / self.cache_file

        self.tolerance = tolerance
        self.max_lost_frames = max_lost_frames
        self.unmatched_threshold = unmatched_threshold
        self.min_crop_size = min_crop_size
        self.reverify_interval = reverify_interval

        # Thread safety
        self._auth_lock = threading.Lock()

        # Known whitelist database
        self.known_encodings: List[np.ndarray] = []
        self.known_identities: List[str] = []

        # Active track state cache: track_id -> dict
        self.authorized_track_ids: Dict[int, Dict[str, Any]] = {}

        # Temporal bypass state cache (persists detection smoothly across frames)
        self._last_auth_time: float = 0.0
        self._last_auth_names: List[str] = []
        self.bypass_duration_sec: float = 2.0

        if not FACE_RECOGNITION_AVAILABLE:
            logger.error("`face_recognition` library is not available! Whitelist verification will be disabled.")
        else:
            self.load_or_generate_whitelist()
            try:
                self.sync_with_db()
            except Exception as e:
                logger.debug(f"Initial DB sync skipped: {e}")

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

    def sync_with_db(self, db_session=None) -> int:
        """
        Synchronizes known face encodings with active AuthorizedUser records in SQLite.
        """
        try:
            from backend.api.database import SessionLocal
            from backend.api.models import AuthorizedUser

            db = db_session or SessionLocal()
            try:
                users = db.query(AuthorizedUser).filter(AuthorizedUser.active == True).all()
                added = 0
                with self._auth_lock:
                    for u in users:
                        if u.face_encoding and len(u.face_encoding) == 128:
                            enc = np.array(u.face_encoding, dtype=np.float64)
                            # Check if already present
                            already_known = False
                            for known_enc, known_id in zip(self.known_encodings, self.known_identities):
                                if known_id.lower() == u.name.lower() and np.linalg.norm(known_enc - enc) < 0.05:
                                    already_known = True
                                    break
                            if not already_known:
                                self.known_encodings.append(enc)
                                self.known_identities.append(u.name)
                                added += 1
                if added > 0:
                    logger.info(f"Synchronized {added} authorized user(s) from database.")
                return len(self.known_identities)
            finally:
                if db_session is None:
                    db.close()
        except Exception as e:
            logger.debug(f"sync_with_db note: {e}")
            return len(self.known_identities)

    def enroll_user(
        self,
        name: str,
        image_input: Union[np.ndarray, bytes, str],
        role: str = "Authorized Personnel"
    ) -> Tuple[bool, str, Optional[List[float]], Optional[str]]:
        """
        Enroll a new authorized user.
        Accepts:
        - OpenCV BGR numpy array
        - Raw image bytes (JPEG/PNG)
        - Base64 data URI or raw base64 string

        Extracts 128-d face embedding, saves image file to authorized_faces/<name>.jpg,
        and updates in-memory and disk caches.
        Returns: (success, message, encoding_list, thumbnail_b64).
        """
        if not FACE_RECOGNITION_AVAILABLE:
            return False, "Face recognition library is not available on this server.", None, None

        # Decode image
        img_bgr = None
        if isinstance(image_input, np.ndarray):
            img_bgr = image_input
        elif isinstance(image_input, bytes):
            nparr = np.frombuffer(image_input, np.uint8)
            img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        elif isinstance(image_input, str):
            clean_b64 = re.sub(r"^data:image\/[a-zA-Z]+;base64,", "", image_input.strip())
            try:
                img_bytes = base64.b64decode(clean_b64)
                nparr = np.frombuffer(img_bytes, np.uint8)
                img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            except Exception as e:
                return False, f"Failed to decode base64 image: {e}", None, None

        if img_bgr is None or img_bgr.size == 0:
            return False, "Invalid image data provided.", None, None

        # Downscale if excessively large
        h, w = img_bgr.shape[:2]
        max_dim = 1280
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)))

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Detect face locations
        boxes = face_recognition.face_locations(img_rgb, model="hog")
        if not boxes:
            return False, "No face detected in the photo. Please make sure face is clearly visible, well-lit, and facing forward.", None, None

        def box_area(b):
            top, right, bottom, left = b
            return (bottom - top) * (right - left)

        best_box = max(boxes, key=box_area)
        encodings = face_recognition.face_encodings(img_rgb, known_face_locations=[best_box])
        if not encodings:
            return False, "Could not extract face embedding from the detected face.", None, None

        face_enc = encodings[0]
        enc_list = [float(x) for x in face_enc]

        # Crop thumbnail around face with padding
        top, right, bottom, left = best_box
        pad_y = int((bottom - top) * 0.35)
        pad_x = int((right - left) * 0.35)
        y1 = max(0, top - pad_y)
        y2 = min(img_bgr.shape[0], bottom + pad_y)
        x1 = max(0, left - pad_x)
        x2 = min(img_bgr.shape[1], right + pad_x)
        crop = img_bgr[y1:y2, x1:x2]
        crop_thumb = cv2.resize(crop, (160, 160))
        _, thumb_buf = cv2.imencode(".jpg", crop_thumb, [cv2.IMWRITE_JPEG_QUALITY, 85])
        thumb_b64 = "data:image/jpeg;base64," + base64.b64encode(thumb_buf).decode("utf-8")

        # Save to authorized_faces folder
        self.faces_dir.mkdir(parents=True, exist_ok=True)
        safe_name = re.sub(r"[^a-zA-Z0-9_\-]", "_", name.strip())
        img_filename = f"{safe_name}.jpg"
        img_path = self.faces_dir / img_filename
        cv2.imwrite(str(img_path), img_bgr)

        with self._auth_lock:
            self.known_encodings.append(face_enc)
            self.known_identities.append(name.strip())

            # Save updated cache file
            try:
                cache_data = {
                    "encodings": self.known_encodings,
                    "identities": self.known_identities,
                    "generated_at": time.time()
                }
                with open(self.cache_file, "wb") as f:
                    pickle.dump(cache_data, f)
            except Exception as e:
                logger.warning(f"Could not update cache file: {e}")

        logger.info(f"Enrolled authorized user '{name}' with 128-d face embedding.")
        return True, f"Authorized user '{name}' enrolled successfully.", enc_list, thumb_b64

    def delete_user(self, name: str) -> bool:
        """Removes all embeddings and face files for given user name."""
        with self._auth_lock:
            new_encs = []
            new_ids = []
            for enc, ident in zip(self.known_encodings, self.known_identities):
                if ident.lower() != name.lower():
                    new_encs.append(enc)
                    new_ids.append(ident)
            self.known_encodings = new_encs
            self.known_identities = new_ids

            try:
                with open(self.cache_file, "wb") as f:
                    pickle.dump({"encodings": self.known_encodings, "identities": self.known_identities, "generated_at": time.time()}, f)
            except Exception:
                pass

        # Remove file from authorized_faces if exists
        safe_name = re.sub(r"[^a-zA-Z0-9_\-]", "_", name.strip())
        for ext in (".jpg", ".jpeg", ".png", ".webp"):
            fp = self.faces_dir / f"{safe_name}{ext}"
            if fp.exists():
                try:
                    fp.unlink()
                except Exception:
                    pass

        logger.info(f"Removed authorized user '{name}' from whitelist.")
        return True

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        detections: Optional[List[Dict[str, Any]]] = None,
        frame_idx: int = 0
    ) -> Dict[str, Any]:
        """
        High-level live frame analysis for authorized personnel detection and incident bypass.

        Returns dict:
        {
            "bypass_incident": bool,       # True if authorized user is detected in the scene
            "authorized_found": bool,     # True if face verified in this frame
            "authorized_names": List[str], # Unique authorized names visible
            "auth_detections": List[Dict], # Bounding boxes and badges for drawing
        }
        """
        now = time.time()
        auth_names: List[str] = []
        auth_detections: List[Dict[str, Any]] = []

        if not FACE_RECOGNITION_AVAILABLE or len(self.known_encodings) == 0 or frame_bgr is None:
            if (now - self._last_auth_time) < self.bypass_duration_sec:
                return {
                    "bypass_incident": True,
                    "authorized_found": True,
                    "authorized_names": self._last_auth_names,
                    "auth_detections": [],
                }
            return {
                "bypass_incident": False,
                "authorized_found": False,
                "authorized_names": [],
                "auth_detections": [],
            }

        # 1. Process person detections from YOLO using track-state cache
        person_boxes = []
        if detections:
            for det in detections:
                if det.get("label") == "person":
                    box = det["box"]
                    track_id = det.get("track_id")
                    if track_id is not None:
                        # Fast track-state cache: face recognition executes ONLY ONCE per new track ID
                        track_state = self.update_track(track_id, frame_bgr, box, frame_idx)
                        if track_state["status"] == TrackStatus.AUTHORIZED.value:
                            auth_names.append(track_state["identity"])
                            auth_detections.append({
                                "box": box,
                                "status": "AUTHORIZED",
                                "identity": track_state["identity"],
                                "track_id": track_id
                            })
                    else:
                        person_boxes.append(box)

            # Purge inactive tracks after max_lost_frames to maintain low memory usage
            if frame_idx % 30 == 0:
                self.cleanup_lost_tracks(frame_idx)

        # 2. Check un-tracked person crops periodically (at most once every 15 frames)
        if (frame_idx % 15 == 0) and person_boxes:
            box = person_boxes[0]
            x1, y1, x2, y2 = box
            h, w = frame_bgr.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            crop = frame_bgr[y1:y2, x1:x2]
            if crop.size > 0:
                is_auth, identity, dist = self.verify_crop(crop)
                if is_auth:
                    auth_names.append(identity)
                    auth_detections.append({
                        "box": (x1, y1, x2, y2),
                        "status": "AUTHORIZED",
                        "identity": identity,
                        "distance": dist,
                    })

        # 3. Update temporal smoothing
        if auth_names:
            self._last_auth_time = now
            self._last_auth_names = list(set(auth_names))
            bypass_active = True
        else:
            if (now - self._last_auth_time) < self.bypass_duration_sec:
                bypass_active = True
                auth_names = self._last_auth_names
            else:
                bypass_active = False

        return {
            "bypass_incident": bypass_active,
            "authorized_found": len(auth_names) > 0,
            "authorized_names": list(set(auth_names)),
            "auth_detections": auth_detections,
        }


# Singleton accessor
_auth_manager_instance: Optional[AuthManager] = None

def get_auth_manager() -> AuthManager:
    """Returns the shared global AuthManager instance."""
    global _auth_manager_instance
    if _auth_manager_instance is None:
        _auth_manager_instance = AuthManager()
    return _auth_manager_instance

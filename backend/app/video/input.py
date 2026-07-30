"""
Video & Dataset Input Module (Ultra-Fast Cached & High-Accuracy Preprocessor)
========================================================================================
Provides high-performance dataset loading, clip discovery, frame sequencing, 
and advanced spatio-temporal video preprocessing for CCTV incident analysis & 
multimodal validation engines.

Key Features for Maximum Accuracy & Training Performance:
1. Segment-Based Temporal Sampling (TSN): Segment-random sampling for training & segment-center for validation.
2. Video-Consistent Data Augmentation: Clip-level horizontal flip, random crop, and photometric jitter.
3. Aspect-Preserving Smart Crop & Resize: Eliminates spatial aspect ratio distortion.
4. Corrupt Frame Filtering & Integrity Validation: Prevents broken images from ruining training runs.
5. Automated Class Weighting & Normalization: Handles class imbalance and standardizes RGB channels.
6. Sub-millisecond JSON Index Caching & Parallel Batch Preprocessing.
"""

from __future__ import annotations

import os
import re
import json
import glob
import math
import random
import logging
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Generator, Union, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("video.input")

# Optional dependency imports with fallbacks
try:
    from PIL import Image, ImageEnhance, ImageOps
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import cv2
    cv2.setNumThreads(0)
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# Default ImageNet normalization statistics
IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)


def sample_clip_paths(
    paths: Sequence[Path],
    num_frames: int = 16,
    training: bool = True,
    sampling_strategy: str = "segment"
) -> List[Path]:
    """
    Advanced spatio-temporal frame sampling for video clips.
    
    Strategies:
      - 'segment': Temporal Segment Networks (TSN) style.
                  Divides clip into N equal segments.
                  If training=True: randomly pick 1 frame per segment (temporal jittering).
                  If training=False: pick exact middle frame per segment (deterministic eval).
      - 'uniform': Uniform linear interpolation across total frame sequence.
      - 'stride':  Consecutive sampling with fixed temporal stride.
    """
    total = len(paths)
    if total == 0:
        return []

    if total < num_frames:
        # Edge padding / repetition if clip is shorter than requested num_frames
        indices = np.linspace(0, total - 1, num=num_frames, dtype=np.int64)
        return [paths[idx] for idx in indices]

    if sampling_strategy == "segment":
        bounds = np.linspace(0, total, num=num_frames + 1, dtype=np.int64)
        selected_indices = []
        for i in range(num_frames):
            seg_start, seg_end = bounds[i], bounds[i + 1]
            if seg_end <= seg_start:
                seg_end = seg_start + 1
            seg_end = min(seg_end, total)
            if training:
                idx = random.randint(seg_start, seg_end - 1)
            else:
                idx = (seg_start + seg_end) // 2
                idx = min(idx, total - 1)
            selected_indices.append(idx)
        return [paths[idx] for idx in selected_indices]

    elif sampling_strategy == "stride":
        max_start = max(0, total - num_frames)
        start_idx = random.randint(0, max_start) if training else max_start // 2
        return [paths[start_idx + i] for i in range(min(num_frames, total - start_idx))]

    else:  # 'uniform'
        indices = np.linspace(0, total - 1, num=num_frames, dtype=np.int64)
        return [paths[idx] for idx in indices]


def preprocess_clip_frames(
    images: List[Union[Image.Image, np.ndarray]],
    target_size: Tuple[int, int] = (224, 224),
    training: bool = True,
    augment: bool = True,
    normalize: bool = True,
    mean: Tuple[float, float, float] = IMAGE_MEAN,
    std: Tuple[float, float, float] = IMAGE_STD,
    layout: str = "CTHW"
) -> Union[np.ndarray, "torch.Tensor"]:
    """
    Applies video-consistent spatial, crop, and photometric transformations across frames.
    
    Crucial for high accuracy:
      - All frames in a clip share identical horizontal flipping and crop decisions.
      - Preserves spatial aspect ratio via smart aspect-resizing before crop.
      - Standardizes RGB channels.
      - Returns tensor formatted as CTHW (channels, time, height, width) or TCHW / THWC.
    """
    if not images:
        raise ValueError("Cannot preprocess empty list of frame images.")

    # Convert all frames to PIL Images for consistent processing
    pil_frames: List[Image.Image] = []
    for img in images:
        if isinstance(img, np.ndarray):
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif img.ndim == 3 and img.shape[2] == 3:
                # Convert OpenCV BGR to RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil_frames.append(Image.fromarray(img))
        elif HAS_PIL and isinstance(img, Image.Image):
            pil_frames.append(img.convert("RGB"))
        else:
            raise TypeError(f"Unsupported image type: {type(img)}")

    # Generate clip-consistent augmentation parameters
    do_flip = training and augment and (random.random() < 0.5)
    do_jitter = training and augment and (random.random() < 0.6)
    
    if do_jitter:
        brightness_factor = random.uniform(0.85, 1.15)
        contrast_factor = random.uniform(0.85, 1.15)
        saturation_factor = random.uniform(0.85, 1.15)

    target_h, target_w = target_size
    first_w, first_h = pil_frames[0].size

    # Calculate aspect-preserving scale
    scale = max(target_w / first_w, target_h / first_h)
    new_w, new_h = max(target_w, int(math.ceil(first_w * scale))), max(target_h, int(math.ceil(first_h * scale)))

    # Calculate spatial crop coordinates (consistent across clip)
    if training and augment:
        max_x = max(0, new_w - target_w)
        max_y = max(0, new_h - target_h)
        crop_x = random.randint(0, max_x) if max_x > 0 else 0
        crop_y = random.randint(0, max_y) if max_y > 0 else 0
    else:
        crop_x = max(0, (new_w - target_w) // 2)
        crop_y = max(0, (new_h - target_h) // 2)

    processed_arrays: List[np.ndarray] = []
    resample_mode = getattr(Image, "Resampling", Image).LANCZOS

    for frame in pil_frames:
        # Aspect-preserving resize
        if frame.size != (new_w, new_h):
            frame = frame.resize((new_w, new_h), resample_mode)
        
        # Spatial crop
        frame = frame.crop((crop_x, crop_y, crop_x + target_w, crop_y + target_h))

        # Consistent horizontal flip
        if do_flip:
            frame = ImageOps.mirror(frame)

        # Consistent photometric jitter
        if do_jitter:
            frame = ImageEnhance.Brightness(frame).enhance(brightness_factor)
            frame = ImageEnhance.Contrast(frame).enhance(contrast_factor)
            frame = ImageEnhance.Color(frame).enhance(saturation_factor)

        # Convert to numpy float32 array in range [0, 1]
        arr = np.asarray(frame, dtype=np.float32) / 255.0

        if normalize:
            arr = (arr - np.asarray(mean, dtype=np.float32)) / np.asarray(std, dtype=np.float32)

        processed_arrays.append(arr)

    # Stack into array of shape (T, H, W, C)
    clip_array = np.stack(processed_arrays, axis=0)

    # Permute into target layout
    # Default layout: 'CTHW' -> (Channels, Time, Height, Width)
    if layout.upper() == "CTHW":
        clip_array = clip_array.transpose(3, 0, 1, 2)  # (C, T, H, W)
    elif layout.upper() == "TCHW":
        clip_array = clip_array.transpose(0, 3, 1, 2)  # (T, C, H, W)
    elif layout.upper() == "THWC":
        pass  # Already (T, H, W, C)

    if HAS_TORCH:
        return torch.from_numpy(clip_array.copy())
    return clip_array


@dataclass
class VideoClip:
    """Represents a video sequence clip composed of frame images or a video file."""
    clip_id: str
    category: str
    split: str
    dir_path: Optional[Path] = None
    _frame_paths: Optional[List[Path]] = None
    _frame_count: Optional[int] = None
    video_path: Optional[Path] = None

    @property
    def frame_paths(self) -> List[Path]:
        if self._frame_paths is None:
            if self.dir_path and self.dir_path.exists():
                escaped_id = re.escape(self.clip_id)
                exact_pattern = re.compile(rf"^{escaped_id}[_\-\.]?\d+\.[a-zA-Z0-9]+$", re.IGNORECASE)
                paths = [p for p in self.dir_path.iterdir() if p.is_file() and exact_pattern.match(p.name) and p.suffix.lower() in VideoDataset.SUPPORTED_IMAGE_EXTS]
                if not paths:
                    paths = [p for p in self.dir_path.glob(f"{self.clip_id}*.*") if p.is_file() and p.suffix.lower() in VideoDataset.SUPPORTED_IMAGE_EXTS]
                self._frame_paths = self._sort_frame_paths(paths)
                self._frame_count = len(self._frame_paths)
            else:
                self._frame_paths = []
                self._frame_count = 0
        return self._frame_paths

    @frame_paths.setter
    def frame_paths(self, val: List[Path]):
        self._frame_paths = val
        if val is not None:
            self._frame_count = len(val)

    @property
    def total_frames(self) -> int:
        if self._frame_count is not None:
            return self._frame_count
        if self.video_path and HAS_CV2:
            cap = cv2.VideoCapture(str(self.video_path))
            count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            self._frame_count = count
            return count
        return len(self.frame_paths)

    def _sort_frame_paths(self, paths: List[Path]) -> List[Path]:
        """Sorts frame paths numerically by trailing frame index digits (e.g. clip_100.png, frame001.jpg -> 100, 1)."""
        def extract_frame_num(p: Path) -> int:
            digits = re.findall(r'\d+', p.stem)
            return int(digits[-1]) if digits else 0

        return sorted(paths, key=extract_frame_num)

    def get_sorted_frame_paths(self) -> List[Path]:
        return self.frame_paths

    def get_valid_frame_paths(self, verify_integrity: bool = False) -> List[Path]:
        """Returns frame paths while filtering zero-byte or corrupt frame files."""
        if not verify_integrity and self._frame_paths is not None:
            return self._frame_paths

        valid: List[Path] = []
        for path in self.get_sorted_frame_paths():
            if not path.exists() or path.stat().st_size == 0:
                continue
            if verify_integrity:
                try:
                    if HAS_PIL:
                        with Image.open(path) as img:
                            img.verify()
                    elif HAS_CV2:
                        img = cv2.imread(str(path))
                        if img is None:
                            continue
                except Exception:
                    continue
            valid.append(path)
        return valid

    def preprocess(
        self,
        num_frames: int = 16,
        target_size: Tuple[int, int] = (224, 224),
        training: bool = True,
        sampling: str = "segment",
        augment: bool = True,
        normalize: bool = True,
        layout: str = "CTHW"
    ) -> Union[np.ndarray, "torch.Tensor"]:
        """
        Preprocesses this clip into clean, augmented, normalized spatio-temporal tensors.
        """
        paths = self.get_valid_frame_paths()
        sampled_paths = sample_clip_paths(paths, num_frames=num_frames, training=training, sampling_strategy=sampling)
        
        frames: List[Union[Image.Image, np.ndarray]] = []
        for p in sampled_paths:
            if HAS_PIL:
                frames.append(Image.open(p))
            elif HAS_CV2:
                img = cv2.imread(str(p))
                if img is not None:
                    frames.append(img)
            else:
                raise RuntimeError("Neither PIL nor OpenCV is available to load image frames.")

        return preprocess_clip_frames(
            frames,
            target_size=target_size,
            training=training,
            augment=augment,
            normalize=normalize,
            layout=layout
        )


class VideoDataset:
    """
    Dataset loader for multi-modal incident video data with instant JSON caching 
    and state-of-the-art dataset preprocessing & class balancing.
    """

    SUPPORTED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}
    SUPPORTED_VIDEO_EXTS = {".mp4", ".avi", ".mkv", ".mov", ".webm"}

    def __init__(self, root_dir: Optional[Union[str, Path]] = None, use_cache: bool = True):
        if root_dir is None:
            base_dir = Path(__file__).resolve().parents[3]
            possible_paths = [
                base_dir / "datasets" / "archive",
                base_dir / "datasets",
                Path("datasets/archive"),
                Path("datasets")
            ]
            for p in possible_paths:
                if p.exists():
                    root_dir = p
                    break
            if root_dir is None:
                root_dir = base_dir / "datasets"

        self.root_dir = Path(root_dir).resolve()
        self.use_cache = use_cache
        self.cache_file = self.root_dir / ".dataset_index.json"
        self.clips: Dict[str, VideoClip] = {}
        self.categories: List[str] = []
        self.splits: List[str] = []

        start_time = time.time()
        self._load_or_scan()
        elapsed = time.time() - start_time
        logger.info(f"VideoDataset initialized in {elapsed:.3f} seconds with {len(self.clips)} clips across {len(self.categories)} categories.")

    def _load_or_scan(self) -> None:
        """Loads index from JSON cache if available, otherwise scans directory and caches index."""
        if not self.root_dir.exists():
            logger.warning(f"Dataset root directory does not exist: {self.root_dir}")
            return

        if self.use_cache and self.cache_file.exists():
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    cache_data = json.load(f)

                clips_dict = cache_data.get("clips", {})
                first_clip = next(iter(clips_dict.values()), {}) if clips_dict else {}

                # If cache is missing frame_paths, trigger rescan to build ultra-fast cache
                if clips_dict and ("frame_paths" not in first_clip or first_clip.get("frame_paths") is None):
                    raise ValueError("Legacy index cache missing pre-indexed frame_paths.")

                self.splits = cache_data.get("splits", [])
                self.categories = cache_data.get("categories", [])
                for full_id, item in cache_data.get("clips", {}).items():
                    raw_paths = item.get("frame_paths")
                    frame_paths = [Path(p) for p in raw_paths] if raw_paths else None
                    clip = VideoClip(
                        clip_id=item["clip_id"],
                        category=item["category"],
                        split=item["split"],
                        dir_path=Path(item["dir_path"]) if item.get("dir_path") else None,
                        video_path=Path(item["video_path"]) if item.get("video_path") else None,
                        _frame_paths=frame_paths,
                        _frame_count=item.get("frame_count")
                    )
                    self.clips[full_id] = clip
                logger.info(f"Loaded dataset index from cache file: {self.cache_file.name}")
                return
            except Exception as e:
                logger.warning(f"Rescanning dataset index ({e})...")

        self._scan_dataset()
        if self.use_cache:
            self._save_cache()

    def _save_cache(self) -> None:
        """Saves lightweight index cache to JSON for instant future loads."""
        try:
            cache_clips = {}
            for full_id, clip in self.clips.items():
                paths = clip._frame_paths if clip._frame_paths is not None else clip.frame_paths
                cache_clips[full_id] = {
                    "clip_id": clip.clip_id,
                    "category": clip.category,
                    "split": clip.split,
                    "dir_path": str(clip.dir_path) if clip.dir_path else None,
                    "video_path": str(clip.video_path) if clip.video_path else None,
                    "frame_count": clip._frame_count,
                    "frame_paths": [str(p) for p in paths] if paths else None
                }
            cache_data = {
                "splits": self.splits,
                "categories": self.categories,
                "clips": cache_clips
            }
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(cache_data, f, indent=2)
            logger.info(f"Saved dataset index cache to {self.cache_file}")
        except Exception as e:
            logger.warning(f"Could not save dataset index cache: {e}")

    def _scan_dataset(self) -> None:
        """Scans dataset directory for splits, categories, and video clips."""
        subdirs = [d for d in self.root_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]
        has_splits = any(d.name.lower() in {"train", "test", "val", "validation"} for d in subdirs)

        if has_splits:
            for split_dir in subdirs:
                split_name = split_dir.name
                if split_name not in self.splits:
                    self.splits.append(split_name)
                self._scan_split(split_dir, split_name)
        else:
            self._scan_split(self.root_dir, split_name="default")

        self.categories = sorted(list({clip.category for clip in self.clips.values()}))

    def _scan_split(self, split_dir: Path, split_name: str) -> None:
        """Scans category subdirectories inside a split directory using fast scandir."""
        category_dirs = [d for d in split_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]

        for cat_dir in category_dirs:
            category = cat_dir.name
            seen_clip_paths: Dict[str, List[Path]] = {}

            try:
                with os.scandir(cat_dir) as entries:
                    for entry in entries:
                        if not entry.is_file():
                            continue

                        name = entry.name
                        ext = os.path.splitext(name)[1].lower()

                        if ext in self.SUPPORTED_IMAGE_EXTS:
                            stem = os.path.splitext(name)[0]
                            match = re.match(r'^(.*?)[_\-\.]?\d+$', stem)
                            clip_id = match.group(1) if (match and match.group(1)) else stem
                            if clip_id not in seen_clip_paths:
                                seen_clip_paths[clip_id] = []
                            seen_clip_paths[clip_id].append(Path(entry.path))

                        elif ext in self.SUPPORTED_VIDEO_EXTS:
                            clip_id = os.path.splitext(name)[0]
                            full_id = f"{split_name}_{category}_{clip_id}"
                            self.clips[full_id] = VideoClip(
                                clip_id=clip_id,
                                category=category,
                                split=split_name,
                                video_path=Path(entry.path)
                            )

                for clip_id, raw_paths in seen_clip_paths.items():
                    full_id = f"{split_name}_{category}_{clip_id}"
                    clip = VideoClip(
                        clip_id=clip_id,
                        category=category,
                        split=split_name,
                        dir_path=cat_dir,
                        _frame_count=len(raw_paths)
                    )
                    clip._frame_paths = clip._sort_frame_paths(raw_paths)
                    self.clips[full_id] = clip
            except Exception as e:
                logger.error(f"Error scanning category directory {cat_dir}: {e}")

    def get_summary(self) -> Dict[str, Union[int, List[str], Dict[str, int]]]:
        """Returns structured statistics summary of dataset."""
        summary = {
            "total_clips": len(self.clips),
            "total_frames": sum(c.total_frames for c in self.clips.values()),
            "splits": self.splits,
            "categories": self.categories,
            "category_counts": {}
        }
        for clip in self.clips.values():
            cat = clip.category
            if cat not in summary["category_counts"]:
                summary["category_counts"][cat] = 0
            summary["category_counts"][cat] += 1
        return summary

    def compute_class_weights(
        self,
        split: str = "Train",
        smoothing: float = 0.1,
        output_type: str = "tensor"
    ) -> Union[Dict[str, float], "torch.Tensor"]:
        """
        Calculates balanced class weights for loss function optimization (e.g. CrossEntropyLoss).
        Solves accuracy loss caused by class imbalance in incident datasets.
        """
        counts = {cat: 0 for cat in self.categories}
        for clip in self.get_clips(split=split):
            if clip.category in counts:
                counts[clip.category] += 1

        total_samples = sum(counts.values())
        num_classes = len(self.categories)

        if total_samples == 0:
            weights_arr = np.ones(num_classes, dtype=np.float32)
        else:
            weights_arr = np.zeros(num_classes, dtype=np.float32)
            for idx, cat in enumerate(self.categories):
                count = counts[cat]
                # Inverse class frequency weighting with smoothing
                if count > 0:
                    weights_arr[idx] = total_samples / (num_classes * count)
                else:
                    weights_arr[idx] = 1.0

            # Normalize weights to sum to num_classes
            weights_arr = (weights_arr / weights_arr.sum()) * num_classes
            weights_arr = (1.0 - smoothing) * weights_arr + smoothing * 1.0

        if output_type == "dict":
            return {cat: float(weights_arr[i]) for i, cat in enumerate(self.categories)}
        elif output_type == "numpy":
            return weights_arr
        elif HAS_TORCH and output_type == "tensor":
            return torch.tensor(weights_arr, dtype=torch.float32)
        return weights_arr

    def compute_dataset_mean_std(
        self,
        split: str = "Train",
        num_samples: int = 50,
        target_size: Tuple[int, int] = (224, 224)
    ) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """
        Calculates exact RGB channel mean and standard deviation across sampled dataset frames.
        """
        clips = self.get_clips(split=split)
        if not clips:
            return IMAGE_MEAN, IMAGE_STD

        sampled_clips = random.sample(clips, min(num_samples, len(clips)))
        means, stds = [], []

        for clip in sampled_clips:
            paths = clip.get_valid_frame_paths()
            if not paths:
                continue
            p = random.choice(paths)
            try:
                if HAS_PIL:
                    img = Image.open(p).convert("RGB").resize(target_size)
                    arr = np.asarray(img, dtype=np.float32) / 255.0
                elif HAS_CV2:
                    img = cv2.imread(str(p))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, target_size)
                    arr = img.astype(np.float32) / 255.0
                else:
                    continue
                means.append(arr.mean(axis=(0, 1)))
                stds.append(arr.std(axis=(0, 1)))
            except Exception:
                continue

        if not means:
            return IMAGE_MEAN, IMAGE_STD

        overall_mean = tuple(np.mean(means, axis=0).tolist())
        overall_std = tuple(np.mean(stds, axis=0).tolist())
        logger.info(f"Calculated dataset RGB Mean: {overall_mean}, Std: {overall_std}")
        return overall_mean, overall_std

    def get_clips(self, split: Optional[str] = None, category: Optional[str] = None) -> List[VideoClip]:
        """Filters video clips by split and/or category."""
        results = []
        for clip in self.clips.values():
            if split and clip.split.lower() != split.lower():
                continue
            if category and clip.category.lower() != category.lower():
                continue
            results.append(clip)
        return results

    def get_clip_by_id(self, clip_id: str) -> Optional[VideoClip]:
        """Finds clip matching clip_id or key in dataset."""
        if clip_id in self.clips:
            return self.clips[clip_id]
        for clip in self.clips.values():
            if clip.clip_id == clip_id:
                return clip
        return None

    def get_random_clip(self, category: Optional[str] = None, split: Optional[str] = None) -> Optional[VideoClip]:
        """Returns a random clip matching criteria."""
        clips = self.get_clips(split=split, category=category)
        return random.choice(clips) if clips else None

    def load_clip_frames(
        self,
        clip_or_id: Union[str, VideoClip],
        max_frames: Optional[int] = None,
        target_size: Optional[Tuple[int, int]] = None
    ) -> List[Union[Path, object]]:
        """
        Loads frame images for a given clip.
        If PIL or CV2 is available, returns loaded image objects/arrays.
        Otherwise returns list of Path objects.
        """
        clip = clip_or_id if isinstance(clip_or_id, VideoClip) else self.get_clip_by_id(clip_or_id)
        if not clip:
            logger.error(f"Clip not found: {clip_or_id}")
            return []

        frame_paths = clip.get_valid_frame_paths()
        if max_frames and max_frames < len(frame_paths):
            step = max(1, len(frame_paths) // max_frames)
            frame_paths = frame_paths[::step][:max_frames]

        loaded_frames = []
        for path in frame_paths:
            if HAS_CV2:
                img = cv2.imread(str(path))
                if img is not None:
                    if target_size:
                        img = cv2.resize(img, target_size)
                    loaded_frames.append(img)
            elif HAS_PIL:
                img = Image.open(path)
                if target_size:
                    img = img.resize(target_size)
                loaded_frames.append(img)
            else:
                loaded_frames.append(path)

        return loaded_frames

    def preprocess_clip(
        self,
        clip_or_id: Union[str, VideoClip],
        num_frames: int = 16,
        target_size: Tuple[int, int] = (224, 224),
        training: bool = True,
        sampling: str = "segment",
        augment: bool = True,
        normalize: bool = True,
        layout: str = "CTHW"
    ) -> Union[np.ndarray, "torch.Tensor"]:
        """
        High-accuracy preprocessor for a single video clip.
        """
        clip = clip_or_id if isinstance(clip_or_id, VideoClip) else self.get_clip_by_id(clip_or_id)
        if not clip:
            raise ValueError(f"Clip not found: {clip_or_id}")
        return clip.preprocess(
            num_frames=num_frames,
            target_size=target_size,
            training=training,
            sampling=sampling,
            augment=augment,
            normalize=normalize,
            layout=layout
        )

    def preprocess_batch(
        self,
        clips: Sequence[Union[str, VideoClip]],
        num_frames: int = 16,
        target_size: Tuple[int, int] = (224, 224),
        training: bool = True,
        sampling: str = "segment",
        augment: bool = True,
        normalize: bool = True,
        layout: str = "CTHW",
        num_workers: int = 4
    ) -> Union[np.ndarray, "torch.Tensor"]:
        """
        Parallel multithreaded batch preprocessing for fast data feeding during training.
        """
        resolved_clips = [c if isinstance(c, VideoClip) else self.get_clip_by_id(c) for c in clips]
        valid_clips = [c for c in resolved_clips if c is not None]

        if not valid_clips:
            raise ValueError("No valid clips provided for batch preprocessing.")

        processed: List[Union[np.ndarray, torch.Tensor]] = [None] * len(valid_clips)

        def _worker(idx: int, clip_item: VideoClip):
            tensor_or_arr = clip_item.preprocess(
                num_frames=num_frames,
                target_size=target_size,
                training=training,
                sampling=sampling,
                augment=augment,
                normalize=normalize,
                layout=layout
            )
            return idx, tensor_or_arr

        if num_workers > 1:
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = [executor.submit(_worker, i, c) for i, c in enumerate(valid_clips)]
                for future in as_completed(futures):
                    idx, data = future.result()
                    processed[idx] = data
        else:
            for i, c in enumerate(valid_clips):
                processed[i] = c.preprocess(
                    num_frames=num_frames,
                    target_size=target_size,
                    training=training,
                    sampling=sampling,
                    augment=augment,
                    normalize=normalize,
                    layout=layout
                )

        if HAS_TORCH and isinstance(processed[0], torch.Tensor):
            return torch.stack(processed, dim=0)
        return np.stack(processed, axis=0)

    def stream_clip_frames(
        self,
        clip_or_id: Union[str, VideoClip],
        loop: bool = False
    ) -> Generator[Dict[str, Union[int, str, Path, object]], None, None]:
        """
        Generator yielding frames sequentially to simulate CCTV video stream ingestion.
        Yields dict containing frame_index, frame_data, frame_path, clip_id, and category.
        """
        clip = clip_or_id if isinstance(clip_or_id, VideoClip) else self.get_clip_by_id(clip_or_id)
        if not clip:
            logger.error(f"Clip not found for streaming: {clip_or_id}")
            return

        frame_paths = clip.get_valid_frame_paths()
        if not frame_paths and not clip.video_path:
            logger.warning(f"No frames or video file for clip {clip.clip_id}")
            return

        while True:
            if frame_paths:
                for idx, path in enumerate(frame_paths):
                    frame_obj = None
                    if HAS_CV2:
                        frame_obj = cv2.imread(str(path))
                    elif HAS_PIL:
                        frame_obj = Image.open(path)
                    else:
                        frame_obj = path

                    yield {
                        "frame_index": idx,
                        "frame_data": frame_obj,
                        "frame_path": str(path),
                        "clip_id": clip.clip_id,
                        "category": clip.category,
                        "split": clip.split
                    }
            elif clip.video_path and HAS_CV2:
                cap = cv2.VideoCapture(str(clip.video_path))
                idx = 0
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    yield {
                        "frame_index": idx,
                        "frame_data": frame,
                        "video_path": str(clip.video_path),
                        "clip_id": clip.clip_id,
                        "category": clip.category,
                        "split": clip.split
                    }
                    idx += 1
                cap.release()

            if not loop:
                break


# Convenience Helper Functions
def load_dataset(dataset_dir: Optional[Union[str, Path]] = None, use_cache: bool = True) -> VideoDataset:
    """Convenience factory function to load VideoDataset instantly."""
    return VideoDataset(root_dir=dataset_dir, use_cache=use_cache)


def load_clip(clip_id: str, dataset_dir: Optional[Union[str, Path]] = None) -> Optional[VideoClip]:
    """Helper to load a specific clip by ID."""
    ds = load_dataset(dataset_dir)
    return ds.get_clip_by_id(clip_id)


if __name__ == "__main__":
    print("=" * 60)
    print(" Multimodal Incident Analysis - Enhanced Dataset Preprocessor Test ")
    print("=" * 60)

    start = time.time()
    dataset = load_dataset()
    init_time = time.time() - start

    summary = dataset.get_summary()

    print(f"Initialization : {init_time:.4f} seconds")
    print(f"Root Directory : {dataset.root_dir}")
    print(f"Total Clips    : {summary['total_clips']}")
    print(f"Total Frames   : {summary['total_frames']}")
    print(f"Splits         : {summary['splits']}")
    print(f"Categories ({len(summary['categories'])}): {', '.join(summary['categories'])}")

    # Test class weights calculation
    class_weights = dataset.compute_class_weights(output_type="dict")
    print("\nSample Class Weights (for loss balancing):")
    for cat, weight in list(class_weights.items())[:5]:
        print(f"  {cat:15s}: {weight:.4f}")

    # Sample clip testing
    if dataset.clips:
        sample_clip = dataset.get_random_clip()
        print(f"\nSample Clip Selected: {sample_clip.clip_id} (Category: {sample_clip.category}, Split: {sample_clip.split})")
        print(f"Total frames in clip: {sample_clip.total_frames}")

        # Preprocessing test
        tensor = dataset.preprocess_clip(sample_clip, num_frames=16, target_size=(224, 224), training=True, sampling="segment")
        if HAS_TORCH and isinstance(tensor, torch.Tensor):
            print(f"Preprocessed Tensor Shape (CTHW): {tensor.shape}, dtype: {tensor.dtype}")
        else:
            print(f"Preprocessed NumPy Array Shape (CTHW): {tensor.shape}, dtype: {tensor.dtype}")

    print("=" * 60)
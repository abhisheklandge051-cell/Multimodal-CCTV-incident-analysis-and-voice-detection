import os
import sys

os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8"
os.environ["AV_LOG_FORCE_NOCOLOR"] = "1"

import json
import hashlib
import random
import logging
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path
from collections import OrderedDict

import cv2

if hasattr(cv2, "setLogLevel"):
    cv2.setLogLevel(0)
if hasattr(cv2, "utils") and hasattr(cv2.utils, "logging"):
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as transforms
import torchvision.transforms.functional as F

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")
logger = logging.getLogger("XD_Violence_Pipeline")

LABEL_MAP: Dict[str, int] = {
    "Abuse": 0,
    "CarAccident": 1,
    "Explosion": 2,
    "Fighting": 3,
    "Normal": 4,
    "Riot": 5,
    "Shooting": 6
}

REVERSE_LABEL_MAP: Dict[int, str] = {v: k for k, v in LABEL_MAP.items()}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def compute_file_hash(filepath: str, block_size: int = 65536) -> str:
    hasher = hashlib.md5()
    try:
        with open(filepath, "rb") as f:
            buf = f.read(block_size)
            while len(buf) > 0:
                hasher.update(buf)
                buf = f.read(block_size)
        return hasher.hexdigest()
    except Exception as e:
        logger.warning(f"Failed to compute hash for {filepath}: {e}")
        return ""


def verify_video_file(filepath: str) -> Optional[Dict[str, Any]]:
    if not os.path.isfile(filepath) or os.path.getsize(filepath) == 0:
        return None

    cap = cv2.VideoCapture(filepath)
    if not cap.isOpened():
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    if total_frames <= 0 or width <= 0 or height <= 0:
        return None

    fps = 30.0 if fps <= 0 or np.isnan(fps) else fps
    duration = total_frames / fps

    return {
        "video_path": os.path.abspath(filepath),
        "total_frames": total_frames,
        "fps": fps,
        "duration": round(duration, 2),
        "resolution": (width, height),
    }


def verify_dataset(dataset_dir: str) -> Tuple[List[Dict[str, Any]], Dict[str, List[str]]]:
    logger.info(f"Starting Dataset Verification in: {dataset_dir}")
    valid_videos = []
    issues = {"corrupted": [], "duplicate": [], "invalid_label": [], "missing": []}
    seen_hashes = {}

    video_extensions = {".mp4", ".avi", ".mkv", ".mov"}
    found_files = []
    root_path = Path(dataset_dir)
    for p in root_path.rglob("*"):
        if p.is_file() and p.suffix.lower() in video_extensions:
            found_files.append(str(p))

    for filepath in found_files:
        path_parts = Path(filepath).parts
        class_name = None
        for part in path_parts:
            if part in LABEL_MAP:
                class_name = part
                break

        if not class_name:
            issues["invalid_label"].append(filepath)
            continue

        meta = verify_video_file(filepath)
        if meta is None:
            issues["corrupted"].append(filepath)
            continue

        file_hash = compute_file_hash(filepath)
        if file_hash in seen_hashes:
            issues["duplicate"].append(filepath)
            continue
        seen_hashes[file_hash] = filepath

        meta["label"] = class_name
        meta["label_id"] = LABEL_MAP[class_name]
        meta["hash"] = file_hash
        valid_videos.append(meta)

    logger.info(f"Verification Complete: {len(valid_videos)} valid videos found.")
    return valid_videos, issues


def split_dataset(
    verified_videos: List[Dict[str, Any]],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> List[Dict[str, Any]]:
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-5, "Ratios must sum to 1.0"
    random.seed(seed)

    label_groups: Dict[int, List[Dict[str, Any]]] = {}
    for video in verified_videos:
        lbl = video["label_id"]
        label_groups.setdefault(lbl, []).append(video)

    split_videos = []
    for lbl, videos in label_groups.items():
        random.shuffle(videos)
        n = len(videos)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        for i, video in enumerate(videos):
            if i < n_train:
                video["split"] = "train"
            elif i < n_train + n_val:
                video["split"] = "val"
            else:
                video["split"] = "test"
            split_videos.append(video)

    return split_videos


def build_clip_index(
    split_videos: List[Dict[str, Any]],
    clip_len: int = 16,
    overlap_ratio: float = 0.50
) -> List[Dict[str, Any]]:
    stride = max(1, int(clip_len * (1.0 - overlap_ratio)))
    clip_index = []

    for video in split_videos:
        total_frames = video["total_frames"]
        vpath = video["video_path"]
        lbl_id = video["label_id"]
        split = video["split"]

        if total_frames < clip_len:
            clip_index.append({
                "video_path": vpath,
                "start_frame": 0,
                "end_frame": total_frames,
                "label": video["label"],
                "label_id": lbl_id,
                "split": split
            })
        else:
            for start in range(0, total_frames - clip_len + 1, stride):
                end = start + clip_len
                clip_index.append({
                    "video_path": vpath,
                    "start_frame": start,
                    "end_frame": end,
                    "label": video["label"],
                    "label_id": lbl_id,
                    "split": split
                })

    return clip_index


class VideoClipTransform:
    def __init__(self, crop_size: int = 224, is_train: bool = True):
        self.crop_size = crop_size
        self.is_train = is_train
        self.normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    def __call__(self, clip: List[np.ndarray]) -> torch.Tensor:
        tensors = [F.to_tensor(f) for f in clip]

        if self.is_train:
            do_flip = random.random() < 0.5
            _, h, w = tensors[0].shape
            if h > self.crop_size and w > self.crop_size:
                top = random.randint(0, h - self.crop_size)
                left = random.randint(0, w - self.crop_size)
            else:
                top, left = 0, 0

            brightness_factor = random.uniform(0.8, 1.2)
            contrast_factor = random.uniform(0.8, 1.2)
            do_blur = random.random() < 0.2

            processed_frames = []
            for t in tensors:
                if do_flip:
                    t = F.hflip(t)
                t = F.crop(t, top, left, self.crop_size, self.crop_size)
                t = F.adjust_brightness(t, brightness_factor)
                t = F.adjust_contrast(t, contrast_factor)
                if do_blur:
                    t = F.gaussian_blur(t, kernel_size=[3, 3])
                t = self.normalize(t)
                processed_frames.append(t)
        else:
            processed_frames = []
            for t in tensors:
                t = F.center_crop(t, (self.crop_size, self.crop_size))
                t = self.normalize(t)
                processed_frames.append(t)

        clip_tensor = torch.stack(processed_frames, dim=0)
        clip_tensor = clip_tensor.permute(1, 0, 2, 3)
        return clip_tensor


class XDViolenceDataset(Dataset):
    def __init__(
        self,
        clip_indices: List[Dict[str, Any]],
        clip_len: int = 16,
        crop_size: int = 224,
        target_short_side: int = 256,
        is_train: bool = True,
        max_cache_clips: int = 2000
    ):
        self.clip_indices = clip_indices
        self.clip_len = clip_len
        self.target_short_side = target_short_side
        self.is_train = is_train
        self.transform = VideoClipTransform(crop_size=crop_size, is_train=is_train)
        self.max_cache_clips = max_cache_clips
        self.clip_cache: OrderedDict[Tuple[str, int, int], List[np.ndarray]] = OrderedDict()

    def __len__(self) -> int:
        return len(self.clip_indices)

    def _read_clip_frames(self, clip_info: Dict[str, Any]) -> List[np.ndarray]:
        vpath = clip_info["video_path"]
        start_frame = clip_info["start_frame"]
        end_frame = clip_info["end_frame"]
        cache_key = (vpath, start_frame, end_frame)

        if cache_key in self.clip_cache:
            self.clip_cache.move_to_end(cache_key)
            return self.clip_cache[cache_key]

        frames = []
        cap = None
        try:
            cap = cv2.VideoCapture(vpath)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                for _ in range(end_frame - start_frame):
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        break
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    h, w, _ = frame.shape

                    if h < w:
                        new_h = self.target_short_side
                        new_w = int(w * (self.target_short_side / h))
                    else:
                        new_w = self.target_short_side
                        new_h = int(h * (self.target_short_side / w))

                    frame_resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                    frames.append(frame_resized)
        except Exception as e:
            logger.warning(f"Error reading frames from {vpath}: {e}")
        finally:
            if cap is not None:
                cap.release()

        if len(frames) == 0:
            dummy_dim = self.target_short_side
            frames = [np.zeros((dummy_dim, dummy_dim, 3), dtype=np.uint8) for _ in range(self.clip_len)]

        while len(frames) < self.clip_len:
            frames.append(frames[-1].copy())

        frames = frames[:self.clip_len]

        if len(self.clip_cache) >= self.max_cache_clips:
            self.clip_cache.popitem(last=False)
        self.clip_cache[cache_key] = frames

        return frames

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        clip_info = self.clip_indices[idx]
        frames = self._read_clip_frames(clip_info)
        clip_tensor = self.transform(frames)
        label_tensor = torch.tensor(clip_info["label_id"], dtype=torch.long)

        meta = {
            "video_path": clip_info["video_path"],
            "start_frame": clip_info["start_frame"],
            "end_frame": clip_info["end_frame"],
            "label": clip_info["label"],
            "split": clip_info["split"]
        }
        return clip_tensor, label_tensor, meta


def create_weighted_sampler(clip_indices: List[Dict[str, Any]]) -> WeightedRandomSampler:
    labels = [clip["label_id"] for clip in clip_indices]
    class_counts = np.bincount(labels, minlength=len(LABEL_MAP))
    class_weights = 1.0 / np.maximum(class_counts, 1)
    sample_weights = [class_weights[lbl] for lbl in labels]
    sample_weights_tensor = torch.tensor(sample_weights, dtype=torch.double)

    return WeightedRandomSampler(
        weights=sample_weights_tensor,
        num_samples=len(sample_weights_tensor),
        replacement=True
    )


def create_dataloaders(
    dataset_dir: str,
    batch_size: int = 8,
    num_workers: int = 0,
    persistent_workers: bool = False,
    prefetch_factor: Optional[int] = None,
    clip_len: int = 16,
    crop_size: int = 224,
    target_short_side: int = 256,
    seed: int = 42,
    use_cache: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, Any]]:
    cache_path = os.path.join(dataset_dir, "dataset_cache.json")

    if use_cache and os.path.exists(cache_path):
        logger.info(f"Loading dataset metadata & clip index from cache: {cache_path}")
        with open(cache_path, "r") as f:
            cache_data = json.load(f)
        verified_videos = cache_data["verified_videos"]
        all_clips = cache_data["all_clips"]
        issues = cache_data.get("issues", {})
    else:
        verified_videos, issues = verify_dataset(dataset_dir)
        if not verified_videos:
            raise ValueError(f"No valid videos found in directory: {dataset_dir}")

        split_videos = split_dataset(verified_videos, seed=seed)
        all_clips = build_clip_index(split_videos, clip_len=clip_len)

        if use_cache:
            try:
                cache_data = {
                    "verified_videos": verified_videos,
                    "all_clips": all_clips,
                    "issues": issues
                }
                with open(cache_path, "w") as f:
                    json.dump(cache_data, f)
                logger.info(f"Saved dataset cache to: {cache_path}")
            except Exception as e:
                logger.warning(f"Failed to write dataset cache: {e}")

    train_clips = [c for c in all_clips if c["split"] == "train"]
    val_clips = [c for c in all_clips if c["split"] == "val"]
    test_clips = [c for c in all_clips if c["split"] == "test"]

    train_dataset = XDViolenceDataset(train_clips, clip_len=clip_len, crop_size=crop_size, target_short_side=target_short_side, is_train=True)
    val_dataset = XDViolenceDataset(val_clips, clip_len=clip_len, crop_size=crop_size, target_short_side=target_short_side, is_train=False)
    test_dataset = XDViolenceDataset(test_clips, clip_len=clip_len, crop_size=crop_size, target_short_side=target_short_side, is_train=False)

    train_sampler = create_weighted_sampler(train_clips) if train_clips else None

    loader_kwargs: Dict[str, Any] = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available()
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = persistent_workers
        if prefetch_factor is not None:
            loader_kwargs["prefetch_factor"] = prefetch_factor

    train_loader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        **loader_kwargs
    )

    val_kwargs = loader_kwargs.copy()
    val_kwargs["shuffle"] = False
    val_loader = DataLoader(
        val_dataset,
        **val_kwargs
    )

    test_kwargs = loader_kwargs.copy()
    test_kwargs["shuffle"] = False
    test_loader = DataLoader(
        test_dataset,
        **test_kwargs
    )

    stats = {
        "total_videos": len(verified_videos),
        "issues": issues,
        "clip_counts": {
            "train": len(train_clips),
            "val": len(val_clips),
            "test": len(test_clips),
        }
    }

    return train_loader, val_loader, test_loader, stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="XD-Violence Data Loading & Processing Pipeline")
    parser.add_argument("--dataset_dir", type=str, default="XD-voilence_dataset", help="Path to dataset")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of worker processes")
    args = parser.parse_args()

    if os.path.exists(args.dataset_dir):
        train_loader, val_loader, test_loader, stats = create_dataloaders(
            dataset_dir=args.dataset_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )
        print(f"Dataset verified: {stats['total_videos']} videos.")
        print(f"Train clips: {stats['clip_counts']['train']}, Val clips: {stats['clip_counts']['val']}, Test clips: {stats['clip_counts']['test']}")

        for batch_clips, batch_labels, _ in train_loader:
            print(f"Batch Tensor Shape: {batch_clips.shape}")
            print(f"Batch Labels: {batch_labels.tolist()}")
            break

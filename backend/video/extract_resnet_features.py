import os
import sys

# Suppress FFmpeg C-level log warnings before importing cv2
os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8"
os.environ["AV_LOG_FORCE_NOCOLOR"] = "1"

import json
import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Any

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
import torchvision.transforms as transforms
import torchvision.transforms.functional as F

from tqdm import tqdm

if hasattr(cv2, "setLogLevel"):
    cv2.setLogLevel(0)
if hasattr(cv2, "utils") and hasattr(cv2.utils, "logging"):
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")
logger = logging.getLogger("ResNet_Feature_Extractor")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class ResNetFeatureExtractor:
    def __init__(self, device: torch.device):
        self.device = device
        logger.info(f"Initializing ResNet50 Feature Extractor on {device}...")
        self.backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.backbone.fc = nn.Identity()
        self.backbone.to(self.device)
        self.backbone.eval()
        self.normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    @torch.no_grad()
    def extract_video_features(self, video_path: str, target_short_side: int = 256, batch_size: int = 32) -> np.ndarray:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return np.zeros((0, 2048), dtype=np.float32)

        frames_tensor_list = []
        all_features = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, _ = frame_rgb.shape

            if h < w:
                new_h = target_short_side
                new_w = int(w * (target_short_side / h))
            else:
                new_w = target_short_side
                new_h = int(h * (target_short_side / w))

            frame_resized = cv2.resize(frame_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            t = F.to_tensor(frame_resized)
            t = F.center_crop(t, (224, 224))
            t = self.normalize(t)
            frames_tensor_list.append(t)

            if len(frames_tensor_list) >= batch_size:
                batch_tensor = torch.stack(frames_tensor_list, dim=0).to(self.device)
                features = self.backbone(batch_tensor).cpu().numpy()
                all_features.append(features)
                frames_tensor_list.clear()

        cap.release()

        if len(frames_tensor_list) > 0:
            batch_tensor = torch.stack(frames_tensor_list, dim=0).to(self.device)
            features = self.backbone(batch_tensor).cpu().numpy()
            all_features.append(features)

        if len(all_features) == 0:
            return np.zeros((0, 2048), dtype=np.float32)

        return np.concatenate(all_features, axis=0).astype(np.float32)


def run_feature_extraction(
    dataset_dir: str = "XD-voilence_dataset",
    output_dir: str = "XD-voilence_dataset/features",
    batch_size: int = 32
) -> str:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = ResNetFeatureExtractor(device=device)

    os.makedirs(output_dir, exist_ok=True)
    cache_path = os.path.join(dataset_dir, "dataset_cache.json")

    if os.path.exists(cache_path):
        logger.info(f"Loading metadata from dataset cache: {cache_path}")
        with open(cache_path, "r") as f:
            cache_data = json.load(f)
        verified_videos = cache_data["verified_videos"]
    else:
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
        try:
            from backend.video.loading_processing import verify_dataset, split_dataset
        except ImportError:
            from loading_processing import verify_dataset, split_dataset
        verified_videos, _ = verify_dataset(dataset_dir)
        verified_videos = split_dataset(verified_videos)

    total_videos = len(verified_videos)
    logger.info(f"Extracting ResNet50 features for {total_videos} videos into '{output_dir}'...")

    extracted_count = 0
    skipped_count = 0

    pbar = tqdm(verified_videos, desc=f"Extracting Features ({total_videos} videos)", unit="video", dynamic_ncols=True)
    for video in pbar:
        vpath = video["video_path"]
        vhash = video.get("hash", hashlib.md5(vpath.encode("utf-8")).hexdigest()[:12])
        out_filename = f"{vhash}.npy"
        out_filepath = os.path.join(output_dir, out_filename)

        video["feature_path"] = out_filepath

        if os.path.exists(out_filepath) and os.path.getsize(out_filepath) > 0:
            skipped_count += 1
            pbar.set_postfix(extracted=extracted_count, skipped=skipped_count)
            continue

        feats = extractor.extract_video_features(vpath, batch_size=batch_size)
        if len(feats) > 0:
            np.save(out_filepath, feats)
            extracted_count += 1
        else:
            logger.warning(f"Could not extract features for: {vpath}")

        pbar.set_postfix(extracted=extracted_count, skipped=skipped_count)

    # Save feature mapping metadata
    feature_meta_path = os.path.join(dataset_dir, "feature_dataset_cache.json")
    with open(feature_meta_path, "w") as f:
        json.dump(verified_videos, f)
    logger.info(f"Feature extraction complete! Saved mapping metadata to '{feature_meta_path}'.")

    return feature_meta_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract ResNet50 Features for XD-Violence Dataset")
    parser.add_argument("--dataset_dir", type=str, default="XD-voilence_dataset", help="Dataset directory")
    parser.add_argument("--output_dir", type=str, default="XD-voilence_dataset/features", help="Output feature directory")
    parser.add_argument("--batch_size", type=int, default=32, help="Extraction batch size")
    args = parser.parse_args()

    run_feature_extraction(dataset_dir=args.dataset_dir, output_dir=args.output_dir, batch_size=args.batch_size)

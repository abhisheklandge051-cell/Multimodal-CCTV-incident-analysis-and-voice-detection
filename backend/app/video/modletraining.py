from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

try:
    import cv2
    cv2.setNumThreads(0)
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    from PIL import Image, ImageEnhance, ImageOps
except ImportError as error:
    raise RuntimeError("Pillow is required to load video frames. Install it with 'pip install pillow'.") from error

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
    HAS_TORCH = True
except Exception as error:
    HAS_TORCH = False
    TORCH_ERROR = error

try:
    import torchvision
    import torchvision.models as models
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False

import sys
try:
    from app.video.input import VideoClip, VideoDataset, load_dataset, sample_clip_paths, preprocess_clip_frames
    from app.video.epochcycle import EpochCycleTracker, classification_metrics, run_baseline_epoch
except ImportError:
    backend_dir = str(Path(__file__).resolve().parents[2])
    if backend_dir not in sys.path:
        sys.path.insert(0, backend_dir)
    try:
        from app.video.input import VideoClip, VideoDataset, load_dataset, sample_clip_paths, preprocess_clip_frames
        from app.video.epochcycle import EpochCycleTracker, classification_metrics, run_baseline_epoch
    except ImportError:
        from input import VideoClip, VideoDataset, load_dataset, sample_clip_paths, preprocess_clip_frames
        from epochcycle import EpochCycleTracker, classification_metrics, run_baseline_epoch


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True)
logger = logging.getLogger("video.modletraining")

IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)


def resolve_training_device(requested_device: Optional[str] = None) -> str:
    """Choose a GPU device when available; otherwise fall back to CPU."""
    if not HAS_TORCH:
        return "cpu"

    if requested_device in {None, "", "auto"}:
        requested_device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        requested_device = str(requested_device).strip().lower()

    if requested_device.startswith("cuda"):
        if not torch.cuda.is_available():
            logger.warning("CUDA was requested but no compatible GPU is available. Falling back to CPU.")
            return "cpu"
        device_name = torch.cuda.get_device_name(requested_device)
        vram_gb = torch.cuda.get_device_properties(requested_device).total_memory / (1024**3)
        logger.info("⚡ [GPU ENABLED] Model training will process on GPU Device: '%s' (%.2f GB VRAM)", device_name, vram_gb)
        return requested_device

    logger.warning("Training on CPU. For faster GPU training, ensure CUDA PyTorch is installed and use --device cuda.")
    return "cpu"


def configure_torch_runtime(device: str) -> None:
    """Enable the best available runtime settings for stable GPU training."""
    if not HAS_TORCH:
        return

    if device.startswith("cuda"):
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")


def set_seed(seed: int) -> None:
    """Make runs repeatable as far as the installed PyTorch version permits."""
    random.seed(seed)
    np.random.seed(seed)
    if HAS_TORCH:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def sample_paths(paths: Sequence[Path], num_frames: int) -> List[Path]:
    """Choose uniformly spaced frames, repeating only for very short clips."""
    if not paths:
        return []
    indices = np.linspace(0, len(paths) - 1, num=num_frames, dtype=np.int64)
    return [paths[index] for index in indices]


if HAS_TORCH:
    class FrameFolderDataset(Dataset):
        """Clip dataset with frame paths resolved once during construction."""

        def __init__(
            self,
            dataset: VideoDataset,
            split: str,
            num_frames: int,
            image_size: int,
            training: bool,
        ) -> None:
            self.categories = sorted(dataset.categories)
            self.class_to_index = {name: index for index, name in enumerate(self.categories)}
            self.num_frames = num_frames
            self.image_size = image_size
            self.training = training
            self.samples: List[Tuple[Tuple[Path, ...], int, str]] = []

            for clip in dataset.get_clips(split=split):
                paths = tuple(clip.frame_paths)
                if not paths:
                    continue
                self.samples.append((paths, self.class_to_index[clip.category], clip.clip_id))
            logger.info("Indexed %d valid clips for %s split.", len(self.samples), split)

        def __len__(self) -> int:
            return len(self.samples)

        def _transform(self, image: Image.Image) -> torch.Tensor:
            """PIL/NumPy preprocessing that deliberately does not need torchvision."""
            resize_mode = getattr(Image, "Resampling", Image).LANCZOS
            image = image.resize((self.image_size, self.image_size), resize_mode)
            if self.training:
                if random.random() < 0.5:
                    image = ImageOps.mirror(image)
                image = ImageEnhance.Brightness(image).enhance(random.uniform(0.88, 1.12))
                image = ImageEnhance.Contrast(image).enhance(random.uniform(0.88, 1.12))
            array = np.asarray(image, dtype=np.float32) / 255.0
            array = (array - np.asarray(IMAGE_MEAN, dtype=np.float32)) / np.asarray(IMAGE_STD, dtype=np.float32)
            return torch.from_numpy(array.transpose(2, 0, 1).copy())

        def __getitem__(self, index: int) -> Tuple[torch.Tensor, int, str]:
            paths, label, clip_id = self.samples[index]
            selected = sample_clip_paths(paths, self.num_frames, training=self.training, sampling_strategy="segment")
            images: List[Image.Image] = []
            for path in selected:
                try:
                    with Image.open(path) as image:
                        images.append(image.convert("RGB"))
                except (OSError, ValueError) as error:
                    logger.warning(f"Could not read frame '{path}': {error}")
            if not images:
                raise RuntimeError(f"No readable frames found for clip '{clip_id}'.")

            clip_tensor = preprocess_clip_frames(
                images,
                target_size=(self.image_size, self.image_size),
                training=self.training,
                augment=self.training,
                normalize=True,
                layout="TCHW"
            )
            return clip_tensor, label, clip_id


    class ConvBlock(nn.Module):
        def __init__(self, input_channels: int, output_channels: int, stride: int = 1) -> None:
            super().__init__()
            self.layers = nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(output_channels),
                nn.GELU(),
                nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(output_channels),
                nn.GELU(),
            )

        def forward(self, image: torch.Tensor) -> torch.Tensor:
            return self.layers(image)


    class ResidualBlock(nn.Module):
        def __init__(self, channels: int) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(channels)
            self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(channels)
            self.act = nn.GELU()

        def forward(self, tensor: torch.Tensor) -> torch.Tensor:
            residual = tensor
            tensor = self.act(self.bn1(self.conv1(tensor)))
            tensor = self.bn2(self.conv2(tensor))
            return self.act(tensor + residual)


    class IncidentVideoClassifier(nn.Module):
        """High-accuracy video classifier using pretrained ResNet backbone & temporal attention."""

        def __init__(self, num_classes: int, pretrained: bool = True, dropout: float = 0.3) -> None:
            super().__init__()
            if pretrained and HAS_TORCHVISION:
                logger.info("⚡ Loading ImageNet Pretrained ResNet-18 Backbone for high accuracy & fast convergence...")
                try:
                    weights = models.ResNet18_Weights.DEFAULT
                    resnet = models.resnet18(weights=weights)
                except Exception:
                    resnet = models.resnet18(pretrained=True)
                feature_dim = resnet.fc.in_features
                resnet.fc = nn.Identity()
                self.backbone = resnet
            else:
                if pretrained:
                    logger.warning("--pretrained requested, but torchvision models not found. Falling back to native CNN.")
                feature_dim = 512
                self.backbone = nn.Sequential(
                    nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
                    nn.BatchNorm2d(64),
                    nn.GELU(),
                    nn.MaxPool2d(2),
                    ConvBlock(64, 64),
                    ResidualBlock(64),
                    nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(128),
                    nn.GELU(),
                    ConvBlock(128, 128),
                    ResidualBlock(128),
                    nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(256),
                    nn.GELU(),
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                    nn.Linear(256, feature_dim),
                    nn.GELU(),
                )

            self.temporal_attention = nn.MultiheadAttention(feature_dim, num_heads=8, batch_first=True)
            self.norm = nn.LayerNorm(feature_dim)
            self.head = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(feature_dim, feature_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(feature_dim // 2, num_classes),
            )

        def forward(self, clips: torch.Tensor) -> torch.Tensor:
            batch, frames, channels, height, width = clips.shape
            frame_features = self.backbone(clips.reshape(batch * frames, channels, height, width))
            features = frame_features.reshape(batch, frames, -1)
            attended, _ = self.temporal_attention(features, features, features, need_weights=False)
            pooled_feature = self.norm(features + attended).mean(dim=1)
            return self.head(pooled_feature)


if HAS_TORCH:
    class TorchTrainer:
        def __init__(
            self,
            dataset: VideoDataset,
            frames: int,
            image_size: int,
            batch_size: int,
            workers: int,
            learning_rate: float,
            pretrained: bool,
            device: Optional[str],
        ) -> None:
            self.categories = sorted(dataset.categories)
            if len(self.categories) < 2:
                raise ValueError("At least two categories are required for classification.")
            self.device = torch.device(resolve_training_device(device))
            configure_torch_runtime(self.device.type)
            self.train_dataset = FrameFolderDataset(dataset, "Train", frames, image_size, training=True)
            self.test_dataset = FrameFolderDataset(dataset, "Test", frames, image_size, training=False)
            if not self.train_dataset or not self.test_dataset:
                raise ValueError("Both Train and Test splits must contain readable image clips.")

            if os.name == "nt" and workers > 0:
                logger.info("Configuring 0 loader workers on Windows to prevent multiprocessing deadlocks.")
                workers = 0

            loader_options: Dict[str, object] = {
                "batch_size": batch_size,
                "num_workers": workers,
                "pin_memory": self.device.type == "cuda",
            }
            if workers > 0:
                loader_options["persistent_workers"] = True
                loader_options["prefetch_factor"] = 2
            self.train_loader = DataLoader(self.train_dataset, shuffle=True, **loader_options)
            self.test_loader = DataLoader(self.test_dataset, shuffle=False, **loader_options)
            self.model = IncidentVideoClassifier(len(self.categories), pretrained=pretrained).to(self.device)
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=1e-3)
            class_weights = dataset.compute_class_weights(split="Train", output_type="tensor").to(self.device)
            self.loss_function = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)
            self.scaler = torch.amp.GradScaler('cuda', enabled=self.device.type == "cuda")
            logger.info("Training on %s with %d loader worker(s).", self.device, workers)

        def _evaluate(self) -> Dict[str, object]:
            self.model.eval()
            targets: List[int] = []
            predictions: List[int] = []
            with torch.inference_mode():
                for clips, labels, _ in self.test_loader:
                    logits = self.model(clips.to(self.device, non_blocking=True))
                    targets.extend(labels.tolist())
                    predictions.extend(logits.argmax(dim=1).cpu().tolist())
            return classification_metrics(targets, predictions, len(self.categories))

        def fit(self, epochs: int, save_dir: Path) -> Path:
            save_dir.mkdir(parents=True, exist_ok=True)
            checkpoint = save_dir / "best_incident_video_classifier.pt"
            metrics_file = save_dir / "best_incident_video_metrics.json"
            tracker = EpochCycleTracker(total_epochs=epochs)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=max(1, epochs))

            total_batches = len(self.train_loader)
            total_steps = max(1, epochs * total_batches)

            logger.info("🚀 Starting GPU model training across %d epoch(s) (%d total steps)...", epochs, total_steps)

            for epoch in range(1, epochs + 1):
                self.model.train()
                cumulative_loss = 0.0
                count = 0
                epoch_start = time.perf_counter()

                for batch_idx, (clips, labels, _) in enumerate(self.train_loader, start=1):
                    clips = clips.to(self.device, non_blocking=True)
                    labels = labels.to(self.device, non_blocking=True)
                    self.optimizer.zero_grad(set_to_none=True)
                    with torch.amp.autocast('cuda', enabled=self.device.type == "cuda"):
                        logits = self.model(clips)
                        loss = self.loss_function(logits, labels)

                    if self.device.type == "cuda":
                        self.scaler.scale(loss).backward()
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                        self.optimizer.step()

                    batch_loss = loss.item()
                    cumulative_loss += batch_loss * labels.size(0)
                    count += labels.size(0)

                    # Calculate live progress percentages
                    epoch_pct = (batch_idx / total_batches) * 100.0
                    global_step = (epoch - 1) * total_batches + batch_idx
                    overall_pct = (global_step / total_steps) * 100.0

                    elapsed = time.perf_counter() - epoch_start
                    fps = count / max(0.001, elapsed)

                    # Log terminal progress bar every few batches & on final batch
                    log_frequency = max(1, total_batches // 10)
                    if batch_idx % log_frequency == 0 or batch_idx == total_batches:
                        bar_len = 20
                        filled = int(bar_len * batch_idx / total_batches)
                        bar = "█" * filled + "░" * (bar_len - filled)
                        logger.info(
                            f"Overall: {overall_pct:5.1f}% | "
                            f"Epoch [{epoch:02d}/{epochs:02d}] {epoch_pct:5.1f}% [{bar}] "
                            f"Batch {batch_idx:03d}/{total_batches:03d} | "
                            f"Loss: {batch_loss:.4f} | Speed: {fps:.1f} clips/s"
                        )

                scheduler.step()
                metrics = self._evaluate()
                train_loss = cumulative_loss / max(1, count)
                is_best = tracker.log_epoch(epoch, train_loss, metrics)

                if is_best:
                    torch.save({
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "categories": self.categories,
                        "image_mean": IMAGE_MEAN,
                        "image_std": IMAGE_STD,
                        "metrics": metrics,
                    }, checkpoint)
                    metrics_file.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
            return checkpoint


class HandCraftedBaseline:
    """Cached, transparent baseline for checking that labels and splits work."""

    FEATURE_DIM = 86  # 43 values/frame, then temporal mean and maximum.

    def __init__(self, frames: int, workers: int, seed: int) -> None:
        if not HAS_CV2:
            raise RuntimeError("Baseline mode requires opencv-python.")
        self.frames = frames
        self.workers = max(1, workers)
        self.seed = seed

    def _feature(self, clip: VideoClip) -> np.ndarray:
        paths = sample_paths(clip.get_sorted_frame_paths(), self.frames)
        if not paths:
            return np.zeros(self.FEATURE_DIM, dtype=np.float32)
        values: List[np.ndarray] = []
        previous: Optional[np.ndarray] = None
        for path in paths:
            image = cv2.imread(str(path), cv2.IMREAD_REDUCED_COLOR_4)
            if image is None:
                values.append(np.zeros(43, dtype=np.float32))
                continue
            image = cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            histograms = [cv2.calcHist([image], [channel], None, [8], [0, 256]).ravel() for channel in range(3)]
            colour = np.concatenate(histograms).astype(np.float32)
            colour /= np.linalg.norm(colour) + 1e-6
            x_gradient = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            y_gradient = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
            magnitude = cv2.magnitude(x_gradient, y_gradient)
            gradient = magnitude.reshape(4, 8, 4, 8).mean(axis=(1, 3)).ravel()
            gradient /= np.linalg.norm(gradient) + 1e-6
            motion = np.zeros(3, dtype=np.float32) if previous is None else np.array([
                np.mean(np.abs(gray.astype(np.float32) - previous)),
                np.std(np.abs(gray.astype(np.float32) - previous)),
                np.max(np.abs(gray.astype(np.float32) - previous)),
            ], dtype=np.float32)
            previous = gray
            values.append(np.concatenate((colour, gradient, motion)))
        features = np.asarray(values, dtype=np.float32)
        result = np.concatenate((features.mean(axis=0), features.max(axis=0)))
        return result / (np.linalg.norm(result) + 1e-6)

    def _extract(self, clips: Sequence[VideoClip], label: str) -> np.ndarray:
        started = time.perf_counter()
        output = np.zeros((len(clips), self.FEATURE_DIM), dtype=np.float32)
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            futures = {pool.submit(self._feature, clip): index for index, clip in enumerate(clips)}
            for completed, future in enumerate(as_completed(futures), start=1):
                output[futures[future]] = future.result()
                if completed % 50 == 0 or completed == len(clips):
                    rate = completed / max(0.001, time.perf_counter() - started)
                    logger.info("%s features: %d/%d clips (%.1f clips/s)", label, completed, len(clips), rate)
        return output

    def fit(self, dataset: VideoDataset, epochs: int, batch_size: int, learning_rate: float, save_dir: Path) -> Path:
        save_dir.mkdir(parents=True, exist_ok=True)
        categories = sorted(dataset.categories)
        category_index = {name: index for index, name in enumerate(categories)}
        train_clips = dataset.get_clips("Train")
        test_clips = dataset.get_clips("Test")
        cache = save_dir / "baseline_features_v3.npz"

        if cache.exists():
            data = np.load(cache)
            if data["categories"].tolist() == categories and int(data["frames"]) == self.frames:
                train_x, train_y, test_x, test_y = data["train_x"], data["train_y"], data["test_x"], data["test_y"]
                logger.info("Loaded baseline features from %s.", cache)
            else:
                cache.unlink()
                return self.fit(dataset, epochs, batch_size, learning_rate, save_dir)
        else:
            train_x = self._extract(train_clips, "Train")
            test_x = self._extract(test_clips, "Test")
            train_y = np.array([category_index[clip.category] for clip in train_clips], dtype=np.int64)
            test_y = np.array([category_index[clip.category] for clip in test_clips], dtype=np.int64)
            np.savez(cache, train_x=train_x, train_y=train_y, test_x=test_x, test_y=test_y, categories=np.array(categories), frames=self.frames)

        # Feature standardization (Z-score normalization)
        mean = train_x.mean(axis=0, keepdims=True)
        std = train_x.std(axis=0, keepdims=True) + 1e-6
        train_x_scaled = (train_x - mean) / std
        test_x_scaled = (test_x - mean) / std

        rng = np.random.default_rng(self.seed)
        weights = rng.normal(0, math.sqrt(2 / self.FEATURE_DIM), size=(self.FEATURE_DIM, len(categories))).astype(np.float32)
        bias = np.zeros((1, len(categories)), dtype=np.float32)

        # Vectorized Adam Optimizer setup
        lr = 0.005 if math.isclose(learning_rate, 3e-4) else learning_rate
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        mW, vW = np.zeros_like(weights), np.zeros_like(weights)
        mb, vb = np.zeros_like(bias), np.zeros_like(bias)
        step_count = 0

        tracker = EpochCycleTracker(total_epochs=epochs)

        for epoch in range(1, epochs + 1):
            weights, bias, mW, vW, mb, vb, step_count = run_baseline_epoch(
                train_x_scaled, train_y, weights, bias, batch_size, lr,
                beta1, beta2, eps, mW, vW, mb, vb, step_count, rng
            )
            metrics = classification_metrics(test_y, (test_x_scaled @ weights + bias).argmax(axis=1), len(categories))
            tracker.log_epoch(epoch, -1.0, metrics)

        model = save_dir / "incident_baseline.json"
        model.write_text(json.dumps({
            "categories": categories,
            "mean": mean.squeeze().tolist(),
            "std": std.squeeze().tolist(),
            "weights": weights.tolist(),
            "bias": bias.tolist(),
            "metrics": tracker.best_metrics
        }, indent=2), encoding="utf-8")
        return model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train an incident video classifier from image-frame folders.")
    parser.add_argument("--dataset-dir", type=Path, default=None, help="Dataset root containing Train/ and Test/.")
    parser.add_argument("--mode", choices=("torch", "baseline"), default="torch")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--frames", type=int, default=16, help="Uniformly sampled frames per clip.")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--workers", type=int, default=min(4, max(0, (os.cpu_count() or 2) - 1)), help="Data-loader/feature workers; use 0 to debug loading.")
    parser.add_argument("--device", default="cuda", help="cuda, cpu, or a CUDA device such as cuda:0.")
    parser.add_argument("--pretrained", action="store_true", default=True, help="Use ImageNet pretrained backbone for fast convergence & high accuracy.")
    parser.add_argument("--save-dir", type=Path, default=Path("models"))
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.frames < 1 or args.batch_size < 1 or args.epochs < 1 or args.workers < 0:
        raise ValueError("epochs, batch-size and frames must be positive; workers must be zero or positive.")
    set_seed(args.seed)
    dataset = load_dataset(args.dataset_dir)
    summary = dataset.get_summary()
    logger.info("Dataset: %d clips, %d frames, %d categories.", summary["total_clips"], summary["total_frames"], len(summary["categories"]))
    if args.mode == "torch" and HAS_TORCH:
        trainer = TorchTrainer(dataset, args.frames, args.image_size, args.batch_size, args.workers, args.lr, args.pretrained, args.device)
        model = trainer.fit(args.epochs, args.save_dir)
    else:
        if args.mode == "torch" and not HAS_TORCH:
            logger.warning("PyTorch initialization bypassed due to system policy/environment (%s).", TORCH_ERROR)
            logger.info("Engaging Native High-Speed OpenCV/NumPy Spatio-Temporal Baseline Engine...")
        model = HandCraftedBaseline(args.frames, max(1, args.workers), args.seed).fit(dataset, args.epochs, args.batch_size, args.lr, args.save_dir)
    logger.info("Training complete. Best model: %s", model.resolve())


if __name__ == "__main__":
    main()

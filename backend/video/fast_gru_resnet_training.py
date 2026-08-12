import os
import sys
import time
import json
import logging
from typing import Dict, Tuple, List, Optional, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, accuracy_score
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
try:
    from backend.video.loading_processing import LABEL_MAP, REVERSE_LABEL_MAP
except ImportError:
    from loading_processing import LABEL_MAP, REVERSE_LABEL_MAP

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")
logger = logging.getLogger("Fast_GRU_ResNet_Training")


class ResNetResidualBlock(nn.Module):
    """
    ResNet-style Residual Block applied after GRU temporal processing
    to refine features before classification.
    """
    def __init__(self, dim: int = 256, dropout_rate: float = 0.3):
        super(ResNetResidualBlock, self).__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out


class FastGRUResNetDetector(nn.Module):
    """
    Incident Detector that first stores/encodes temporal sequence data using a GRU network,
    followed by a ResNet residual block for feature refining and final classification.
    Enhanced with robust LayerNorm, multi-stage Dropout, and temporal pooling to prevent overfitting.
    """
    def __init__(
        self,
        feature_dim: int = 2048,
        hidden_dim: int = 256,
        num_classes: int = 7,
        bidirectional: bool = True,
        dropout_rate: float = 0.4
    ):
        super(FastGRUResNetDetector, self).__init__()
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        gru_out_dim = hidden_dim * 2 if bidirectional else hidden_dim

        # 0. Layer Normalization & Feature Dropout to scale pre-extracted features
        self.input_norm = nn.LayerNorm(feature_dim)
        self.input_dropout = nn.Dropout(p=0.3)

        # 1. GRU Temporal Network to store and process temporal video frame features
        self.temporal_net = nn.GRU(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=0.4
        )

        # 2. Linear projection adapter if bidirectional GRU output is doubled
        self.adapter = nn.Linear(gru_out_dim, hidden_dim) if gru_out_dim != hidden_dim else nn.Identity()

        # 3. ResNet Residual Block applied after training GRU temporal representations
        self.resnet_head = ResNetResidualBlock(dim=hidden_dim, dropout_rate=dropout_rate)

        # 4. Dropout before final classification
        self.head_dropout = nn.Dropout(p=dropout_rate)

        # 5. Final Incident Classification Layer
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: (Batch, Time, Feature_Dim) = (B, 16, 2048)
        normed_x = self.input_norm(x)
        normed_x = self.input_dropout(normed_x)

        gru_out, _ = self.temporal_net(normed_x)  # (B, T, gru_out_dim)

        # Combine last step, max temporal pool, and mean temporal pool for rich context
        last_step = gru_out[:, -1, :]
        max_step, _ = torch.max(gru_out, dim=1)
        avg_step = torch.mean(gru_out, dim=1)
        temporal_representation = (last_step + max_step + avg_step) / 3.0

        # Adapter projection
        adapted = self.adapter(temporal_representation)

        # Refine temporal features through ResNet Residual Block
        refined_features = self.resnet_head(adapted)
        refined_features = self.head_dropout(refined_features)

        # Final class logits
        logits = self.classifier(refined_features)
        return logits


# Alias for backward compatibility
FastIncidentDetector = FastGRUResNetDetector


class FocalLoss(nn.Module):
    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 1.5, label_smoothing: float = 0.05):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction="none", weight=self.alpha, label_smoothing=self.label_smoothing)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


class ResNetFeatureDataset(Dataset):
    def __init__(self, clip_indices: List[Dict[str, Any]], clip_len: int = 16, is_train: bool = False):
        self.clip_indices = clip_indices
        self.clip_len = clip_len
        self.is_train = is_train

    def __len__(self) -> int:
        return len(self.clip_indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        clip_info = self.clip_indices[idx]
        fpath = clip_info["feature_path"]
        start_frame = clip_info["start_frame"]
        end_frame = clip_info["end_frame"]

        if not (os.path.exists(fpath) and os.path.getsize(fpath) > 0):
            clip_feats = np.zeros((self.clip_len, 2048), dtype=np.float32)
        else:
            try:
                if fpath.endswith('.pt'):
                    feats = torch.load(fpath, map_location="cpu").float().numpy()
                    clip_feats = feats[start_frame:end_frame]
                else:
                    mmap_feats = np.load(fpath, mmap_mode="r")
                    clip_feats = np.array(mmap_feats[start_frame:end_frame], dtype=np.float32)
            except Exception as e:
                logger.warning(f"Error loading slice from {fpath}: {e}")
                clip_feats = np.zeros((self.clip_len, 2048), dtype=np.float32)

        if len(clip_feats) == 0:
            clip_feats = np.zeros((self.clip_len, 2048), dtype=np.float32)

        while len(clip_feats) < self.clip_len:
            pad = clip_feats[-1:] if len(clip_feats) > 0 else np.zeros((1, 2048), dtype=np.float32)
            clip_feats = np.concatenate([clip_feats, pad], axis=0)

        clip_feats = clip_feats[:self.clip_len]

        # Feature-level Data Augmentation during Training to prevent Overfitting
        if self.is_train:
            # 1. Additive Gaussian Feature Noise
            noise = np.random.normal(0.0, 0.02, size=clip_feats.shape).astype(np.float32)
            clip_feats = clip_feats + noise

            # 2. Random Temporal Frame Masking (zero out random 1 frame out of 16)
            if np.random.rand() < 0.3:
                mask_idx = np.random.randint(0, self.clip_len)
                clip_feats[mask_idx] = 0.0

            # 3. Channel Feature Dropout (zero out random 10% channels)
            if np.random.rand() < 0.3:
                chan_mask = (np.random.rand(2048) > 0.10).astype(np.float32)
                clip_feats = clip_feats * chan_mask

        feat_tensor = torch.from_numpy(clip_feats)
        label_tensor = torch.tensor(clip_info["label_id"], dtype=torch.long)

        return feat_tensor, label_tensor


def build_feature_clip_index(
    verified_videos: List[Dict[str, Any]],
    clip_len: int = 16,
    overlap_ratio: float = 0.50
) -> List[Dict[str, Any]]:
    stride = max(1, int(clip_len * (1.0 - overlap_ratio)))
    clip_index = []

    for video in verified_videos:
        total_frames = video["total_frames"]
        fpath = video["feature_path"]
        lbl_id = video["label_id"]
        split = video["split"]

        if total_frames < clip_len:
            clip_index.append({
                "feature_path": fpath,
                "start_frame": 0,
                "end_frame": total_frames,
                "label_id": lbl_id,
                "split": split
            })
        else:
            for start in range(0, total_frames - clip_len + 1, stride):
                end = start + clip_len
                clip_index.append({
                    "feature_path": fpath,
                    "start_frame": start,
                    "end_frame": end,
                    "label_id": lbl_id,
                    "split": split
                })

    return clip_index


def load_stored_features_index(
    dataset_dir: str = "XD-voilence_dataset",
    features_dir: str = "XD-voilence_dataset/features"
) -> List[Dict[str, Any]]:
    """
    Loads features ONLY for files actually stored in features_dir.
    Does not read from or write to external JSON cache files for training.
    """
    logger.info(f"Scanning stored feature files directly in '{features_dir}'...")

    if not os.path.exists(features_dir):
        raise RuntimeError(f"Features directory '{features_dir}' not found.")

    # Read dataset_cache.json if available solely to map stored hashes to labels/splits
    cache_path = os.path.join(dataset_dir, "dataset_cache.json")
    meta_lookup = {}
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r") as f:
                data = json.load(f)
                verified = data.get("verified_videos", [])
                for v in verified:
                    if "hash" in v:
                        meta_lookup[v["hash"]] = v
        except Exception as e:
            logger.warning(f"Could not load metadata from {cache_path}: {e}")

    feature_files = [f for f in os.listdir(features_dir) if f.endswith(".npy") or f.endswith(".pt")]
    logger.info(f"Directly loaded {len(feature_files)} stored feature files from '{features_dir}' for training.")

    valid_entries = []
    for fname in feature_files:
        fpath = os.path.join(features_dir, fname)
        fhash = os.path.splitext(fname)[0]

        if fhash in meta_lookup:
            video_info = dict(meta_lookup[fhash])
            video_info["feature_path"] = fpath
            valid_entries.append(video_info)
        else:
            try:
                if fname.endswith(".npy"):
                    arr = np.load(fpath, mmap_mode="r")
                    n_frames = arr.shape[0]
                else:
                    arr = torch.load(fpath, map_location="cpu")
                    n_frames = arr.shape[0]
                valid_entries.append({
                    "feature_path": fpath,
                    "total_frames": n_frames,
                    "label_id": 0,
                    "split": "train"
                })
            except Exception as e:
                logger.warning(f"Skipping unreadable feature file {fpath}: {e}")

    if len(valid_entries) == 0:
        raise RuntimeError(f"No valid stored feature files found in '{features_dir}'.")

    return valid_entries


def print_dataset_distribution(
    verified_videos: List[Dict[str, Any]],
    train_clips: List[Dict[str, Any]],
    val_clips: List[Dict[str, Any]],
    test_clips: List[Dict[str, Any]]
) -> None:
    """
    Calculates and prints a detailed summary table of video and clip distributions
    across all 7 classes before starting training.
    """
    vid_counts = {lbl_id: {"train": 0, "val": 0, "test": 0} for lbl_id in range(len(LABEL_MAP))}
    for v in verified_videos:
        lbl_id = v.get("label_id", 0)
        split = v.get("split", "train")
        if lbl_id in vid_counts and split in vid_counts[lbl_id]:
            vid_counts[lbl_id][split] += 1

    clip_counts = {lbl_id: {"train": 0, "val": 0, "test": 0} for lbl_id in range(len(LABEL_MAP))}
    for c in train_clips:
        lbl_id = c.get("label_id", 0)
        clip_counts[lbl_id]["train"] += 1
    for c in val_clips:
        lbl_id = c.get("label_id", 0)
        clip_counts[lbl_id]["val"] += 1
    for c in test_clips:
        lbl_id = c.get("label_id", 0)
        clip_counts[lbl_id]["test"] += 1

    print("\n" + "=" * 105)
    print(f"{'DATASET CLASS, VIDEO & CLIP DISTRIBUTION SUMMARY (ALL 7 CLASSES)':^105}")
    print("=" * 105)
    header = f"{'ID':<3} | {'Class Name':<13} | {'Train Vid':<9} | {'Val Vid':<7} | {'Test Vid':<8} | {'Total Vid':<9} | {'Train Clips':<11} | {'Val Clips':<9} | {'Test Clips':<10}"
    print(header)
    print("-" * 105)

    tot_tr_v, tot_val_v, tot_ts_v, tot_all_v = 0, 0, 0, 0
    tot_tr_c, tot_val_c, tot_ts_c = 0, 0, 0

    for lbl_id in range(len(LABEL_MAP)):
        c_name = REVERSE_LABEL_MAP.get(lbl_id, f"Class_{lbl_id}")
        v_tr = vid_counts[lbl_id]["train"]
        v_val = vid_counts[lbl_id]["val"]
        v_ts = vid_counts[lbl_id]["test"]
        v_tot = v_tr + v_val + v_ts

        c_tr = clip_counts[lbl_id]["train"]
        c_val = clip_counts[lbl_id]["val"]
        c_ts = clip_counts[lbl_id]["test"]

        tot_tr_v += v_tr
        tot_val_v += v_val
        tot_ts_v += v_ts
        tot_all_v += v_tot

        tot_tr_c += c_tr
        tot_val_c += c_val
        tot_ts_c += c_ts

        print(f"{lbl_id:<3} | {c_name:<13} | {v_tr:<9,d} | {v_val:<7,d} | {v_ts:<8,d} | {v_tot:<9,d} | {c_tr:<11,d} | {c_val:<9,d} | {c_ts:<10,d}")

    print("-" * 105)
    print(f"{'TOT':<3} | {'ALL CLASSES':<13} | {tot_tr_v:<9,d} | {tot_val_v:<7,d} | {tot_ts_v:<8,d} | {tot_all_v:<9,d} | {tot_tr_c:<11,d} | {tot_val_c:<9,d} | {tot_ts_c:<10,d}")
    print("=" * 105 + "\n")


def create_feature_dataloaders(
    dataset_dir: str = "XD-voilence_dataset",
    features_dir: str = "XD-voilence_dataset/features",
    batch_size: int = 64,
    num_workers: int = 4,
    samples_per_epoch: int = 100000
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, Any]]:
    verified_videos = load_stored_features_index(dataset_dir=dataset_dir, features_dir=features_dir)

    all_clips = build_feature_clip_index(verified_videos, clip_len=16)

    train_clips = [c for c in all_clips if c["split"] == "train"]
    val_clips = [c for c in all_clips if c["split"] == "val"]
    test_clips = [c for c in all_clips if c["split"] == "test"]

    # Print video & clip distribution across all 7 classes before training
    print_dataset_distribution(verified_videos, train_clips, val_clips, test_clips)

    train_dataset = ResNetFeatureDataset(train_clips, clip_len=16, is_train=True)
    val_dataset = ResNetFeatureDataset(val_clips, clip_len=16, is_train=False)
    test_dataset = ResNetFeatureDataset(test_clips, clip_len=16, is_train=False)

    # Class-balanced sampling with smoothed weights (square root inverse frequency)
    labels = [c["label_id"] for c in train_clips]
    class_counts = np.bincount(labels, minlength=len(LABEL_MAP))
    class_weights = 1.0 / np.sqrt(np.maximum(class_counts, 1))
    sample_weights = [class_weights[lbl] for lbl in labels]

    # Capped epoch budget to prevent hyper-oversampling loops
    num_samples = min(len(train_clips), samples_per_epoch) if samples_per_epoch > 0 else len(train_clips)
    sampler = WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.double),
        num_samples=num_samples,
        replacement=True
    )

    loader_kwargs = {"batch_size": batch_size, "num_workers": num_workers, "pin_memory": torch.cuda.is_available()}
    val_kwargs = {"batch_size": batch_size * 2, "num_workers": num_workers, "pin_memory": torch.cuda.is_available()}

    train_loader = DataLoader(train_dataset, sampler=sampler, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **val_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **val_kwargs)

    stats = {
        "train_clips": len(train_clips),
        "val_clips": len(val_clips),
        "test_clips": len(test_clips),
        "epoch_samples": num_samples
    }
    return train_loader, val_loader, test_loader, stats


def train_fast_gru_resnet_model(
    dataset_dir: str = "XD-voilence_dataset",
    output_model_path: str = "best_fast_gru_resnet_model.pth",
    epochs: int = 30,
    batch_size: int = 64,
    lr: float = 3e-4,
    weight_decay: float = 1e-2,
    samples_per_epoch: int = 100000
) -> Dict[str, Any]:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using Device: {DEVICE}")

    train_loader, val_loader, test_loader, stats = create_feature_dataloaders(
        dataset_dir=dataset_dir,
        batch_size=batch_size,
        samples_per_epoch=samples_per_epoch
    )
    logger.info(f"Loaded Feature DataLoaders: Train Clips={stats['train_clips']} (Sampled {stats['epoch_samples']}/epoch), Val Clips={stats['val_clips']}, Test Clips={stats['test_clips']}")

    model = FastGRUResNetDetector(
        feature_dim=2048,
        hidden_dim=256,
        num_classes=len(LABEL_MAP),
        dropout_rate=0.4
    ).to(DEVICE)

    criterion = FocalLoss(alpha=None, gamma=1.5, label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_macro_f1 = 0.0
    patience = 8
    patience_counter = 0

    print("\n" + "=" * 80)
    print("STARTING FAST GRU + RESNET FEATURE TRAINING")
    print("=" * 80 + "\n")

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        model.train()
        running_loss = 0.0

        train_pbar = tqdm(train_loader, desc=f"Epoch [{epoch:02d}/{epochs:02d}] Training", unit="batch", leave=False, dynamic_ncols=True)
        for inputs, targets in train_pbar:
            inputs, targets = inputs.to(DEVICE, non_blocking=True), targets.to(DEVICE, non_blocking=True)

            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, targets)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            batch_loss = loss.item()
            running_loss += batch_loss * inputs.size(0)
            train_pbar.set_postfix(loss=f"{batch_loss:.4f}")

        scheduler.step()
        train_loss = running_loss / len(train_loader.dataset) if len(train_loader.dataset) > 0 else 0.0

        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []

        val_pbar = tqdm(val_loader, desc=f"Epoch [{epoch:02d}/{epochs:02d}] Validation", unit="batch", leave=False, dynamic_ncols=True)
        with torch.no_grad():
            for inputs, targets in val_pbar:
                inputs, targets = inputs.to(DEVICE, non_blocking=True), targets.to(DEVICE, non_blocking=True)
                logits = model(inputs)
                loss = criterion(logits, targets)

                val_loss += loss.item() * inputs.size(0)
                preds = torch.argmax(logits, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(targets.cpu().numpy())

        val_loss = val_loss / len(val_loader.dataset) if len(val_loader.dataset) > 0 else 0.0
        acc = accuracy_score(all_labels, all_preds)
        prec, rec, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="macro", zero_division=0)
        elapsed = time.time() - start_time

        print(f"Epoch [{epoch:02d}/{epochs:02d}] ({elapsed:.2f}s) | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {acc*100:.2f}% | Val Macro F1: {f1:.4f}")

        if f1 > best_macro_f1:
            best_macro_f1 = f1
            patience_counter = 0
            torch.save(model.state_dict(), output_model_path)
            print(f"  [SUCCESS] Improved Val Macro F1 ({best_macro_f1:.4f}). Saved model to '{output_model_path}'.")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("[WARNING] Early Stopping Triggered!")
                break

    print("\n[INFO] Training Complete.")
    print(f"[INFO] Best Validation Macro F1 Score: {best_macro_f1:.4f}")
    return {"best_macro_f1": best_macro_f1, "model_path": output_model_path}


# Alias for backward compatibility
train_fast_resnet_model = train_fast_gru_resnet_model


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fast GRU + ResNet Feature Training")
    parser.add_argument("--dataset_dir", type=str, default="XD-voilence_dataset", help="Dataset path")
    parser.add_argument("--epochs", type=int, default=30, help="Epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--samples_per_epoch", type=int, default=100000, help="Samples per epoch")
    args = parser.parse_args()

    train_fast_gru_resnet_model(
        dataset_dir=args.dataset_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        samples_per_epoch=args.samples_per_epoch
    )

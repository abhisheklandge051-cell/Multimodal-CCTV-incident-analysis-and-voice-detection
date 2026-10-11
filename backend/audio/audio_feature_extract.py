import os
import sys

# Suppress TensorFlow C++ logs and oneDNN warnings before importing tf
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np

# Try importing soundfile and librosa
try:
    import soundfile as sf
except ImportError:
    sf = None

try:
    import librosa
except ImportError:
    librosa = None

try:
    import scipy.signal
except ImportError:
    scipy = None

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        return iterable

# Configure logging
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")
logger = logging.getLogger("YAMNet_Feature_Extractor")

# 7 XD-Violence classes (consistent with audio & video pipelines)
CLASSES = [
    "Abuse",
    "CarAccident",
    "Explosion",
    "Fighting",
    "Normal",
    "Riot",
    "Shooting"
]

LABEL_MAP = {cls_name: idx for idx, cls_name in enumerate(CLASSES)}
REVERSE_LABEL_MAP = {idx: cls_name for idx, cls_name in enumerate(CLASSES)}

# YAMNet specifications
TARGET_SAMPLE_RATE = 16000       # YAMNet requires exactly 16 kHz audio
YAMNET_WINDOW_SAMPLES = 15360    # 0.96 seconds @ 16 kHz
YAMNET_HOP_SAMPLES = 7680        # 0.48 seconds @ 16 kHz
DEFAULT_EMBEDDING_DIM = 1024     # YAMNet per-frame embedding size
DEFAULT_AUDIOSET_CLASSES = 521   # AudioSet class predictions
DEFAULT_SPECTROGRAM_BINS = 64    # YAMNet log-mel spectrogram bins

DEFAULT_INPUT_SPLIT_DIR = Path("xd_violence_audio_split")
DEFAULT_INPUT_RAW_DIR = Path("xd_violence_audio")
DEFAULT_OUTPUT_DIR = Path("xd_violence_audio_features")
DEFAULT_MANIFEST_PATH = Path("xd_violence_audio_split/dataset_split_manifest.json")
YAMNET_TFHUB_URL = "https://tfhub.dev/google/yamnet/1"

 
class YAMNetFeatureExtractor:
    """
    YAMNet Audio Feature Extractor for incident analysis and classification.
    Loads the pretrained Google YAMNet model from TensorFlow Hub and extracts:
      - 'embeddings': Frame-level 1024-dimensional feature embeddings [T, 1024] (default)
      - 'pooled': Global temporal mean-pooled 1024-dimensional vector [1024]
      - 'mean_max_pooled': Concatenated mean + max pooled 2048-dimensional vector [2048]
      - 'spectrogram': Log-mel spectrogram representation [M, 64]
      - 'scores': AudioSet 521-class prediction scores [T, 521]
      - 'all': Dictionary containing all representations
    """

    def __init__(
        self,
        model_url: str = YAMNET_TFHUB_URL,
        feature_type: str = "embeddings",
        pooling: Optional[str] = None,
        normalize: bool = True,
        min_duration_sec: float = 0.96
    ):
        self.model_url = model_url
        self.feature_type = feature_type.lower()
        self.pooling = pooling.lower() if pooling else None
        self.normalize = normalize
        self.min_duration_samples = int(min_duration_sec * TARGET_SAMPLE_RATE)
        self.model = None

        self._load_model()

    def _load_model(self):
        """Loads YAMNet model from TensorFlow Hub with caching."""
        logger.info(f"Loading pretrained YAMNet model from: {self.model_url}")
        try:
            import tensorflow as tf
            import tensorflow_hub as hub

            # Check for GPU devices
            gpus = tf.config.list_physical_devices("GPU")
            if gpus:
                try:
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    logger.info(f"TensorFlow GPU acceleration enabled ({len(gpus)} GPU(s)).")
                except Exception as e:
                    logger.warning(f"Could not set GPU memory growth: {e}")
            else:
                logger.info("TensorFlow running on CPU mode.")

            self.model = hub.load(self.model_url)
            logger.info("YAMNet model loaded successfully.")

        except Exception as e:
            logger.error(f"Failed to load YAMNet model: {e}")
            raise RuntimeError(
                f"Could not load YAMNet from '{self.model_url}'. "
                f"Please ensure 'tensorflow' and 'tensorflow_hub' are installed. Error: {e}"
            )

    def load_and_preprocess_audio(self, audio_path: Union[str, Path]) -> Tuple[np.ndarray, float]:
        """
        Loads an audio file, converts to mono, resamples to 16 kHz if necessary,
        normalizes peak amplitude to [-1.0, 1.0], and zero-pads short clips to >= 0.96s.
        Returns:
            waveform (np.ndarray): 1D float32 array in range [-1.0, 1.0]
            duration_sec (float): Duration in seconds before padding
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        audio = None
        sr = TARGET_SAMPLE_RATE

        # Read audio using soundfile or librosa
        if sf is not None:
            try:
                audio, sr = sf.read(str(audio_path))
            except Exception:
                audio = None

        if audio is None and librosa is not None:
            audio, sr = librosa.load(str(audio_path), sr=None, mono=False)

        if audio is None:
            raise RuntimeError(f"Unable to read audio file with soundfile or librosa: {audio_path}")

        # Convert to mono if multi-channel
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1 if audio.shape[0] > audio.shape[1] else 0)

        # Ensure float32
        if audio.dtype != np.float32:
            if np.issubdtype(audio.dtype, np.integer):
                max_int = np.iinfo(audio.dtype).max
                audio = (audio / max_int).astype(np.float32)
            else:
                audio = audio.astype(np.float32)

        # Resample to 16000 Hz if needed
        if sr != TARGET_SAMPLE_RATE:
            if librosa is not None:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SAMPLE_RATE)
            elif scipy is not None:
                gcd = np.gcd(sr, TARGET_SAMPLE_RATE)
                up = TARGET_SAMPLE_RATE // gcd
                down = sr // gcd
                audio = scipy.signal.resample_poly(audio, up, down).astype(np.float32)
            else:
                logger.warning(f"Audio {audio_path.name} sr={sr} != {TARGET_SAMPLE_RATE}, librosa missing for resample.")

        duration_sec = float(len(audio)) / TARGET_SAMPLE_RATE

        # Peak normalization
        if self.normalize:
            max_val = np.max(np.abs(audio)) if len(audio) > 0 else 0.0
            if max_val > 0.0:
                audio = audio / max_val

        # Pad with zeros if shorter than minimum YAMNet window (15360 samples = 0.96s)
        if len(audio) < self.min_duration_samples:
            pad_len = self.min_duration_samples - len(audio)
            audio = np.pad(audio, (0, pad_len), mode="constant")

        return audio, duration_sec

    def extract_features(
        self,
        audio_input: Union[str, Path, np.ndarray],
        feature_type: Optional[str] = None,
        pooling: Optional[str] = None
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """
        Extracts YAMNet features for given audio file path or 1D waveform array.
        Returns:
            np.ndarray or dict depending on feature_type and pooling.
        """
        feat_type = (feature_type or self.feature_type).lower()
        pool_mode = (pooling or self.pooling)

        if isinstance(audio_input, (str, Path)):
            waveform, _ = self.load_and_preprocess_audio(audio_input)
        else:
            waveform = audio_input.astype(np.float32)
            if len(waveform) < self.min_duration_samples:
                pad_len = self.min_duration_samples - len(waveform)
                waveform = np.pad(waveform, (0, pad_len), mode="constant")

        # Run YAMNet inference
        scores, embeddings, spectrogram = self.model(waveform)

        # Convert tensors to numpy arrays
        emb_np = embeddings.numpy().astype(np.float32)       # [T, 1024]
        scores_np = scores.numpy().astype(np.float32)         # [T, 521]
        spec_np = spectrogram.numpy().astype(np.float32)     # [M, 64]

        # Apply temporal pooling if requested
        def apply_pool(feats: np.ndarray, mode: str) -> np.ndarray:
            if mode == "mean":
                return np.mean(feats, axis=0) if len(feats) > 0 else np.zeros((feats.shape[-1],), dtype=np.float32)
            elif mode == "max":
                return np.max(feats, axis=0) if len(feats) > 0 else np.zeros((feats.shape[-1],), dtype=np.float32)
            elif mode in ("mean_max", "mean+max"):
                if len(feats) == 0:
                    return np.zeros((feats.shape[-1] * 2,), dtype=np.float32)
                return np.concatenate([np.mean(feats, axis=0), np.max(feats, axis=0)], axis=0)
            return feats

        if pool_mode and feat_type in ("embeddings", "scores"):
            emb_np = apply_pool(emb_np, pool_mode)
            scores_np = apply_pool(scores_np, pool_mode)

        if feat_type == "embeddings":
            return emb_np
        elif feat_type == "pooled":
            return np.mean(emb_np, axis=0) if emb_np.ndim > 1 else emb_np
        elif feat_type == "mean_max_pooled":
            if emb_np.ndim > 1:
                return np.concatenate([np.mean(emb_np, axis=0), np.max(emb_np, axis=0)], axis=0)
            return np.concatenate([emb_np, emb_np], axis=0)
        elif feat_type == "spectrogram":
            return spec_np
        elif feat_type == "scores":
            return scores_np
        elif feat_type == "all":
            pooled_mean = np.mean(emb_np, axis=0) if emb_np.ndim > 1 else emb_np
            return {
                "embeddings": emb_np,
                "pooled": pooled_mean,
                "spectrogram": spec_np,
                "scores": scores_np
            }
        else:
            raise ValueError(
                f"Unknown feature_type: '{feat_type}'. "
                f"Supported types: 'embeddings', 'pooled', 'mean_max_pooled', 'spectrogram', 'scores', 'all'."
            )


# PyTorch Dataset Loader for training models on extracted .npy features
try:
    import torch
    from torch.utils.data import Dataset

    class AudioFeatureDataset(Dataset):
        """
        PyTorch Dataset for loading YAMNet .npy features and labels.
        Supports sequence truncation/padding and attention masks for temporal models (GRU, LSTM, Transformer).
        """
        def __init__(
            self,
            manifest_path: Union[str, Path],
            split: Optional[str] = "train",
            max_seq_len: int = 64,
            return_mask: bool = True
        ):
            self.manifest_path = Path(manifest_path)
            self.split = split
            self.max_seq_len = max_seq_len
            self.return_mask = return_mask
            self.samples = []

            with open(self.manifest_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if isinstance(data, list):
                if split:
                    self.samples = [d for d in data if d.get("split") == split]
                else:
                    self.samples = data
            elif isinstance(data, dict) and "records" in data:
                all_records = data["records"]
                if split:
                    self.samples = [d for d in all_records if d.get("split") == split]
                else:
                    self.samples = all_records

            logger.info(f"AudioFeatureDataset initialized with {len(self.samples)} samples (split='{split}').")

        def __len__(self) -> int:
            return len(self.samples)

        def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
            item = self.samples[idx]
            feat_path = item["feature_path"]
            label_id = item["label_id"]

            features = np.load(feat_path)  # shape: [T, 1024] or [1024]

            # If 1D pooled vector
            if features.ndim == 1:
                feat_tensor = torch.from_numpy(features).float()
                label_tensor = torch.tensor(label_id, dtype=torch.long)
                return feat_tensor, label_tensor

            # If 2D sequence [T, feature_dim]
            seq_len, feat_dim = features.shape
            mask = np.zeros(self.max_seq_len, dtype=np.float32)

            if seq_len >= self.max_seq_len:
                # Truncate
                padded_feats = features[:self.max_seq_len]
                mask[:] = 1.0
            else:
                # Zero-pad
                padded_feats = np.zeros((self.max_seq_len, feat_dim), dtype=np.float32)
                padded_feats[:seq_len] = features
                mask[:seq_len] = 1.0

            feat_tensor = torch.from_numpy(padded_feats).float()
            label_tensor = torch.tensor(label_id, dtype=torch.long)
            mask_tensor = torch.from_numpy(mask).float()

            if self.return_mask:
                return feat_tensor, label_tensor, mask_tensor
            return feat_tensor, label_tensor

except ImportError:
    AudioFeatureDataset = None


def resolve_split_or_raw_files(
    input_dir: Optional[Union[str, Path]] = None,
    manifest_path: Optional[Union[str, Path]] = None,
    split_filter: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Discovers all audio files to process from:
    1) Existing split manifest JSON if available
    2) Or split folder structure (xd_violence_audio_split/train/<class>/...)
    3) Or class folder structure (xd_violence_audio/<class>/...)
    """
    file_records = []

    # 1. Try reading manifest file
    man_path = Path(manifest_path) if manifest_path else DEFAULT_MANIFEST_PATH
    if man_path.exists():
        logger.info(f"Loading split records from manifest: {man_path}")
        with open(man_path, "r", encoding="utf-8") as f:
            records = json.load(f)
        for r in records:
            s = r.get("split", "train")
            if split_filter and split_filter.lower() != "all" and s != split_filter.lower():
                continue
            src = Path(r.get("dst_path", r.get("src_path", "")))
            if not src.exists():
                src = Path(r.get("src_path", ""))
            if src.exists():
                file_records.append({
                    "audio_path": src,
                    "filename": src.name,
                    "class_name": r.get("class_name"),
                    "label_id": r.get("label_id", LABEL_MAP.get(r.get("class_name"), -1)),
                    "split": s
                })
        if file_records:
            return file_records

    # 2. Try split directory
    inp_dir = Path(input_dir) if input_dir else (DEFAULT_INPUT_SPLIT_DIR if DEFAULT_INPUT_SPLIT_DIR.exists() else DEFAULT_INPUT_RAW_DIR)
    inp_dir = inp_dir.resolve()

    if not inp_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {inp_dir}")

    # Check if directory has splits (train, val, test)
    split_dirs = [s for s in ["train", "val", "test"] if (inp_dir / s).is_dir()]
    if split_dirs:
        for s in split_dirs:
            if split_filter and split_filter.lower() != "all" and s != split_filter.lower():
                continue
            for cls_name in CLASSES:
                cls_dir = inp_dir / s / cls_name
                if cls_dir.is_dir():
                    for wav_file in sorted(cls_dir.glob("*.wav")):
                        file_records.append({
                            "audio_path": wav_file,
                            "filename": wav_file.name,
                            "class_name": cls_name,
                            "label_id": LABEL_MAP[cls_name],
                            "split": s
                        })
    else:
        # Check if direct class directories
        for cls_name in CLASSES:
            cls_dir = inp_dir / cls_name
            if cls_dir.is_dir():
                for wav_file in sorted(cls_dir.glob("*.wav")):
                    file_records.append({
                        "audio_path": wav_file,
                        "filename": wav_file.name,
                        "class_name": cls_name,
                        "label_id": LABEL_MAP[cls_name],
                        "split": "train"
                    })

    return file_records


def run_audio_feature_extraction(
    input_dir: Optional[Union[str, Path]] = None,
    output_dir: Union[str, Path] = DEFAULT_OUTPUT_DIR,
    manifest_path: Optional[Union[str, Path]] = None,
    feature_type: str = "embeddings",
    pooling: Optional[str] = None,
    split_filter: Optional[str] = None,
    overwrite: bool = False,
    max_samples: Optional[int] = None,
    normalize: bool = True
) -> str:
    """
    Main execution pipeline:
    1) Loads YAMNet feature extractor
    2) Discovers audio files across train, val, and test splits
    3) Extracts features and saves as .npy files matching the dataset structure
    4) Generates a comprehensive audio_feature_manifest.json with all sample metadata
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    extractor = YAMNetFeatureExtractor(
        feature_type=feature_type,
        pooling=pooling,
        normalize=normalize
    )

    records = resolve_split_or_raw_files(
        input_dir=input_dir,
        manifest_path=manifest_path,
        split_filter=split_filter
    )

    if not records:
        logger.error(f"No WAV files found in input directories.")
        return ""

    if max_samples and max_samples > 0:
        records = records[:max_samples]
        logger.info(f"Limiting processing to first {max_samples} samples.")

    total_files = len(records)
    logger.info(f"Starting YAMNet feature extraction ({feature_type}) for {total_files} files into '{output_dir}'...")

    manifest_output = []
    extracted_count = 0
    skipped_count = 0
    error_count = 0

    pbar = tqdm(records, desc=f"Extracting YAMNet features", unit="file", dynamic_ncols=True)

    start_time = time.time()
    for rec in pbar:
        audio_path = rec["audio_path"]
        class_name = rec["class_name"]
        split_name = rec["split"]
        filename = rec["filename"]

        # Target output file path: output_dir / <split> / <class_name> / <stem>.npy
        target_subfolder = output_dir / split_name / class_name
        target_subfolder.mkdir(parents=True, exist_ok=True)
        out_filepath = target_subfolder / f"{audio_path.stem}.npy"

        rec_entry = {
            "filename": filename,
            "audio_path": str(audio_path.resolve()),
            "feature_path": str(out_filepath.resolve()),
            "class_name": class_name,
            "label_id": rec["label_id"],
            "split": split_name
        }

        # Check if already extracted
        if out_filepath.exists() and out_filepath.stat().st_size > 0 and not overwrite:
            skipped_count += 1
            try:
                data = np.load(out_filepath, mmap_mode="r")
                rec_entry["num_frames"] = int(data.shape[0]) if data.ndim > 1 else 1
                rec_entry["feature_dim"] = int(data.shape[-1])
            except Exception:
                rec_entry["num_frames"] = 0
                rec_entry["feature_dim"] = DEFAULT_EMBEDDING_DIM
            manifest_output.append(rec_entry)
            pbar.set_postfix(done=extracted_count, skipped=skipped_count, err=error_count)
            continue

        try:
            feats = extractor.extract_features(audio_path)
            np.save(out_filepath, feats)

            rec_entry["num_frames"] = int(feats.shape[0]) if feats.ndim > 1 else 1
            rec_entry["feature_dim"] = int(feats.shape[-1])
            manifest_output.append(rec_entry)
            extracted_count += 1

        except Exception as e:
            logger.warning(f"Error extracting features for {audio_path.name}: {e}")
            error_count += 1

        pbar.set_postfix(done=extracted_count, skipped=skipped_count, err=error_count)

    elapsed = time.time() - start_time
    rate = total_files / elapsed if elapsed > 0 else 0.0

    # Save central manifest
    manifest_file = output_dir / "audio_feature_manifest.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest_output, f, indent=2)

    # Save split-specific manifests for easy loading
    for s in ["train", "val", "test"]:
        split_records = [r for r in manifest_output if r.get("split") == s]
        if split_records:
            split_manifest_file = output_dir / f"audio_feature_manifest_{s}.json"
            with open(split_manifest_file, "w", encoding="utf-8") as f:
                json.dump(split_records, f, indent=2)

    logger.info(f"\n=======================================================")
    logger.info(f" YAMNet Feature Extraction Completed!")
    logger.info(f" Total Files:     {total_files}")
    logger.info(f" Extracted:       {extracted_count}")
    logger.info(f" Skipped:         {skipped_count}")
    logger.info(f" Errors:          {error_count}")
    logger.info(f" Time Elapsed:    {elapsed:.2f}s ({rate:.1f} files/sec)")
    logger.info(f" Manifest saved:  {manifest_file}")
    logger.info(f"=======================================================")

    print_extraction_summary(manifest_output)
    return str(manifest_file.resolve())


def print_extraction_summary(manifest_records: List[Dict[str, Any]]):
    """Prints a clean summary table of extracted feature files per split and per class."""
    if not manifest_records:
        return

    splits = sorted(list(set(r.get("split", "unknown") for r in manifest_records)))
    counts = {s: {cls: 0 for cls in CLASSES} for s in splits}

    for r in manifest_records:
        s = r.get("split", "unknown")
        c = r.get("class_name")
        if s in counts and c in counts[s]:
            counts[s][c] += 1

    print("\n" + "=" * 70)
    header = f"{'Class Name':<15} | " + " | ".join([f"{s.capitalize():<10}" for s in splits]) + " | Total"
    print(header)
    print("=" * 70)

    totals = {s: 0 for s in splits}
    for cls in CLASSES:
        row = f"{cls:<15} | "
        cls_total = 0
        for s in splits:
            cnt = counts[s][cls]
            row += f"{cnt:<10} | "
            totals[s] += cnt
            cls_total += cnt
        row += f"{cls_total:<10}"
        print(row)

    print("-" * 70)
    total_row = f"{'Total':<15} | "
    grand_total = 0
    for s in splits:
        total_row += f"{totals[s]:<10} | "
        grand_total += totals[s]
    total_row += f"{grand_total:<10}"
    print(total_row)
    print("=" * 70 + "\n")


def verify_extracted_features(output_dir: Union[str, Path] = DEFAULT_OUTPUT_DIR) -> bool:
    """Verifies that extracted .npy files exist, can be loaded, and have valid shapes."""
    output_dir = Path(output_dir)
    manifest_path = output_dir / "audio_feature_manifest.json"

    if not manifest_path.exists():
        logger.error(f"Manifest not found at: {manifest_path}")
        return False

    with open(manifest_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    logger.info(f"Verifying {len(records)} feature files from {manifest_path}...")

    verified = 0
    corrupted = 0
    sample_shapes = []

    for r in records[:50]:  # Verify first 50 as spot check
        p = Path(r["feature_path"])
        if not p.exists():
            corrupted += 1
            continue
        try:
            arr = np.load(p)
            if np.isnan(arr).any() or np.isinf(arr).any():
                corrupted += 1
            else:
                verified += 1
                if len(sample_shapes) < 3:
                    sample_shapes.append((r["filename"], arr.shape, arr.dtype))
        except Exception:
            corrupted += 1

    logger.info(f"Verification sample check: {verified} valid, {corrupted} errors.")
    for name, shape, dtype in sample_shapes:
        logger.info(f"  Sample: {name} -> Shape: {shape}, Dtype: {dtype}")

    return corrupted == 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract YAMNet Audio Features (.npy) for Model Training")
    parser.add_argument("--input_dir", type=str, default=None, help="Input directory (split or raw audio)")
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Output directory for .npy features")
    parser.add_argument("--manifest_path", type=str, default=None, help="Split manifest json path")
    parser.add_argument("--feature_type", type=str, default="embeddings",
                        choices=["embeddings", "pooled", "mean_max_pooled", "spectrogram", "scores", "all"],
                        help="Type of YAMNet features to extract (default: 'embeddings' -> [T, 1024])")
    parser.add_argument("--pooling", type=str, default=None, choices=[None, "mean", "max", "mean_max"],
                        help="Optional temporal pooling to apply across time frames")
    parser.add_argument("--split", type=str, default=None, choices=[None, "train", "val", "test", "all"],
                        help="Filter to process a specific split (default: all available splits)")
    parser.add_argument("--max_samples", type=int, default=None, help="Process only the first N samples (for quick tests)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing extracted .npy files")
    parser.add_argument("--verify", action="store_true", help="Verify existing extracted .npy features")

    args = parser.parse_args()

    if args.verify:
        verify_extracted_features(output_dir=args.output_dir)
    else:
        run_audio_feature_extraction(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            manifest_path=args.manifest_path,
            feature_type=args.feature_type,
            pooling=args.pooling,
            split_filter=args.split,
            overwrite=args.overwrite,
            max_samples=args.max_samples
        )

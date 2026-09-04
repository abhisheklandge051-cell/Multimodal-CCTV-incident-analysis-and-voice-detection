"""
YAMNet Audio Incident Classifier Training Script
=================================================
Trains a deep classification head on YAMNet audio embeddings for multimodal
incident analysis and voice detection (XD-Violence dataset).

Features:
  1. Multi-source Data Loading: Automatically discovers and loads features from
     'xd_violence_audio_features' manifests (train/val/test splits), split folder
     hierarchies, pre-aggregated X_embeddings.npy / y_labels.npy, or fast cached matrices.
  2. Temporal Feature Pooling: Supports 'mean_max' (2048-dim, concatenating sustained
     background context + impulsive peak sound bursts like gunshots/crashes),
     'mean' (1024-dim), 'max' (1024-dim), and 'stats' (3072-dim).
  3. Class Imbalance Compensation: Damped balanced class weighting prevents gradient
     destabilization from minority classes (e.g. Abuse) while retaining high sensitivity.
  4. Modern Deep Architectures: Offers Residual MLP and Multi-stage Dropout MLP with
     Layer Normalization and L2 regularization to prevent overfitting.
  5. Learning Rate Scheduling & Early Stopping: Automatically reduces LR on plateau and
     restores the best checkpoint based on validation accuracy.
  6. Comprehensive Evaluation & Reporting: Outputs detailed classification report,
     confusion matrix, and saves 'audio_evaluation_report.json' matching the video pipeline.
  7. Late Fusion Ready: Exports .keras checkpoint and includes AudioIncidentPredictor
     for seamless multimodal inference and decision fusion.
"""

import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union

# Suppress TensorFlow C++ logs and oneDNN warnings before importing tf
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support
)

# Configure logging
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")
logger = logging.getLogger("YAMNet_Training")

# 7 XD-Violence classes (consistent across audio, video, and multimodal pipelines)
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

# Project root path resolution
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_FEATURES_DIR = PROJECT_ROOT / "xd_violence_audio_features"
DEFAULT_CHECKPOINTS_DIR = PROJECT_ROOT / "backend" / "checkpoints"


def pool_audio_features(features: np.ndarray, pooling: str = "mean_max") -> np.ndarray:
    """
    Applies temporal pooling to frame-level YAMNet embeddings [T, 1024].

    Modes:
      - 'mean_max': Concatenates temporal mean and max pooling -> [2048]
                    (Best for audio incidents: captures sustained ambient sound + transient bursts)
      - 'mean':     Temporal mean pooling -> [1024]
      - 'max':      Temporal max pooling -> [1024]
      - 'stats':    Concatenates mean, max, and std -> [3072]
    """
    pooling = pooling.lower()
    if features.ndim == 1:
        # Already 1D vector
        if pooling in ("mean_max", "mean+max"):
            return np.concatenate([features, features], axis=0) if len(features) == 1024 else features
        return features

    if len(features) == 0:
        feat_dim = features.shape[-1] if features.shape[-1] > 0 else 1024
        multiplier = 2 if pooling in ("mean_max", "mean+max") else (3 if pooling == "stats" else 1)
        return np.zeros((feat_dim * multiplier,), dtype=np.float32)

    mean_pool = np.mean(features, axis=0)

    if pooling == "mean":
        return mean_pool.astype(np.float32)
    elif pooling == "max":
        return np.max(features, axis=0).astype(np.float32)
    elif pooling in ("mean_max", "mean+max"):
        max_pool = np.max(features, axis=0)
        return np.concatenate([mean_pool, max_pool], axis=0).astype(np.float32)
    elif pooling == "stats":
        max_pool = np.max(features, axis=0)
        std_pool = np.std(features, axis=0)
        return np.concatenate([mean_pool, max_pool, std_pool], axis=0).astype(np.float32)
    else:
        raise ValueError(f"Unsupported pooling mode: '{pooling}'. Choose from: 'mean_max', 'mean', 'max', 'stats'.")


def load_split_from_manifest(
    manifest_path: Path,
    pooling: str = "mean_max",
    max_samples: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads and pools audio features specified in a manifest JSON file.
    Returns:
        X: np.ndarray of shape (N, feature_dim)
        y: np.ndarray of shape (N,)
    """
    with open(manifest_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    if max_samples and max_samples > 0:
        records = records[:max_samples]

    X_list = []
    y_list = []

    for r in records:
        fpath = Path(r.get("feature_path", ""))
        if not fpath.exists():
            # Try resolving relative to project root or manifest parent
            alt_path = manifest_path.parent / fpath.name
            if alt_path.exists():
                fpath = alt_path
            else:
                continue

        try:
            arr = np.load(fpath)
            pooled = pool_audio_features(arr, pooling=pooling)
            label_id = r.get("label_id", LABEL_MAP.get(r.get("class_name"), -1))
            if label_id >= 0:
                X_list.append(pooled)
                y_list.append(label_id)
        except Exception as e:
            logger.warning(f"Error loading feature file {fpath}: {e}")

    if not X_list:
        raise RuntimeError(f"No valid features loaded from manifest: {manifest_path}")

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int32)


def load_or_cache_dataset(
    features_dir: Path,
    pooling: str = "mean_max",
    cache_dir: Optional[Path] = None,
    force_recompute: bool = False
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Loads train, validation, and test datasets with caching for instant startup.
    Returns:
        dict with keys 'train', 'val', 'test', each containing (X, y)
    """
    features_dir = Path(features_dir)
    if cache_dir is None:
        cache_dir = features_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    cache_prefix = f"cached_{pooling}"
    splits = ["train", "val", "test"]
    dataset = {}

    # Check if all cached matrices exist
    all_cached = not force_recompute and all(
        (cache_dir / f"{cache_prefix}_X_{s}.npy").exists() and
        (cache_dir / f"{cache_prefix}_y_{s}.npy").exists()
        for s in splits
    )

    if all_cached:
        logger.info(f"Loading cached pre-aggregated features from '{cache_dir}' (pooling='{pooling}')...")
        for s in splits:
            X = np.load(cache_dir / f"{cache_prefix}_X_{s}.npy")
            y = np.load(cache_dir / f"{cache_prefix}_y_{s}.npy")
            dataset[s] = (X, y)
            logger.info(f"  {s.upper()}: X={X.shape}, y={y.shape}")
        return dataset

    # Otherwise, extract from manifests or directory structure
    logger.info(f"Aggregating audio features from '{features_dir}' (pooling='{pooling}')...")
    for s in splits:
        manifest_file = features_dir / f"audio_feature_manifest_{s}.json"
        if manifest_file.exists():
            logger.info(f"Processing split '{s}' from manifest: {manifest_file.name}...")
            X_s, y_s = load_split_from_manifest(manifest_file, pooling=pooling)
        else:
            # Check split subfolder directly: features_dir / s / <class> / *.npy
            split_folder = features_dir / s
            if not split_folder.exists():
                raise FileNotFoundError(f"Neither manifest '{manifest_file}' nor folder '{split_folder}' found.")

            X_list, y_list = [], []
            for cls_name, cls_id in LABEL_MAP.items():
                cls_dir = split_folder / cls_name
                if cls_dir.exists():
                    for npy_file in sorted(cls_dir.glob("*.npy")):
                        try:
                            arr = np.load(npy_file)
                            X_list.append(pool_audio_features(arr, pooling=pooling))
                            y_list.append(cls_id)
                        except Exception as e:
                            logger.warning(f"Error loading {npy_file}: {e}")

            if not X_list:
                raise RuntimeError(f"No .npy feature files found in '{split_folder}'.")
            X_s = np.array(X_list, dtype=np.float32)
            y_s = np.array(y_list, dtype=np.int32)

        dataset[s] = (X_s, y_s)
        logger.info(f"  {s.upper()}: X={X_s.shape}, y={y_s.shape}")

        # Save to cache for subsequent runs
        try:
            np.save(cache_dir / f"{cache_prefix}_X_{s}.npy", X_s)
            np.save(cache_dir / f"{cache_prefix}_y_{s}.npy", y_s)
        except Exception as e:
            logger.warning(f"Could not write cache file: {e}")

    logger.info("Dataset aggregation and caching completed successfully.")
    return dataset


def compute_damped_class_weights(
    y_train: np.ndarray,
    num_classes: int = 7,
    damp_factor: float = 0.8,
    min_weight: float = 0.5,
    max_weight: float = 8.0
) -> Dict[int, float]:
    """
    Computes balanced class weights with damping and clipping.
    Extreme raw weights (e.g. 17.0x for rare Abuse class) can destabilize SGD/Adam
    gradients, leading to divergence or false positives. Damping (weight^damp_factor)
    preserves minority class sensitivity while ensuring stable optimization.
    """
    classes_present = np.unique(y_train)
    raw_weights = compute_class_weight(
        class_weight="balanced",
        classes=classes_present,
        y=y_train
    )

    weight_dict = {}
    for cls_idx, raw_w in zip(classes_present, raw_weights):
        damped_w = float(np.clip(raw_w ** damp_factor, min_weight, max_weight))
        weight_dict[int(cls_idx)] = damped_w

    # Ensure all classes have a default weight of 1.0 if not present in train set
    for i in range(num_classes):
        if i not in weight_dict:
            weight_dict[i] = 1.0

    return weight_dict


def build_yamnet_classifier(
    input_dim: int,
    num_classes: int = len(CLASSES),
    architecture: str = "residual",
    dropout_rate: float = 0.3,
    l2_reg: float = 1e-4
) -> tf.keras.Model:
    """
    Builds the deep classification head for pooled YAMNet feature embeddings.

    Architectures:
      - 'residual': LayerNorm -> Dense(256) -> Residual Block(256) -> Dense(128) -> Softmax
      - 'mlp':      LayerNorm -> Dense(512) -> Dense(256) -> Dense(128) -> Softmax
      - 'light':    Standard 2-layer MLP (Dense 256 -> Dense 128 -> Softmax)
    """
    regularizer = tf.keras.regularizers.l2(l2_reg) if l2_reg > 0 else None
    inputs = tf.keras.layers.Input(shape=(input_dim,), name="audio_embedding_input")

    if architecture == "residual":
        # Input layer normalization
        x = tf.keras.layers.LayerNormalization(name="input_norm")(inputs)
        x = tf.keras.layers.Dense(256, activation="relu", kernel_regularizer=regularizer, name="dense_proj")(x)
        x = tf.keras.layers.Dropout(dropout_rate, name="dropout_proj")(x)

        # Residual Block: preserves gradient flow and refines acoustic patterns
        res = x
        x = tf.keras.layers.Dense(256, activation="relu", kernel_regularizer=regularizer, name="res_dense_1")(x)
        x = tf.keras.layers.LayerNormalization(name="res_norm_1")(x)
        x = tf.keras.layers.Dropout(dropout_rate, name="res_dropout")(x)
        x = tf.keras.layers.Dense(256, activation="relu", kernel_regularizer=regularizer, name="res_dense_2")(x)
        x = tf.keras.layers.LayerNormalization(name="res_norm_2")(x)
        x = tf.keras.layers.Add(name="res_add")([x, res])

        # Bottleneck and classifier
        x = tf.keras.layers.Dense(128, activation="relu", kernel_regularizer=regularizer, name="dense_neck")(x)
        x = tf.keras.layers.Dropout(max(0.1, dropout_rate - 0.1), name="dropout_neck")(x)
        outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="incident_probabilities")(x)

    elif architecture == "mlp":
        x = tf.keras.layers.LayerNormalization(name="input_norm")(inputs)
        x = tf.keras.layers.Dense(512, activation="relu", kernel_regularizer=regularizer, name="dense_512")(x)
        x = tf.keras.layers.Dropout(dropout_rate + 0.1, name="dropout_1")(x)
        x = tf.keras.layers.Dense(256, activation="relu", kernel_regularizer=regularizer, name="dense_256")(x)
        x = tf.keras.layers.Dropout(dropout_rate, name="dropout_2")(x)
        x = tf.keras.layers.Dense(128, activation="relu", kernel_regularizer=regularizer, name="dense_128")(x)
        x = tf.keras.layers.Dropout(max(0.1, dropout_rate - 0.1), name="dropout_3")(x)
        outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="incident_probabilities")(x)

    elif architecture == "light":
        x = tf.keras.layers.LayerNormalization(name="input_norm")(inputs)
        x = tf.keras.layers.Dense(256, activation="relu", kernel_regularizer=regularizer, name="dense_256")(x)
        x = tf.keras.layers.Dropout(dropout_rate, name="dropout_1")(x)
        x = tf.keras.layers.Dense(128, activation="relu", kernel_regularizer=regularizer, name="dense_128")(x)
        x = tf.keras.layers.Dropout(max(0.1, dropout_rate - 0.1), name="dropout_2")(x)
        outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="incident_probabilities")(x)

    else:
        raise ValueError(f"Unknown architecture: '{architecture}'. Choose 'residual', 'mlp', or 'light'.")

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=f"YAMNet_IncidentHead_{architecture.capitalize()}")
    return model


def train_yamnet_model(
    features_dir: Union[str, Path] = DEFAULT_FEATURES_DIR,
    output_dir: Union[str, Path] = DEFAULT_CHECKPOINTS_DIR,
    model_name: str = "audio_incident_head.keras",
    pooling: str = "mean_max",
    architecture: str = "residual",
    batch_size: int = 32,
    epochs: int = 60,
    learning_rate: float = 0.001,
    dropout_rate: float = 0.3,
    label_smoothing: float = 0.05,
    damp_factor: float = 0.8,
    force_recompute: bool = False
) -> Tuple[tf.keras.Model, Dict[str, Any]]:
    """
    Main training execution function:
    1. Loads train/val/test splits (or discovers them automatically)
    2. Calculates damped class weights
    3. Builds and compiles deep classifier
    4. Trains with EarlyStopping and ReduceLROnPlateau
    5. Performs comprehensive evaluation on untouched test set
    6. Saves checkpoint, evaluation report, and metadata
    """
    start_time = time.time()
    features_dir = Path(features_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = output_dir / model_name

    logger.info("=" * 70)
    logger.info(" STARTING YAMNET AUDIO CLASSIFIER TRAINING")
    logger.info(f" Features Directory:  {features_dir}")
    logger.info(f" Checkpoint Output:   {best_model_path}")
    logger.info(f" Pooling Strategy:    {pooling}")
    logger.info(f" Architecture:        {architecture}")
    logger.info(f" Initial LR:          {learning_rate}")
    logger.info(f" Batch Size:          {batch_size}")
    logger.info(f" Max Epochs:          {epochs}")
    logger.info("=" * 70)

    # 1. Check for legacy standalone X_embeddings.npy / y_labels.npy vs dataset directory
    legacy_x_path = Path("X_embeddings.npy")
    legacy_y_path = Path("y_labels.npy")

    if legacy_x_path.exists() and legacy_y_path.exists() and not features_dir.exists():
        logger.info("Loading pre-extracted legacy embeddings: 'X_embeddings.npy' & 'y_labels.npy'...")
        X = np.load(legacy_x_path)
        y = np.load(legacy_y_path)
        num_classes = len(np.unique(y))

        # Perform stratified 70 / 15 / 15 split
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.30, random_state=42, stratify=y
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
        )
    else:
        # Load from structured features directory with predefined splits
        dataset = load_or_cache_dataset(
            features_dir=features_dir,
            pooling=pooling,
            force_recompute=force_recompute
        )
        X_train, y_train = dataset["train"]
        X_val, y_val = dataset["val"]
        X_test, y_test = dataset["test"]
        num_classes = len(CLASSES)

    input_dim = X_train.shape[1]
    logger.info(f"Input Feature Dimension: {input_dim} | Classes: {num_classes}")

    # One-hot encode labels for Keras Categorical Crossentropy
    y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes)
    y_val_cat = tf.keras.utils.to_categorical(y_val, num_classes)
    y_test_cat = tf.keras.utils.to_categorical(y_test, num_classes)

    # 2. Calculate Damped Class Weights
    class_weights = compute_damped_class_weights(
        y_train=y_train,
        num_classes=num_classes,
        damp_factor=damp_factor
    )
    logger.info("Computed Damped Class Weights:")
    for cls_idx, w in class_weights.items():
        cls_name = REVERSE_LABEL_MAP.get(cls_idx, str(cls_idx))
        sample_count = int(np.sum(y_train == cls_idx))
        logger.info(f"  [{cls_idx}] {cls_name:<12}: weight={w:.3f} (train samples: {sample_count})")

    # 3. Build & Compile Model
    model = build_yamnet_classifier(
        input_dim=input_dim,
        num_classes=num_classes,
        architecture=architecture,
        dropout_rate=dropout_rate
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing)

    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=["accuracy"]
    )

    # 4. Training Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=15,
            restore_best_weights=True,
            mode="max",
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(best_model_path),
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1
        )
    ]

    # 5. Train Model
    logger.info("Starting model training...")
    history = model.fit(
        X_train, y_train_cat,
        validation_data=(X_val, y_val_cat),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )

    # Ensure best weights are preserved and explicitly saved
    model.save(str(best_model_path))
    logger.info(f"Trained model successfully saved to '{best_model_path}'")

    # Also save a copy to project root for seamless backwards compatibility
    root_model_path = PROJECT_ROOT / model_name
    if root_model_path.resolve() != best_model_path.resolve():
        try:
            model.save(str(root_model_path))
            logger.info(f"Saved root compatibility checkpoint to '{root_model_path}'")
        except Exception as e:
            logger.warning(f"Could not save root copy: {e}")

    # 6. Evaluation on Untouched Test Set
    logger.info("\n" + "=" * 70)
    logger.info(" EVALUATING ON UNTOUCHED TEST SET")
    logger.info("=" * 70)

    test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=0)
    logger.info(f"Test Loss:     {test_loss:.4f}")
    logger.info(f"Test Accuracy: {test_acc * 100:.2f}%\n")

    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    target_names = [CLASSES[i] for i in range(num_classes)]
    clf_report_str = classification_report(
        y_test, y_pred,
        target_names=target_names,
        digits=4,
        zero_division=0
    )
    print("Classification Report:")
    print(clf_report_str)

    # Compute detailed metrics
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0
    )
    prec_wt, rec_wt, f1_wt, _ = precision_recall_fscore_support(
        y_test, y_pred, average="weighted", zero_division=0
    )
    cm = confusion_matrix(y_test, y_pred, labels=list(range(num_classes)))

    # Formatted Confusion Matrix
    print(f"\n{'CONFUSION MATRIX':^95}")
    print("=" * 95)
    col_title = "Actual \\ Pred"
    header = f"{col_title:<15}" + "".join([f"{c[:11]:>11} " for c in CLASSES])
    print(header)
    print("-" * 95)
    for idx, row in enumerate(cm):
        row_str = f"{CLASSES[idx]:<15}" + "".join([f"{val:>11} " for val in row])
        print(row_str)
    print("=" * 95 + "\n")

    # Per-class metrics dictionary
    per_class_metrics = {}
    p_arr, r_arr, f_arr, s_arr = precision_recall_fscore_support(
        y_test, y_pred, labels=list(range(num_classes)), zero_division=0
    )
    for idx, name in enumerate(CLASSES):
        per_class_metrics[name] = {
            "precision": float(p_arr[idx]),
            "recall": float(r_arr[idx]),
            "f1_score": float(f_arr[idx]),
            "support": int(s_arr[idx])
        }

    # Save Evaluation Report JSON (consistent with video pipeline)
    eval_report = {
        "model_path": str(best_model_path),
        "modality": "audio",
        "pooling": pooling,
        "architecture": architecture,
        "input_dim": input_dim,
        "split": "test",
        "total_samples": int(len(y_test)),
        "accuracy": float(test_acc),
        "macro_precision": float(prec_macro),
        "macro_recall": float(rec_macro),
        "macro_f1": float(f1_macro),
        "weighted_f1": float(f1_wt),
        "per_class": per_class_metrics,
        "confusion_matrix": cm.tolist()
    }

    eval_report_file = output_dir / "audio_evaluation_report.json"
    with open(eval_report_file, "w", encoding="utf-8") as f:
        json.dump(eval_report, f, indent=2)
    logger.info(f"Evaluation report saved to: '{eval_report_file}'")

    # Save Model Metadata for Multimodal Inference
    metadata = {
        "classes": CLASSES,
        "label_map": LABEL_MAP,
        "reverse_label_map": REVERSE_LABEL_MAP,
        "feature_dim": input_dim,
        "pooling": pooling,
        "architecture": architecture,
        "accuracy": float(test_acc),
        "macro_f1": float(f1_macro),
        "weighted_f1": float(f1_wt),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    metadata_file = output_dir / "audio_incident_head_metadata.json"
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Model metadata saved to: '{metadata_file}'")

    elapsed = time.time() - start_time
    logger.info(f"Training and evaluation completed in {elapsed:.1f}s.")
    print("\nModel saved! Ready for multimodal late fusion.")

    return model, eval_report


# ==============================================================================
# Multimodal Late Fusion Predictor
# ==============================================================================
class AudioIncidentPredictor:
    """
    Inference helper for Multimodal Late Fusion.
    Loads the trained .keras classification head and produces normalized incident probabilities.
    """
    def __init__(
        self,
        model_path: Union[str, Path] = DEFAULT_CHECKPOINTS_DIR / "audio_incident_head.keras",
        metadata_path: Optional[Union[str, Path]] = None,
        pooling: str = "mean_max"
    ):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            alt = PROJECT_ROOT / self.model_path.name
            if alt.exists():
                self.model_path = alt
            else:
                raise FileNotFoundError(f"Trained audio model checkpoint not found: {model_path}")

        logger.info(f"Loading trained audio classification model from '{self.model_path}'...")
        self.model = tf.keras.models.load_model(str(self.model_path))

        self.pooling = pooling
        self.classes = CLASSES
        self.label_map = LABEL_MAP

        # Try loading metadata
        meta_file = metadata_path or self.model_path.parent / "audio_incident_head_metadata.json"
        if Path(meta_file).exists():
            try:
                with open(meta_file, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                    self.classes = meta.get("classes", CLASSES)
                    self.label_map = meta.get("label_map", LABEL_MAP)
                    self.pooling = meta.get("pooling", self.pooling)
            except Exception as e:
                logger.warning(f"Could not load metadata: {e}")

    def predict_features(self, features: np.ndarray) -> Dict[str, Any]:
        """
        Takes raw YAMNet embeddings [T, 1024] or pre-pooled vector and returns
        predicted incident class, confidence score, and probability distribution.
        """
        pooled = pool_audio_features(features, pooling=self.pooling)
        input_tensor = np.expand_dims(pooled, axis=0)  # [1, D]
        probs = self.model.predict(input_tensor, verbose=0)[0]  # [num_classes]

        top_idx = int(np.argmax(probs))
        top_class = self.classes[top_idx]
        confidence = float(probs[top_idx])

        prob_dict = {cls_name: float(probs[i]) for i, cls_name in enumerate(self.classes)}

        return {
            "top_class": top_class,
            "confidence": confidence,
            "probabilities": prob_dict,
            "logits_vector": probs.tolist()
        }

    def predict_audio_file(self, audio_path: Union[str, Path], feature_extractor=None) -> Dict[str, Any]:
        """
        Extracts YAMNet features directly from a .wav audio file and returns predictions.
        Requires YAMNetFeatureExtractor from backend.audio.audio_feature_extract.
        """
        if feature_extractor is None:
            try:
                from backend.audio.audio_feature_extract import YAMNetFeatureExtractor
            except ImportError:
                from audio_feature_extract import YAMNetFeatureExtractor
            feature_extractor = YAMNetFeatureExtractor(feature_type="embeddings")

        feats = feature_extractor.extract_features(audio_path)
        return self.predict_features(feats)


# ==============================================================================
# CLI Entry Point
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Train YAMNet Deep Classification Head on XD-Violence Audio Features for Multimodal Fusion"
    )
    parser.add_argument(
        "--features_dir",
        type=str,
        default=str(DEFAULT_FEATURES_DIR),
        help="Path to extracted features directory containing manifests or train/val/test splits"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(DEFAULT_CHECKPOINTS_DIR),
        help="Directory to save trained model checkpoints and evaluation reports"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="audio_incident_head.keras",
        help="Saved model filename (default: audio_incident_head.keras)"
    )
    parser.add_argument(
        "--pooling",
        type=str,
        default="mean_max",
        choices=["mean_max", "mean", "max", "stats"],
        help="Temporal pooling mode: 'mean_max' (2048-dim, recommended), 'mean' (1024-dim), 'max', 'stats'"
    )
    parser.add_argument(
        "--architecture",
        type=str,
        default="residual",
        choices=["residual", "mlp", "light"],
        help="Classifier head architecture (default: 'residual')"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=60,
        help="Maximum training epochs (default: 60)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Training batch size (default: 32)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Initial learning rate (default: 0.001)"
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.3,
        help="Dropout rate (default: 0.3)"
    )
    parser.add_argument(
        "--label_smoothing",
        type=float,
        default=0.05,
        help="Categorical crossentropy label smoothing factor (default: 0.05)"
    )
    parser.add_argument(
        "--damp_weights",
        type=float,
        default=0.8,
        help="Power factor for damped class weights to avoid gradient spikes on minority classes"
    )
    parser.add_argument(
        "--force_recompute",
        action="store_true",
        help="Force recomputing pooled matrices from individual .npy files rather than using cache"
    )

    args = parser.parse_args()

    train_yamnet_model(
        features_dir=args.features_dir,
        output_dir=args.output_dir,
        model_name=args.model_name,
        pooling=args.pooling,
        architecture=args.architecture,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        dropout_rate=args.dropout,
        label_smoothing=args.label_smoothing,
        damp_factor=args.damp_weights,
        force_recompute=args.force_recompute
    )


if __name__ == "__main__":
    main()

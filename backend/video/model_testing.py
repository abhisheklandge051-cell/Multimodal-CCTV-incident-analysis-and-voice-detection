import os
import sys
import json
import logging
from typing import Dict, Any, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
try:
    from backend.video.loading_processing import LABEL_MAP, REVERSE_LABEL_MAP
    from backend.video.fast_gru_resnet_training import (
        FastGRUResNetDetector,
        ResNetFeatureDataset,
        load_stored_features_index,
        build_feature_clip_index
    )
except ImportError:
    from loading_processing import LABEL_MAP, REVERSE_LABEL_MAP
    from fast_gru_resnet_training import (
        FastGRUResNetDetector,
        ResNetFeatureDataset,
        load_stored_features_index,
        build_feature_clip_index
    )

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")
logger = logging.getLogger("Model_Testing")


def evaluate_model(
    model_path: str = "best_fast_gru_resnet_model.pth",
    dataset_dir: str = "XD-voilence_dataset",
    features_dir: str = "XD-voilence_dataset/features",
    batch_size: int = 128,
    split: str = "test",
    output_report_path: Optional[str] = "test_evaluation_report.json"
) -> Dict[str, Any]:
    """
    Evaluates the trained FastGRUResNetDetector model on the specified split (default: test).
    Prints comprehensive classification metrics, per-class breakdown, and confusion matrix.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using Device for Evaluation: {device}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model weights file not found: '{model_path}'. Please train the model first.")

    # 1. Load stored features & build clip index for target split
    logger.info(f"Loading stored features from '{features_dir}' for split='{split}'...")
    verified_videos = load_stored_features_index(dataset_dir=dataset_dir, features_dir=features_dir)
    all_clips = build_feature_clip_index(verified_videos, clip_len=16)

    target_clips = [c for c in all_clips if c["split"] == split]
    if len(target_clips) == 0:
        logger.warning(f"No clips found for split='{split}'. Evaluating on all available stored clips.")
        target_clips = all_clips

    logger.info(f"Loaded {len(target_clips)} clips for evaluation.")

    dataset = ResNetFeatureDataset(target_clips, clip_len=16, is_train=False)
    loader_workers = 4 if os.name != 'nt' else 0
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=loader_workers,
        pin_memory=torch.cuda.is_available()
    )

    # 2. Instantiate model and load checkpoint
    num_classes = len(LABEL_MAP)
    model = FastGRUResNetDetector(
        feature_dim=2048,
        hidden_dim=256,
        num_classes=num_classes,
        dropout_rate=0.4
    ).to(device)

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    logger.info(f"Successfully loaded trained model weights from '{model_path}'.")

    # 3. Model Inference
    all_preds = []
    all_labels = []

    pbar = tqdm(loader, desc=f"Evaluating ({split})", unit="batch", dynamic_ncols=True)
    with torch.no_grad():
        for inputs, targets in pbar:
            inputs = inputs.to(device, non_blocking=True)
            logits = model(inputs)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(targets.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # 4. Compute Metrics
    acc = accuracy_score(all_labels, all_preds)
    macro_prec, macro_rec, macro_f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="macro", zero_division=0
    )
    weighted_prec, weighted_rec, weighted_f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="weighted", zero_division=0
    )

    per_class_p, per_class_r, per_class_f1, per_class_sup = precision_recall_fscore_support(
        all_labels, all_preds, average=None, labels=list(range(num_classes)), zero_division=0
    )

    cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))

    # 5. Format & Display Results
    print("\n" + "=" * 90)
    print(f"{'MODEL EVALUATION RESULTS (' + split.upper() + ' SPLIT)':^90}")
    print("=" * 90)
    print(f"Overall Accuracy:       {acc * 100:.2f}%")
    print(f"Macro Precision:        {macro_prec:.4f}")
    print(f"Macro Recall:           {macro_rec:.4f}")
    print(f"Macro F1-Score:         {macro_f1:.4f}")
    print(f"Weighted F1-Score:      {weighted_f1:.4f}")
    print("-" * 90)

    print("\n" + "=" * 90)
    print(f"{'PER-CLASS PERFORMANCE BREAKDOWN':^90}")
    print("=" * 90)
    print(f"{'ID':<3} | {'Class Name':<14} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Support':<8}")
    print("-" * 90)
    class_metrics = {}
    for i in range(num_classes):
        c_name = REVERSE_LABEL_MAP.get(i, f"Class_{i}")
        p_val, r_val, f1_val, sup_val = per_class_p[i], per_class_r[i], per_class_f1[i], int(per_class_sup[i])
        print(f"{i:<3} | {c_name:<14} | {p_val:<10.4f} | {r_val:<10.4f} | {f1_val:<10.4f} | {sup_val:<8,d}")
        class_metrics[c_name] = {
            "precision": float(p_val),
            "recall": float(r_val),
            "f1_score": float(f1_val),
            "support": sup_val
        }
    print("=" * 90)

    print("\n" + "=" * 90)
    print(f"{'CONFUSION MATRIX':^90}")
    print("=" * 90)
    matrix_title = "True \\ Pred"
    header_str = f"{matrix_title:<15} | " + " | ".join([f"{REVERSE_LABEL_MAP.get(i, str(i)):^8}" for i in range(num_classes)])
    print(header_str)
    print("-" * len(header_str))
    for i in range(num_classes):
        c_name = REVERSE_LABEL_MAP.get(i, f"Class_{i}")
        row_str = f"{c_name:<15} | " + " | ".join([f"{cm[i, j]:^8d}" for j in range(num_classes)])
        print(row_str)
    print("=" * 90 + "\n")

    results = {
        "model_path": model_path,
        "split": split,
        "total_clips": len(all_labels),
        "accuracy": float(acc),
        "macro_precision": float(macro_prec),
        "macro_recall": float(macro_rec),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "per_class": class_metrics,
        "confusion_matrix": cm.tolist()
    }

    if output_report_path:
        with open(output_report_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved evaluation metrics report to '{output_report_path}'.")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate Fast GRU + ResNet Model")
    parser.add_argument("--model_path", type=str, default="best_fast_gru_resnet_model.pth", help="Model checkpoint path")
    parser.add_argument("--dataset_dir", type=str, default="XD-voilence_dataset", help="Dataset directory")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"], help="Dataset split to evaluate")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--output", type=str, default="test_evaluation_report.json", help="Output report JSON file")
    args = parser.parse_args()

    evaluate_model(
        model_path=args.model_path,
        dataset_dir=args.dataset_dir,
        split=args.split,
        batch_size=args.batch_size,
        output_report_path=args.output
    )

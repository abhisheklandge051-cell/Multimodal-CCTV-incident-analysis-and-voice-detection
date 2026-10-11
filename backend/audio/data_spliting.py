import os
import json
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from concurrent.futures import ThreadPoolExecutor

# Target classes (all 7 XD-Violence categories, consistent with video pipeline)
CLASSES = [
    "Abuse",
    "CarAccident",
    "Explosion",
    "Fighting",
    "Normal",
    "Riot",
    "Shooting"
]

LABEL_MAP = {
    "Abuse": 0,
    "CarAccident": 1,
    "Explosion": 2,
    "Fighting": 3,
    "Normal": 4,
    "Riot": 5,
    "Shooting": 6
}

DEFAULT_INPUT_DIR = Path("D:/multimodal incident analysis and voice validation/xd_violence_audio_preprocessed")
DEFAULT_RAW_DIR = Path("D:/multimodal incident analysis and voice validation/xd_violence_audio")
DEFAULT_DATASET_DIR = Path("D:/multimodal incident analysis and voice validation/XD-voilence_dataset")
DEFAULT_OUTPUT_DIR = Path("./xd_violence_audio_split")

def set_seed(seed: int = 42):
    random.seed(seed)

def setup_split_directories(output_dir: Path) -> Dict[str, Path]:
    """Creates directory structure for train, val, validation, and test splits."""
    output_dir = Path(output_dir)
    splits = ["train", "val", "validation", "test"]
    split_paths = {}

    for split in splits:
        for cls_name in CLASSES:
            path = output_dir / split / cls_name
            path.mkdir(parents=True, exist_ok=True)
            split_paths[f"{split}/{cls_name}"] = path

    return split_paths

def get_audio_files_by_class(input_dir: Path) -> Dict[str, List[Path]]:
    """Scans input directory for WAV files grouped by class."""
    input_dir = Path(input_dir)
    class_files = {cls_name: [] for cls_name in CLASSES}

    for cls_name in CLASSES:
        cls_dir = input_dir / cls_name
        if cls_dir.exists():
            wav_files = sorted(list(cls_dir.glob("*.wav")))
            class_files[cls_name] = wav_files
        else:
            print(f"Warning: Class directory '{cls_dir}' not found.")

    return class_files

def split_dataset_stratified(
    class_files: Dict[str, List[Path]],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> Dict[str, Dict[str, List[Path]]]:
    """
    Performs stratified train/val/test split across all classes.
    Returns dictionary mapping split_name -> class_name -> list of file paths.
    """
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-5, "Ratios must sum to 1.0"
    set_seed(seed)

    split_result = {
        "train": {cls: [] for cls in CLASSES},
        "val": {cls: [] for cls in CLASSES},
        "test": {cls: [] for cls in CLASSES}
    }

    for cls_name, files in class_files.items():
        shuffled = files.copy()
        random.shuffle(shuffled)
        
        n = len(shuffled)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train_files = shuffled[:n_train]
        val_files = shuffled[n_train:n_train + n_val]
        test_files = shuffled[n_train + n_val:]

        split_result["train"][cls_name] = train_files
        split_result["val"][cls_name] = val_files
        split_result["test"][cls_name] = test_files

    return split_result

def copy_split_files(
    split_result: Dict[str, Dict[str, List[Path]]],
    output_dir: Path,
    use_hardlink: bool = True
) -> List[Dict[str, Any]]:
    """
    Copies (or hardlinks on NTFS for 0 extra disk space) files into target split directories
    and builds split manifest.
    """
    output_dir = Path(output_dir)
    setup_split_directories(output_dir)

    manifest_records = []
    copy_tasks = []

    for split_name in ["train", "val", "test"]:
        for cls_name in CLASSES:
            files = split_result[split_name][cls_name]
            dst_dir = output_dir / split_name / cls_name
            # Create duplicate alias for validation directory
            dst_dir_alias = output_dir / "validation" / cls_name if split_name == "val" else None

            for src_path in files:
                dst_path = dst_dir / src_path.name
                copy_tasks.append((src_path, dst_path))
                if dst_dir_alias:
                    copy_tasks.append((src_path, dst_dir_alias / src_path.name))

                manifest_records.append({
                    "filename": src_path.name,
                    "class_name": cls_name,
                    "label_id": LABEL_MAP[cls_name],
                    "split": split_name,
                    "src_path": str(src_path.resolve()),
                    "dst_path": str(dst_path.resolve())
                })

    method_str = "Hardlinking / copying" if use_hardlink else "Copying"
    print(f"{method_str} audio files into train, val/validation, and test directories...")
    
    def copy_file(args):
        src, dst = args
        if dst.exists():
            try:
                dst.unlink()
            except Exception:
                pass
        if use_hardlink:
            try:
                os.link(src, dst)
                return
            except Exception:
                pass
        shutil.copy2(src, dst)

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(copy_file, copy_tasks))

    # Save manifest
    manifest_path = output_dir / "dataset_split_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_records, f, indent=2)

    print(f"Manifest successfully saved to '{manifest_path}'")
    return manifest_records

def print_split_summary(split_result: Dict[str, Dict[str, List[Path]]]):
    """Prints clean table of dataset split distribution."""
    print("\n" + "=" * 65)
    print(f"{'Class Name':<15} | {'Train':<10} | {'Val':<10} | {'Test':<10} | {'Total':<10}")
    print("=" * 65)

    totals = {"train": 0, "val": 0, "test": 0, "total": 0}

    for cls_name in CLASSES:
        tr = len(split_result["train"][cls_name])
        va = len(split_result["val"][cls_name])
        te = len(split_result["test"][cls_name])
        tot = tr + va + te

        totals["train"] += tr
        totals["val"] += va
        totals["test"] += te
        totals["total"] += tot

        print(f"{cls_name:<15} | {tr:<10} | {va:<10} | {te:<10} | {tot:<10}")

    print("-" * 65)
    print(f"{'Total':<15} | {totals['train']:<10} | {totals['val']:<10} | {totals['test']:<10} | {totals['total']:<10}")
    print("=" * 65 + "\n")

def resolve_input_dir(input_dir: Optional[Path] = None) -> Path:
    """Finds existing input directory containing audio files, checking preprocessed then raw."""
    if input_dir is not None:
        p = Path(input_dir)
        if p.exists() and any(p.rglob("*.wav")):
            return p

    for candidate in [DEFAULT_INPUT_DIR, Path("./xd_violence_audio_preprocessed")]:
        if candidate.exists() and any(candidate.rglob("*.wav")):
            return candidate

    for candidate in [DEFAULT_RAW_DIR, Path("./xd_violence_audio")]:
        if candidate.exists() and any(candidate.rglob("*.wav")):
            return candidate

    return DEFAULT_INPUT_DIR

def extract_all_audio(
    dataset_dir: Path = DEFAULT_DATASET_DIR,
    output_dir: Path = DEFAULT_RAW_DIR,
    max_workers: int = 8,
    preprocess: bool = True
):
    """Invokes the extraction pipeline from audio_dataload to extract all 7 classes."""
    try:
        from backend.audio.audio_dataload import extract_audio_pipeline
    except ImportError:
        from audio_dataload import extract_audio_pipeline

    print("=== Extracting Audio for all 7 classes from XD-voilence_dataset ===")
    extract_audio_pipeline(
        input_dir=dataset_dir,
        output_dir=output_dir,
        max_workers=max_workers,
        preprocess=preprocess
    )

def run_data_splitting(
    input_dir: Optional[Path] = None,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    extract_first: bool = False,
    max_workers: int = 8
):
    print(f"=== Audio Dataset Splitting Pipeline (7 Classes) ===")

    if extract_first:
        extract_all_audio(max_workers=max_workers, preprocess=True)

    resolved_input = resolve_input_dir(input_dir)
    print(f"Input Directory: {resolved_input}")
    print(f"Output Directory: {output_dir}")
    print(f"Ratios -> Train: {train_ratio:.0%}, Val: {val_ratio:.0%}, Test: {test_ratio:.0%}")

    class_files = get_audio_files_by_class(resolved_input)
    total_found = sum(len(f) for f in class_files.values())
    print(f"Total audio files found: {total_found}")

    if total_found == 0:
        if not extract_first:
            print("No audio files found. Automatically triggering audio extraction first...")
            extract_all_audio(max_workers=max_workers, preprocess=True)
            resolved_input = resolve_input_dir(input_dir)
            class_files = get_audio_files_by_class(resolved_input)
            total_found = sum(len(f) for f in class_files.values())
            print(f"Audio files available after extraction: {total_found}")

        if total_found == 0:
            print("Error: Still no WAV files found in input directory.")
            return

    split_result = split_dataset_stratified(
        class_files,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed
    )

    print_split_summary(split_result)
    copy_split_files(split_result, output_dir=output_dir)
    print("Dataset splitting completed successfully!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Split XD-Violence audio dataset into train/val/test splits.")
    parser.add_argument("--extract", action="store_true", help="Extract audio from dataset before splitting")
    parser.add_argument("--input_dir", type=str, default=None, help="Input directory of audio files")
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Output split directory")
    parser.add_argument("--train_ratio", type=float, default=0.70)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    input_path = Path(args.input_dir) if args.input_dir else None
    run_data_splitting(
        input_dir=input_path,
        output_dir=Path(args.output_dir),
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        extract_first=args.extract,
        max_workers=args.workers
    )

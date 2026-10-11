import os
import re
import sys
import shutil
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np

# Try importing audio libraries
try:
    import soundfile as sf
except ImportError:
    sf = None

try:
    import librosa
except ImportError:
    librosa = None

try:
    import imageio_ffmpeg
    FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    FFMPEG_EXE = shutil.which("ffmpeg") or "ffmpeg"

# Target dictionary for all 7 XD-Violence classes
CODE2LABEL = {
    "A": "Normal",
    "B1": "Fighting",
    "B2": "Shooting",
    "B4": "Riot",
    "B5": "Abuse",
    "B6": "CarAccident",
    "G": "Explosion"
}

DEFAULT_INPUT_DIR = Path("D:/multimodal incident analysis and voice validation/XD-voilence_dataset")
DEFAULT_OUTPUT_DIR = Path("./xd_violence_audio")
PREPROCESSED_OUTPUT_DIR = Path("./xd_violence_audio_preprocessed")

def setup_directories(base_dir=DEFAULT_OUTPUT_DIR):
    """Creates the base output directory and target subfolders for all 7 classes."""
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    for class_name in set(CODE2LABEL.values()):
        (base_dir / class_name).mkdir(exist_ok=True)
    return base_dir

def get_class_from_filename(filename):
    """
    Scans the filename or parent directory for XD-Violence label pattern
    (e.g., label_A, label_B1, label_B2, label_B4, label_B5, label_B6-0-0, label_G).
    Returns the target class label if matched with CODE2LABEL, otherwise returns None.
    """
    path_obj = Path(filename)
    match = re.search(r'label_([A-Za-z0-9]+)', path_obj.name)
    if match:
        code = match.group(1).upper()
        if code in CODE2LABEL:
            return CODE2LABEL[code]

    # Check parent directory name
    if path_obj.parent.name in CODE2LABEL.values():
        return path_obj.parent.name

    # Check filename prefix before first underscore
    prefix = path_obj.name.split('_')[0]
    for label in CODE2LABEL.values():
        if label.lower() == prefix.lower():
            return label

    return None

def extract_single_audio(video_path, output_path, sample_rate=16000, preprocess=False):
    """
    Extracts mono WAV audio at target sample rate using FFmpeg.
    If preprocess=True, applies silence trimming and normalization directly before saving.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        FFMPEG_EXE,
        "-y",
        "-i", str(video_path),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", str(sample_rate),
        "-ac", "1",
        "-loglevel", "error",
        str(output_path)
    ]
    subprocess.run(cmd, check=True)

    if preprocess:
        try:
            if sf is not None:
                audio, sr = sf.read(str(output_path))
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)
                audio = trim_silence(audio, sr=sr)
                audio = normalize_audio(audio)
                sf.write(str(output_path), audio, sr)
        except Exception as e:
            print(f"Warning: Preprocessing failed for {output_path.name}: {e}")

def extract_audio_pipeline(
    input_dir=DEFAULT_INPUT_DIR,
    output_dir=DEFAULT_OUTPUT_DIR,
    max_workers=16,
    overwrite=False,
    preprocess=True
):
    """
    Finds all target videos in input_dir and extracts 16kHz mono audio WAVs into target class subfolders.
    Supports all 7 XD-Violence classes.
    """
    input_dir = Path(input_dir)
    output_dir = setup_directories(output_dir)
    
    if not input_dir.exists():
        print(f"Error: Input directory '{input_dir}' does not exist.")
        return
        
    video_files = list(input_dir.rglob("*.mp4"))
    print(f"Found {len(video_files)} total videos in '{input_dir}'. Filtering for 7 target classes...")
    
    target_tasks = []
    skipped_count = 0
    already_extracted_count = 0

    for video_path in video_files:
        target_class = get_class_from_filename(video_path)
        if target_class is None:
            skipped_count += 1
            continue
            
        output_filename = f"{video_path.stem}.wav"
        output_path = output_dir / target_class / output_filename
        
        if output_path.exists() and not overwrite:
            already_extracted_count += 1
            continue
            
        target_tasks.append((video_path, output_path))
        
    print(f"Target videos to extract: {len(target_tasks)} (Already extracted: {already_extracted_count}, Skipped non-target: {skipped_count})")
    
    if not target_tasks:
        print("No new audio extractions needed.")
        return
        
    def task_wrapper(args):
        v_path, o_path = args
        try:
            extract_single_audio(v_path, o_path, preprocess=preprocess)
            return True
        except Exception as e:
            print(f"Error extracting {v_path.name}: {e}")
            return False

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(task_wrapper, target_tasks))
        
    success_count = sum(1 for r in results if r)
    error_count = sum(1 for r in results if not r)
    
    print("\n--- Audio Extraction Summary ---")
    print(f"Successfully extracted: {success_count} files")
    print(f"Errors: {error_count} files")
    print(f"Total ready in '{output_dir}': {already_extracted_count + success_count} files")

# Audio Preprocessing Functions

def normalize_audio(audio, target_db=-3.0):
    """Normalizes peak amplitude of audio signal to target dB."""
    max_val = np.max(np.abs(audio))
    if max_val == 0:
        return audio
    target_amplitude = 10 ** (target_db / 20.0)
    return audio * (target_amplitude / max_val)

def trim_silence(audio, sr=16000, top_db=25):
    """Trims leading and trailing silence from audio array using fast energy thresholding."""
    if len(audio) == 0:
        return audio
    energy = np.abs(audio)
    threshold = np.max(energy) * (10 ** (-top_db / 20.0))
    non_silent = np.where(energy > threshold)[0]
    if len(non_silent) > 0:
        return audio[non_silent[0]:non_silent[-1] + 1]
    return audio

def preprocess_audio_file(wav_path, output_path=None, sample_rate=16000, target_duration=None, top_db=25):
    """
    Loads a WAV file, trims silence, normalizes amplitude, and optionally pads/truncates to target_duration (seconds).
    Saves preprocessed WAV to output_path if provided, and returns (audio_data, sample_rate).
    """
    wav_path = Path(wav_path)
    if sf is not None:
        audio, sr = sf.read(str(wav_path))
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != sample_rate and librosa is not None:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)
            sr = sample_rate
    elif librosa is not None:
        audio, sr = librosa.load(str(wav_path), sr=sample_rate, mono=True)
    else:
        raise RuntimeError("Neither soundfile nor librosa is installed for loading audio.")

    # 1. Trim silence
    audio = trim_silence(audio, sr=sr, top_db=top_db)

    # 2. Peak normalization
    audio = normalize_audio(audio)

    # 3. Optional fixed length padding / truncation
    if target_duration is not None:
        target_length = int(target_duration * sr)
        if len(audio) < target_length:
            pad_len = target_length - len(audio)
            audio = np.pad(audio, (0, pad_len), mode='constant')
        elif len(audio) > target_length:
            audio = audio[:target_length]

    # Save processed audio if output_path specified
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if sf is not None:
            sf.write(str(output_path), audio, sr)
        elif librosa is not None:
            import scipy.io.wavfile as wavfile
            wavfile.write(str(output_path), sr, (audio * 32767).astype(np.int16))

    return audio, sr

def preprocess_all_audio(input_dir=DEFAULT_OUTPUT_DIR, output_dir=PREPROCESSED_OUTPUT_DIR, max_workers=8, target_duration=None):
    """
    Preprocesses all extracted audio files in input_dir and saves preprocessed WAVs to output_dir.
    """
    input_dir = Path(input_dir)
    output_dir = setup_directories(output_dir)

    wav_files = list(input_dir.rglob("*.wav"))
    print(f"Preprocessing {len(wav_files)} extracted WAV files from '{input_dir}'...")

    def process_wrapper(wav_path):
        rel_path = wav_path.relative_to(input_dir)
        out_path = output_dir / rel_path
        try:
            preprocess_audio_file(wav_path, output_path=out_path, target_duration=target_duration)
            return True
        except Exception as e:
            print(f"Error preprocessing {wav_path.name}: {e}")
            return False

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(process_wrapper, wav_files))

    success = sum(1 for r in results if r)
    print(f"\n--- Audio Preprocessing Summary ---")
    print(f"Preprocessed {success} / {len(wav_files)} files saved to '{output_dir}'.")

if __name__ == "__main__":
    print("=== XD-Violence Audio Extraction & Preprocessing Pipeline (7 Classes) ===")
    # Extract and preprocess audio for all 7 target classes
    extract_audio_pipeline(preprocess=True, max_workers=16)
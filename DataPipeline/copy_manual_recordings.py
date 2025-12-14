#!/usr/bin/env python3
from pathlib import Path
import shutil
import librosa
import os
import soundfile as sf
import numpy as np

# Initializing paths
SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR / "p001_manual_recordings"
DST_DIR = SCRIPT_DIR / "data_recordings" / "accepted" / "p001"

# Length of audio segments
SEGMENT_LENGTH = 1.0


def copy_manual(src_dir: Path = SRC_DIR, dst_dir: Path = DST_DIR) -> Path:
    # Function for copying and segmenting manual recordings

    # Initializing paths
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)

    # Making directories
    if not src_dir.exists():
        raise FileNotFoundError(f"{src_dir} not found")
    dst_dir.mkdir(parents=True, exist_ok=True)


    copied = 0

    # Iterating through all audio files of a speaker
    for p in sorted(src_dir.glob("*.wav")):
        dst = dst_dir / p.name
        if dst.exists():
            continue

        # Segment audio file
        _write_segments_for_member_manual(src_dir/p.name, dst, SEGMENT_LENGTH)
        
        # Keeping track of number of copied files
        copied += 1
    print(f"[copy] p001: {copied} files -> {dst_dir}")
    return dst_dir


def _write_segments_for_member_manual(
    name: str,
    dst_pattern: str,
    seg_length: float
    ):
    # Function to segment audio file

    # Loading audio data
    data, sr = librosa.load(name)

    # Making audio data monochannel
    if getattr(data, "ndim", 1) > 1:
        data = data.mean(axis=1)


    # Removing silence
    energies = np.abs(data)
    threshold = np.percentile(energies, 10)
    mask = energies > threshold
    data = data[mask]


    frames = int(seg_length * sr)

    # If audio files too short for segments
    if len(data) < frames:
        return 0.0
    

    # max_total_frames = int(max_total_seconds * sr)
    total_written_frames = 0
    base, ext = os.path.splitext(dst_pattern)
    idx = 0
    i = 0

    while len(data) - i >= frames:

        # Segmenting the audio file
        segment = data[i : i + frames]

        # Keeping track of used up frames in audio file
        i += frames

        # Saving segment (if not existent)
        dst_path = f"{base}_{idx:03d}{ext}"
        if not os.path.exists(dst_path):
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            sf.write(dst_path, segment, sr)
            total_written_frames += len(segment)
        
        # Index of segment
        idx += 1
    return total_written_frames / float(sr)


if __name__ == "__main__":
    copy_manual()

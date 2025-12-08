#!/usr/bin/env python3
from pathlib import Path
import shutil
import librosa
import os
import soundfile as sf
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR / "p001_manual_recordings"
DST_DIR = SCRIPT_DIR / "data_recordings" / "accepted" / "p001"

SEGMENT_LENGTH = 1.0


def copy_manual(src_dir: Path = SRC_DIR, dst_dir: Path = DST_DIR) -> Path:
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)
    if not src_dir.exists():
        raise FileNotFoundError(f"{src_dir} not found")
    dst_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for p in sorted(src_dir.glob("*.wav")):
        dst = dst_dir / p.name
        if dst.exists():
            continue

        _write_segments_for_member_manual(src_dir/p.name, dst, SEGMENT_LENGTH)
        
        copied += 1
    print(f"[copy] p001: {copied} files -> {dst_dir}")
    return dst_dir

if __name__ == "__main__":
    copy_manual()


def _write_segments_for_member_manual(
    name: str,
    dst_pattern: str,
    seg_len: float
    ):

    data, sr = librosa.load(name)
    if getattr(data, "ndim", 1) > 1:
        data = data.mean(axis=1)

    energies = np.abs(data)
    threshold = np.percentile(energies, 10)

    mask = energies > threshold
    data = data[mask]

    frames = int(seg_len * sr)
    if len(data) < frames:
        return 0.0
    # max_total_frames = int(max_total_seconds * sr)
    total_written_frames = 0
    base, ext = os.path.splitext(dst_pattern)
    idx = 0
    n = len(data)
    i = 0
    while n - i >= frames:
        #seg_len = min(frames_max, n - i, remaining_frames)
        seg_len = frames
        if seg_len < frames and total_written_frames >= 0:
            break
        segment = data[i : i + seg_len]
        i += seg_len
        dst_path = f"{base}_{idx:03d}{ext}"
        if not os.path.exists(dst_path):
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            sf.write(dst_path, segment, sr)
            total_written_frames += len(segment)
            idx += 1
        else:
            idx += 1
    return total_written_frames / float(sr)

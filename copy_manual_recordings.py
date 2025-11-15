#!/usr/bin/env python3
from pathlib import Path
import shutil

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR / "p001_manual_recordings"
DST_DIR = SCRIPT_DIR / "data_recordings" / "accepted" / "p001"

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
        shutil.copy2(p, dst)
        copied += 1
    print(f"[copy] p001: {copied} files -> {dst_dir}")
    return dst_dir

if __name__ == "__main__":
    copy_manual()

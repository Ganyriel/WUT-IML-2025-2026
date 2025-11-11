#!/usr/bin/env python3
# Build accepted/rejected speaker sets from CN-Celeb (OpenSLR SLR82), no HF, no trust_remote_code.
# Output:
#   data/accepted/<speaker_id>/clip_XXXX.wav  (5 speakers, ~15 min each)
#   data/rejected/<speaker_id>/clip_XXXX.wav  (20 speakers, ~5 min each)

import os, tarfile, shutil, re
from pathlib import Path
from collections import defaultdict
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

import numpy as np
import soundfile as sf
from tqdm import tqdm

# ================== CONFIG (edit if you really need) ==================
OUT_DIR = Path("data")
CACHE_DIR = Path(".cache_cnceleb")

# Minimal download: CN-Celeb1 updated bundle (~22 GB). Mirrors from OpenSLR SLR82.
CNCELEB_ARCHIVE_URLS = [
    # EU mirror(s) – if one fails, the next is tried:
    "https://openslr.trmal.net/resources/82/cn-celeb_v2.tar.gz",
    "https://openslr.elda.org/resources/82/cn-celeb_v2.tar.gz",
    # CN mirror (fallback):
    "https://openslr.magicdatatech.com/resources/82/cn-celeb_v2.tar.gz",
]

ACCEPTED_TARGET_SPEAKERS = 5
ACCEPTED_TARGET_MIN      = 15.0  # per accepted speaker
REJECTED_TARGET_SPEAKERS = 20
REJECTED_TARGET_MIN      = 5.0   # per rejected speaker
# =====================================================================

def download(urls, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    last_err = None
    for url in urls:
        try:
            req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urlopen(req) as r, open(dst, "wb") as f, tqdm(
                total=int(r.headers.get("Content-Length", 0)) or None,
                unit="B", unit_scale=True, desc=f"Downloading {dst.name}"
            ) as pbar:
                while True:
                    chunk = r.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
                    pbar.update(len(chunk))
            return
        except (HTTPError, URLError) as e:
            last_err = e
            continue
    raise RuntimeError(f"All mirrors failed. Last error: {last_err}")

def safe_extract(tar_path: Path, to_dir: Path):
    to_dir.mkdir(parents=True, exist_ok=True)
    marker = to_dir / ".extracted_ok"
    if marker.exists():
        return
    with tarfile.open(tar_path, "r:gz") as tar:
        members = [m for m in tar.getmembers() if m.name.endswith(".wav")]
        for m in tqdm(members, desc="Extracting wavs", unit="file"):
            tar.extract(m, path=to_dir)
    marker.touch()

def wav_seconds(p: Path) -> float:
    info = sf.info(str(p))
    if info.frames and info.samplerate:
        return info.frames / float(info.samplerate)
    y, sr = sf.read(str(p))
    return len(y) / float(sr)

def infer_speaker_id(p: Path) -> str:
    # CN-Celeb layout inside the archive typically includes a speaker directory.
    # We'll use the closest parent directory that looks like a speaker ID:
    # e.g., cn-celeb/*/<spk_id>/*/*.wav  -> use that <spk_id>
    parts = p.parts
    # search backwards for a dir name matching CN-Celeb speaker pattern (often alphanum)
    for comp in reversed(parts[:-1]):
        if re.fullmatch(r"[A-Za-z0-9_\-]+", comp):
            # Heuristic: skip common folder names
            if comp.lower() in {"wav", "audio", "dev", "train", "cn-celeb", "cn-celeb1", "cn-celeb2"}:
                continue
            return comp
    # Fallback to parent name
    return p.parent.name

def index_wavs(root: Path):
    wavs = list(root.rglob("*.wav"))
    if not wavs:
        raise RuntimeError(f"No WAVs found under {root.resolve()}")
    by_spk = defaultdict(list)
    for w in wavs:
        spk = infer_speaker_id(w)
        by_spk[spk].append(w)
    return by_spk

def pick_top(by_spk, n):
    totals = []
    for spk, files in by_spk.items():
        s = 0.0
        for f in files:
            try: s += wav_seconds(f)
            except Exception: pass
        totals.append((spk, s))
    totals.sort(key=lambda t: t[1], reverse=True)
    return [spk for spk,_ in totals[:n]], dict(totals)

def copy_until_minutes(spk2files, target_min, dest: Path):
    dest.mkdir(parents=True, exist_ok=True)
    need = target_min * 60.0
    res = {}
    for spk, files in tqdm(spk2files.items(), desc=f"Writing {dest.name}"):
        spk_dir = dest / spk
        spk_dir.mkdir(parents=True, exist_ok=True)
        total = 0.0
        k = 0
        # longest-first to reach minutes fast
        files_sorted = sorted(files, key=lambda p: wav_seconds(p), reverse=True)
        for src in files_sorted:
            if total >= need:
                break
            try:
                d = wav_seconds(src)
            except Exception:
                continue
            dst = spk_dir / f"clip_{k:04d}.wav"
            with open(src, "rb") as r, open(dst, "wb") as w:
                shutil.copyfileobj(r, w)
            total += d
            k += 1
        res[spk] = (total, k)
    return res

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "accepted").mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "rejected").mkdir(parents=True, exist_ok=True)

    # 1) Download + extract minimal CN-Celeb1 bundle (~22 GB)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tar_path = CACHE_DIR / "cn-celeb_v2.tar.gz"
    download(CNCELEB_ARCHIVE_URLS, tar_path)
    extracted = CACHE_DIR / "cn-celeb_v2"
    safe_extract(tar_path, extracted)

    # 2) Index wavs -> group by speaker
    by_spk = index_wavs(extracted)

    # 3) Choose speakers: top-N by available minutes
    acc_ids, totals = pick_top(by_spk, ACCEPTED_TARGET_SPEAKERS)
    remaining = {spk:files for spk,files in by_spk.items() if spk not in set(acc_ids)}
    rej_ids, _ = pick_top(remaining, REJECTED_TARGET_SPEAKERS)

    # 4) Export audio to your structure
    acc_res = copy_until_minutes({s:by_spk[s] for s in acc_ids}, ACCEPTED_TARGET_MIN, OUT_DIR / "accepted")
    rej_res = copy_until_minutes({s:by_spk[s] for s in rej_ids}, REJECTED_TARGET_MIN, OUT_DIR / "rejected")

    # 5) Summary
    def fmt(sec): m,s = divmod(int(sec), 60); return f"{m:02d}m{s:02d}s"
    print("\n=== Accepted ===")
    for spk in acc_ids:
        sec, n = acc_res.get(spk,(0.0,0))
        print(f"{spk}: ~{fmt(sec)}, clips={n} (available ~{fmt(totals.get(spk,0))})")
    print("\n=== Rejected ===")
    for spk in rej_ids:
        sec, n = rej_res.get(spk,(0.0,0))
        print(f"{spk}: ~{fmt(sec)}, clips={n} (available ~{fmt(totals.get(spk,0))})")

    print(f"\nDone. Files under: {OUT_DIR.resolve()}")
    print("Tip: delete the cache to reclaim space:  Remove-Item -Recurse -Force .cache_cnceleb (on Windows PowerShell)")
    
if __name__ == "__main__":
    main()

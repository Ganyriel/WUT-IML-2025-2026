import os
import sys
import io
import re
import zipfile
import random
import requests
import numpy as np
import soundfile as sf
from collections import defaultdict
from tqdm import tqdm

# -------------------- CONFIG --------------------
URL = "https://datashare.ed.ac.uk/bitstream/handle/10283/3038/DR-VCTK.zip"
ACCEPTED_N = 5
REJECTED_N = 20
ACCEPTED_MIN, ACCEPTED_MAX = 15*60, 20*60   # seconds
REJECTED_MIN, REJECTED_MAX = 5*60, 6*60     # seconds (small headroom)
SEED = 42
# ------------------------------------------------

random.seed(SEED)
np.random.seed(SEED)

# Resolve paths RELATIVE TO THIS SCRIPT, not the shell CWD.
SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
ZIP_PATH   = os.path.join(SCRIPT_DIR, "DR-VCTK.zip")
OUT_DIR    = os.path.join(SCRIPT_DIR, "dr_vctk_subset")
os.makedirs(OUT_DIR, exist_ok=True)

def human(n):
    for u in ['B','KB','MB','GB','TB']:
        if n < 1024: return f"{n:.1f} {u}"
        n /= 1024
    return f"{n:.1f} PB"

def download_with_progress(url, dst):
    """Stream download with progress; verify non-empty; return bytes written."""
    with requests.get(url, stream=True, allow_redirects=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        # Write to a temp file first
        tmp = dst + ".part"
        with open(tmp, "wb") as f, tqdm(
            total=total if total > 0 else None,
            unit="B", unit_scale=True, desc="Downloading DR-VCTK"
        ) as bar:
            for chunk in r.iter_content(chunk_size=1024*512):
                if not chunk:
                    continue
                f.write(chunk)
                if total > 0:
                    bar.update(len(chunk))
        # Basic sanity: non-empty + zip signature
        size = os.path.getsize(tmp)
        if size == 0:
            raise RuntimeError("Download resulted in 0 bytes. Check your connection/firewall.")
        # ZIP signature check (PK\x03\x04 at start of first entry OR PK\x05\x06 empty archive)
        with open(tmp, "rb") as f:
            sig = f.read(4)
        if sig[:2] != b"PK":
            raise RuntimeError("Downloaded file does not look like a ZIP (bad signature).")
        os.replace(tmp, dst)
        return size

def ensure_zip():
    if os.path.exists(ZIP_PATH):
        print(f"Found existing ZIP at:\n  {ZIP_PATH}  ({human(os.path.getsize(ZIP_PATH))})")
        return
    print("ZIP not found; starting fresh download…")
    size = download_with_progress(URL, ZIP_PATH)
    print(f"Saved ZIP → {ZIP_PATH}  ({human(size)})")

def list_wavs(zf: zipfile.ZipFile):
    """Return (wav_paths, speakers) inside the ZIP, robust to folder variations."""
    wavs = [n for n in zf.namelist() if n.lower().endswith(".wav")]
    speakers = set()
    for p in wavs:
        m = re.search(r"/(p\d{3})/", p, re.IGNORECASE)
        if m:
            speakers.add(m.group(1).lower())
    speakers = sorted(speakers)
    return wavs, speakers

def main():
    # 1) Download (or reuse) the ZIP
    ensure_zip()

    # 2) Open and index speakers
    try:
        zf = zipfile.ZipFile(ZIP_PATH)
    except zipfile.BadZipFile:
        raise SystemExit("The ZIP is corrupted. Delete DR-VCTK.zip and run again.")
    wav_paths, speakers = list_wavs(zf)
    if not wav_paths:
        raise SystemExit("No .wav files found inside the ZIP. The download may be incomplete.")
    if len(speakers) == 0:
        raise SystemExit("Could not parse speaker IDs. ZIP layout unexpected.")

    random.shuffle(speakers)
    accepted_spk = speakers[:ACCEPTED_N]
    rejected_spk = speakers[ACCEPTED_N:ACCEPTED_N+REJECTED_N]
    print(f"Speakers found: {len(speakers)} | Using accepted={accepted_spk} rejected={rejected_spk[:5]}...")

    targets = {
        "accepted": {"spks": accepted_spk, "min": ACCEPTED_MIN, "max": ACCEPTED_MAX},
        "rejected": {"spks": rejected_spk, "min": REJECTED_MIN, "max": REJECTED_MAX},
    }
    dur = defaultdict(float)

    def read_audio_from_zip(member):
        with zf.open(member) as f:
            data, sr = sf.read(io.BytesIO(f.read()), dtype="float32", always_2d=False)
        return data, sr

    # 3) Build subsets
    for tag, conf in targets.items():
        for spk in conf["spks"]:
            # Collect this speaker's files
            spk_files = [p for p in wav_paths if re.search(fr"/{spk}/", p, re.IGNORECASE)]
            random.shuffle(spk_files)
            for p in spk_files:
                if dur[(tag, spk)] >= conf["min"]:
                    break
                data, sr = read_audio_from_zip(p)
                remaining = conf["max"] - dur[(tag, spk)]
                if remaining <= 0:
                    break
                clip_sec = len(data) / sr
                take = min(clip_sec, remaining)
                if take <= 0:
                    continue
                if clip_sec > take:
                    data = data[: int(take * sr)]
                    clip_sec = take
                out_dir = os.path.join(OUT_DIR, tag, spk)
                os.makedirs(out_dir, exist_ok=True)
                out_path = os.path.join(out_dir, os.path.basename(p))
                sf.write(out_path, data, sr)
                dur[(tag, spk)] += clip_sec
            print(f"{tag} {spk}: {dur[(tag, spk)]:.1f}s")

    # 4) Final checks
    for spk in accepted_spk:
        if dur[("accepted", spk)] < ACCEPTED_MIN:
            raise SystemExit(f"Not enough audio for accepted {spk} "
                             f"({dur[('accepted', spk)]:.1f}s). Try re-running or adjust caps.")
    for spk in rejected_spk:
        if dur[("rejected", spk)] < REJECTED_MIN:
            raise SystemExit(f"Not enough audio for rejected {spk} "
                             f"({dur[('rejected', spk)]:.1f}s). Try re-running or adjust caps.")

    print("\n✅ DONE")
    print(f"Output folder:\n  {OUT_DIR}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # Make errors painfully obvious
        print("\nERROR:", e)
        print(f"\nDebug info:\n  Script dir: {SCRIPT_DIR}\n  ZIP path:   {ZIP_PATH}\n  Out dir:    {OUT_DIR}")
        raise

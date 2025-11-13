#!/usr/bin/env python3
import io, os, re, random, zipfile
from collections import defaultdict
from typing import Dict, Iterable, List, Tuple
import soundfile as sf
from tqdm import tqdm

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
ZIP_PATH = os.path.join(SCRIPT_DIR, "DR-VCTK.zip")
OUT_DIR = os.path.join(SCRIPT_DIR, "data_recordings")

PREFERRED_DIR_MARKER = "device-recorded_trainset_wav_16k"
ACCEPTED_N = 5
REJECTED_N = 20
ACCEPTED_MIN, ACCEPTED_MAX = 15 * 60, 20 * 60
REJECTED_MIN, REJECTED_MAX = 5 * 60, 6 * 60
RESERVED_SPEAKERS = {"p001"}
SEED = 42
random.seed(SEED)

def _speaker_id(path: str) -> str:
    b = os.path.basename(path)
    m = re.match(r"([pP]\d{3})[_-]", b)
    if m: return m.group(1).lower()
    m2 = re.search(r"/(p\d{3})/", path, re.IGNORECASE)
    return m2.group(1).lower() if m2 else ""

def _pick_trainset_dirs(names: Iterable[str]) -> List[str]:
    dirs = set()
    for n in names:
        if n.lower().endswith(".wav") and PREFERRED_DIR_MARKER in n:
            parts = n.split("/")
            for i in range(1, len(parts)):
                sub = "/".join(parts[:i])
                if PREFERRED_DIR_MARKER in sub:
                    dirs.add(sub); break
    return sorted(dirs)

def _iter_wavs(zf: zipfile.ZipFile, use_dirs: List[str]) -> Iterable[str]:
    for n in zf.namelist():
        if n.lower().endswith(".wav") and any(n.startswith(d) for d in use_dirs):
            yield n

def _duration_from_member(zf: zipfile.ZipFile, member: str) -> float:
    with zf.open(member) as f:
        data, sr = sf.read(io.BytesIO(f.read()), dtype="float32", always_2d=False)
    return len(data) / float(sr)

def _write_truncated(zf: zipfile.ZipFile, member: str, dst_path: str, max_seconds: float) -> float:
    if os.path.exists(dst_path): 
        return 0.0
    with zf.open(member) as f:
        data, sr = sf.read(io.BytesIO(f.read()), dtype="float32", always_2d=False)
    max_frames = int(max_seconds * sr)
    if max_frames < len(data):
        data = data[:max_frames]
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    sf.write(dst_path, data, sr)
    return len(data) / float(sr)

def _select_speakers(spk_to_files: Dict[str, List[str]], need: int, min_seconds: float, zf: zipfile.ZipFile) -> List[str]:
    c = [s for s in spk_to_files.keys() if s not in RESERVED_SPEAKERS]
    random.shuffle(c)
    chosen: List[str] = []
    for spk in c:
        if len(chosen) >= need: break
        total = 0.0
        for m in spk_to_files[spk]:
            total += _duration_from_member(zf, m)
            if total >= min_seconds:
                chosen.append(spk); break
    return chosen

def build_subset(
    zip_path: str = ZIP_PATH,
    out_dir: str = OUT_DIR,
    accepted_n: int = ACCEPTED_N,
    rejected_n: int = REJECTED_N,
    accepted_caps: Tuple[int, int] = (ACCEPTED_MIN, ACCEPTED_MAX),
    rejected_caps: Tuple[int, int] = (REJECTED_MIN, REJECTED_MAX),
) -> str:
    zip_path = os.path.abspath(zip_path)
    out_dir = os.path.abspath(out_dir)
    acc_dir = os.path.join(out_dir, "accepted")
    rej_dir = os.path.join(out_dir, "rejected")
    os.makedirs(acc_dir, exist_ok=True)
    os.makedirs(rej_dir, exist_ok=True)
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"ZIP not found: {zip_path}")

    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        use_dirs = _pick_trainset_dirs(names)
        if not use_dirs:
            raise RuntimeError(f"Expected '{PREFERRED_DIR_MARKER}' in ZIP contents.")
        spk_to_files: Dict[str, List[str]] = defaultdict(list)
        for member in _iter_wavs(zf, use_dirs):
            spk = _speaker_id(member)
            if spk: spk_to_files[spk].append(member)

        accepted = _select_speakers(spk_to_files, accepted_n, accepted_caps[0], zf)
        remaining = [s for s in spk_to_files.keys() if s not in accepted and s not in RESERVED_SPEAKERS]
        spk_to_files_remaining = {s: spk_to_files[s] for s in remaining}
        rejected = _select_speakers(spk_to_files_remaining, rejected_n, rejected_caps[0], zf)
        if len(accepted) < accepted_n or len(rejected) < rejected_n:
            raise RuntimeError("Not enough speakers with required minutes. Adjust caps or counts.")

        caps = {"accepted": (accepted, accepted_caps), "rejected": (rejected, rejected_caps)}
        written = defaultdict(float)
        for tag, (spk_list, (min_s, max_s)) in caps.items():
            for spk in spk_list:
                files = spk_to_files[spk][:]
                random.shuffle(files)
                for member in tqdm(files, desc=f"{tag} {spk}", leave=False):
                    if written[(tag, spk)] >= min_s: break
                    remaining_s = max_s - written[(tag, spk)]
                    if remaining_s <= 0: break
                    dst = os.path.join(out_dir, tag, spk, os.path.basename(member))
                    added = _write_truncated(zf, member, dst, remaining_s)
                    written[(tag, spk)] += added
                print(f"{tag} {spk}: {written[(tag, spk)]:.1f}s")
        print("[build] Output:", out_dir)
        return out_dir

if __name__ == "__main__":
    build_subset()

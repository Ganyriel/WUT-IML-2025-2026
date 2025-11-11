#!/usr/bin/env python3
import os, io, re, zipfile, random
from collections import defaultdict
import soundfile as sf
from tqdm import tqdm

# ------------------------- CONFIG -------------------------
SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
ZIP_PATH   = os.path.join(SCRIPT_DIR, "DR-VCTK.zip")
OUT_DIR    = os.path.join(SCRIPT_DIR, "dr_vctk_subset")

# selection targets
ACCEPTED_N = 5
REJECTED_N = 20
ACCEPTED_MIN, ACCEPTED_MAX = 15*60, 20*60   # seconds
REJECTED_MIN, REJECTED_MAX = 5*60,  6*60

SEED = 42                 # change to reshuffle speakers
PREFER = "trainset"       # prefer device-recorded_trainset_wav_16k
# ----------------------------------------------------------

random.seed(SEED)

def _seconds(frames, sr): return frames / float(sr)

def _pick_device_dirs(namelist):
    """
    Return list of directory prefixes to use, preferring:
      device-recorded_trainset_wav_16k  then device-recorded_testset_wav_16k
    Works regardless of an extra top-level 'DR-VCTK/' folder.
    """
    dirs = set()
    for n in namelist:
        if not n.lower().endswith(".wav"): 
            continue
        parts = n.split("/")
        # keep two/three leading parts to identify the bucket name
        for i in range(1, min(4, len(parts))):
            sub = "/".join(parts[:i]).lower()
            if "device-recorded" in sub and "wav_16k" in sub:
                dirs.add("/".join(parts[:i]))
    # Prefer trainset
    train = [d for d in dirs if "device-recorded_trainset_wav_16k" in d.lower()]
    test  = [d for d in dirs if "device-recorded_testset_wav_16k" in d.lower()]
    chosen = train if (PREFER == "trainset" and train) else (test if test else train)
    return sorted(chosen), sorted(list(dirs))

def _speaker_from_filename(path):
    # filenames like p232_003.wav → speaker = p232
    m = re.match(r"([pP]\d{3})[_-]", os.path.basename(path))
    return m.group(1).lower() if m else None

def _iter_wavs_in_dirs(zf, use_dirs):
    """Yield (zip_member_path, speaker_id) for wavs inside selected directories."""
    for n in zf.namelist():
        if not n.lower().endswith(".wav"):
            continue
        if any(n.startswith(d) for d in use_dirs):
            spk = _speaker_from_filename(n)
            if spk:
                yield n, spk

def _duration_of_member(zf, member):
    """Return duration (seconds) without writing to disk."""
    with zf.open(member) as f:
        data, sr = sf.read(io.BytesIO(f.read()), dtype="float32", always_2d=False)
    return len(data) / sr

def _write_member_truncated(zf, member, dst_path, max_seconds=None):
    """Write WAV to dst_path, truncating to max_seconds if set. Return seconds written."""
    with zf.open(member) as f:
        data, sr = sf.read(io.BytesIO(f.read()), dtype="float32", always_2d=False)
    if max_seconds is not None:
        max_frames = int(max_seconds * sr)
        if len(data) > max_frames:
            data = data[:max_frames]
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    sf.write(dst_path, data, sr)
    return len(data) / sr

def main():
    if not os.path.exists(ZIP_PATH):
        raise SystemExit(f"[error] DR-VCTK.zip not found next to the script:\n  {ZIP_PATH}")

    with zipfile.ZipFile(ZIP_PATH) as zf:
        names = zf.namelist()
        use_dirs, all_device_dirs = _pick_device_dirs(names)
        if not use_dirs:
            if all_device_dirs:
                raise SystemExit(f"[error] Found device-recorded dirs {all_device_dirs} but couldn't pick one.")
            raise SystemExit("[error] No device-recorded *_wav_16k folders found in the ZIP.")
        print("[info] Using device directories:")
        for d in use_dirs: print("  -", d)

        # Map speaker -> list of member paths (device-recorded only)
        spk_to_files = defaultdict(list)
        for member, spk in _iter_wavs_in_dirs(zf, use_dirs):
            spk_to_files[spk].append(member)

        speakers = sorted(spk_to_files.keys())
        if len(speakers) == 0:
            raise SystemExit("[error] No speakers detected (filenames should start with p###_).")

        # Shuffle for reproducibility
        random.shuffle(speakers)

        # Helper to check if a speaker has at least X seconds available
        def has_minutes(spk, min_seconds):
            tot = 0.0
            # iterate a few files until we exceed the threshold (avoid summing all files)
            for m in spk_to_files[spk]:
                tot += _duration_of_member(zf, m)
                if tot >= min_seconds:
                    return True
            return False

        # Pick accepted speakers (need >= ACCEPTED_MIN available)
        accepted = []
        for spk in speakers:
            if len(accepted) >= ACCEPTED_N:
                break
            if has_minutes(spk, ACCEPTED_MIN):
                accepted.append(spk)

        # Pick rejected speakers from the remaining set
        remaining = [s for s in speakers if s not in accepted]
        rejected = []
        for spk in remaining:
            if len(rejected) >= REJECTED_N:
                break
            if has_minutes(spk, REJECTED_MIN):
                rejected.append(spk)

        if len(accepted) < ACCEPTED_N:
            raise SystemExit(f"[error] Only found {len(accepted)}/{ACCEPTED_N} speakers with >= {ACCEPTED_MIN/60:.0f} min.")
        if len(rejected) < REJECTED_N:
            raise SystemExit(f"[error] Only found {len(rejected)}/{REJECTED_N} speakers with >= {REJECTED_MIN/60:.0f} min.")

        print(f"[plan] accepted: {accepted}")
        print(f"[plan] rejected: {rejected}")

        # Write out audio with caps (truncate last file if needed)
        caps = {
            "accepted": (ACCEPTED_MIN, ACCEPTED_MAX, accepted),
            "rejected": (REJECTED_MIN, REJECTED_MAX, rejected),
        }
        written = defaultdict(float)

        for tag, (min_s, max_s, spk_list) in caps.items():
            for spk in spk_list:
                files = spk_to_files[spk][:]
                random.shuffle(files)
                bar = tqdm(files, desc=f"{tag} {spk}", leave=False)
                for member in bar:
                    if written[(tag, spk)] >= min_s:
                        break
                    remaining_sec = max_s - written[(tag, spk)]
                    if remaining_sec <= 0:
                        break
                    dst_dir = os.path.join(OUT_DIR, tag, spk)
                    dst_path = os.path.join(dst_dir, os.path.basename(member))
                    added = _write_member_truncated(zf, member, dst_path, max_seconds=remaining_sec)
                    written[(tag, spk)] += added
                print(f"{tag} {spk}: {written[(tag, spk)]:.1f}s")

        # Final checks
        for spk in accepted:
            if written[("accepted", spk)] < ACCEPTED_MIN:
                raise SystemExit(f"[fail] accepted {spk} only {written[('accepted', spk)]:.1f}s")
        for spk in rejected:
            if written[("rejected", spk)] < REJECTED_MIN:
                raise SystemExit(f"[fail] rejected {spk} only {written[('rejected', spk)]:.1f}s")

    print("\n✅ DONE")
    print(f"Output:\n  {OUT_DIR}")
    print("Layout:\n  dr_vctk_subset/accepted/<p###>/*.wav\n  dr_vctk_subset/rejected/<p###>/*.wav")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import io, os, re, zipfile, shutil
from collections import defaultdict
from typing import Dict, Iterable, List, Tuple
import soundfile as sf
from tqdm import tqdm
import numpy as np
import librosa

# Directories
SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
ZIP_PATH = os.path.join(SCRIPT_DIR, "../DR-VCTK.zip")
OUT_DIR = os.path.join(SCRIPT_DIR, "../data_recordings")

PREFERRED_DIR_MARKER = "device-recorded_trainset_wav_16k"

# Accepted speakers + selection
ACCEPTED_N = 5
REJECTED_N = 20
ACCEPTED_MIN, ACCEPTED_MAX = 15 * 60, 16 * 60
REJECTED_MIN, REJECTED_MAX = 5 * 60, 6 * 60

ACCEPTED_SELECT_MARGIN = 1200.0 # default : 600.0
REJECTED_SELECT_MARGIN = 400.0 # default : 200.0

# Add manual recordings here
RESERVED_SPEAKERS = {"p001"}

# Length of audio segments 
SEGMENT_LENGTH = 1.0 # default 1.0, can be changed to 3.0

def _pick_trainset_dirs(names: Iterable[str]) -> List[str]:
    # Function for finding the data from the online dataset zip file

    dirs = set()
    for n in names:
        if PREFERRED_DIR_MARKER in n:
            parts = n.split("/")
            if parts:
                dirs.add(parts[0])
    return sorted(dirs)


def _iter_wavs(zf: zipfile.ZipFile, use_dirs: Iterable[str]) -> Iterable[str]:
    use_dirs = tuple(use_dirs)
    for n in sorted(zf.namelist()):
        if not n.lower().endswith(".wav"):
            continue
        if use_dirs and not any(n.startswith(d + "/") for d in use_dirs):
            continue
        yield n


def _speaker_id(member: str) -> str:
    # Function for categorizing audio files

    m = re.search(r"/(p\d{3})/", member)
    if m:
        return m.group(1)
    m = re.search(r"(p\d{3})", member)
    if m:
        return m.group(1)
    return ""


def _probe_speaker_duration(
    zf: zipfile.ZipFile,
    members: Iterable[str],
    target_seconds: float,
    seg_length: float,
) -> float:
    # File for probing speaker duration (without cutting silence)
    # TODO see why it shows high number for available audio files

    total = 0.0
    members = sorted(members)
    
    # Iterating through all files of a speaker
    for member in members:

        # Taking the audio data from the zip file
        with zf.open(member) as f:
            data, sr = sf.read(io.BytesIO(f.read()), dtype="float32", always_2d=False)

        # Making the data monochannel   
        if getattr(data, "ndim", 1) > 1:
            data = data.mean(axis=1)   

        # Set length of audio segment in frames
        frames = int(seg_length * sr)
        # Calculate length of audio file in frames
        n = len(data)
        
        # if audio too short
        if n < frames:
            continue

        # Count for used up frames in audio file
        i = 0

        # Counting the number of segments that fit into the audio file
        while n - i >= frames and total < target_seconds:
            total += seg_length
            i += frames
            if total >= target_seconds:
                break
        if total >= target_seconds:
            break
    return total


def _write_segments_for_member(
    zf: zipfile.ZipFile,
    member: str,
    dst_pattern: str,
    seg_length: float,
    max_total_seconds: float,
) -> float:
    # File for segmenting speaker files into segments

    # If speaker gets no duration
    if max_total_seconds <= 0.0:
        return 0.0
    
    # Extracting data from files
    with zf.open(member) as f:
        data, sr = sf.read(io.BytesIO(f.read()), dtype="float32", always_2d=False)

    # Making the audio data monochannel
    if getattr(data, "ndim", 1) > 1:
        data = data.mean(axis=1)

    # Filtering silence
    energies = np.abs(data)

    # Experimental silence filtering for audio files with only silence/noise (not tested)
    # thresh_quiet = 0.8
    # if np.max(energies) <= thresh_quiet : 
    #     return 0.0

    threshold = np.percentile(energies, 10)
    mask = energies > threshold
    # print("Before: ", len(data)/sr)
    data = data[mask]
    # print("After: ", len(data)/sr)

    # Setting segment length in frames
    frames = int(seg_length * sr)

    # Length of audio file in frames
    n = len(data)

    # If audio file too short
    if n < frames:
        return 0.0

    # Maximal amount of frames we want
    max_total_frames = int(max_total_seconds * sr)

    # Amount of written frames
    total_written_frames = 0

    # base of path
    base, ext = os.path.splitext(dst_pattern)

    # index of audio file for speaker
    idx = 0
    
    # Written frames from audio file
    i = 0

    # Segmenting audio file
    while n - i >= frames and total_written_frames < max_total_frames:

        # remaining frames we need
        remaining_frames = max_total_frames - total_written_frames

        # We will have more frames than we need
        # TODO this part needed?
        if remaining_frames <= frames:
            break

        # Segmenting
        segment = data[i : i + frames]

        # Keeping count of frames used
        i += frames

        # Adjusting path
        dst_path = f"{base}_{idx:03d}{ext}"

        # Categorizing
        if not os.path.exists(dst_path):
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            sf.write(dst_path, segment, sr)
            total_written_frames += len(segment)
            
        idx += 1
    return total_written_frames / float(sr)


def _select_speakers(
    zf: zipfile.ZipFile,
    spk_to_files: Dict[str, List[str]],
    need: int,
    min_seconds: float,
) -> List[str]:
    # A function to choose speakers who have more recordings than minimum

    candidates = sorted(s for s in spk_to_files.keys() if s not in RESERVED_SPEAKERS)
    chosen: List[str] = []
    for spk in candidates:
        if len(chosen) >= need:
            break
        files = spk_to_files[spk]
        total = _probe_speaker_duration(zf, files, min_seconds, SEGMENT_LENGTH)
        print(f"speaker: {spk}, total duration: {total}")
        print("min_seconds: ", min_seconds)
        print()
        if total >= min_seconds:
            chosen.append(spk)
    return chosen


def build_subset(
    zip_path: str = ZIP_PATH,
    out_dir: str = OUT_DIR,
    accepted_n: int = ACCEPTED_N,
    rejected_n: int = REJECTED_N,
    accepted_caps: Tuple[int, int] = (ACCEPTED_MIN, ACCEPTED_MAX),
    rejected_caps: Tuple[int, int] = (REJECTED_MIN, REJECTED_MAX),
) -> str:
    # Function to build the directories and copying and segmenting the audio files form zip file

    # Initializing paths
    zip_path = os.path.abspath(zip_path)
    out_dir = os.path.abspath(out_dir)
    acc_dir = os.path.join(out_dir, "accepted")
    rej_dir = os.path.join(out_dir, "rejected")

    # Creating directories
    for d in (acc_dir, rej_dir):
        if os.path.exists(d):
            shutil.rmtree(d)
    os.makedirs(acc_dir, exist_ok=True)
    os.makedirs(rej_dir, exist_ok=True)

    # If zip not found
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"ZIP not found: {zip_path}")
    
    # Checking if values for accepted/rejected speakers are reasonable
    if accepted_caps[1] < accepted_caps[0]:
        raise ValueError("ACCEPTED_MAX must be >= ACCEPTED_MIN")
    if rejected_caps[1] < rejected_caps[0]:
        raise ValueError("REJECTED_MAX must be >= REJECTED_MIN")
    

    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        use_dirs = _pick_trainset_dirs(names)
        if not use_dirs:
            raise RuntimeError(f"Expected '{PREFERRED_DIR_MARKER}' in ZIP contents.")
        spk_to_files: Dict[str, List[str]] = defaultdict(list)
        for member in _iter_wavs(zf, use_dirs):
            spk = _speaker_id(member)
            if spk:
                spk_to_files[spk].append(member)
        accepted_select_min = accepted_caps[0] + ACCEPTED_SELECT_MARGIN
        rejected_select_min = rejected_caps[0] + REJECTED_SELECT_MARGIN

        
        accepted = _select_speakers(zf, spk_to_files, accepted_n, accepted_select_min)
        remaining = [s for s in spk_to_files.keys() if s not in accepted and s not in RESERVED_SPEAKERS]
        spk_to_files_remaining = {s: spk_to_files[s] for s in remaining}
        rejected = _select_speakers(zf, spk_to_files_remaining, rejected_n, rejected_select_min)
        print("Number of accepted: ", len(accepted))
        print()
        print("Number of rejected: ", len(rejected))
        print()
        if len(accepted) < accepted_n or len(rejected) < rejected_n:
            raise RuntimeError("Not enough speakers with required minutes. Adjust caps or counts.")
        caps = {"accepted": (accepted, accepted_caps), "rejected": (rejected, rejected_caps)}
        written = defaultdict(float)
        for tag, (spk_list, (min_s, max_s)) in caps.items():
            for spk in spk_list:
                files = sorted(spk_to_files[spk])
                for member in tqdm(files, desc=f"{tag} {spk}", leave=False):
                    if written[(tag, spk)] >= min_s:
                        break
                    remaining_s = max_s - written[(tag, spk)]
                    if remaining_s <= 0:
                        break
                    dst = os.path.join(out_dir, tag, spk, os.path.basename(member))
                    added = _write_segments_for_member(
                        zf,
                        member,
                        dst,
                        SEGMENT_LENGTH,
                        remaining_s
                    )
                    written[(tag, spk)] += added
                if written[(tag, spk)] < min_s:
                    raise RuntimeError(f"Speaker {spk} in {tag} did not reach minimum seconds: {written[(tag, spk)]:.1f}s")
                print(f"{tag} {spk}: {written[(tag, spk)]:.1f}s")
        print("[build] Output:", out_dir)
        return out_dir


if __name__ == "__main__":
    build_subset()

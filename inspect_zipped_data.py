#!/usr/bin/env python3
import os, sys, zipfile, re
from collections import defaultdict, Counter

# ------------------------------------------------------------
# defaults
# ------------------------------------------------------------
SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
ZIP_PATH   = os.path.join(SCRIPT_DIR, "DR-VCTK.zip")
SAMPLE_PER_DIR = 10       # how many file names to show per folder
ONLY_WAVS = False          # set True to hide non-wav files
# ------------------------------------------------------------

def human(n):
    for u in ["B","KB","MB","GB","TB"]:
        if n < 1024: return f"{n:.1f}{u}"
        n /= 1024
    return f"{n:.1f}PB"

def build_index(z):
    names = [n.rstrip("/") for n in z.namelist() if n]
    dirs = set()
    files_by_dir = defaultdict(list)
    sizes = {}
    for n in names:
        if n.endswith("/"): continue
        d = os.path.dirname(n)
        files_by_dir[d].append(os.path.basename(n))
        try:
            sizes[n] = z.getinfo(n).file_size
        except KeyError:
            sizes[n] = 0
        parts = d.split("/") if d else []
        for i in range(1, len(parts)+1):
            dirs.add("/".join(parts[:i]))
    dirs.add("")  # root
    return sorted(dirs, key=lambda p:(p.count("/"),p)), files_by_dir, sizes

def print_tree(dir_list, files_by_dir, sizes, sample_per_dir, only_wavs):
    for d in dir_list:
        depth = 0 if not d else d.count("/")
        indent = "  " * depth
        label = d if d else "<ZIP-ROOT>"
        files = files_by_dir.get(d, [])
        if only_wavs:
            files = [f for f in files if f.lower().endswith(".wav")]
        files_sorted = sorted(files)
        n = len(files_sorted)
        sample = files_sorted[:sample_per_dir]
        clipped = n > sample_per_dir
        print(f"{indent}{label}/  (files: {n})")
        for f in sample:
            path = f"{d}/{f}" if d else f
            print(f"{indent}  - {f}  [{human(sizes.get(path,0))}]")
        if clipped:
            print(f"{indent}  … +{n-len(sample)} more")

def detect_summary(z):
    names = [n for n in z.namelist() if n.lower().endswith(".wav")]
    print("\n[SUMMARY]")
    print(f"  total .wav files: {len(names)}")
    top = sorted({n.split('/')[0] for n in names if '/' in n})
    if top:
        print("  top-level dirs:", ", ".join(top))
    subsets = Counter(n.split('/')[1] for n in names if n.count('/')>=1)
    if subsets:
        print("  first-level dirs with wavs:")
        for k,v in subsets.most_common():
            print(f"    - {k}: {v} files")
    ids=set()
    for n in names:
        base=os.path.basename(n)
        m=re.match(r"([pP]\d{3})[_-]",base)
        if m: ids.add(m.group(1).lower())
        else:
            for seg in n.split("/"):
                if re.fullmatch(r"[pP]\d{3}",seg):
                    ids.add(seg.lower()); break
    ex=sorted(list(ids))[:20]
    print(f"  detected speaker ids: {len(ids)} (examples: {', '.join(ex) if ex else 'none'})")

def main():
    if not os.path.exists(ZIP_PATH):
        sys.exit(f"[error] DR-VCTK.zip not found in {SCRIPT_DIR}")
    with zipfile.ZipFile(ZIP_PATH) as z:
        dirs, files_by_dir, sizes = build_index(z)
        print_tree(dirs, files_by_dir, sizes, SAMPLE_PER_DIR, ONLY_WAVS)
        detect_summary(z)
    print("\n✅ inspection done")

if __name__ == "__main__":
    main()

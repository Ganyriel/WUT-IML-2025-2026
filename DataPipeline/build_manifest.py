from pathlib import Path
import csv
import os

SCRIPT_DIR = Path(__file__).resolve().parent
SUBSET_ROOT = os.path.join(SCRIPT_DIR, "../data_recordings")
CSV_PATH = os.path.join(SUBSET_ROOT, "manifest.csv")

def build_manifest(subset_root: Path = SUBSET_ROOT, csv_path: Path = CSV_PATH) -> Path:
    subset_root = Path(subset_root).resolve()
    csv_path = Path(csv_path).resolve()
    if not subset_root.exists():
        raise FileNotFoundError(f"{subset_root} not found")
    rows = []
    for label_dir, label in [("accepted", 1), ("rejected", 0)]:
        base = subset_root / label_dir
        if not base.exists():
            base.mkdir(parents=True, exist_ok=True)
        for wav in sorted(base.rglob("*.wav")):
            speaker_id = wav.parent.name
            rel_path = wav.relative_to(subset_root).as_posix()
            rows.append((speaker_id, rel_path, label))
    rows.sort(key=lambda r: (r[2], r[0], r[1]))
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["speaker_id", "file_path", "label"])
        w.writerows(rows)
    print(f"[manifest] {len(rows)} rows -> {csv_path}")
    return csv_path

if __name__ == "__main__":
    build_manifest()

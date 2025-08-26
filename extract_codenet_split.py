# extract_codenet_split.py
# Location: ErrorDetectionModel/extract_codenet_split.py
#
# Reads split ID files from data/2021-12-29-f=0.01/
# Resolves each (problem_id, submission_id) to a real source file under
#   <CODENET_PATH>/data/<problem_id>/<language>/<submission_id>.<filename_ext>
# using <CODENET_PATH>/metadata/<problem_id>.csv.
# Writes consolidated JSONL to data/codenet_extract/ and optionally copies raw files.

import os
import json
import csv
import re
import shutil
from pathlib import Path
from typing import Dict, Tuple, List, Optional

# -------------------
# CONFIG (your setup)
# -------------------
# IMPORTANT: set CODENET_PATH to the Project_CodeNet ROOT that contains:
#   <CODENET_PATH>/
#     data/  metadata/  problem_descriptions/  derived/  README
CODENET_ROOT = Path(os.environ["CODENET_PATH"])   # e.g., E:/Project_CodeNet
SPLIT_DIR    = Path("data/2021-12-29-f=0.01")     # where train-ids.json, valid-ids.json, test-ids.json live
OUT_DIR      = Path("data/codenet_extract")       # where outputs go
COPY_RAW     = True                                # also copy original source files for inspection
FILTER_LANGS = None                                # e.g., set to {"C++"} to keep only C++

# Create output dirs
OUT_DIR.mkdir(parents=True, exist_ok=True)
(OUT_DIR / "raw").mkdir(parents=True, exist_ok=True)

# Accept either "p00023_s006384060" or {"problem_id":"p00023","submission_id":"s006384060"}
ID_PAT = re.compile(r'^(p\d{5})_s(\d{1,})$')

def load_id_list(fpath: Path) -> List[Tuple[str, str]]:
    """Load a split file and return list of (problem_id, submission_id)."""
    with fpath.open("r", encoding="utf-8") as f:
        data = json.load(f)

    ids: List[Tuple[str, str]] = []
    for item in data:
        # Case 1: string "p00023_s006384060"
        if isinstance(item, str):
            m = ID_PAT.match(item)
            if not m:
                raise ValueError(f"Unrecognized id format (string): {item}")
            pid, sid = m.group(1), "s" + m.group(2)
            ids.append((pid, sid))
            continue

        # Case 2: dict {"problem_id": "...", "submission_id": "..."}
        if isinstance(item, dict):
            pid = (item.get("problem_id") or "").strip()
            sid = (item.get("submission_id") or "").strip()
            if not pid or not sid:
                raise ValueError(f"Dict id missing keys: {item}")
            if not sid.startswith("s"):
                sid = "s" + sid  # normalize if missing prefix
            ids.append((pid, sid))
            continue

        # Case 3: list/tuple ["p00023","s006384060"]
        if isinstance(item, (list, tuple)) and len(item) == 2:
            pid = str(item[0]).strip()
            sid = str(item[1]).strip()
            if not pid or not sid:
                raise ValueError(f"List id missing values: {item}")
            if not sid.startswith("s"):
                sid = "s" + sid  # normalize
            ids.append((pid, sid))
            continue

        raise ValueError(f"Unsupported id entry: {item}")

    return ids


def build_submission_index(problem_id: str) -> Dict[str, Tuple[str, str, Dict[str, str]]]:
    """
    Build index for a problem:
      submission_id -> (language, filename, full_metadata_row)

    NOTE: metadata has 'filename_ext' (e.g., 'cpp'), not a full 'filename'.
    We derive filename as f"{submission_id}.{filename_ext}".
    """
    meta_csv = CODENET_ROOT / "metadata" / f"{problem_id}.csv"
    if not meta_csv.exists():
        raise FileNotFoundError(f"Missing metadata csv: {meta_csv}")

    idx: Dict[str, Tuple[str, str, Dict[str, str]]] = {}
    with meta_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = row.get("submission_id") or row.get("id")
            if not sid:
                continue
            # submission_id is used here to match IDs from split files
            # to their corresponding language & filename (via filename_ext) in metadata.
            language = (row.get("language") or "").strip()            # e.g., "C++"
            if FILTER_LANGS and language not in FILTER_LANGS:
                continue
            ext = (row.get("filename_ext") or "").strip().lstrip(".") # e.g., "cpp"
            filename = f"{sid}.{ext}" if ext else sid                  # derived filename
            idx[sid] = (language, filename, row)
    return idx

def resolve_source_path(problem_id: str, submission_id: str, language: str, filename: str) -> Path:
    # submission_id is NOT needed here; it's encoded inside the filename already (e.g., s300682070.cpp).
    # Canonical CodeNet path:
    #   data/<problem_id>/<language>/<filename>
    base = CODENET_ROOT / "data" / problem_id / language
    path = base / filename
    if path.exists():
        return path
    # Safety fallback: if extension is somehow missing in the filesystem, try without it.
    try_no_ext = base / submission_id
    if try_no_ext.exists():
        return try_no_ext
    return path  # will fail later with exists() check

def map_status_to_label(status: str) -> str:
    """
    OPTIONAL: Map CodeNet 'status' to a coarse label. Tweak to match your trainer.
    """
    if not status:
        return "unknown"
    s = status.strip().lower()
    mapping = {
        "accepted": "no_error",
        "runtime error": "runtime_error",
        "time limit exceeded": "timeout",
        "memory limit exceeded": "memory_limit",
        "compile error": "compile_error",
        "wrong answer": "wrong_answer",
        "output limit exceeded": "output_limit",
        "judge not available": "infra_issue",
        "internal error": "infra_issue",
        "judge system error": "infra_issue",
        "wa: presentation error": "presentation_error",
        "waiting for judging": "pending",
        "waiting for re-judging": "pending",
    }
    return mapping.get(s, s.replace(" ", "_"))

def extract_split(split_name: str, id_file: Path) -> None:
    print(f"[{split_name}] Reading IDs from {id_file}")
    if not id_file.exists():
        raise FileNotFoundError(f"ID file not found: {id_file}")

    ids = load_id_list(id_file)
    print(f"[{split_name}] Loaded {len(ids)} IDs")

    out_jsonl = OUT_DIR / f"{split_name}.jsonl"
    raw_root = OUT_DIR / "raw" / split_name

    count_written = 0
    missing_meta = 0
    missing_file = 0

    # Group by problem to avoid reopening the same CSV many times
    by_problem: Dict[str, List[str]] = {}
    for pid, sid in ids:
        by_problem.setdefault(pid, []).append(sid)

    with out_jsonl.open("w", encoding="utf-8") as out:
        for pid, sids in by_problem.items():
            try:
                idx = build_submission_index(pid)
            except FileNotFoundError as e:
                print(f"[WARN] {e}")
                missing_meta += len(sids)
                continue

            for sid in sids:
                if sid not in idx:
                    print(f"[WARN] {pid}/{sid} not found in metadata; skipping")
                    missing_meta += 1
                    continue

                language, filename, meta = idx[sid]
                src_path = resolve_source_path(pid, sid, language, filename)
                if not src_path.exists():
                    print(f"[WARN] Missing file: {src_path}")
                    missing_file += 1
                    continue

                # Read code (robust to encoding issues)
                try:
                    code = src_path.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    code = src_path.read_text(encoding="latin-1", errors="ignore")

                status = meta.get("status") or meta.get("result") or ""
                label = map_status_to_label(status)

                rec = {
                    "problem_id": pid,
                    "submission_id": sid,
                    "language": language,
                    "filename": src_path.name,   # store actual on-disk name
                    "status": status,            # raw CodeNet status
                    "label": label,              # coarse label
                    "code": code
                }
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                count_written += 1

                if COPY_RAW:
                    dest_dir = raw_root / pid / language
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src_path, dest_dir / src_path.name)

    print(f"[{split_name}] Wrote {count_written} records → {out_jsonl}")
    if COPY_RAW:
        print(f"[{split_name}] Raw copies under {raw_root}")
    if missing_meta or missing_file:
        print(f"[{split_name}] Missing metadata rows: {missing_meta}, missing files: {missing_file}")

def main():
    extract_split("train", SPLIT_DIR / "train-ids.json")
    extract_split("valid", SPLIT_DIR / "valid-ids.json")
    extract_split("test",  SPLIT_DIR / "test-ids.json")

if __name__ == "__main__":
    # Preflight checks so failures are obvious
    if not CODENET_ROOT.exists():
        raise SystemExit(f"CODENET_PATH does not exist: {CODENET_ROOT}")
    if not (CODENET_ROOT / "data").exists() or not (CODENET_ROOT / "metadata").exists():
        raise SystemExit(
            "Expected 'data/' and 'metadata/' under CODENET_PATH, but they were not found at:\n"
            f"  {CODENET_ROOT}\n"
            "Make sure CODENET_PATH points to the Project_CodeNet ROOT."
        )
    if not SPLIT_DIR.exists():
        raise SystemExit(f"SPLIT_DIR not found: {SPLIT_DIR}")
    main()

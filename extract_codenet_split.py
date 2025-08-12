import os, json, csv, re, shutil
from pathlib import Path
from typing import Dict, Tuple, List

# --- CONFIG ---
CODENET_ROOT = Path(os.environ["CODENET_PATH"]) #location of the data itself on a different drive
SPLIT_DIR    = Path("data/2021-12-29-f=0.01") #grabbing ids here
OUT_DIR      = Path("data/codenet_extract")   #outputting data here
COPY_RAW     = True # also copy raw files for inspection

OUT_DIR.mkdir(parents=True, exist_ok=True)
(OUT_DIR / "raw").mkdir(exist_ok=True)

# Optional: label mapping (if your *-ids files include a label per id, adapt here)
# By default we only extract code; labels can be joined later from Google’s prepared metadata.

# Accept either "p00023_s006384060" or {"problem_id":"p00023","submission_id":"s006384060"}
ID_PAT = re.compile(r'^(p\d{5})_s(\d+)$')

def load_id_list(fpath: Path) -> List[Tuple[str, str]]:
    with fpath.open("r", encoding="utf-8") as f:
        data = json.load(f)
    ids: List[Tuple[str, str]] = []
    for item in data:
        if isinstance(item, str):
            m = ID_PAT.match(item)
            if not m:
                raise ValueError(f"Unrecognized id format: {item}")
            ids.append((m.group(1), "s" + m.group(2)))
        elif isinstance(item, dict):
            pid = item.get("problem_id")
            sid = item.get("submission_id")
            if not (pid and sid):
                raise ValueError(f"Dict id missing keys: {item}")
            ids.append((pid, sid))
        else:
            raise ValueError(f"Unsupported id entry: {item}")
    return ids

def build_submission_index(problem_id: str) -> Dict[str, Tuple[str, str, Dict[str, str]]]:
    """
    Read metadata/<problem_id>.csv and build a dict:
      submission_id -> (language, filename, full_row_dict)
    """
    meta_csv = CODENET_ROOT / "Project_CodeNet" / "metadata" / f"{problem_id}.csv"
    if not meta_csv.exists():
        raise FileNotFoundError(f"Missing metadata csv: {meta_csv}")

    idx: Dict[str, Tuple[str, str, Dict[str, str]]] = {}
    with meta_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = row.get("submission_id") or row.get("id")
            if not sid:
                continue
                #submission_id used her to match ids from the split file to the right language and filename in metadata
            language = (row.get("language") or "").strip()
            filename = (row.get("filename") or "").strip()
            if not filename:
                # Fallback if filename absent: guess from language
                # (most C++ entries are .cpp; adjust if needed)
                ext = ".cpp"
                if "Python" in language:
                    ext = ".py"
                elif "C" in language and "C++" not in language:
                    ext = ".c"
                filename = f"{sid}{ext}"
            idx[sid] = (language, filename, row)
    return idx

def resolve_source_path(problem_id: str, submission_id: str, language: str, filename: str) -> Path:
    # Canonical CodeNet structure: data/<problem_id>/<language>/<filename>
    # submission_id is nopt used here, but needed earlier in build_submission_index so it has to be a part of problem_id
    return CODENET_ROOT / "Project_CodeNet" / "data" / problem_id / language / filename

def map_status_to_label(status: str) -> str:
    """
    OPTIONAL: map CodeNet 'status' to a coarse label you can refine later.
    Adjust to match your training pipeline expectations.
    """
    if not status:
        return "unknown"
    s = status.strip().upper()
    if s in {"AC"}:
        return "no_error"
    if s in {"RE"}:
        return "runtime_error"
    if s in {"TLE"}:
        return "timeout"
    if s in {"MLE"}:
        return "memory_limit"
    if s in {"CE"}:
        return "compile_error"
    if s in {"WA"}:
        return "wrong_answer"
    return s.lower()

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

    # Group by problem to avoid repeatedly reading CSV
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

                # Optional status/label (keep raw status too for later mapping if needed)
                status = meta.get("status") or meta.get("result") or ""
                label = map_status_to_label(status)

                rec = {
                    "problem_id": pid,
                    "submission_id": sid,
                    "language": language,
                    "filename": filename,
                    "status": status,     # raw CodeNet status
                    "label": label,       # coarse label (adjust as needed)
                    "code": code
                }
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                count_written += 1

                if COPY_RAW:
                    dest_dir = raw_root / pid / language
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    # keep original filename to avoid collisions
                    shutil.copy2(src_path, dest_dir / filename)

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
    # Basic sanity checks before running
    if not CODENET_ROOT.exists():
        raise SystemExit(f"CODENET_PATH does not exist: {CODENET_ROOT}")
    if not (CODENET_ROOT / "Project_CodeNet" / "data").exists() or not (CODENET_ROOT / "Project_CodeNet" / "metadata").exists():
        raise SystemExit(f"Expected 'data/' and 'metadata/' under CODENET_PATH, but they were not found at: {CODENET_ROOT}")
    if not SPLIT_DIR.exists():
        raise SystemExit(f"SPLIT_DIR not found: {SPLIT_DIR}")
    main()
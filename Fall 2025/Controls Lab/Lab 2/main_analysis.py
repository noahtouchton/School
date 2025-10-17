# main_analysis.py
# Driver for EML 4314C Lab 2 analysis (multi-sheet, multi-run)
# ---------------------------------------------------------------------------------
import os
import glob
import pandas as pd
from helpers import analyze_file_multi, parse_condition_from_filename

# --- Paths (resolve relative to this script) ---
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(BASE_DIR, "data")
OUT_DIR   = os.path.join(BASE_DIR, "outputs")
TABLE_DIR = os.path.join(OUT_DIR, "tables")

# --- Case-insensitive waveform override for files that DON'T say 'sine'/'square' ---
# Edit as needed. Keys must be lowercased file names.
WAVE_OVERRIDE = {
    "bb_sine_data.xlsx": "sine",
    "bb_square_data.xlsx": "square",
    "pid_sine_kp_data.xlsx": "sine",
    "pid_square_kp_data.xlsx": "square",
    "pid_ack_tuning.xlsx": "square",        # Step 6 design/tuning uses square
    "pid_manual_tuning.xlsx": "square",
}

def _wave_or_infer(filename: str) -> str | None:
    """Return override if present; otherwise infer from filename tokens."""
    key = filename.lower()
    if key in WAVE_OVERRIDE:
        return WAVE_OVERRIDE[key]
    ctrl, wave = parse_condition_from_filename(filename)
    return wave if wave in ("sine", "square") else None

def ensure_dirs():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(TABLE_DIR, exist_ok=True)

def main():
    ensure_dirs()
    print(f"[INFO] Script directory: {BASE_DIR}")
    print(f"[INFO] Data directory:   {DATA_DIR}")

    # Discover xlsx files (skip Excel lock files)
    paths = sorted(
        p for p in glob.glob(os.path.join(DATA_DIR, "*.xlsx"))
        if not os.path.basename(p).startswith("~$")
    )
    if not paths:
        print("[ERR] No .xlsx files found in data/.")
        print("     Put your files here:", DATA_DIR)
        return

    rows = []
    for path in paths:
        fname = os.path.basename(path)
        wave = _wave_or_infer(fname)
        try:
            # Will raise if waveform is still unknown
            result_rows = analyze_file_multi(path, waveform_override=wave)
            rows.extend(result_rows)
            print(f"[OK] {fname} (+{len(result_rows)} runs)")
        except Exception as e:
            print(f"[ERR] {fname}: {e}")

    if not rows:
        print("[ERR] No results produced.")
        return

    # Build DataFrame and nice column ordering
    df = pd.DataFrame(rows)
    preferred_cols = [
        "file", "sheet", "run_index", "controller", "waveform",
        "deadband", "duty", "effort",  # optional if present
        "error_mean", "error_std", "error_rms",
        "command_mean", "command_std", "command_rms",
        "cycle_duration", "N",
        "rise_time_10_90", "percent_overshoot", "settling_time_2pct",
    ]
    # Keep preferred order first, then append any remaining columns
    cols = [c for c in preferred_cols if c in df.columns] + [c for c in df.columns if c not in preferred_cols]
    df = df[cols]

    # Save outputs
    csv_path  = os.path.join(TABLE_DIR, "lab2_summary.csv")
    xlsx_path = os.path.join(TABLE_DIR, "lab2_summary.xlsx")
    df.to_csv(csv_path, index=False)
    with pd.ExcelWriter(xlsx_path) as w:
        df.to_excel(w, sheet_name="Summary", index=False)

    print("\n[INFO] Saved summary:")
    print(" -", csv_path)
    print(" -", xlsx_path)

if __name__ == "__main__":
    main()

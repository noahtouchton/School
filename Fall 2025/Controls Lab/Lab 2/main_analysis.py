# main_analysis.py
# Minimal driver that uses helpers.py

import os
import glob
import pandas as pd
from helpers import analyze_file

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(BASE_DIR, "data")
OUT_DIR   = os.path.join(BASE_DIR, "outputs")
TABLE_DIR = os.path.join(OUT_DIR, "tables")

# Explicit waveform for each file that doesn't say "sine"/"square" in the name
WAVE_OVERRIDE = {
    "bb_sine_data.xlsx": "sine",
    "bb_square_data.xlsx": "square",
    "pid_sine_kp_data.xlsx": "sine",
    "pid_square_kp_data.xlsx": "square",
    "pid_ack_tuning.xlsx": "square",       # per Lab 2 step-6 specs
    "pid_manual_tuning.xlsx": "square",
}

def main():
    os.makedirs(TABLE_DIR, exist_ok=True)
    print(f"[INFO] Script directory: {BASE_DIR}")
    print(f"[INFO] Data directory:   {DATA_DIR}")

    paths = [p for p in glob.glob(os.path.join(DATA_DIR, "*.xlsx"))
             if not os.path.basename(p).startswith("~$")]

    rows = []
    for path in sorted(paths):
        fname = os.path.basename(path)
        wf = WAVE_OVERRIDE.get(fname.lower())
        try:
            row = analyze_file(path, waveform_override=wf)
            rows.append(row)
            print(f"[OK] {fname}")
        except Exception as e:
            print(f"[ERR] {fname}: {e}")

    if not rows:
        print("[ERR] No results produced.")
        return

    summary = pd.DataFrame(rows)
    csv_path  = os.path.join(TABLE_DIR, "lab2_summary.csv")
    xlsx_path = os.path.join(TABLE_DIR, "lab2_summary.xlsx")
    summary.to_csv(csv_path, index=False)
    with pd.ExcelWriter(xlsx_path) as w:
        summary.to_excel(w, sheet_name="Summary", index=False)

    print("\n[INFO] Saved summary:")
    print(" -", csv_path)
    print(" -", xlsx_path)

if __name__ == "__main__":
    main()

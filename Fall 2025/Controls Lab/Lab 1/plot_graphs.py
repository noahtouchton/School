#!/usr/bin/env python3
"""
Batch parser & plotter for Controls Lab frequency-sweep logs across multiple amplitudes.

- Expects Excel files with header at line 4 (Excel UI; 0-based pandas header=3) and data from line 5 onward.
- Columns (matched by fuzzy substrings):
    "Time (s)", "Command Signal", "Filtered Platen Rotation Angle",
    "Platen Velocity", "Amplitude", "Frequency"
- Frequency is piecewise-constant in long contiguous blocks (~10k samples).
- Produces separate plots (not overlaid) for Command, Angle, and (optional) Velocity.

Folder structure:
plots/
  amp_200/
    block_summary.csv
    freq_0.50Hz/
      01_command.png
      02_angle.png
      03_velocity.png  (if enabled)
      block_info.csv
  amp_250/
    ...

Requires: pandas, numpy, matplotlib, openpyxl (for .xlsx)
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ----------------------- USER SETTINGS -----------------------
BASE_DIR = Path(r"C:\Users\noaht\School\Fall 2025\Controls Lab\Lab 1")
FILES = [
    "amp_200.xlsx",
    "amp_250.xlsx",
    "amp_300.xlsx",
    "amp_350.xlsx",
    "amp_400.xlsx",
]
HEADER_ROW = 3          # 0-based; Excel row 4 is the header line
OUTROOT = BASE_DIR / "plots"
PLOT_VELOCITY = False   # set True if you also want velocity plots
FREQ_TOL = 1e-6         # tolerance for frequency changes
MIN_BLOCK_LEN = 100     # ignore tiny blocks
# -------------------------------------------------------------

def load_data(xlsx_path: Path, header_row: int) -> pd.DataFrame:
    # engine openpyxl for .xlsx
    df = pd.read_excel(xlsx_path, header=header_row, engine="openpyxl")

    # Fuzzy pick of expected columns
    cols_lower = {c.lower(): c for c in df.columns}

    def pick(*cands):
        lcands = [s.lower() for s in cands]
        for lc, orig in cols_lower.items():
            if any(s in lc for s in lcands):
                return orig
        raise KeyError(f"Missing expected column like: {cands}")

    col_time  = pick("Time (s)", "Time")
    col_cmd   = pick("Command Signal", "Command")
    col_angle = pick("Filtered Platen Rotation Angle", "Rotation Angle", "Platen Angle", "Filtered")
    col_vel   = pick("Platen Velocity", "Velocity")
    col_amp   = pick("Amplitude", "Amp")
    col_freq  = pick("Frequency", "Freq")

    df = df[[col_time, col_cmd, col_angle, col_vel, col_amp, col_freq]].copy()
    df.columns = ["time", "command", "angle", "velocity", "amplitude", "frequency"]

    # Ensure numeric where expected
    for c in ["time", "command", "angle", "velocity", "amplitude", "frequency"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Drop rows with no time or frequency
    df = df.dropna(subset=["time", "frequency"]).reset_index(drop=True)
    return df

def find_frequency_blocks(freq: pd.Series, tol: float, min_len: int):
    f = freq.to_numpy(dtype=float)
    n = len(f)
    if n == 0:
        return []

    # New block whenever abs diff > tol
    change = np.concatenate(([True], np.abs(np.diff(f)) > tol, [True]))
    idx = np.flatnonzero(change)
    blocks = []
    for i in range(len(idx) - 1):
        start = idx[i]
        end_excl = idx[i+1]
        length = end_excl - start
        if length >= min_len:
            fval = float(np.median(f[start:end_excl]))
            blocks.append((start, end_excl - 1, fval))
    return blocks

def plot_single_series(t: np.ndarray, y: np.ndarray, title: str, ylabel: str, outpath: Path):
    # time re-zero for readability
    t0 = t[0] if len(t) else 0.0
    plt.figure(figsize=(10, 5.2))
    plt.plot(t - t0, y, linewidth=1.2)
    plt.xlabel("Time (s)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, linewidth=0.4, alpha=0.6)
    plt.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=200)
    plt.close()

def save_block_info(df_block: pd.DataFrame, outdir: Path, freq_val: float):
    t0, t1 = df_block["time"].iloc[0], df_block["time"].iloc[-1]
    info = pd.DataFrame({
        "frequency_hz": [freq_val],
        "num_samples": [len(df_block)],
        "time_start_s": [float(t0)],
        "time_end_s": [float(t1)],
        "duration_s": [float(t1 - t0)],
        "amplitude_set": [float(np.median(df_block["amplitude"].dropna())) if "amplitude" in df_block else np.nan],
        "command_peak_est": [float(np.nanmax(np.abs(df_block["command"])))],
        "angle_peak_est": [float(np.nanmax(np.abs(df_block["angle"])))],
        "velocity_peak_est": [float(np.nanmax(np.abs(df_block["velocity"])))]
    })
    info.to_csv(outdir / "block_info.csv", index=False)

def process_file(xlsx_path: Path, outroot: Path):
    # amp name (folder) inferred from file stem
    amp_name = xlsx_path.stem  # e.g., "amp_200"
    outdir_amp = outroot / amp_name
    outdir_amp.mkdir(parents=True, exist_ok=True)

    df = load_data(xlsx_path, HEADER_ROW)
    blocks = find_frequency_blocks(df["frequency"], FREQ_TOL, MIN_BLOCK_LEN)

    if not blocks:
        print(f"[WARN] No frequency blocks found in {xlsx_path.name}")
        return

    # Build overall summary
    rows = []
    for k, (i0, i1, fval) in enumerate(blocks, start=1):
        sub = df.iloc[i0:i1+1].reset_index(drop=True)
        t0, t1 = sub["time"].iloc[0], sub["time"].iloc[-1]
        rows.append({
            "block_index": k,
            "start_idx": i0,
            "end_idx": i1,
            "num_samples": len(sub),
            "frequency_hz": fval,
            "time_start_s": float(t0),
            "time_end_s": float(t1),
            "duration_s": float(t1 - t0),
            "amplitude_set": float(np.median(sub["amplitude"].dropna())),
        })
    summary = pd.DataFrame(rows).sort_values(by=["frequency_hz", "block_index"]).reset_index(drop=True)
    summary.to_csv(outdir_amp / "block_summary.csv", index=False)

    # Plot per block into freq-named subfolders
    for k, (i0, i1, fval) in enumerate(blocks, start=1):
        sub = df.iloc[i0:i1+1].reset_index(drop=True)
        freq_folder = outdir_amp / f"freq_{fval:.4g}Hz"
        freq_folder.mkdir(parents=True, exist_ok=True)

        t = sub["time"].to_numpy()
        # 01 Command
        plot_single_series(
            t, sub["command"].to_numpy(),
            title=f"{amp_name} | Command @ {fval:g} Hz  (N={len(sub)})",
            ylabel="Command (arb.)",
            outpath=freq_folder / "01_command.png"
        )
        # 02 Angle
        plot_single_series(
            t, sub["angle"].to_numpy(),
            title=f"{amp_name} | Filtered Rotation Angle @ {fval:g} Hz  (N={len(sub)})",
            ylabel="Angle (rad or deg)",
            outpath=freq_folder / "02_angle.png"
        )
        # 03 Velocity (optional)
        if PLOT_VELOCITY:
            plot_single_series(
                t, sub["velocity"].to_numpy(),
                title=f"{amp_name} | Platen Velocity @ {fval:g} Hz  (N={len(sub)})",
                ylabel="Velocity (rad/s)",
                outpath=freq_folder / "03_velocity.png"
            )

        save_block_info(sub, freq_folder, fval)

def main():
    OUTROOT.mkdir(parents=True, exist_ok=True)

    for fname in FILES:
        path = BASE_DIR / fname
        # Guard common missing-suffix mistake like "amp_250" (no extension)
        if not path.suffix:
            path = path.with_suffix(".xlsx")
        if not path.exists():
            print(f"[WARN] Missing: {path}")
            continue
        print(f"[INFO] Processing {path.name} ...")
        process_file(path, OUTROOT)
    print("[DONE] All files processed.")

if __name__ == "__main__":
    main()

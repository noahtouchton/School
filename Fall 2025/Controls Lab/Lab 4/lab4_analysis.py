#!/usr/bin/env python3
"""
Lab 4 Furuta Pendulum analysis script (one plot per case, both angles).

Behavior:
- Computes summary metrics per scenario (RMS errors, gains, etc.).
- For each scenario (and segment for tape_no_tape), creates ONE plot that shows:
    - desired platen angle
    - actual platen angle
    - pendulum angle
  on the same axes.
- No titles on plots, only axis labels and legends.
- Saves outputs (plots + summary CSV) into a subfolder next to this script.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ----------------------------- Utility functions ----------------------------- #

def script_directory() -> str:
    """Return the directory where this script resides."""
    return os.path.dirname(os.path.abspath(__file__))


def ensure_output_directory(base_dir: str, folder_name: str = "lab4_output") -> str:
    """Create / get an output folder inside base_dir."""
    out_dir = os.path.join(base_dir, folder_name)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def load_excel(path: str) -> pd.DataFrame:
    """
    Load an Excel file with header on row 4 (1-indexed -> header=3).
    """
    df = pd.read_excel(path, header=3)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def split_segments_by_time_gap(df: pd.DataFrame, threshold: float = 0.5):
    """
    Split a DataFrame into segments whenever there is a jump in time larger than `threshold` seconds.
    Returns a list of segment DataFrames.
    """
    time = df["time"].to_numpy()
    dt = np.diff(time)
    split_indices = np.where(dt > threshold)[0] + 1

    segments = []
    start = 0
    for idx in split_indices:
        seg = df.iloc[start:idx].reset_index(drop=True)
        if len(seg) > 0:
            segments.append(seg)
        start = idx
    last_seg = df.iloc[start:].reset_index(drop=True)
    if len(last_seg) > 0:
        segments.append(last_seg)

    return segments


def rms(x: np.ndarray) -> float:
    """Root-mean-square, ignoring NaNs."""
    x = np.asarray(x, dtype=float)
    return float(np.sqrt(np.nanmean(x ** 2)))


def summarize_gains(df: pd.DataFrame):
    """
    Compute representative gains for a segment using the median.
    """
    gains = {}
    for k in ["k1", "k2", "k3", "k4", "k5"]:
        if k in df.columns:
            gains[k] = float(df[k].median())
    return gains


def choose_command_column(df: pd.DataFrame) -> np.ndarray:
    """
    Decide which column to use as the 'motor command'.
    Preference: 'u' if present and non-all-NaN, otherwise 'command'.
    """
    if "u" in df.columns and not df["u"].isna().all():
        return df["u"].to_numpy()
    elif "command" in df.columns:
        return df["command"].to_numpy()
    else:
        raise KeyError("No 'u' or 'command' column found for motor command.")


# ------------------------- Segment analysis / plotting ----------------------- #

def analyze_segment(df: pd.DataFrame,
                    scenario_name: str,
                    description: str,
                    out_dir: str,
                    segment_index: int | None = None):
    """
    Compute metrics for a single segment and create a single plot with both angles.
    Returns a dict with summary metrics.

    For the taped-pendulum case (task1_taped), the pendulum was not zeroed in hardware,
    so we treat the first pendulum angle sample as the zero reference and offset the
    entire signal by that value before plotting and RMS calculations.
    """

    # Basic signals
    t = df["time"].to_numpy()
    theta_des = df["desired position"].to_numpy()
    theta_platen = df["platen rotation"].to_numpy()
    theta_pend = df["pend rot"].to_numpy().copy()
    u = choose_command_column(df)

    # ----- FIX FOR TAPED CASE: zero pendulum based on first sample -----
    if scenario_name == "task1_taped" and len(theta_pend) > 0:
        theta_pend = theta_pend - theta_pend[0]

    # Errors / RMS metrics
    theta1_err = theta_des - theta_platen        # platen tracking error
    theta2_dev = theta_pend                      # pendulum angle (offset-corrected for taped case)
    rms_theta1_err = rms(theta1_err)
    rms_theta2 = rms(theta2_dev)
    rms_u = rms(u)

    gains = summarize_gains(df)

    # Build a suffix for filenames so we can distinguish segments
    seg_suffix = "" if segment_index is None else f"_seg{segment_index}"

    # Single plot: desired platen, actual platen, pendulum
    t0 = t - t[0] if len(t) > 0 else t

    plt.figure()
    plt.plot(t0, theta_des, label="platen desired")
    plt.plot(t0, theta_platen, label="platen actual", linestyle="--")
    plt.plot(t0, theta_pend, label="pendulum angle")
    plt.xlabel("time [s]")
    plt.ylabel("angle [deg]")
    plt.legend()
    fname = f"{scenario_name}{seg_suffix}_angles.png"
    plt.savefig(os.path.join(out_dir, fname), dpi=300, bbox_inches="tight")
    plt.close()

    # Summary dict for table
    summary = {
        "scenario": scenario_name,
        "segment_index": segment_index,
        "description": description,
        "rms_theta1_error_deg": rms_theta1_err,
        "rms_theta2_deg": rms_theta2,
        "rms_motor_command": rms_u,
    }
    summary.update(gains)
    return summary



# ------------------------------- Main pipeline ------------------------------ #

def main():
    base_dir = script_directory()
    out_dir = ensure_output_directory(base_dir, folder_name="lab4_output")

    file_configs = [
        {
            "filename": "tape_no_tape.xlsx",
            "multi_segments": True,
            "scenario_names": ["task1_taped", "task1_free"],
            "descriptions": [
                "Task 1: platen tracking with pendulum taped",
                "Task 1: platen tracking with pendulum free (same gains)"
            ],
        },
        {
            "filename": "allfour.xlsx",
            "multi_segments": False,
            "scenario_name": "task2_allfour",
            "description": "Task 2: manual tuning of all four gains",
        },
        {
            "filename": "lqrtuned.xlsx",
            "multi_segments": False,
            "scenario_name": "task3_lqr",
            "description": "Task 3: LQR / model-based controller",
        },
        {
            "filename": "lqr_angle_2.xlsx",
            "multi_segments": False,
            "scenario_name": "task4a_lqr_angle2",
            "description": "Task 4a: modified Q / poles to reduce pendulum angle",
        },
        {
            "filename": "r1_10.xlsx",
            "multi_segments": False,
            "scenario_name": "task4b_R0p1",
            "description": "Task 4b: LQR with R scaled by 0.1",
        },
        {
            "filename": "r10.xlsx",
            "multi_segments": False,
            "scenario_name": "task4b_R10",
            "description": "Task 4b: LQR with R scaled by 10",
        },
    ]

    all_summaries = []

    for cfg in file_configs:
        path = os.path.join(base_dir, cfg["filename"])
        if not os.path.isfile(path):
            print(f"[WARN] File not found, skipping: {path}")
            continue

        print(f"\n=== Processing file: {cfg['filename']} ===")
        df = load_excel(path)

        if cfg["multi_segments"]:
            segments = split_segments_by_time_gap(df, threshold=0.5)
            if len(segments) != len(cfg["scenario_names"]):
                print(
                    f"[WARN] Expected {len(cfg['scenario_names'])} segments in "
                    f"{cfg['filename']}, but found {len(segments)}. Proceeding anyway."
                )

            for i, seg in enumerate(segments):
                scenario_idx = min(i, len(cfg["scenario_names"]) - 1)
                scenario_name = cfg["scenario_names"][scenario_idx]
                description = cfg["descriptions"][scenario_idx]
                summary = analyze_segment(
                    seg,
                    scenario_name=scenario_name,
                    description=description,
                    out_dir=out_dir,
                    segment_index=i,
                )
                all_summaries.append(summary)

        else:
            scenario_name = cfg["scenario_name"]
            description = cfg["description"]
            summary = analyze_segment(
                df,
                scenario_name=scenario_name,
                description=description,
                out_dir=out_dir,
                segment_index=None,
            )
            all_summaries.append(summary)

    if not all_summaries:
        print("No data processed. Check file names and locations.")
        return

    # Build and save summary table
    summary_df = pd.DataFrame(all_summaries)

    col_order = [
        "scenario",
        "segment_index",
        "description",
        "k1",
        "k2",
        "k3",
        "k4",
        "k5",
        "rms_theta1_error_deg",
        "rms_theta2_deg",
        "rms_motor_command",
    ]
    cols = [c for c in col_order if c in summary_df.columns] + [
        c for c in summary_df.columns if c not in col_order
    ]
    summary_df = summary_df[cols]

    summary_path = os.path.join(out_dir, "lab4_summary_metrics.csv")
    summary_df.to_csv(summary_path, index=False)

    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", None)
    print("\n=== Summary metrics (for report) ===")
    print(summary_df.to_string(index=False))

    print(f"\nAll plots and summary saved in: {out_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Automatic first-order step-response analysis (no arguments).
- Loads: "Fall 2025/Controls Lab/Lab 3/basic_data.xlsx"
- Detects alternating-sign command steps
- Computes per-step Gain (K) and Time Constant (tau via 63.2% method)
- Saves CSVs to the same folder as the data
- Pops up plots for each step (close each window to advance)

Columns expected (header row is 4 -> pandas header=3):
  'Time', 'Command Signal', 'Filtered Platen Rotation Angle', 'Platen Velocity', 'Amplitude', 'Frequency'
"""

import numpy as np
import pandas as pd
import warnings
from pathlib import Path

# Plotting (matplotlib, no custom styles/colors)
import matplotlib.pyplot as plt

def moving_median(series, window_pts):
    if window_pts <= 1:
        return series.copy()
    return series.rolling(window_pts, center=True, min_periods=max(1, window_pts//3)).median()

def detect_steps(time, cmd, deadband_frac=0.1):
    """
    Detect indices where the command changes sign with a hysteresis deadband.
    deadband_frac is relative to the max(|cmd|). Default 10%.
    Returns: list of indices k where a step begins at index k (transition from k-1 -> k).
    """
    cmd = np.asarray(cmd)
    amp = np.nanmax(np.abs(cmd))
    if amp == 0 or np.isnan(amp):
        return []

    deadband = deadband_frac * amp

    def signed_region(val):
        if val > deadband:
            return 1
        elif val < -deadband:
            return -1
        else:
            return 0

    regions = np.array([signed_region(v) for v in cmd], dtype=int)
    steps = []
    for i in range(1, len(regions)):
        if regions[i] != regions[i-1] and regions[i] != 0 and regions[i-1] != 0:
            steps.append(i)
    return steps

def interp_time_at_level(t, y, target):
    """
    Find the time when y(t) first crosses 'target' after the start, using linear interpolation.
    Returns None if not crossed.
    """
    for i in range(1, len(y)):
        y0, y1 = y[i-1], y[i]
        if (y0 - target) * (y1 - target) <= 0 and not np.isnan(y0) and not np.isnan(y1):
            if y1 == y0:
                return t[i]
            frac = (target - y0) / (y1 - y0)
            return t[i-1] + frac * (t[i] - t[i-1])
    return None

def analyze_steps(df, time_col, cmd_col, y_col, edge_frac=0.15, min_step_frac=0.2, plot=True, plot_dir=None):
    """
    edge_frac: fraction of each step-interval used at the start and end to estimate steady values (medians).
    min_step_frac: minimum fraction of the interval to consider for 63.2% crossing search (avoids immediate noise).
    plot: when True, generates matplotlib figures (pops up later via plt.show()). If plot_dir is provided, also saves PNGs.
    """
    time = df[time_col].to_numpy(dtype=float)
    cmd  = df[cmd_col].to_numpy(dtype=float)
    y    = df[y_col].to_numpy(dtype=float)

    # Light smoothing for detection only
    cmd_s = moving_median(pd.Series(cmd), max(3, int(0.01*len(cmd)))).to_numpy()
    step_idxs = detect_steps(time, cmd_s, deadband_frac=0.1)
    if len(step_idxs) < 1:
        raise RuntimeError("No steps detected. Check the command column or adjust detection parameters.")

    boundaries = [0] + step_idxs + [len(time)-1]
    step_records = []

    if plot_dir:
        Path(plot_dir).mkdir(parents=True, exist_ok=True)

    for si in range(1, len(boundaries)-1):
        start = boundaries[si]
        end   = boundaries[si+1]
        if end - start < 5:
            continue

        t_seg = time[start:end]
        y_seg = y[start:end]

        prev = boundaries[si-1]
        left_start = max(prev, start - int(edge_frac * (start - prev)))
        left_end   = start
        y0 = np.nanmedian(y[left_start:left_end]) if left_end > left_start else y[start]

        right_start = start + int((1.0 - edge_frac) * (end - start))
        right_start = min(max(right_start, start+1), end-1)
        right_end   = end
        y_ss = np.nanmedian(y[right_start:right_end]) if right_end > right_start else y[end-1]

        u0 = np.nanmedian(cmd[max(prev, start-5):start]) if start - prev >= 5 else cmd[start-1] if start>0 else cmd[start]
        u1 = np.nanmedian(cmd[right_start:right_end]) if right_end > right_start else cmd[end-1]
        du = u1 - u0
        dy = y_ss - y0
        if du == 0 or np.isnan(du) or np.isnan(dy):
            continue

        K_i = dy / du

        target = y0 + 0.632 * dy
        start_idx = start + int(min_step_frac * (end - start))
        start_idx = min(start_idx, end-2)
        t_632 = interp_time_at_level(time[start_idx:end], y[start_idx:end], target)
        tau_i = (t_632 - time[start]) if t_632 is not None else np.nan

        record = {
            "step_index": si,
            "t_step": time[start],
            "t_next": time[end-1],
            "u0": u0,
            "u1": u1,
            "du": du,
            "y0": y0,
            "y_ss": y_ss,
            "dy": dy,
            "gain_K": K_i,
            "tau_63": tau_i,
            "segment_len": end - start
        }
        step_records.append(record)

        if plot:
            fig = plt.figure()
            plt.plot(t_seg, y_seg, label="Angle")
            plt.axvline(time[start], linestyle="--", label="Step start")
            plt.axhline(y0, linestyle=":", label="y0")
            plt.axhline(y_ss, linestyle=":", label="y_ss")
            plt.axhline(target, linestyle="--", label="63.2% level")
            if t_632 is not None:
                plt.axvline(t_632, linestyle="--", label="t_63")
            plt.xlabel("Time (s)"); plt.ylabel("Angle")
            plt.title(f"Step {si}: K={K_i:.4g}, tau={tau_i if not np.isnan(tau_i) else float('nan'):.4g}")
            plt.legend(loc="best")
            if plot_dir:
                out_path = Path(plot_dir) / f"step_{si:02d}.png"
                fig.savefig(out_path, dpi=150, bbox_inches="tight")
            # Do NOT close; we'll let plt.show() pop them all up

    if not step_records:
        raise RuntimeError("Detected steps, but no valid segments for analysis.")

    results = pd.DataFrame(step_records)
    agg = {
        "gain_K_median": float(np.nanmedian(results["gain_K"])),
        "gain_K_mean": float(np.nanmean(results["gain_K"])),
        "gain_K_std": float(np.nanstd(results["gain_K"], ddof=1)) if len(results) > 1 else float("nan"),
        "tau_median": float(np.nanmedian(results["tau_63"])),
        "tau_mean": float(np.nanmean(results["tau_63"])),
        "tau_std": float(np.nanstd(results["tau_63"], ddof=1)) if len(results) > 1 else float("nan"),
        "n_steps": int(len(results))
    }
    return results, agg

def main():
    # Hard-coded relative path (with safe Path composition for spaces)
    file_path = Path("Fall 2025") / "Controls Lab" / "Lab 3" / "basic_data.xlsx"

    if not file_path.exists():
        print(f"[!] File not found: {file_path.resolve()}")
        return

    # Read: header row is 4 => header=3
    df = pd.read_excel(file_path, header=3)

    time_col = "Time"
    cmd_col = "Command Signal"
    y_col   = "Filtered Platen Rotation Angle"

    # Outputs to the SAME folder as the data
    out_dir = file_path.parent
    plots_dir = out_dir / "plots"

    results, agg = analyze_steps(
        df,
        time_col=time_col,
        cmd_col=cmd_col,
        y_col=y_col,
        edge_frac=0.15,
        min_step_frac=0.2,
        plot=True,
        plot_dir=str(plots_dir)
    )

    # Save CSVs next to the data
    results.to_csv(out_dir / "step_results.csv", index=False)
    pd.DataFrame([agg]).to_csv(out_dir / "summary_results.csv", index=False)

    # Console summary
    print("\nPer-step (first 10 shown):")
    show_cols = ["step_index","t_step","du","dy","gain_K","tau_63"]
    print(results[show_cols].head(10).to_string(index=False))

    print("\nAggregates:")
    for k, v in agg.items():
        print(f"  {k}: {v}")

    print(f"\nSaved: {out_dir/'step_results.csv'}")
    print(f"Saved: {out_dir/'summary_results.csv'}")
    print(f"Saved plots in: {plots_dir}")

    # Pop up all figures at the end
    plt.show()

if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        main()

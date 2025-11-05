#!/usr/bin/env python3
"""
Estimate pendulum viscous damping b2 from free-swing data (motor OFF).

Input file: "raw_pend_data.xlsx" (header row 4 -> pandas header=3)
Columns needed:
  - 'Time'
  - 'Pend Rotation'  (pendulum angle; degrees or radians)

Outputs next to the input file:
  - peaks.csv
  - summary_b2.csv
  - Two plots (shown): time series with peaks+envelope; log amplitude vs peak index.

Edit the CONSTANTS section if you know J2, or m2/l2 to approximate J2.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import warnings
import math
import matplotlib.pyplot as plt

# -------------------- CONSTANTS (edit here) --------------------
J2_KGM2 = 0.00002929   # If known, set e.g. 0.00234 (kg·m^2). Else None.
m2_KG = 0.00377    # If J2 unknown but you know mass (kg), set this and l2_M to estimate J2 ≈ m2*l2^2
l2_M  = 0.1472    # Distance from pivot to COM (m)
LOGDEC_K = 1     # Use 1 for adjacent same-type peaks
SMOOTH_FRAC = 0.005  # 0.5% of samples (min 5 points) for moving median
PROM_FRAC = 0.03     # Peak prominence threshold as fraction of robust amplitude
INPUT_PATH = Path("Fall 2025/Controls Lab/Lab 3/raw_pend_data.xlsx")
# ---------------------------------------------------------------

def moving_median(x, win_pts):
    if win_pts <= 1:
        return x.copy()
    y = pd.Series(x).rolling(win_pts, center=True, min_periods=max(1, win_pts//3)).median()
    return y.to_numpy()

def detect_local_extrema(t, y, min_prom):
    y = np.asarray(y); t = np.asarray(t)
    n = len(y)
    if n < 3:
        return pd.DataFrame(columns=["t","theta","type","amp"])

    baseline = np.nanmedian(y)
    scale = np.nanmedian(np.abs(y - baseline)) * 1.4826
    if scale == 0 or np.isnan(scale):
        scale = np.nanstd(y) if np.nanstd(y) > 0 else 1.0
    prom_thresh = min_prom * max(scale, 1e-9)

    rows = []
    for i in range(1, n-1):
        if np.isnan(y[i-1]) or np.isnan(y[i]) or np.isnan(y[i+1]):
            continue
        if y[i] >= y[i-1] and y[i] >= y[i+1]:
            amp = abs(y[i] - baseline)
            if amp >= prom_thresh:
                rows.append((t[i], y[i], "max", amp))
        if y[i] <= y[i-1] and y[i] <= y[i+1]:
            amp = abs(y[i] - baseline)
            if amp >= prom_thresh:
                rows.append((t[i], y[i], "min", amp))

    dfp = pd.DataFrame(rows, columns=["t","theta","type","amp"]).sort_values("t").reset_index(drop=True)

    if not dfp.empty:
        start_idx = int(np.argmax(np.abs(dfp["theta"].to_numpy())))
        start_type = dfp.loc[start_idx, "type"]
        keep = [start_idx]
        last_type = start_type; last_t = dfp.loc[start_idx, "t"]
        for j in range(start_idx+1, len(dfp)):
            if dfp.loc[j,"type"] != last_type and (dfp.loc[j,"t"] - last_t) > 0:
                keep.append(j); last_type = dfp.loc[j,"type"]; last_t = dfp.loc[j,"t"]
        last_type = start_type; last_t = dfp.loc[start_idx, "t"]
        back = []
        for j in range(start_idx-1, -1, -1):
            if dfp.loc[j,"type"] != last_type and (last_t - dfp.loc[j,"t"]) > 0:
                back.append(j); last_type = dfp.loc[j,"type"]; last_t = dfp.loc[j,"t"]
        keep = back[::-1] + keep
        dfp = dfp.loc[keep].reset_index(drop=True)

    return dfp

def estimate_log_dec_and_period(peaks_df, k=1):
    if peaks_df.empty or len(peaks_df) < (2+k):
        raise RuntimeError("Not enough peaks detected to estimate damping.")
    out_rows = []
    for ptype in ["max", "min"]:
        sub = peaks_df[peaks_df["type"] == ptype].reset_index(drop=True)
        if len(sub) < (2+k):
            continue
        A = np.abs(sub["theta"].to_numpy())
        t = sub["t"].to_numpy()
        deltas, periods = [], []
        for i in range(len(sub) - k):
            if A[i] > 0 and A[i+k] > 0:
                deltas.append( (1.0/k) * math.log( A[i] / A[i+k] ) )
                periods.append( t[i+k] - t[i] )
        if deltas:
            out_rows.append( (np.mean(deltas), np.mean(periods)) )
    if not out_rows:
        raise RuntimeError("Could not form valid same-type peak pairs for logarithmic decrement.")
    delta_mean = float(np.mean([r[0] for r in out_rows]))
    T_mean     = float(np.mean([r[1] for r in out_rows]))
    zeta = float( delta_mean / math.sqrt(4*math.pi**2 + delta_mean**2) )
    omega_d = float( 2*math.pi / T_mean )
    omega_n = float( omega_d / math.sqrt(max(1e-12, 1 - zeta**2)) )
    return delta_mean, zeta, T_mean, omega_d, omega_n

def main():
    if not INPUT_PATH.exists():
        print(f"[!] File not found: {INPUT_PATH.resolve()}")
        return
    df = pd.read_excel(INPUT_PATH, header=3)
    time_col = "time"; theta_col = "Pend Rotation"
    for col in [time_col, theta_col]:
        if col not in df.columns:
            raise KeyError(f"Expected column '{col}' not found. Columns present: {list(df.columns)}")
    t = df[time_col].to_numpy(dtype=float)
    theta_raw = df[theta_col].to_numpy(dtype=float)

    # Degrees->radians if necessary
    theta = np.deg2rad(theta_raw) if np.nanmax(np.abs(theta_raw)) > 6.5 else theta_raw.copy()

    # Smooth for peak detection
    N = len(theta); win = max(5, int(SMOOTH_FRAC * max(1, N)))
    theta_s = moving_median(theta, win_pts=win)

    # Detect peaks
    _baseline = np.nanmedian(theta_s)
    peaks_df = detect_local_extrema(t, theta_s, min_prom=PROM_FRAC)
    if len(peaks_df) < 4:
        raise RuntimeError("Too few peaks detected; adjust PROM_FRAC/SMOOTH_FRAC or verify free-swing data.")

    # Logarithmic decrement & period
    delta_mean, zeta, T_mean, omega_d, omega_n = estimate_log_dec_and_period(peaks_df, k=LOGDEC_K)

    # J2
    if J2_KGM2 is not None:
        J2 = float(J2_KGM2); J2_source = "provided J2_KGM2"
    elif m2_KG is not None and l2_M is not None:
        J2 = float(m2_KG * (l2_M**2)); J2_source = "computed J2 = m2*l2^2 (point-mass approx)"
    else:
        J2 = None; J2_source = "unknown"

    b2 = float(2.0 * zeta * omega_n * J2) if J2 is not None else None

    # Save peaks & summary near the input
    peaks_out = peaks_df.rename(columns={"t":"Time","theta":"Theta","type":"PeakType","amp":"AbsAmp"})
    peaks_out_path = INPUT_PATH.parent / "peaks.csv"
    peaks_out.to_csv(peaks_out_path, index=False)

    summary = {
        "delta_mean": delta_mean,
        "zeta": zeta,
        "T_mean": T_mean,
        "omega_d": omega_d,
        "omega_n": omega_n,
        "J2_source": J2_source,
        "J2": J2 if J2 is not None else "",
        "m2": m2_KG if m2_KG is not None else "",
        "l2": l2_M if l2_M is not None else "",
        "b2": b2 if b2 is not None else ""
    }
    summary_df = pd.DataFrame([summary])
    summary_out = INPUT_PATH.parent / "summary_b2.csv"
    summary_df.to_csv(summary_out, index=False)

    # Plots
    plt.figure()
    plt.plot(t, theta, label="θ (raw)")
    plt.plot(peaks_df["t"], peaks_df["theta"], marker="o", linestyle="none", label="Detected peaks")
    try:
        t0 = float(peaks_df["t"].iloc[0]); A0 = float(abs(peaks_df["theta"].iloc[0]))
        env = A0 * np.exp(-zeta * omega_n * (t - t0))
        plt.plot(t, env, label="Envelope")
        plt.plot(t, -env, label="Envelope (-)")
    except Exception:
        pass
    plt.xlabel("Time (s)"); plt.ylabel("θ (rad)")
    plt.title("Free-swing pendulum with detected peaks and exponential envelope")
    plt.legend(loc="best")

    plt.figure()
    amps = np.abs(peaks_df["theta"].to_numpy())
    idx  = np.arange(len(amps))
    amps_safe = np.where(amps <= 1e-12, np.nan, amps)
    plt.plot(idx, np.log(amps_safe), marker="o", linestyle="none", label="ln |θ_peak|")
    plt.xlabel("Peak index"); plt.ylabel("ln |θ_peak|")
    plt.title("Logarithmic decrement check")
    plt.legend(loc="best")

    print("\n=== SUMMARY (b2 estimation) ===")
    for k,v in summary.items():
        print(f"{k}: {v}")
    print(f"\nSaved: {peaks_out_path}")
    print(f"Saved: {summary_out}")

    plt.show()

if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        main()

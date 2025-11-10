#!/usr/bin/env python3
"""
Robust b2 estimator (Bro. Noah edition)
- Reads 'time' and 'Pend Rotation' from the Lab 3 Excel file.
- Forces min peak spacing = 0.50 s to lock onto the physical swing.
- Uses LOGDEC_K=2 (every other peak) for more stable damping ratio.
- Adds a clean θ(t) plot saved as PNG.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import warnings
import math
import matplotlib.pyplot as plt

# -------------------- CONSTANTS --------------------
J2_KGM2 = 0.00002929
m2_KG = 0.00377
l2_M  = 0.1472

LOGDEC_K = 2
SMOOTH_FRAC = 0.02
PROM_FRAC   = 0.1
MIN_SEP_S   = 0.50  # force at least half a second between same-type peaks

INPUT_PATH = Path("Fall 2025/Controls Lab/Lab 3/raw_pend_data.xlsx")
# ---------------------------------------------------

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
    return pd.DataFrame(rows, columns=["t","theta","type","amp"]).sort_values("t").reset_index(drop=True)

def enforce_min_sep(peaks_df, min_sep_s):
    out = []
    for ptype in ["max", "min"]:
        sub = peaks_df[peaks_df["type"] == ptype].sort_values("t").reset_index(drop=True)
        if sub.empty:
            continue
        kept = [sub.loc[0]]
        for i in range(1, len(sub)):
            curr = sub.loc[i]; last = kept[-1]
            if (curr["t"] - last["t"]) < min_sep_s:
                if abs(curr["theta"]) > abs(last["theta"]):
                    kept[-1] = curr
            else:
                kept.append(curr)
        out.append(pd.DataFrame(kept))
    if out:
        return pd.concat(out, axis=0).sort_values("t").reset_index(drop=True)
    return peaks_df

def estimate_log_dec_and_period(peaks_df, k=1):
    if peaks_df.empty or len(peaks_df) < (2+k):
        raise RuntimeError("Not enough peaks detected.")
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
                deltas.append( (1.0/k) * math.log(A[i] / A[i+k]) )
                periods.append( t[i+k] - t[i] )
        if deltas:
            out_rows.append( (np.mean(deltas), np.mean(periods)) )
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
            raise KeyError(f"Expected column '{col}' not found. Columns: {list(df.columns)}")

    t = df[time_col].to_numpy(dtype=float)
    theta_raw = df[theta_col].to_numpy(dtype=float)
    theta = np.deg2rad(theta_raw) if np.nanmax(np.abs(theta_raw)) > 6.5 else theta_raw.copy()

    N = len(theta)
    win = max(7, int(SMOOTH_FRAC * max(1, N)))
    theta_s = moving_median(theta, win_pts=win)

    peaks0 = detect_local_extrema(t, theta_s, min_prom=PROM_FRAC)
    if len(peaks0) < 4:
        raise RuntimeError("Too few peaks detected; adjust PROM_FRAC/SMOOTH_FRAC.")

    peaks = enforce_min_sep(peaks0, min_sep_s=MIN_SEP_S)
    if len(peaks) < 4:
        raise RuntimeError("Too few peaks after enforcing min_sep_s=0.50 s.")

    delta_mean, zeta, T_mean, omega_d, omega_n = estimate_log_dec_and_period(peaks, k=LOGDEC_K)

    J2 = float(J2_KGM2)
    b2 = float(2.0 * zeta * omega_n * J2)

    peaks_out = peaks.rename(columns={"t":"Time","theta":"Theta","type":"PeakType","amp":"AbsAmp"})
    peaks_out_path = INPUT_PATH.parent / "peaks.csv"
    peaks_out.to_csv(peaks_out_path, index=False)

    summary = {
        "min_sep_s_used": MIN_SEP_S,
        "delta_mean": float(delta_mean),
        "zeta": float(zeta),
        "T_mean": float(T_mean),
        "omega_d": float(omega_d),
        "omega_n": float(omega_n),
        "J2": J2,
        "m2": m2_KG,
        "l2": l2_M,
        "b2": b2
    }
    summary_df = pd.DataFrame([summary])
    summary_out = INPUT_PATH.parent / "summary_b2.csv"
    summary_df.to_csv(summary_out, index=False)

    # Plot θ(t)
    plt.figure()
    plt.plot(t, theta)
    plt.xlabel("Time (s)")
    plt.ylabel("θ (rad)")
    plt.title("Pendulum Angle vs Time")
    plt.tight_layout()
    out_fig = INPUT_PATH.parent / "theta_vs_time.png"
    plt.savefig(out_fig, dpi=150)

    # Annotated plot with peaks and envelope
    plt.figure()
    plt.plot(t, theta, label="θ (raw)")
    plt.plot(peaks_out["Time"], peaks_out["Theta"], marker="o", linestyle="none", label="Detected peaks")
    try:
        t0 = float(peaks_out["Time"].iloc[0])
        A0 = float(abs(peaks_out["Theta"].iloc[0]))
        env = A0 * np.exp(-zeta * omega_n * (t - t0))
        plt.plot(t, env, label="Envelope")
        plt.plot(t, -env, label="Envelope (-)")
    except Exception:
        pass
    plt.xlabel("Time (s)"); plt.ylabel("θ (rad)")
    plt.title("Free-swing pendulum with filtered peaks")
    plt.legend()

    plt.figure()
    amps = np.abs(peaks_out["Theta"].to_numpy())
    idx = np.arange(len(amps))
    plt.plot(idx, np.log(amps), "o")
    plt.xlabel("Peak index"); plt.ylabel("ln |θ_peak|")
    plt.title("Logarithmic Decrement Check (k=2)")

    print("\n=== SUMMARY (b2 estimation, Bro. Noah edition) ===")
    for k,v in summary.items():
        print(f"{k}: {v}")
    print(f"\nSaved: {peaks_out_path}")
    print(f"Saved: {summary_out}")
    print(f"Saved: {out_fig}")

    plt.show()

if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        main()

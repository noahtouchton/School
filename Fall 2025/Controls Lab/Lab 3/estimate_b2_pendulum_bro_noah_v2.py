#!/usr/bin/env python3
"""
Robust b2 estimator (Bro. Noah v2)
- Reads 'time' and 'Pend Rotation' from: Fall 2025/Controls Lab/Lab 3/raw_pend_data.xlsx
- Non-blocking plots by default (no hang). Toggle SHOW_PLOTS = True if you want pop-ups.
- Enforces min same-type peak spacing (starts at 0.50 s), with auto-relax if too few peaks.
- Uses every-other peak (LOGDEC_K = 2) for damping ratio.
- Saves: peaks.csv, summary_b2.csv, theta_vs_time.png, step_debug.png

If peaks are still < 4 after relaxation, it will save plots and exit gracefully.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import warnings
import math
import matplotlib
import matplotlib.pyplot as plt

# -------------------- CONSTANTS --------------------
INPUT_PATH = Path("Fall 2025") / "Controls Lab" / "Lab 3" / "raw_pend_data.xlsx"
TIME_COL   = "time"
THETA_COL  = "Pend Rotation"

J2_KGM2 = 0.00002929
m2_KG = 0.00377
l2_M  = 0.1472

LOGDEC_K = 2
SMOOTH_FRAC_INIT = 0.02
PROM_FRAC_INIT   = 0.10
MIN_SEP_INIT     = 0.50   # seconds

# Toggle pop-up windows here (non-blocking by default)
SHOW_PLOTS = False
# ---------------------------------------------------

def moving_median(x, win_pts):
    if win_pts <= 1: return x.copy()
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
        if np.isnan(y[i-1]) or np.isnan(y[i]) or np.isnan(y[i+1]): continue
        if y[i] >= y[i-1] and y[i] >= y[i+1]:
            amp = abs(y[i] - baseline)
            if amp >= prom_thresh: rows.append((t[i], y[i], "max", amp))
        if y[i] <= y[i-1] and y[i] <= y[i+1]:
            amp = abs(y[i] - baseline)
            if amp >= prom_thresh: rows.append((t[i], y[i], "min", amp))
    return pd.DataFrame(rows, columns=["t","theta","type","amp"]).sort_values("t").reset_index(drop=True)

def enforce_min_sep(peaks_df, min_sep_s):
    out = []
    for ptype in ["max", "min"]:
        sub = peaks_df[peaks_df["type"] == ptype].sort_values("t").reset_index(drop=True)
        if sub.empty: continue
        kept = [sub.loc[0]]
        for i in range(1, len(sub)):
            curr = sub.loc[i]; last = kept[-1]
            if (curr["t"] - last["t"]) < min_sep_s:
                if abs(curr["theta"]) > abs(last["theta"]): kept[-1] = curr
            else:
                kept.append(curr)
        out.append(pd.DataFrame(kept))
    if out: return pd.concat(out, axis=0).sort_values("t").reset_index(drop=True)
    return peaks_df

def estimate_log_dec_and_period(peaks_df, k=1):
    if peaks_df.empty or len(peaks_df) < (2+k):
        raise RuntimeError("Not enough peaks detected.")
    out_rows = []
    for ptype in ["max", "min"]:
        sub = peaks_df[peaks_df["type"] == ptype].reset_index(drop=True)
        if len(sub) < (2+k): continue
        A = np.abs(sub["theta"].to_numpy())
        t = sub["t"].to_numpy()
        deltas, periods = [], []
        for i in range(len(sub) - k):
            if A[i] > 0 and A[i+k] > 0:
                deltas.append( (1.0/k) * math.log(A[i] / A[i+k]) )
                periods.append( t[i+k] - t[i] )
        if deltas: out_rows.append( (np.mean(deltas), np.mean(periods)) )
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
    for col in [TIME_COL, THETA_COL]:
        if col not in df.columns:
            print(f"[!] Missing column '{col}'. Available: {list(df.columns)}")
            return

    t = df[TIME_COL].to_numpy(dtype=float)
    theta_raw = df[THETA_COL].to_numpy(dtype=float)
    theta = np.deg2rad(theta_raw) if np.nanmax(np.abs(theta_raw)) > 6.5 else theta_raw.copy()

    # θ(t) quick plot (saved)
    plt.figure()
    plt.plot(t, theta)
    plt.xlabel("Time (s)")
    plt.ylabel("θ (rad)")
    plt.title("Pendulum Angle vs Time")
    plt.tight_layout()
    out_theta = INPUT_PATH.parent / "theta_vs_time.png"
    plt.savefig(out_theta, dpi=150)
    if SHOW_PLOTS: plt.show(block=False)

    # Try peak detection with progressive relaxation
    N = len(theta)
    attempts = [
        dict(smooth=SMOOTH_FRAC_INIT,   prom=PROM_FRAC_INIT,   minsep=MIN_SEP_INIT),
        dict(smooth=SMOOTH_FRAC_INIT,   prom=PROM_FRAC_INIT/2, minsep=MIN_SEP_INIT),
        dict(smooth=SMOOTH_FRAC_INIT/2, prom=PROM_FRAC_INIT/2, minsep=MIN_SEP_INIT*0.7),
        dict(smooth=SMOOTH_FRAC_INIT/3, prom=PROM_FRAC_INIT/3, minsep=MIN_SEP_INIT*0.5),
    ]

    chosen = None
    for cfg in attempts:
        win = max(7, int(cfg["smooth"] * max(1, N)))
        theta_s = moving_median(theta, win_pts=win)
        peaks0 = detect_local_extrema(t, theta_s, min_prom=cfg["prom"])
        peaks  = enforce_min_sep(peaks0, min_sep_s=cfg["minsep"])
        if len(peaks) >= 4:
            chosen = (cfg, peaks, theta_s)
            break

    if chosen is None:
        print("[!] Could not find >= 4 peaks even after relaxation.")
        peaks = pd.DataFrame(columns=["t","theta","type","amp"])
        save_and_done(t, theta, peaks, None, None, None, None, None, None, "No valid peaks")
        return

    cfg, peaks, theta_s = chosen
    print(f"[i] Using smooth={cfg['smooth']}, prom={cfg['prom']}, minsep={cfg['minsep']}s; peaks found: {len(peaks)}")

    # Compute damping & period
    delta_mean, zeta, T_mean, omega_d, omega_n = estimate_log_dec_and_period(peaks, k=LOGDEC_K)
    J2 = float(J2_KGM2)
    b2 = float(2.0 * zeta * omega_n * J2)

    # Save CSVs
    peaks_out = peaks.rename(columns={"t":"Time","theta":"Theta","type":"PeakType","amp":"AbsAmp"})
    peaks_out_path = INPUT_PATH.parent / "peaks.csv"
    peaks_out.to_csv(peaks_out_path, index=False)

    summary = {
        "smooth_used": cfg["smooth"],
        "prom_used": cfg["prom"],
        "min_sep_s_used": cfg["minsep"],
        "n_peaks_used": int(len(peaks)),
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
    summary_out = INPUT_PATH.parent / "summary_b2.csv"
    pd.DataFrame([summary]).to_csv(summary_out, index=False)

    # Debug/annotated plot
    plt.figure()
    plt.plot(t, theta, label="θ (raw)")
    try:
        plt.plot(peaks_out["Time"], peaks_out["Theta"], "o", label="peaks")
    except Exception:
        pass
    t0 = float(peaks_out["Time"].iloc[0]) if len(peaks_out) else t[0]
    A0 = float(abs(peaks_out["Theta"].iloc[0])) if len(peaks_out) else abs(theta[0])
    env = A0 * np.exp(-zeta * omega_n * (t - t0)) if np.isfinite(zeta) and np.isfinite(omega_n) else None
    if env is not None:
        plt.plot(t, env, label="Envelope")
        plt.plot(t, -env, label="Envelope (-)")
    plt.xlabel("Time (s)"); plt.ylabel("θ (rad)")
    plt.title("Free-swing peaks + envelope (auto-relaxed)")
    plt.legend()
    out_debug = INPUT_PATH.parent / "step_debug.png"
    plt.savefig(out_debug, dpi=150)
    if SHOW_PLOTS: plt.show(block=False)

    print("\n=== SUMMARY (b2 estimation, v2 non-blocking) ===")
    for k,v in summary.items():
        print(f"{k}: {v}")
    print(f"\nSaved: {peaks_out_path}")
    print(f"Saved: {summary_out}")
    print(f"Saved: {out_theta}")
    print(f"Saved: {out_debug}")

def save_and_done(t, theta, peaks_df, delta, zeta, T_mean, omega_d, omega_n, b2, reason):
    # Always save plots for inspection
    out_dir = INPUT_PATH.parent
    plt.figure()
    plt.plot(t, theta)
    plt.xlabel("Time (s)"); plt.ylabel("θ (rad)"); plt.title("Pendulum Angle vs Time")
    plt.tight_layout(); plt.savefig(out_dir/"theta_vs_time.png", dpi=150)

    plt.figure()
    if not peaks_df.empty:
        plt.plot(peaks_df["t"], peaks_df["theta"], "o", label="peaks")
    plt.xlabel("Time (s)"); plt.ylabel("θ (rad)"); plt.title("Peaks (none/too few)")
    plt.legend(); plt.tight_layout(); plt.savefig(out_dir/"step_debug.png", dpi=150)

    summary = {
        "reason": reason,
        "delta_mean": "" if delta is None else delta,
        "zeta": "" if zeta is None else zeta,
        "T_mean": "" if T_mean is None else T_mean,
        "omega_d": "" if omega_d is None else omega_d,
        "omega_n": "" if omega_n is None else omega_n,
        "b2": "" if b2 is None else b2
    }
    pd.DataFrame([summary]).to_csv(out_dir/"summary_b2.csv", index=False)
    print(f"[!] {reason}. Saved debug plots and summary_b2.csv.")

if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        main()

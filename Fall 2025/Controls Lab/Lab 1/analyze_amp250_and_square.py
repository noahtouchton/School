#!/usr/bin/env python3
"""
Analyze Controls Lab data for amplitude=250:

1) Bode (voltage -> velocity) from amp_250.xlsx
   - Segments contiguous frequency blocks
   - Fits input (volts) and output (velocity) sinusoids at the known block frequency
   - Computes Gain (linear + dB) and Phase (deg) vs frequency
   - Saves bode_data.csv and bode_plot.png

2) Step (square wave) from square.xlsx
   - Converts command -> volts
   - Detects step transitions
   - For each step: computes K = Δω_ss / ΔV and τ via 63.2% crossing
   - Saves square_step_summary.csv and step_xxx.png per transition

Assumptions
- Excel header row is line 4 (1-indexed) => pandas header=3
- Data start on line 5
- Columns matched by fuzzy names:
    "Time (s)", "Command Signal", "Filtered Platen Rotation Angle",
    "Platen Velocity", "Amplitude", "Frequency"
- Command range is ±1023 mapping to ±V_SUPPLY (set below)
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================ USER CONFIG ============================
BASE_DIR = Path(r"C:\Users\noaht\School\Fall 2025\Controls Lab\Lab 1")
FREQ_FILE = BASE_DIR / "amp_250.xlsx"
STEP_FILE = BASE_DIR / "square.xlsx"

RESULTS_DIR = BASE_DIR / "results_amp_250"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

HEADER_ROW = 3               # Excel row 4 is header -> pandas header=3
V_SUPPLY = 12.0              # H-bridge supply (Volts) — change if you used a different Vmax
FREQ_TOL = 1e-3              # tolerance for frequency block changes (Hz)
MIN_BLOCK_LEN = 1000         # ignore tiny frequency blocks

# Step detection / plots
ROLL_MED_WIN = 101           # rolling median window (samples) to smooth command for step detection
STEP_THRESH_FRAC = 0.3       # threshold fraction of global command swing to detect a step
PRE_SAMPLES = 300            # samples before step to estimate y0 and V0 plateaus
POST_SAMPLES = 600           # samples after step to estimate y_inf and V1 plateaus
# ====================================================================


# --------------------- Utilities and helpers -------------------------
def read_excel(path: Path, header_row: int) -> pd.DataFrame:
    df = pd.read_excel(path, header=header_row, engine="openpyxl")
    # Fuzzy pick columns
    cols = {c.lower(): c for c in df.columns}

    def pick(*names):
        targets = [n.lower() for n in names]
        for l, orig in cols.items():
            if any(t in l for t in targets):
                return orig
        raise KeyError(f"Missing expected column like: {names}")

    col_time  = pick("Time (s)", "Time")
    col_cmd   = pick("Command Signal", "Command")
    col_angle = pick("Filtered Platen Rotation Angle", "Rotation Angle", "Platen Angle", "Filtered")
    col_vel   = pick("Platen Velocity", "Velocity")
    col_amp   = pick("Amplitude", "Amp")
    col_freq  = pick("Frequency", "Freq")

    df = df[[col_time, col_cmd, col_angle, col_vel, col_amp, col_freq]].copy()
    df.columns = ["time", "command", "angle", "velocity", "amplitude", "frequency"]

    # numeric coercion
    for c in ["time", "command", "angle", "velocity", "amplitude", "frequency"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["time"]).reset_index(drop=True)
    return df


def command_to_volts(cmd: np.ndarray, v_supply: float = V_SUPPLY) -> np.ndarray:
    # ±1023 -> ±V_SUPPLY
    return (cmd / 1023.0) * v_supply


def find_frequency_blocks(freq: np.ndarray, tol: float, min_len: int):
    """Return list of (start_idx, end_idx_incl, f_med)."""
    f = np.asarray(freq, dtype=float)
    if len(f) == 0:
        return []
    change = np.concatenate(([True], np.abs(np.diff(f)) > tol, [True]))
    idx = np.flatnonzero(change)
    blocks = []
    for i in range(len(idx) - 1):
        i0, i1_excl = idx[i], idx[i+1]
        if i1_excl - i0 >= min_len:
            f_med = float(np.median(f[i0:i1_excl]))
            blocks.append((i0, i1_excl - 1, f_med))
    return blocks


def sine_fit_known_freq(t: np.ndarray, y: np.ndarray, f_hz: float):
    """
    Fit y ~ a*sin(wt) + b*cos(wt) + c at known frequency f_hz.
    Returns amplitude A, phase deg (y ≈ A*sin(ωt + φ) + c), offset c, and fitted vector yf.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    w = 2 * np.pi * f_hz

    # Design matrix: [sin(wt), cos(wt), 1]
    S = np.sin(w * t)
    C = np.cos(w * t)
    X = np.column_stack((S, C, np.ones_like(t)))

    # Least-squares
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    a, b, c = beta
    A = float(np.hypot(a, b))
    # A*sin(ωt + φ) = a*sin(ωt) + b*cos(ωt) -> φ = atan2(b, a)
    phi = float(np.degrees(np.arctan2(b, a)))
    y_fit = (a * S) + (b * C) + c
    return A, phi, c, y_fit


def unwrap_phase_deg(delta_deg: float) -> float:
    """Wrap to (-180, 180]."""
    d = (delta_deg + 180.0) % 360.0 - 180.0
    return d


# --------------------- Bode from amp_250.xlsx ------------------------
def build_bode(freq_df: pd.DataFrame, outdir: Path):
    df = freq_df.dropna(subset=["frequency"]).reset_index(drop=True)
    blocks = find_frequency_blocks(df["frequency"].values, FREQ_TOL, MIN_BLOCK_LEN)
    if not blocks:
        raise RuntimeError("No usable frequency blocks found. Adjust FREQ_TOL or MIN_BLOCK_LEN.")

    rows = []
    for i0, i1, f_med in blocks:
        sub = df.iloc[i0:i1+1].reset_index(drop=True)
        t = sub["time"].values
        u_cmd = sub["command"].values
        v = sub["velocity"].values

        u_volt = command_to_volts(u_cmd, V_SUPPLY)

        # Fit input and output at f_med
        Au, phi_u_deg, cu, u_fit = sine_fit_known_freq(t, u_volt, f_med)
        Ay, phi_y_deg, cy, y_fit = sine_fit_known_freq(t, v,       f_med)

        if Au <= 1e-9:
            continue  # avoid division by zero

        G_lin = Ay / Au                         # rad/s per Volt
        G_dB = 20.0 * np.log10(G_lin)
        phase_deg = unwrap_phase_deg(phi_y_deg - phi_u_deg)

        # store metrics
        rows.append({
            "f_hz": f_med,
            "omega_rad_s": 2 * np.pi * f_med,
            "Au_volts": Au,
            "Ay_rad_per_s": Ay,
            "Gain_linear_rad_per_s_per_V": G_lin,
            "Gain_dB": G_dB,
            "Phase_deg": phase_deg,
            "N_samples": len(sub)
        })

    bode = pd.DataFrame(rows).sort_values("f_hz").reset_index(drop=True)
    csv_path = outdir / "bode_data.csv"
    bode.to_csv(csv_path, index=False)
    print(f"[BODE] Wrote {csv_path}")

    # Plot magnitude and phase
    fig1 = plt.figure(figsize=(10, 6))
    ax1 = fig1.add_subplot(2, 1, 1)
    ax2 = fig1.add_subplot(2, 1, 2)

    # magnitude (dB)
    ax1.semilogx(bode["f_hz"], bode["Gain_dB"], marker="o", linewidth=1.2)
    ax1.set_ylabel("Magnitude (dB)")
    ax1.grid(True, which="both", linewidth=0.4, alpha=0.6)

    # phase (deg)
    ax2.semilogx(bode["f_hz"], bode["Phase_deg"], marker="o", linewidth=1.2)
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Phase (deg)")
    ax2.grid(True, which="both", linewidth=0.4, alpha=0.6)

    plt.tight_layout()
    fig_path = outdir / "bode_plot.png"
    plt.savefig(fig_path, dpi=200)
    plt.close(fig1)
    print(f"[BODE] Saved {fig_path}")

    # Rough readouts: K ~ low-f req linear gain; tau from -3 dB corner
    if len(bode) >= 3:
        K_est = float(bode.loc[0:2, "Gain_linear_rad_per_s_per_V"].median())
        # find closest to -3 dB from low-f plateau
        plateau_dB = float(20 * np.log10(K_est))
        corner_target = plateau_dB - 3.0
        idx = int(np.argmin(np.abs(bode["Gain_dB"].values - corner_target)))
        f_c = float(bode.loc[idx, "f_hz"])
        tau_est = 1.0 / (2 * np.pi * f_c)
        print(f"[BODE] Approx K ~ {K_est:.4g} rad/s/V, tau ~ {tau_est:.4g} s (from -3 dB near {f_c:.3g} Hz)")

    return bode


# -------------------- Step analysis from square.xlsx -----------------
def detect_steps(t: np.ndarray, u_volt: np.ndarray):
    """
    Detect step start indices by thresholding changes in a rolling-median-smoothed command.
    Returns list of indices where a step transition begins.
    """
    series = pd.Series(u_volt)
    smooth = series.rolling(window=ROLL_MED_WIN, center=True, min_periods=1).median().to_numpy()
    du = np.diff(smooth, prepend=smooth[0])
    swing = float(np.nanmax(smooth) - np.nanmin(smooth))
    thresh = STEP_THRESH_FRAC * max(swing, 1e-6)
    candidates = np.flatnonzero(np.abs(du) > thresh)

    # Keep only leading edges separated by some gap
    steps = []
    min_gap = max(PRE_SAMPLES + POST_SAMPLES, 100)  # ensure room around the step
    last = -min_gap
    for idx in candidates:
        if idx - last >= min_gap:
            steps.append(idx)
            last = idx
    return steps


def analyze_step_at_index(t: np.ndarray, y: np.ndarray, u_volt: np.ndarray, idx: int):
    """
    Compute K and tau for one step centered at idx using:
      - K = Δω_ss / ΔV (steady states from pre/post plateaus)
      - tau via 63.2% crossing on the rising/falling segment
    Returns dict with metrics and optional plot data.
    """
    n = len(t)
    i0 = max(0, idx - PRE_SAMPLES)
    i1 = min(n - 1, idx + POST_SAMPLES)

    t_seg = t[i0:i1+1]
    y_seg = y[i0:i1+1]
    u_seg = u_volt[i0:i1+1]

    # Plateau estimates
    y0 = float(np.median(y[max(0, idx - PRE_SAMPLES):idx]))
    y1 = float(np.median(y[idx:min(n, idx + POST_SAMPLES)]))
    u0 = float(np.median(u_volt[max(0, idx - PRE_SAMPLES):idx]))
    u1 = float(np.median(u_volt[idx:min(n, idx + POST_SAMPLES)]))

    dV = u1 - u0
    dω = y1 - y0
    if abs(dV) < 1e-9:
        return None

    K = dω / dV

    # 63.2% cross time from the instant of the step
    # determine target level depending on rise or fall
    target = y0 + 0.632 * dω
    post_t = t[idx:i1+1]
    post_y = y[idx:i1+1]

    # find first crossing of target
    crossing = None
    for k in range(1, len(post_t)):
        y_prev, y_curr = post_y[k-1], post_y[k]
        if (y_prev - target) * (y_curr - target) <= 0:
            # linear interpolation for better estimate
            t_prev, t_curr = post_t[k-1], post_t[k]
            if y_curr != y_prev:
                frac = (target - y_prev) / (y_curr - y_prev)
            else:
                frac = 0.0
            crossing = t_prev + frac * (t_curr - t_prev)
            break

    tau = None
    if crossing is not None:
        tau = float(crossing - t[idx])  # τ is time from the step edge

    return {
        "index": idx,
        "t_step": float(t[idx]),
        "y0": y0,
        "y1": y1,
        "u0_V": u0,
        "u1_V": u1,
        "dV": dV,
        "domega": dω,
        "K_rad_per_s_per_V": K,
        "tau_s": tau
    }, (t_seg, y_seg, u_seg, target)


def plot_step_segment(t_seg, y_seg, u_seg, t_step, target, outpath: Path, title: str):
    fig = plt.figure(figsize=(10, 5.5))
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2, sharex=ax1)

    ax1.plot(t_seg, y_seg, linewidth=1.2)
    ax1.axvline(t_step, linestyle="--", alpha=0.7)
    ax1.axhline(target, linestyle=":", alpha=0.7)
    ax1.set_ylabel("Velocity (rad/s)")
    ax1.set_title(title)
    ax1.grid(True, linewidth=0.4, alpha=0.6)

    ax2.plot(t_seg, u_seg, linewidth=1.2)
    ax2.axvline(t_step, linestyle="--", alpha=0.7)
    ax2.set_ylabel("Command (Volts)")
    ax2.set_xlabel("Time (s)")
    ax2.grid(True, linewidth=0.4, alpha=0.6)

    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close(fig)


def analyze_square(step_df: pd.DataFrame, outdir: Path):
    df = step_df.dropna(subset=["time", "command", "velocity"]).reset_index(drop=True)
    t = df["time"].values
    y = df["velocity"].values
    u_volt = command_to_volts(df["command"].values, V_SUPPLY)

    step_indices = detect_steps(t, u_volt)
    if not step_indices:
        raise RuntimeError("No steps detected in square.xlsx. Try adjusting ROLL_MED_WIN/STEP_THRESH_FRAC.")

    summary = []
    step_dir = outdir / "square_step_plots"
    step_dir.mkdir(parents=True, exist_ok=True)

    for j, idx in enumerate(step_indices, start=1):
        result = analyze_step_at_index(t, y, u_volt, idx)
        if result is None:
            continue
        metrics, plot_data = result
        summary.append(metrics)

        t_seg, y_seg, u_seg, target = plot_data
        title = f"Step #{j} @ t={metrics['t_step']:.3f}s  |  K={metrics['K_rad_per_s_per_V']:.4g}, τ={metrics['tau_s'] if metrics['tau_s'] else np.nan:.4g}s"
        outpath = step_dir / f"step_{j:02d}.png"
        plot_step_segment(t_seg, y_seg, u_seg, metrics["t_step"], target, outpath, title)
        print(f"[STEP] Saved {outpath}")

    if not summary:
        raise RuntimeError("Steps found, but none yielded valid metrics (ΔV ~ 0?).")

    df_sum = pd.DataFrame(summary)
    csv_path = outdir / "square_step_summary.csv"
    df_sum.to_csv(csv_path, index=False)
    print(f"[STEP] Wrote {csv_path}")

    # Overall estimates (robust)
    K_med = float(df_sum["K_rad_per_s_per_V"].median())
    tau_med = float(df_sum["tau_s"].dropna().median()) if df_sum["tau_s"].notna().any() else np.nan
    print(f"[STEP] Median K ~ {K_med:.4g} rad/s/V, median τ ~ {tau_med:.4g} s")

    return df_sum


# ------------------------------ Main -------------------------------
def main():
    print(f"[INFO] Reading frequency file: {FREQ_FILE}")
    df_freq = read_excel(FREQ_FILE, HEADER_ROW)

    # Bode (use VELOCITY as output)
    bode = build_bode(df_freq, RESULTS_DIR)

    print(f"[INFO] Reading square file: {STEP_FILE}")
    df_step = read_excel(STEP_FILE, HEADER_ROW)

    # Step analysis (use VELOCITY as output)
    step_summary = analyze_square(df_step, RESULTS_DIR)

    # Write a short text summary
    summary_txt = RESULTS_DIR / "summary.txt"
    with open(summary_txt, "w") as f:
        f.write("=== Bode (from amp_250.xlsx) ===\n")
        if len(bode):
            K_est = float(bode.loc[0:2, "Gain_linear_rad_per_s_per_V"].median())
            plateau_dB = float(20*np.log10(K_est))
            # corner approx
            idx = int(np.argmin(np.abs(bode["Gain_dB"].values - (plateau_dB - 3.0))))
            f_c = float(bode.loc[idx, "f_hz"])
            tau_est = 1.0/(2*np.pi*f_c)
            f.write(f"K_lowf_est ~ {K_est:.6g} rad/s/V\n")
            f.write(f"Corner ~ {f_c:.6g} Hz -> tau ~ {tau_est:.6g} s\n")
        f.write("\n=== Steps (from square.xlsx) ===\n")
        K_med = float(step_summary["K_rad_per_s_per_V"].median())
        tau_med = float(step_summary["tau_s"].dropna().median()) if step_summary["tau_s"].notna().any() else np.nan
        f.write(f"K_median ~ {K_med:.6g} rad/s/V\n")
        f.write(f"tau_median ~ {tau_med:.6g} s\n")
    print(f"[INFO] Wrote {summary_txt}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Controls Lab: Bode + Step with model overlays

What this does:
1) Builds Bode (voltage -> velocity) from amp_250.xlsx
   - Fits sinusoids per frequency block
   - Outputs bode_data.csv
   - Plots ONE Bode figure with experimental points and TWO model approximations
     (first-order K, tau) from:
       a) Bode-derived estimates
       b) Step-derived estimates
     -> saved as results_amp_250/bode_with_models.png

2) Analyzes steps from square.xlsx
   - Detects up/down steps
   - For every UP step, normalizes response to "rad/s per Volt"
     and overlays ALL up-steps on the SAME plot with the two model step responses
     -> saved as results_amp_250/up_steps_vs_models.png
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
from matplotlib import rcParams

# ---- bigger, clearer fonts everywhere ----
rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "font.size": 15,        # base size (ticks)
    "axes.labelsize": 17,   # x/y labels
    "axes.titlesize": 18,   # titles
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fontsize": 15   # legend text
})


# ============================ USER CONFIG ============================
BASE_DIR = Path(r"C:\Users\noaht\School\Fall 2025\Controls Lab\Lab 1")
FREQ_FILE = BASE_DIR / "amp_250.xlsx"
STEP_FILE = BASE_DIR / "square.xlsx"

RESULTS_DIR = BASE_DIR / "results_amp_250"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

HEADER_ROW = 3               # Excel row 4 is header -> pandas header=3
V_SUPPLY = 12.0              # H-bridge supply (Volts)
FREQ_TOL = 1e-3              # tolerance for frequency block changes (Hz)
MIN_BLOCK_LEN = 1000         # ignore tiny frequency blocks

# Step detection / plots
ROLL_MED_WIN = 101           # rolling median window (samples)
STEP_THRESH_FRAC = 0.3       # fraction of global swing to detect a step
PRE_SAMPLES = 300            # samples before step to estimate y0/V0
POST_SAMPLES = 600           # samples after step to estimate y_inf/V1
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
    # 0..1023 -> 0..V_SUPPLY (per your lab note)
    return (cmd / 1023.0) * v_supply

def vel_degps_to_radps(vel_degps: np.ndarray) -> np.ndarray:
    # deg/s -> rad/s
    return vel_degps * (np.pi / 180.0)


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

    S = np.sin(w * t)
    C = np.cos(w * t)
    X = np.column_stack((S, C, np.ones_like(t)))

    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    a, b, c = beta
    A = float(np.hypot(a, b))
    phi = float(np.degrees(np.arctan2(b, a)))  # deg
    y_fit = (a * S) + (b * C) + c
    return A, phi, c, y_fit


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
        v_degps = sub["velocity"].values

        u_volt = command_to_volts(u_cmd, V_SUPPLY)          # Volts
        v = vel_degps_to_radps(v_degps)                     # rad/s

        # Fit input and output at f_med
        Au, phi_u_deg, cu, _ = sine_fit_known_freq(t, u_volt, f_med)
        Ay, phi_y_deg, cy, _ = sine_fit_known_freq(t, v,      f_med)

        if Au <= 1e-9:
            continue  # avoid division by zero

        G_lin = Ay / Au                         # rad/s per Volt
        G_dB  = 20.0 * np.log10(max(G_lin, 1e-15))
        phase_deg = (phi_y_deg - phi_u_deg)
        # unwrap later with numpy unwrap; store raw here
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

    bode = pd.DataFrame(rows).sort_values("omega_rad_s").reset_index(drop=True)

    # Phase unwrap across frequency (keep in degrees)
    if len(bode):
        bode["Phase_deg"] = np.degrees(np.unwrap(np.radians(bode["Phase_deg"].to_numpy())))

    csv_path = outdir / "bode_data.csv"
    bode.to_csv(csv_path, index=False)
    print(f"[BODE] Wrote {csv_path}")
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

    steps = []
    min_gap = max(PRE_SAMPLES + POST_SAMPLES, 100)
    last = -min_gap
    for idx in candidates:
        if idx - last >= min_gap:
            steps.append(idx)
            last = idx
    return steps


def analyze_step_at_index(t: np.ndarray, y: np.ndarray, u_volt: np.ndarray, idx: int):
    """
    Compute K and tau for one step centered at idx using:
      - K = Δω_ss / ΔV (plateaus)
      - tau via 63.2% crossing
    Also return a normalized segment for plotting (per-Volt, time aligned at t=0).
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
    direction = "up" if dV > 0 else "down"

    # 63.2% crossing
    target = y0 + 0.632 * dω
    post_t = t[idx:i1+1]
    post_y = y[idx:i1+1]
    crossing = None
    for k in range(1, len(post_t)):
        y_prev, y_curr = post_y[k-1], post_y[k]
        if (y_prev - target) * (y_curr - target) <= 0:
            t_prev, t_curr = post_t[k-1], post_t[k]
            frac = 0.0 if y_curr == y_prev else (target - y_prev) / (y_curr - y_prev)
            crossing = t_prev + frac * (t_curr - t_prev)
            break
    tau = float(crossing - t[idx]) if crossing is not None else np.nan

    # Build normalized per-Volt post-step trace for overlay: y_norm = (y - y0)/dV
    t_post = t[idx:i1+1] - t[idx]
    y_post_norm = (y[idx:i1+1] - y0) / dV  # units: (rad/s)/V

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
        "tau_s": tau,
        "direction": direction
    }, (t_post, y_post_norm)


def analyze_square(step_df: pd.DataFrame, outdir: Path):
    df = step_df.dropna(subset=["time", "command", "velocity"]).reset_index(drop=True)
    t = df["time"].values
    y = vel_degps_to_radps(df["velocity"].values)              # rad/s
    u_volt = command_to_volts(df["command"].values, V_SUPPLY)  # V

    step_indices = detect_steps(t, u_volt)
    if not step_indices:
        raise RuntimeError("No steps detected in square.xlsx. Adjust ROLL_MED_WIN/STEP_THRESH_FRAC.")

    summary = []
    up_traces = []  # list of (t_post, y_post_norm) for UP steps only

    for idx in step_indices:
        result = analyze_step_at_index(t, y, u_volt, idx)
        if result is None:
            continue
        metrics, (t_post, y_post_norm) = result
        summary.append(metrics)
        if metrics["direction"] == "up":
            up_traces.append((t_post, y_post_norm))

    if not summary:
        raise RuntimeError("Steps found, but none yielded valid metrics (ΔV ~ 0?).")

    df_sum = pd.DataFrame(summary)
    csv_path = outdir / "square_step_summary.csv"
    df_sum.to_csv(csv_path, index=False)
    print(f"[STEP] Wrote {csv_path}")

    # Robust overall estimates
    K_med = float(df_sum["K_rad_per_s_per_V"].median())
    tau_med = float(df_sum["tau_s"].dropna().median()) if df_sum["tau_s"].notna().any() else np.nan
    print(f"[STEP] Median K ~ {K_med:.4g} rad/s/V, median τ ~ {tau_med:.4g} s")

    return df_sum, up_traces


# ---------------------- Model estimation helpers ---------------------
def estimate_K_tau_from_bode(bode: pd.DataFrame):
    """Estimate K (low-ω plateau) and τ from -3 dB point in Bode data."""
    if bode.empty:
        return np.nan, np.nan
    b = bode.sort_values("omega_rad_s").reset_index(drop=True)
    # K ≈ average of first few linear magnitudes
    n_lo = min(3, len(b))
    Kdc = float(b.loc[:n_lo-1, "Gain_linear_rad_per_s_per_V"].mean())
    plateau_dB = float(20*np.log10(max(Kdc, 1e-12)))

    # target -3 dB
    target = plateau_dB - 3.0
    mag_db = b["Gain_dB"].to_numpy()
    w = b["omega_rad_s"].to_numpy()

    # find crossing by log-frequency interpolation
    wc = np.nan
    for i in range(len(w) - 1):
        y1, y2 = mag_db[i], mag_db[i+1]
        if (y1 - target) * (y2 - target) <= 0:
            x1, x2 = np.log(w[i]), np.log(w[i+1])
            m = (y2 - y1) / (x2 - x1)
            x_star = x1 + (target - y1) / m if m != 0 else x1
            wc = float(np.exp(x_star))
            break
    tau = 1.0 / wc if np.isfinite(wc) else np.nan
    return Kdc, tau


def bode_mag_phase_first_order(K, tau, w):
    """Magnitude (linear) and phase (deg) of K/(1 + j*w*tau)."""
    mag = K / np.sqrt(1.0 + (w * tau)**2)
    phase = -np.degrees(np.arctan(w * tau))
    return mag, phase


# ------------------------------ Plots --------------------------------
def plot_bode_with_models(bode: pd.DataFrame, K_bode, tau_bode, K_step, tau_step, outdir: Path):
    """One Bode figure with experimental points + two model approximations."""
    if bode.empty:
        return
    # Frequency grid spanning experimental data
    w_min = max(1e-2, float(bode["omega_rad_s"].min())/1.5)
    w_max = float(bode["omega_rad_s"].max())*1.5
    w = np.logspace(np.log10(w_min), np.log10(w_max), 600)

    # Models
    mag_b, ph_b = bode_mag_phase_first_order(K_bode, tau_bode, w) if np.isfinite(tau_bode) else (None, None)
    mag_s, ph_s = bode_mag_phase_first_order(K_step, tau_step, w) if np.isfinite(tau_step) else (None, None)

    # Build figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    # Experimental magnitude (points) in dB
    ax1.semilogx(bode["omega_rad_s"], bode["Gain_dB"], marker="o", linestyle="none",
                 label="Experimental (mag, dB)")
    # Model magnitudes (lines) in dB
    if mag_b is not None:
        ax1.semilogx(w, 20*np.log10(np.maximum(mag_b, 1e-15)), label=f"Model (Bode fit): K={K_bode:.3g}, τ={tau_bode:.3g}s")
    if mag_s is not None:
        ax1.semilogx(w, 20*np.log10(np.maximum(mag_s, 1e-15)), label=f"Model (Step fit): K={K_step:.3g}, τ={tau_step:.3g}s")
    ax1.set_ylabel("Magnitude (dB)")
    ax1.grid(True, which="both", linewidth=0.4, alpha=0.6)
    ax1.legend(framealpha=0.95, fontsize=15)

    # Experimental phase (points) in deg
    ax2.semilogx(bode["omega_rad_s"], bode["Phase_deg"], marker="x", linestyle="none",
                 label="Experimental (phase, deg)")
    # Model phases (lines) in deg
    if ph_b is not None:
        ax2.semilogx(w, ph_b, label="Model (Bode fit)")
    if ph_s is not None:
        ax2.semilogx(w, ph_s, label="Model (Step fit)")
    ax2.set_xlabel("Angular frequency ω (rad/s)")
    ax2.set_ylabel("Phase (deg)")
    ax2.grid(True, which="both", linewidth=0.4, alpha=0.6)
    ax2.legend(framealpha=0.95, fontsize=15)

    plt.tight_layout()
    out = outdir / "bode_with_models.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"[PLOT] Saved {out}")


def plot_up_steps_vs_models(up_traces, K_bode, tau_bode, K_step, tau_step, outdir: Path):
    """
    Overlay ALL 'up' steps (normalized per Volt) with BOTH model step responses.
    x-axis is limited to the data extent (slight padding), so lines end where the blue data end.
    """
    if not up_traces:
        print("[PLOT] No UP steps found to overlay.")
        return

    # --- time window LIMITED BY DATA (not by model taus) ---
    t_max_data = max(float(tr[0].max()) for tr in up_traces if tr[0].size)
    pad = 0.02 * t_max_data  # 2% padding for aesthetics
    t_end = max(t_max_data + pad, 1e-3)
    t = np.linspace(0, t_end, 600)

    # Model responses over the same, data-limited time base (1 V step)
    y_b = K_bode * (1.0 - np.exp(-t / tau_bode)) if np.isfinite(tau_bode) else None
    y_s = K_step * (1.0 - np.exp(-t / tau_step)) if np.isfinite(tau_step) else None

    fig, ax = plt.subplots(figsize=(10, 5))

    # Blue up-step data: more pronounced + single legend entry
    first_label_done = False
    for t_post, y_norm in up_traces:
        label = "Up-step data (normalized)" if not first_label_done else "_nolegend_"
        ax.plot(
            t_post,
            y_norm,
            color="C0",
            linewidth=1.8,
            alpha=0.75,
            solid_capstyle="round",
            zorder=1,
            label=label
        )
        first_label_done = True

    # Models on SAME time axis (data-limited)
    if y_b is not None:
        ax.plot(t, y_b, color="C1", linewidth=2.4, zorder=2,
                label=f"Model (Bode fit): K={K_bode:.3g}, τ={tau_bode:.3g}s")
    if y_s is not None:
        ax.plot(t, y_s, color="C2", linewidth=2.4, linestyle="--", zorder=2,
                label=f"Model (Step fit): K={K_step:.3g}, τ={tau_step:.3g}s")

    ax.set_xlim(0, t_end)  # <-- keep axis tight to data extent
    ax.set_xlabel("Time since step (s)")
    ax.set_ylabel("Velocity per Volt (rad/s per V)")
    ax.set_title("All UP steps (normalized) vs model step responses")
    ax.grid(True, linewidth=0.4, alpha=0.6)
    ax.legend(framealpha=0.95)
    plt.tight_layout()
    out = outdir / "up_steps_vs_models.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"[PLOT] Saved {out}")




# ------------------------------ Main -------------------------------
def main():
    print(f"[INFO] Reading frequency file: {FREQ_FILE}")
    df_freq = read_excel(FREQ_FILE, HEADER_ROW)

    # Bode
    bode = build_bode(df_freq, RESULTS_DIR)

    # Bode-derived K, tau
    K_bode, tau_bode = estimate_K_tau_from_bode(bode)
    print(f"[BODE] K≈{K_bode:.4g} rad/s/V, τ≈{tau_bode:.4g} s")

    print(f"[INFO] Reading square file: {STEP_FILE}")
    df_step = read_excel(STEP_FILE, HEADER_ROW)

    # Steps (collect UP traces too)
    step_summary, up_traces = analyze_square(df_step, RESULTS_DIR)

    # Step-derived K, tau (robust medians)
    K_step = float(step_summary["K_rad_per_s_per_V"].median())
    tau_step = float(step_summary["tau_s"].dropna().median()) if step_summary["tau_s"].notna().any() else np.nan
    print(f"[STEP]  K≈{K_step:.4g} rad/s/V, τ≈{tau_step:.4g} s")

    # One Bode plot with experimental & both models
    plot_bode_with_models(bode, K_bode, tau_bode, K_step, tau_step, RESULTS_DIR)

    # One overlay plot: all UP steps vs both model step responses
    plot_up_steps_vs_models(up_traces, K_bode, tau_bode, K_step, tau_step, RESULTS_DIR)

    # Write a short text summary (kept)
    summary_txt = RESULTS_DIR / "summary.txt"
    with open(summary_txt, "w") as f:
        f.write("=== Bode (from amp_250.xlsx) ===\n")
        f.write(f"K_bode ~ {K_bode:.6g} rad/s/V\n")
        f.write(f"tau_bode ~ {tau_bode:.6g} s\n")
        f.write("\n=== Steps (from square.xlsx) ===\n")
        f.write(f"K_step_median ~ {K_step:.6g} rad/s/V\n")
        f.write(f"tau_step_median ~ {tau_step:.6g} s\n")
    print(f"[INFO] Wrote {summary_txt}")


if __name__ == "__main__":
    main()

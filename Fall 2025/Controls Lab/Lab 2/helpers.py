# helpers.py
# EML 4314C Lab 2 analysis helpers (strict to your 6 columns)

from __future__ import annotations
from typing import Dict, Iterable, Optional, Tuple
import numpy as np
import pandas as pd

# ----- Canonical schema we will produce for analysis -----
REQUIRED_COLS = ("time", "theta_des", "theta_act", "error", "command")

# ----- Exact expected headers in your spreadsheets -----
EXACT_HEADERS = {
    "Time (s)": "time",
    "Command Signal": "command",
    "Filtered Platen Rotation Angle": "theta_act",
    "Platen Velocity": "theta_dot",   # optional, not used in metrics
    "Amplitude": "amplitude",
    "Frequency": "frequency",
}

# ----------------------------- LOAD + STANDARDIZE -----------------------------

def load_timeseries_xlsx(
    path: str,
    waveform: Optional[str],          # 'sine' or 'square' (required for theta_des)
    sheet_name: Optional[int | str] = None,
    header_row_index: int = 3,        # your header is on line 4 (0-based -> 3)
) -> pd.DataFrame:
    """
    Load a dataset that has exactly these columns:
        Time (s), Command Signal, Filtered Platen Rotation Angle, Platen Velocity, Amplitude, Frequency
    Build a tidy DataFrame with columns:
        time [s], theta_des [deg], theta_act [deg], error [deg], command [counts]
    """
    # Read sheet(s) with the correct header row
    xls = pd.read_excel(path, sheet_name=None if sheet_name is None else sheet_name, header=header_row_index)
    sheets = xls if isinstance(xls, dict) else {sheet_name or 0: xls}

    picked_df = None
    picked_name = None

    for name, df0 in sheets.items():
        if df0 is None or df0.empty:
            continue

        # rename strictly from your exact headers
        ren = {col: EXACT_HEADERS[col] for col in df0.columns if col in EXACT_HEADERS}
        df = df0.rename(columns=ren)

        # must contain the 6 mapped columns (theta_dot optional)
        needed = {"time", "command", "theta_act", "amplitude", "frequency"}
        if needed <= set(df.columns):
            picked_df = df
            picked_name = name
            break

    if picked_df is None:
        # give feedback on what we saw
        seen = {name: list(df.columns) for name, df in sheets.items() if df is not None}
        raise ValueError(f"Could not find a qualifying sheet in {path}. Columns seen: {seen}")

    df = picked_df.copy()
    df = df.dropna().sort_values("time").reset_index(drop=True)

    # Rebuild desired trajectory (requires waveform)
    if not waveform:
        raise ValueError("waveform must be 'sine' or 'square' to reconstruct theta_des")

    df["theta_des"] = build_theta_des(
        t=df["time"].to_numpy(float),
        A=df["amplitude"].to_numpy(float),
        f=df["frequency"].to_numpy(float),
        waveform=waveform,
        phase=0.0,
    )

    # Compute error
    df["error"] = df["theta_des"] - df["theta_act"]

    # Return standardized columns
    return df[list(REQUIRED_COLS)]


def build_theta_des(t: np.ndarray, A: np.ndarray, f: np.ndarray, waveform: str, phase: float = 0.0) -> np.ndarray:
    """
    Build desired angle in degrees given time t [s], amplitude A [deg], frequency f [Hz].
    waveform ∈ {'sine','square'}
    """
    arg = 2 * np.pi * f * t + phase
    w = waveform.lower().strip()
    if w == "sine":
        return A * np.sin(arg)
    elif w == "square":
        s = np.sign(np.sin(arg))
        s[s == 0.0] = 1.0
        return A * s
    else:
        raise ValueError(f"Unknown waveform '{waveform}'")


# ----------------------------- METRICS -----------------------------------------

def basic_stats(x: Iterable[float]) -> Dict[str, float]:
    x = np.asarray(x, dtype=float)
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x, ddof=1)) if x.size > 1 else 0.0,
        "rms": float(np.sqrt(np.mean(x**2))),
    }


def compute_cycle_metrics(df: pd.DataFrame, idx: slice) -> Dict[str, float]:
    sub = df.iloc[idx]
    err = basic_stats(sub["error"].to_numpy())
    cmd = basic_stats(sub["command"].to_numpy())
    out: Dict[str, float] = {f"error_{k}": v for k, v in err.items()}
    out.update({f"command_{k}": v for k, v in cmd.items()})
    out["cycle_duration"] = float(sub["time"].iloc[-1] - sub["time"].iloc[0])
    out["N"] = int(len(sub))
    return out


def step_metrics(
    time: Iterable[float],
    y: Iterable[float],
    target_high: float,
    target_low: float,
    side: str = "rising",
) -> Dict[str, float]:
    """
    10–90% rise time, % overshoot, 2% settling time on a single transition.
    """
    t = np.asarray(time, dtype=float)
    y = np.asarray(y, dtype=float)
    y0, y1 = (target_low, target_high) if side == "rising" else (target_high, target_low)
    yn = (y - y0) / (y1 - y0 + 1e-12)

    def first_cross(level: float) -> float:
        idx = np.where(yn >= level)[0] if side == "rising" else np.where(yn <= 1 - level)[0]
        return float(t[idx[0]]) if idx.size else float("nan")

    t10, t90 = first_cross(0.10), first_cross(0.90)
    tr = (t90 - t10) if np.isfinite(t10) and np.isfinite(t90) else float("nan")
    OS = max(0.0, (np.max(yn) - 1.0) * 100.0) if side == "rising" else max(0.0, (1.0 - np.min(yn)) * 100.0)

    within = np.where(np.abs(yn - 1.0) <= 0.02)[0] if side == "rising" else np.where(np.abs(yn - 0.0) <= 0.02)[0]
    Ts = float(t[within[0]] - t[0]) if within.size else float("nan")

    return {
        "rise_time_10_90": float(tr),
        "percent_overshoot": float(OS),
        "settling_time_2pct": float(Ts),
    }


# ----------------------------- CYCLE PICKERS ------------------------------------

def find_square_cycle(df: pd.DataFrame, col: str = "theta_des", min_period_s: float = 0.3) -> slice:
    """
    rising-edge to next rising-edge window for one full square cycle
    """
    t = df["time"].to_numpy()
    y = df[col].to_numpy()
    mid = 0.5 * (np.nanmax(y) + np.nanmin(y))
    hi = y > mid
    rising = np.where((~hi[:-1]) & (hi[1:]))[0] + 1
    if rising.size < 2:
        raise RuntimeError("Not enough rising edges for a full square cycle.")
    for i in range(rising.size - 1):
        t0, t1 = t[rising[i]], t[rising[i + 1]]
        if (t1 - t0) >= min_period_s:
            mask = (t >= t0) & (t < t1)
            idx = np.where(mask)[0]
            return slice(idx[0], idx[-1] + 1)
    raise RuntimeError("Could not find a valid square-wave cycle.")


def find_sine_cycle(df: pd.DataFrame, col: str = "theta_des", prefer_positive_slope: bool = True) -> slice:
    """
    zero-crossing to next zero-crossing window for one sine period
    """
    t = df["time"].to_numpy()
    y = df[col].to_numpy()
    zc = np.where(np.diff(np.signbit(y)))[0]
    if zc.size < 2:
        raise RuntimeError("Not enough zero-crossings for a full sine cycle.")
    for i in range(zc.size - 1):
        a, b = zc[i], zc[i + 1]
        if prefer_positive_slope and (y[b] - y[a]) <= 0:
            continue
        return slice(a + 1, b + 1)
    return slice(zc[0] + 1, zc[1] + 1)


# ----------------------------- HIGH-LEVEL ONE-FILE ANALYSIS ---------------------

def parse_condition_from_filename(fname: str) -> Tuple[str, str]:
    s = fname.lower()
    ctrl = "pid" if "pid" in s else ("bang-bang" if "bb" in s else "unknown")
    wave = "square" if "square" in s else ("sine" if "sine" in s else "unknown")
    return ctrl, wave


def analyze_file(path: str, waveform_override: Optional[str] = None) -> Dict[str, float | str]:
    """
    Load -> reconstruct theta_des -> pick one cycle -> compute cycle & (if square) step metrics.
    Returns a row dict for the summary table.
    """
    fname = path.split("\\")[-1].split("/")[-1]
    ctrl, wave = parse_condition_from_filename(fname)
    if waveform_override:
        wave = waveform_override
    if wave not in ("sine", "square"):
        # Require explicit choice if filename doesn't indicate it
        raise ValueError(f"Waveform unknown for {fname}. Pass waveform_override='sine' or 'square'.")

    df = load_timeseries_xlsx(path, waveform=wave, sheet_name=None, header_row_index=3)

    if wave == "square":
        idx = find_square_cycle(df, col="theta_des", min_period_s=0.3)
        step = step_metrics(
            time=df.iloc[idx]["time"].to_numpy(),
            y=df.iloc[idx]["theta_act"].to_numpy(),
            target_high=float(df.iloc[idx]["theta_des"].max()),
            target_low=float(df.iloc[idx]["theta_des"].min()),
            side="rising" if df.iloc[idx]["theta_des"].iloc[-1] > df.iloc[idx]["theta_des"].iloc[0] else "falling",
        )
    else:
        idx = find_sine_cycle(df, col="theta_des", prefer_positive_slope=True)
        step = {"rise_time_10_90": np.nan, "percent_overshoot": np.nan, "settling_time_2pct": np.nan}

    cyc = compute_cycle_metrics(df, idx)

    out: Dict[str, float | str] = {
        "file": fname,
        "controller": ctrl,
        "waveform": wave,
    }
    out.update(cyc)
    out.update(step)
    return out

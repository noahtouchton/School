# helpers.py
# EML 4314C Lab 2 analysis helpers tailored to 6 specific columns
# ----------------------------------------------------------------
from __future__ import annotations

import os
from typing import Dict, Iterable, Optional, Tuple, List

import numpy as np
import pandas as pd

# ========================== Canonical schema / expected headers ==========================

# Final schema we return for downstream metrics
REQUIRED_COLS = ("time", "theta_des", "theta_act", "error", "command")

# Exact column names expected in the Excel files (header on line 4 -> header=3)
# We only rename THESE specific headers; anything else is ignored.
EXACT_HEADERS: Dict[str, str] = {
    "Time (s)": "time",
    "Command Signal": "command",
    "Filtered Platen Rotation Angle": "theta_act",
    "Platen Velocity": "theta_dot",  # optional; not used in metrics
    "Amplitude": "amplitude",
    "Frequency": "frequency",
}

# Optional meta columns (if you ever add them, they'll show up in the summary)
OPTIONAL_META_HEADERS: Dict[str, str] = {
    "Deadband": "deadband",
    "Deadband (deg)": "deadband",
    "Duty": "duty",
    "Duty (%)": "duty",
    "Duty Cycle": "duty",
    "Effort": "effort",
    "Effort (%)": "effort",
}

# Merge optionals into the rename map (safe: only applied if present)
EXACT_HEADERS.update(OPTIONAL_META_HEADERS)


# ========================== Desired trajectory reconstruction ==========================

def build_theta_des(
    t: np.ndarray, A: np.ndarray, f: np.ndarray, waveform: str, phase: float = 0.0
) -> np.ndarray:
    """
    Build desired angle in DEGREES given:
      - t [s], A [deg], f [Hz]
      - waveform ∈ {'sine', 'square'}
    """
    arg = 2 * np.pi * f * t + phase
    w = waveform.lower().strip() if waveform else ""
    if w == "sine":
        return A * np.sin(arg)
    if w == "square":
        s = np.sign(np.sin(arg))
        s[s == 0.0] = 1.0  # avoid zeros that break edge detection
        return A * s
    raise ValueError(f"Unknown waveform '{waveform}' (expected 'sine' or 'square').")


# ========================== Robust Excel loading ==========================

def _standardize_sheet(df0: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Rename only known headers; return a standardized DataFrame or None if it doesn't qualify.
    """
    if df0 is None or df0.empty:
        return None
    # Strict rename from provided headers
    ren = {col: EXACT_HEADERS.get(col, col) for col in df0.columns}
    df = df0.rename(columns=ren)

    # Must contain the 5 core inputs (theta_dot is optional)
    needed = {"time", "command", "theta_act", "amplitude", "frequency"}
    if not (needed <= set(df.columns)):
        return None

    df = df.dropna().sort_values("time").reset_index(drop=True)
    return df


def load_all_runs_xlsx(
    path: str,
    waveform: str,
    header_row_index: int = 3,
) -> List[Tuple[str, pd.DataFrame]]:
    """
    Read ALL sheets in the workbook, standardize columns, rebuild theta_des and error.
    Returns list of (sheet_label, df).
    """
    xls = pd.read_excel(path, sheet_name=None, header=header_row_index)
    out: List[Tuple[str, pd.DataFrame]] = []

    for name, df0 in xls.items():
        df = _standardize_sheet(df0)
        if df is None:
            continue

        # Rebuild theta_des (requires waveform)
        df["theta_des"] = build_theta_des(
            t=df["time"].to_numpy(float),
            A=df["amplitude"].to_numpy(float),
            f=df["frequency"].to_numpy(float),
            waveform=waveform,
            phase=0.0,
        )
        # Compute error
        df["error"] = df["theta_des"] - df["theta_act"]

        out.append((str(name) if name is not None else "sheet0", df))

    if not out:
        raise ValueError(
            f"No qualifying data sheets found in {path}. "
            f"Expected headers include: {list(EXACT_HEADERS.keys())}"
        )
    return out


# ========================== Metrics ==========================

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


# ========================== Cycle pickers ==========================

def find_square_cycle(df: pd.DataFrame, col: str = "theta_des", min_period_s: float = 0.3) -> slice:
    """
    One full square-wave cycle: rising-edge to next rising-edge.
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
    One full sine period: zero-crossing to next zero-crossing.
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


# ========================== Multi-run segmentation ==========================

def split_runs_by_time(df: pd.DataFrame, freq_hz: float | None = None) -> List[slice]:
    """
    Split a single sheet into multiple runs based on:
      - time resets (t decreases)
      - large time gaps (gap > max(0.25s, 0.5 * period) if frequency known; else 0.25s)
    Returns a list of index slices, one per run.
    """
    t = df["time"].to_numpy(float)
    if t.size < 3:
        return [slice(0, len(t))]

    dt = np.diff(t)
    T = (1.0 / freq_hz) if (freq_hz and freq_hz > 0) else None
    gap_thresh = max(0.25, 0.5 * T) if T else 0.25

    starts = [0]
    for i, dti in enumerate(dt):
        if dti < 0 or dti > gap_thresh:
            starts.append(i + 1)
    starts.append(len(t))

    slices: List[slice] = []
    for a, b in zip(starts[:-1], starts[1:]):
        if b - a >= 20:  # avoid tiny fragments
            slices.append(slice(a, b))
    if not slices:
        slices = [slice(0, len(t))]
    return slices


# ========================== High-level analyzers ==========================

def parse_condition_from_filename(fname: str) -> Tuple[str, str]:
    """
    Infer (controller, waveform) from file name tokens.
    """
    s = fname.lower()
    ctrl = "pid" if "pid" in s else ("bang-bang" if "bb" in s else "unknown")
    wave = "square" if "square" in s else ("sine" if "sine" in s else "unknown")
    return ctrl, wave


def analyze_file(
    path: str,
    waveform_override: Optional[str] = None
) -> Dict[str, float | str]:
    """
    Legacy single-run analyzer (kept for compatibility).
    Requires a clear waveform (either in filename or via override).
    """
    fname = os.path.basename(path)
    ctrl, wave = parse_condition_from_filename(fname)
    if waveform_override:
        wave = waveform_override
    if wave not in ("sine", "square"):
        raise ValueError(f"Waveform unknown for {fname}. Set waveform_override='sine' or 'square'.")

    # Load first qualifying sheet only
    sheets = load_all_runs_xlsx(path, waveform=wave, header_row_index=3)
    sheet_name, df = sheets[0]

    # Choose cycle & step metrics
    if wave == "square":
        cyc = find_square_cycle(df, col="theta_des", min_period_s=0.3)
        step = step_metrics(
            time=df.iloc[cyc]["time"].to_numpy(),
            y=df.iloc[cyc]["theta_act"].to_numpy(),
            target_high=float(df.iloc[cyc]["theta_des"].max()),
            target_low=float(df.iloc[cyc]["theta_des"].min()),
            side="rising" if df.iloc[cyc]["theta_des"].iloc[-1] > df.iloc[cyc]["theta_des"].iloc[0] else "falling",
        )
    else:
        cyc = find_sine_cycle(df, col="theta_des", prefer_positive_slope=True)
        step = {"rise_time_10_90": np.nan, "percent_overshoot": np.nan, "settling_time_2pct": np.nan}

    cyc_stats = compute_cycle_metrics(df, cyc)

    out: Dict[str, float | str] = {
        "file": fname,
        "sheet": sheet_name,
        "run_index": 1,
        "controller": ctrl,
        "waveform": wave,
    }
    # include optional meta if present
    for meta in ("deadband", "duty", "effort"):
        if meta in df.columns:
            out[meta] = float(df[meta].iloc[0])

    out.update(cyc_stats)
    out.update(step)
    return out


def analyze_file_multi(
    path: str,
    waveform_override: Optional[str] = None
) -> List[Dict[str, float | str]]:
    """
    Analyze EVERY qualifying sheet and EVERY run (split by time resets/gaps).
    Returns a list of summary rows (one per run).
    """
    fname = os.path.basename(path)
    ctrl, wave = parse_condition_from_filename(fname)
    if waveform_override:
        wave = waveform_override
    if wave not in ("sine", "square"):
        raise ValueError(f"Waveform unknown for {fname}. Set waveform_override='sine' or 'square'.")

    rows: List[Dict[str, float | str]] = []
    sheet_runs = load_all_runs_xlsx(path, waveform=wave, header_row_index=3)

    for sheet_name, df in sheet_runs:
        f_med = float(np.median(df["frequency"].to_numpy())) if "frequency" in df.columns else None
        segments = split_runs_by_time(df, freq_hz=f_med)

        for k, idx in enumerate(segments, start=1):
            sub = df.iloc[idx].reset_index(drop=True)

            if wave == "square":
                cyc = find_square_cycle(sub, col="theta_des", min_period_s=0.3)
                step = step_metrics(
                    time=sub.iloc[cyc]["time"].to_numpy(),
                    y=sub.iloc[cyc]["theta_act"].to_numpy(),
                    target_high=float(sub.iloc[cyc]["theta_des"].max()),
                    target_low=float(sub.iloc[cyc]["theta_des"].min()),
                    side="rising" if sub.iloc[cyc]["theta_des"].iloc[-1] > sub.iloc[cyc]["theta_des"].iloc[0] else "falling",
                )
            else:
                cyc = find_sine_cycle(sub, col="theta_des", prefer_positive_slope=True)
                step = {"rise_time_10_90": np.nan, "percent_overshoot": np.nan, "settling_time_2pct": np.nan}

            cyc_stats = compute_cycle_metrics(sub, cyc)

            row: Dict[str, float | str] = {
                "file": fname,
                "sheet": sheet_name,
                "run_index": k,
                "controller": ctrl,
                "waveform": wave,
            }
            # Include optional meta if present (constant per run recommended)
            for meta in ("deadband", "duty", "effort"):
                if meta in sub.columns:
                    row[meta] = float(sub[meta].iloc[0])

            row.update(cyc_stats)
            row.update(step)
            rows.append(row)

    return rows

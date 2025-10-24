# analyze_lab2_mixed.py
# EML 4314C Lab 2 — handles MIXED sine/square & PID/BB segments within the same file.
# Computes per-cycle error/command stats and square step metrics; writes tidy CSV/XLSX tables.

from __future__ import annotations

import os, glob, re, unicodedata
from typing import Dict, Iterable, List, Optional, Tuple
import numpy as np
import pandas as pd

# ------------------------------- Paths & Files -------------------------------

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(BASE_DIR, "data")
OUT_DIR   = os.path.join(BASE_DIR, "outputs")
TABLE_DIR = os.path.join(OUT_DIR, "tables")
os.makedirs(TABLE_DIR, exist_ok=True)

# Your five files; script will also match case variants
EXPECTED_FILES = [
    "BB_Sine.xlsx",
    "BB_Square.xlsx",
    "PID_ack.xlsx",
    "PID_Manual.xlsx",
    "PID_prop.xlsx",
]

HEADER_ROW_INDEX = 3  # header is on line 4 in Excel

# ------------------------------- Header Mapping ------------------------------

def _norm_col(s: str) -> str:
    if not isinstance(s, str): s = str(s)
    s = unicodedata.normalize("NFKC", s).replace("\u200b","").replace("\ufeff","").strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^a-z0-9/ ]+", "", s)
    return s

REN_MAP = {
    "time": "time",
    "command": "command",
    "command signal": "command",
    "filtered platen rotation angle": "theta_act",
    "filtered platen": "theta_act",
    "platen rotation angle": "theta_act",
    "platen velocity": "theta_dot",
    "platen vel": "theta_dot",
    "amplitude": "amplitude",
    "frequency": "frequency",
    "sine/square": "is_sine_flag",
    "sine/squa": "is_sine_flag",
    "error": "error",
    "pid/bb": "is_pid_flag",
    "deadband": "deadband",
    "duty cycle": "duty_cycle",
    "duty": "duty_cycle",
    "kp": "Kp",
    "ki": "Ki",
    "kd": "Kd",
}

def _rename_columns_fuzzy(df_raw: pd.DataFrame) -> pd.DataFrame:
    normed = {c: _norm_col(c) for c in df_raw.columns}
    out = {}
    for orig, nc in normed.items():
        if nc in REN_MAP:
            out[orig] = REN_MAP[nc]; continue
        for k,v in REN_MAP.items():
            if nc.startswith(k):
                out[orig] = v; break
    return df_raw.rename(columns=out)

# ------------------------------- Utilities -----------------------------------

def _as_np(a: Iterable[float]) -> np.ndarray:
    return np.asarray(list(a), dtype=float)

def basic_stats(x: Iterable[float]) -> Dict[str, float]:
    x = _as_np(x)
    if x.size == 0:
        return {"mean": np.nan, "std": np.nan, "rms": np.nan}
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x, ddof=1)) if x.size > 1 else 0.0,
        "rms": float(np.sqrt(np.mean(x**2))),
    }

def build_theta_des_rowwise(t: np.ndarray, A: np.ndarray, f: np.ndarray, is_sine: np.ndarray) -> np.ndarray:
    arg = 2*np.pi*f*t
    sine_des   = A*np.sin(arg)
    square_des = A*np.sign(np.sin(arg))
    square_des[square_des == 0.0] = 1.0
    return np.where(is_sine, sine_des, square_des)

def find_sine_cycle(theta_des: np.ndarray) -> slice:
    zc = np.where(np.diff(np.signbit(theta_des)))[0]
    if zc.size < 2: return slice(0, len(theta_des))
    for i in range(zc.size - 1):
        a, b = zc[i], zc[i+1]
        if theta_des[b] - theta_des[a] > 0:
            return slice(a+1, b+1)
    return slice(zc[0]+1, zc[1]+1)

def find_square_cycle(theta_des: np.ndarray, t: np.ndarray, min_period_s: float = 0.3) -> slice:
    mid = 0.5*(np.nanmax(theta_des)+np.nanmin(theta_des))
    hi = theta_des > mid
    rising = np.where((~hi[:-1]) & (hi[1:]))[0] + 1
    if rising.size < 2: return slice(0, len(theta_des))
    for i in range(rising.size - 1):
        a,b = rising[i], rising[i+1]
        if (t[b]-t[a]) >= min_period_s:
            return slice(a, b)
    return slice(rising[0], rising[1])

def step_metrics(time: np.ndarray, y: np.ndarray, y_des: np.ndarray) -> Dict[str, float]:
    t = _as_np(time); y = _as_np(y); yd = _as_np(y_des)
    if t.size < 3:
        return {"rise_time_10_90": np.nan, "percent_overshoot": np.nan, "settling_time_2pct": np.nan}
    y_low, y_high = float(np.nanmin(yd)), float(np.nanmax(yd))
    rising = (y_high - y_low) > 0 and yd[-1] > yd[0]
    y0, y1 = (y_low, y_high) if rising else (y_high, y_low)
    yn = (y - y0) / (y1 - y0 + 1e-12)
    def first_cross(level: float) -> float:
        idx = np.where(yn >= level)[0] if rising else np.where(yn <= (1-level))[0]
        return float(t[idx[0]]) if idx.size else np.nan
    t10, t90 = first_cross(0.10), first_cross(0.90)
    tr = (t90 - t10) if np.isfinite(t10) and np.isfinite(t90) else np.nan
    OS = max(0.0, (np.max(yn) - 1.0)*100.0) if rising else max(0.0, (1.0 - np.min(yn))*100.0)
    band = np.abs(yn - 1.0) <= 0.02 if rising else np.abs(yn - 0.0) <= 0.02
    idx = np.where(band)[0]
    Ts = float(t[idx[0]] - t[0]) if idx.size else np.nan
    return {"rise_time_10_90": float(tr), "percent_overshoot": float(OS), "settling_time_2pct": float(Ts)}

# ------------------------------- Cleaning ------------------------------------

def clean_and_standardize(df_raw: pd.DataFrame, *, filename: str, validate_error: bool = True) -> pd.DataFrame:
    df = _rename_columns_fuzzy(df_raw)

    required = {"time","command","theta_act","amplitude","frequency"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns after renaming: {missing} (got: {list(df.columns)})")

    # numerics
    for c in ("time","command","theta_act","theta_dot","amplitude","frequency","error",
              "deadband","duty_cycle","Kp","Ki","Kd","is_sine_flag","is_pid_flag"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # waveform/controller flags: row-wise booleans (default from filename if missing)
    fn = filename.lower()
    if "is_sine_flag" in df.columns:
        # Flip the sine/square flag: 1→square, 0→sine
        is_sine = (~(df["is_sine_flag"].fillna(0) > 0.5)).astype(bool).to_numpy()
        # IMPORTANT: write flipped flag back so downstream uses the corrected labeling
        df["is_sine_flag"] = is_sine.astype(int)
    else:
        default_sine = not (("sine" in fn) and ("square" not in fn))
        is_sine = np.full(len(df), default_sine, dtype=bool)
        # Store the (unflipped) original meaning? No—store the final, correct sine flag
        df["is_sine_flag"] = is_sine.astype(int)

    if "is_pid_flag" in df.columns:
        is_pid = (df["is_pid_flag"].fillna(0) > 0.5).astype(bool).to_numpy()
    else:
        default_pid = ("pid" in fn) and ("bb" not in fn)
        is_pid = np.full(len(df), default_pid, dtype=bool)
        df["is_pid_flag"] = is_pid.astype(int)

    # Build theta_des ROW-WISE using the per-sample (flipped-correct) waveform flag
    df["theta_des"] = build_theta_des_rowwise(
        df["time"].to_numpy(float),
        df["amplitude"].to_numpy(float),
        df["frequency"].to_numpy(float),
        is_sine=is_sine
    )

    # error: prefer LabVIEW column
    error_source = "LabVIEW" if "error" in df.columns and not df["error"].isna().all() else "computed"
    if error_source == "computed":
        df["error"] = df["theta_des"] - df["theta_act"]
    elif validate_error:
        err_lv = df["error"].to_numpy(float)
        err_calc = (df["theta_des"] - df["theta_act"]).to_numpy(float)
        diff_rms = float(np.sqrt(np.nanmean((err_lv - err_calc)**2)))
        base_rms = float(np.sqrt(np.nanmean(err_lv**2))) if np.isfinite(err_lv).any() else np.nan
        if np.isfinite(diff_rms) and np.isfinite(base_rms) and (diff_rms > 0.1*base_rms and diff_rms > 1.0):
            print(f"[WARN] {filename}: Error column differs from θ_des-θ_act (diff_rms≈{diff_rms:.2f}, base_rms≈{base_rms:.2f}). Using LabVIEW 'Error'.")
    df["error_source"] = error_source

    df = df.dropna(subset=["time","command","theta_act","amplitude","frequency","error"]).sort_values("time").reset_index(drop=True)
    return df


# ----------------------- Segmentation (time + flags + params) ----------------

def _changed(a: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Return boolean of same length as a, True where value changes from previous (first is False)."""
    a = np.asarray(a)
    out = np.zeros(len(a), dtype=bool)
    if len(a) <= 1: return out
    if a.dtype.kind in "f":
        out[1:] = np.abs(np.diff(a)) > eps
    else:
        out[1:] = np.diff(a) != 0
    return out

def split_runs(df: pd.DataFrame) -> List[slice]:
    """
    Split into segments when:
      - time resets or big gaps
      - Sine/Square flag changes
      - PID/BB flag changes
      - deadband or duty_cycle changes (bang-bang)
      - Kp/Ki/Kd changes (PID)
    """
    t = df["time"].to_numpy(float)
    if len(t) < 3:
        return [slice(0, len(df))]

    # time-based boundaries
    dt = np.diff(t)
    f_med = float(np.nanmedian(df["frequency"])) if "frequency" in df.columns else np.nan
    T = 1.0 / f_med if np.isfinite(f_med) and f_med > 0 else None
    gap_thresh = max(0.25, 0.5*T) if T else 0.25
    cut = np.zeros(len(df), dtype=bool)
    cut[1:] |= dt < 0
    cut[1:] |= dt > gap_thresh

    # flag boundaries
    if "is_sine_flag" in df.columns:
        cut |= _changed((df["is_sine_flag"].fillna(0) > 0.5).astype(int).to_numpy())
    if "is_pid_flag" in df.columns:
        cut |= _changed((df["is_pid_flag"].fillna(0) > 0.5).astype(int).to_numpy())

    # parameter boundaries
    for c in ("deadband","duty_cycle","Kp","Ki","Kd"):
        if c in df.columns:
            cut |= _changed(df[c].fillna(method="ffill").fillna(0).to_numpy(), eps=1e-9)

    # build slices
    idx = np.where(cut)[0]
    starts = [0] + idx.tolist()
    starts = sorted(set(starts))
    ends = starts[1:] + [len(df)]
    segments: List[slice] = []
    for a,b in zip(starts, ends):
        if b - a >= 20:
            segments.append(slice(a,b))
    return segments or [slice(0, len(df))]

# ------------------------------- Analysis ------------------------------------

def analyze_cycle(df: pd.DataFrame) -> Dict[str, float | str | int]:
    # decide dominant waveform for this segment
    is_sine_seg = (df["is_sine_flag"].fillna(0) > 0.5).mean() > 0.5
    wave = "sine" if is_sine_seg else "square"

    t = df["time"].to_numpy(float)
    des = df["theta_des"].to_numpy(float)
    if wave == "square":
        cyc = find_square_cycle(des, t, min_period_s=0.3)
    else:
        cyc = find_sine_cycle(des)

    sub = df.iloc[cyc].reset_index(drop=True)

    err_s = basic_stats(sub["error"])
    cmd_s = basic_stats(sub["command"])

    row: Dict[str, float | str | int] = {
        "waveform": wave,
        "controller": "pid" if (df["is_pid_flag"].fillna(0) > 0.5).mean() > 0.5 else "bang-bang",
        "error_mean": err_s["mean"],
        "error_std": err_s["std"],
        "error_rms": err_s["rms"],
        "command_mean": cmd_s["mean"],
        "command_std": cmd_s["std"],
        "command_rms": cmd_s["rms"],
        "cycle_duration": float(sub["time"].iloc[-1] - sub["time"].iloc[0]) if len(sub) > 1 else np.nan,
        "N": int(len(sub)),
        "error_source": df["error_source"].iloc[0],
        # carry knobs (take first non-NaN)
        "deadband": float(sub["deadband"].dropna().iloc[0]) if "deadband" in sub.columns and sub["deadband"].notna().any() else None,
        "duty_cycle": float(sub["duty_cycle"].dropna().iloc[0]) if "duty_cycle" in sub.columns and sub["duty_cycle"].notna().any() else None,
        "Kp": float(sub["Kp"].dropna().iloc[0]) if "Kp" in sub.columns and sub["Kp"].notna().any() else None,
        "Ki": float(sub["Ki"].dropna().iloc[0]) if "Ki" in sub.columns and sub["Ki"].notna().any() else None,
        "Kd": float(sub["Kd"].dropna().iloc[0]) if "Kd" in sub.columns and sub["Kd"].notna().any() else None,
    }

    if wave == "square":
        sm = step_metrics(sub["time"].to_numpy(float),
                          sub["theta_act"].to_numpy(float),
                          sub["theta_des"].to_numpy(float))
        row.update(sm)
    else:
        row.update({"rise_time_10_90": np.nan, "percent_overshoot": np.nan, "settling_time_2pct": np.nan})
    return row

def analyze_file(path: str) -> List[Dict[str, float | str | int]]:
    fname = os.path.basename(path)
    xls = pd.read_excel(path, sheet_name=None, header=HEADER_ROW_INDEX)
    if not isinstance(xls, dict): xls = {"Sheet1": xls}

    rows: List[Dict[str, float | str | int]] = []
    for sheet_name, df0 in xls.items():
        if df0 is None or df0.empty: continue
        df = clean_and_standardize(df0, filename=fname, validate_error=True)

        segments = split_runs(df)
        for i, sl in enumerate(segments, start=1):
            sub = df.iloc[sl].reset_index(drop=True)
            row = analyze_cycle(sub)
            row.update({"file": fname, "sheet": str(sheet_name), "run_index": i})
            rows.append(row)
    return rows

# ------------------------------- Driver --------------------------------------

def main():
    print(f"[INFO] Data directory: {DATA_DIR}")

    paths: List[str] = []
    for base in EXPECTED_FILES:
        p = os.path.join(DATA_DIR, base)
        if os.path.isfile(p):
            paths.append(p)
        else:
            matches = glob.glob(os.path.join(DATA_DIR, f"*{os.path.splitext(base)[0]}*.xlsx"))
            paths.extend(matches)
    paths = sorted(set(paths), key=lambda s: s.lower())

    if not paths:
        print("[ERR] No expected files found in data/.")
        return

    all_rows: List[Dict[str, float | str | int]] = []
    for p in paths:
        try:
            rows = analyze_file(p)
            all_rows.extend(rows)
            print(f"[OK] {os.path.basename(p)} (+{len(rows)} segments)")
        except Exception as e:
            print(f"[ERR] {os.path.basename(p)}: {e}")

    if not all_rows:
        print("[ERR] No results produced.")
        return

    df = pd.DataFrame(all_rows)

    # Preferred column order
    preferred = [
        "file","sheet","run_index","controller","waveform",
        "deadband","duty_cycle","Kp","Ki","Kd",
        "error_source",
        "error_mean","error_std","error_rms",
        "command_mean","command_std","command_rms",
        "cycle_duration","N",
        "rise_time_10_90","percent_overshoot","settling_time_2pct",
    ]
    cols = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]
    df = df[cols]

    # Save main table
    os.makedirs(TABLE_DIR, exist_ok=True)
    csv_all  = os.path.join(TABLE_DIR, "metrics_all_runs.csv")
    xlsx_all = os.path.join(TABLE_DIR, "metrics_all_runs.xlsx")
    df.to_csv(csv_all, index=False)
    with pd.ExcelWriter(xlsx_all) as w:
        df.to_excel(w, index=False, sheet_name="AllRuns")

    # Square-only second-order table
    square = df[df["waveform"] == "square"].copy()
    second_order_cols = [
        "file","sheet","run_index","controller",
        "deadband","duty_cycle","Kp","Ki","Kd",
        "rise_time_10_90","percent_overshoot","settling_time_2pct",
    ]
    square = square[[c for c in second_order_cols if c in square.columns]]
    csv_sq  = os.path.join(TABLE_DIR, "metrics_square_second_order.csv")
    xlsx_sq = os.path.join(TABLE_DIR, "metrics_square_second_order.xlsx")
    square.to_csv(csv_sq, index=False)
    with pd.ExcelWriter(xlsx_sq) as w:
        square.to_excel(w, index=False, sheet_name="Square2ndOrder")

    # Aggregations by condition
    group_keys = ["controller","waveform"]
    for c in ("deadband","duty_cycle"): 
        if c in df.columns and df[c].notna().any(): group_keys.append(c)
    for c in ("Kp","Ki","Kd"):
        if c in df.columns and df[c].notna().any(): group_keys.append(c)

    agg_cols = {
        "error_rms": ["mean","min","max"],
        "command_rms": ["mean","min","max"],
        "rise_time_10_90": ["mean"],
        "percent_overshoot": ["mean"],
        "settling_time_2pct": ["mean"],
    }
    grouped = df.groupby(group_keys).agg(agg_cols)
    grouped.columns = [f"{a}_{b}" if b else a for a,b in grouped.columns]
    grouped = grouped.reset_index()

    csv_grp  = os.path.join(TABLE_DIR, "metrics_by_condition.csv")
    xlsx_grp = os.path.join(TABLE_DIR, "metrics_by_condition.xlsx")
    grouped.to_csv(csv_grp, index=False)
    with pd.ExcelWriter(xlsx_grp) as w:
        grouped.to_excel(w, index=False, sheet_name="ByCondition")

    print("\n[INFO] Saved:")
    print(" -", csv_all)
    print(" -", xlsx_all)
    print(" -", csv_sq)
    print(" -", xlsx_sq)
    print(" -", csv_grp)
    print(" -", xlsx_grp)

if __name__ == "__main__":
    main()

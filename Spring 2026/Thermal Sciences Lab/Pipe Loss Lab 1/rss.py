# rss.py
import math
import numpy as np
import sympy as sp
import pandas as pd
from pathlib import Path


# ============================================================
# Data class
# ============================================================
class Data:
    def __init__(self, name, var, value, uncertainty=0.0):
        self.name = name
        self.var = var
        self.value = float(value) if np.isscalar(value) else value
        self.uncertainty = float(uncertainty) if np.isscalar(uncertainty) else uncertainty

    def get_error(self):
        print(f"{self.name}: {self.value:.6f} ± {self.uncertainty:.6f}")

    def __repr__(self):
        return f"{self.name}: {self.value:.6f} ± {self.uncertainty:.6f}"


# ============================================================
# RSS uncertainty propagation (kept simple, your style)
# ============================================================
def RSS(name, function, data_list):
    """
    function example: "y = a*x + b"
    data_list: list of Data objects, where Data.var is a string name for the variable
    """
    function = function.replace(" ", "")
    if "=" not in function:
        raise ValueError("RSS() expects an '=' in the function string.")

    lhs, rhs = function.split("=", 1)

    # Build symbol map and substitution map
    sym_map = {}
    subs = {}

    for d in data_list:
        s = sp.Symbol(str(d.var))
        sym_map[str(d.var)] = s
        subs[s] = d.value

    expr = sp.sympify(rhs, locals=sym_map)

    # Nominal value
    ordered_syms = list(sym_map.values())
    f = sp.lambdify(ordered_syms, expr, modules="numpy")
    nominal = f(*[subs[s] for s in ordered_syms])

    # RSS
    rss_sq = 0.0
    for d in data_list:
        s = sym_map[str(d.var)]
        dfdx = sp.diff(expr, s)
        dfdx_f = sp.lambdify(ordered_syms, dfdx, modules="numpy")
        dfdx_val = dfdx_f(*[subs[sym] for sym in ordered_syms])
        rss_sq += (dfdx_val * d.uncertainty) ** 2

    unc = np.sqrt(rss_sq)

    # Return as Data
    return Data(name, lhs, float(nominal), float(unc))


# ============================================================
# Pipe Loss Lab constants
# ============================================================
PCTS = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

# Flowmeter full scales (LPM) based on your Proteus list:
# - Small Flowmeter 0205C24: 1.9–9.5 LPM  => FS = 9.5
# - Medium Flowmeter 0250B* : 6–45 LPM    => FS = 45
# - Large Flowmeter 0270P24 : 35–225 LPM  => FS = 225
FLOW_FS_LOW  = 9.5
FLOW_FS_MED  = 45.0
FLOW_FS_HIGH = 225.0

FLOW_ACC_FS = 0.03  # ±3% of full scale (Proteus stated)


# ============================================================
# Run container (arrays)
# ============================================================
class RunArrays:
    def __init__(self, name):
        self.name = name
        self.pcts = PCTS

        self.Q_lpm = []
        self.dp_large_psid = []
        self.dp_small_psid = []
        self.dp_elbow_psid = []
        self.T_C = []


# ============================================================
# Data loading (robust for tab-dump "xls")
# ============================================================
def _make_unique_headers(headers):
    """
    Your files repeat header names like 'Diff press (PSID)' 3 times.
    Pandas will allow duplicates, but it breaks later selection.
    This makes them unique: 'Diff press (PSID)', 'Diff press (PSID)_1', ...
    """
    seen = {}
    out = []
    for h in headers:
        h = (h or "").strip()
        if h == "":
            h = "col"

        if h not in seen:
            seen[h] = 0
            out.append(h)
        else:
            seen[h] += 1
            out.append(f"{h}_{seen[h]}")
    return out


def load_lab_dataframe(file_path):
    """
    Reads your 'fake .xls' (actually tab-delimited text) and returns a clean DataFrame.

    - finds the real header row (starts with 'Iteration' and contains 'Time')
    - splits rows on tabs
    - makes headers unique
    - pads/truncates rows to header length
    - converts numeric values
    """
    file_path = Path(file_path)
    raw = file_path.read_text(errors="ignore")

    lines = [ln for ln in raw.splitlines() if ln.strip()]
    rows = [ln.split("\t") for ln in lines]

    header_idx = None
    for i, r in enumerate(rows):
        # Find the row that contains Iteration and Time (seconds)
        joined = " ".join(r)
        if joined.startswith("Iteration") and ("Time" in joined):
            header_idx = i
            break

    if header_idx is None:
        # fallback: any row containing Iteration and Time
        for i, r in enumerate(rows):
            joined = " ".join(r)
            if ("Iteration" in joined) and ("Time" in joined):
                header_idx = i
                break

    if header_idx is None:
        raise ValueError(f"Could not find header row in: {file_path}")

    header = _make_unique_headers([h.strip() for h in rows[header_idx]])
    data_rows = rows[header_idx + 1 :]
    ncol = len(header)

    fixed = []
    for r in data_rows:
        r = [x.strip() for x in r]
        if len(r) < ncol:
            r = r + [""] * (ncol - len(r))
        else:
            r = r[:ncol]
        fixed.append(r)

    df = pd.DataFrame(fixed, columns=header)

    # Convert to numeric when possible
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Drop empty columns
    df = df.dropna(axis=1, how="all")

    return df


# ============================================================
# Helpers
# ============================================================
def mean_std(series):
    s = series.dropna()
    if len(s) < 3:
        return float("nan"), float("nan"), 0
    return float(s.mean()), float(s.std(ddof=1)), int(len(s))


def pick_active_flow(df):
    """
    Picks the flow column with largest mean absolute value.
    Closed meters hover near 0 / negative noise.
    """
    cols = [c for c in df.columns if c.startswith("Flowrate") and "(lpm)" in c]
    if not cols:
        raise ValueError("No flowrate columns found (expected 'Flowrate * (lpm)').")

    scores = []
    for c in cols:
        s = df[c].dropna()
        scores.append(float(np.mean(np.abs(s))) if len(s) else -np.inf)

    return cols[int(np.argmax(scores))]


def flow_inst_unc_from_col(col_name):
    """
    Returns ±3% of the correct full-scale range depending on low/medium/high meter.
    """
    c = col_name.lower()

    if "low" in c:
        fs = FLOW_FS_LOW
    elif "medium" in c:
        fs = FLOW_FS_MED
    elif "high" in c:
        fs = FLOW_FS_HIGH
    else:
        # If the column naming ever changes, fail loudly
        raise ValueError(f"Cannot determine flow range from column name: {col_name}")

    return FLOW_ACC_FS * fs


def dp_columns(df):
    """
    Returns the three diff pressure columns in the order they appear in the file:
      Large Pipe, Small Pipe, Elbow
    (because your header row lists them in that order)
    """
    cols = [c for c in df.columns if c.startswith("Diff press (PSID)")]
    if len(cols) < 3:
        raise ValueError(f"Expected 3 diff pressure columns, found: {cols}")

    ordered = [c for c in df.columns if c in cols]
    return ordered[0], ordered[1], ordered[2]


# ============================================================
# Operating point extraction
# ============================================================
def operating_point(file_path, label, dp_inst_unc, temp_inst_unc):
    df = load_lab_dataframe(file_path)

    flow_col = pick_active_flow(df)
    dpL_col, dpS_col, dpE_col = dp_columns(df)

    Qm, Qs, _ = mean_std(df[flow_col])
    dLm, dLs, _ = mean_std(df[dpL_col])
    dSm, dSs, _ = mean_std(df[dpS_col])
    dEm, dEs, _ = mean_std(df[dpE_col])

    if "deg C" not in df.columns:
        raise ValueError("Temperature column 'deg C' not found.")
    Tm, Ts, _ = mean_std(df["deg C"])

    # Flow instrument uncertainty depends on which meter is active
    flow_inst_unc = flow_inst_unc_from_col(flow_col)

    # Total uncertainty = RSS(instrument, sample std)
    # (you said you want no steady trimming; this uses all samples)
    Q_unc  = math.sqrt(flow_inst_unc**2 + Qs**2)
    dL_unc = math.sqrt(dp_inst_unc**2 + dLs**2)
    dS_unc = math.sqrt(dp_inst_unc**2 + dSs**2)
    dE_unc = math.sqrt(dp_inst_unc**2 + dEs**2)
    T_unc  = math.sqrt(temp_inst_unc**2 + Ts**2)

    Q  = Data(f"{label} Flow Rate", "Q_lpm", Qm, Q_unc)
    dpL = Data(f"{label} dP Large", "dpL_psid", dLm, dL_unc)
    dpS = Data(f"{label} dP Small", "dpS_psid", dSm, dS_unc)
    dpE = Data(f"{label} dP Elbow", "dpE_psid", dEm, dE_unc)
    T  = Data(f"{label} Water Temp", "T_C", Tm, T_unc)

    return Q, dpL, dpS, dpE, T


# ============================================================
# Run loaders (arrays)
# ============================================================
def load_large_run(run_num, base_dir="data", dp_inst_unc=0.0, temp_inst_unc=0.0):
    run = RunArrays(f"Large Run {run_num}")
    base_dir = Path(base_dir)

    run_dir = base_dir / "Large Pipe" / f"Run {run_num}"

    for pct in PCTS:
        f = run_dir / f"largepipe_run{run_num}_{pct}.xls"
        label = f"Large Run {run_num} {pct}%"

        Q, dpL, dpS, dpE, T = operating_point(f, label, dp_inst_unc, temp_inst_unc)

        run.Q_lpm.append(Q)
        run.dp_large_psid.append(dpL)
        run.dp_small_psid.append(dpS)
        run.dp_elbow_psid.append(dpE)
        run.T_C.append(T)

    return run


def load_small_run1(base_dir="data", dp_inst_unc=0.0, temp_inst_unc=0.0):
    run = RunArrays("Small Run 1")
    base_dir = Path(base_dir)

    run_dir = base_dir / "Small Pipe"

    for pct in PCTS:
        f = run_dir / f"smallpipe_run1_{pct}.xls"
        label = f"Small Run 1 {pct}%"

        Q, dpL, dpS, dpE, T = operating_point(f, label, dp_inst_unc, temp_inst_unc)

        run.Q_lpm.append(Q)
        run.dp_large_psid.append(dpL)
        run.dp_small_psid.append(dpS)
        run.dp_elbow_psid.append(dpE)
        run.T_C.append(T)

    return run
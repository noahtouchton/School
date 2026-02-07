# rss.py
import math
import numpy as np
import sympy as sp
import pandas as pd
from pathlib import Path

# ============================================================
# Data class (now stores components: std + instrument)
# ============================================================
class Data:
    def __init__(self, name, var, value, uncertainty=0.0, std=0.0, inst_unc=0.0):
        self.name = name
        self.var = var
        self.value = float(value) if np.isscalar(value) else value

        # Total uncertainty (RSS of components)
        self.uncertainty = float(uncertainty) if np.isscalar(uncertainty) else uncertainty

        # Components (for lab tables)
        self.std = float(std) if np.isscalar(std) else std               # sample standard deviation
        self.inst_unc = float(inst_unc) if np.isscalar(inst_unc) else inst_unc  # instrument-only

    def get_error(self):
        print(f"{self.name}: {self.value:.6f} ± {self.uncertainty:.6f}")

    def __repr__(self):
        return f"{self.name}: {self.value:.6f} ± {self.uncertainty:.6f}"


# ============================================================
# RSS uncertainty propagation
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

    sym_map = {}
    subs = {}

    for d in data_list:
        s = sp.Symbol(str(d.var))
        sym_map[str(d.var)] = s
        subs[s] = d.value

    expr = sp.sympify(rhs, locals=sym_map)

    ordered_syms = list(sym_map.values())
    f = sp.lambdify(ordered_syms, expr, modules="numpy")
    nominal = f(*[subs[s] for s in ordered_syms])

    rss_sq = 0.0
    for d in data_list:
        s = sym_map[str(d.var)]
        dfdx = sp.diff(expr, s)
        dfdx_f = sp.lambdify(ordered_syms, dfdx, modules="numpy")
        dfdx_val = dfdx_f(*[subs[sym] for sym in ordered_syms])
        rss_sq += (dfdx_val * d.uncertainty) ** 2

    unc = np.sqrt(rss_sq)

    return Data(name, lhs, float(nominal), float(unc))


# ============================================================
# Pipe Loss Lab constants
# ============================================================
PCTS = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

# Flowmeter full scales (LPM)
FLOW_FS_LOW  = 9.5
FLOW_FS_MED  = 45.0
FLOW_FS_HIGH = 225.0

FLOW_ACC_FS = 0.03  # ±3% of full scale


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
        self.rho = []
        self.head_loss = []
        self.d = []
        self.A = []
        self.v = []
        self.Re = []
        self.f = []
        self.hl_elbow = []
        self.kl = []


# ============================================================
# Data loading (tab-dump "xls")
# ============================================================
def _make_unique_headers(headers):
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
    file_path = Path(file_path)
    raw = file_path.read_text(errors="ignore")

    lines = [ln for ln in raw.splitlines() if ln.strip()]
    rows = [ln.split("\t") for ln in lines]

    header_idx = None
    for i, r in enumerate(rows):
        joined = " ".join(r)
        if joined.startswith("Iteration") and ("Time" in joined):
            header_idx = i
            break

    if header_idx is None:
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

    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

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
    cols = [c for c in df.columns if c.startswith("Flowrate") and "(lpm)" in c]
    if not cols:
        raise ValueError("No flowrate columns found (expected 'Flowrate * (lpm)').")

    scores = []
    for c in cols:
        s = df[c].dropna()
        scores.append(float(np.mean(np.abs(s))) if len(s) else -np.inf)

    return cols[int(np.argmax(scores))]


def flow_inst_unc_from_col(col_name):
    c = col_name.lower()
    if "low" in c:
        fs = FLOW_FS_LOW
    elif "medium" in c:
        fs = FLOW_FS_MED
    elif "high" in c:
        fs = FLOW_FS_HIGH
    else:
        raise ValueError(f"Cannot determine flow range from column name: {col_name}")
    return FLOW_ACC_FS * fs


def dp_columns(df):
    cols = [c for c in df.columns if c.startswith("Diff press (PSID)")]
    if len(cols) < 3:
        raise ValueError(f"Expected 3 diff pressure columns, found: {cols}")
    ordered = [c for c in df.columns if c in cols]
    return ordered[0], ordered[1], ordered[2]


# ============================================================
# Operating point extraction
# ============================================================
def operating_point(file_path, label, dp_inst_unc, temp_inst_unc):
    """
    dp_inst_unc can be:
      - float: same instrument uncertainty for all DP channels
      - (dpL_inst, dpS_inst, dpE_inst): per-channel instrument uncertainty
    """
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

    flow_inst_unc = flow_inst_unc_from_col(flow_col)

    if isinstance(dp_inst_unc, (list, tuple)) and len(dp_inst_unc) == 3:
        dpL_inst, dpS_inst, dpE_inst = dp_inst_unc
    else:
        dpL_inst = dpS_inst = dpE_inst = float(dp_inst_unc)

    # Total uncertainty = RSS(instrument, sample std)
    Q_unc  = math.sqrt(flow_inst_unc**2 + Qs**2)
    dL_unc = math.sqrt(dpL_inst**2 + dLs**2)
    dS_unc = math.sqrt(dpS_inst**2 + dSs**2)
    dE_unc = math.sqrt(dpE_inst**2 + dEs**2)
    T_unc  = math.sqrt(temp_inst_unc**2 + Ts**2)

    # Vars match your equations: Q, dp, T
    Q   = Data(f"{label} Flow Rate",  "Qv",  Qm,  Q_unc,  std=Qs, inst_unc=flow_inst_unc)
    dpL = Data(f"{label} dP Large",   "dp", dLm, dL_unc, std=dLs, inst_unc=dpL_inst)
    dpS = Data(f"{label} dP Small",   "dp", dSm, dS_unc, std=dSs, inst_unc=dpS_inst)
    dpE = Data(f"{label} dP Elbow",   "dp", dEm, dE_unc, std=dEs, inst_unc=dpE_inst)
    T   = Data(f"{label} Water Temp", "T",  Tm,  T_unc,  std=Ts, inst_unc=temp_inst_unc)

    return Q, dpL, dpS, dpE, T


# ============================================================
# Run loaders
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

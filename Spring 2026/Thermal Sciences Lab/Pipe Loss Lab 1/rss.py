#rss.py
import math
import numpy as np
import sympy as sp
from numpy.polynomial import Polynomial
from pathlib import Path
import pandas as pd

class Data:
    def __init__(self, name, var, value, uncertainty=0.0):
        self.name = name
        self.var = var
        self.value = value
        self.uncertainty = uncertainty

    def get_error(self):
        """
        Pretty-print nominal value(s) and uncertainty(ies).

        - If value & uncertainty are both scalar -> print single line.
        - If value is array-like -> print table of index, value, uncertainty.
          Scalar uncertainty is broadcast to all entries.
        """
        # Convert to arrays for uniform handling
        v = np.asarray(self.value, dtype=float)
        u = np.asarray(self.uncertainty, dtype=float)

        # Case 1: true scalar (0-d array or Python scalar) AND scalar uncertainty
        if v.ndim == 0 and u.ndim == 0:
            print(f"{self.name}: {float(v):.6f} ± {float(u):.6f}")
            return

        # Case 2: array-like values
        v = v.ravel()  # flatten to 1D

        # Broadcast uncertainty if needed
        if u.ndim == 0:          # scalar uncertainty -> same for all
            u = np.full_like(v, float(u))
        else:
            u = u.ravel()
            if u.size == 1:      # 1-element array -> treat as scalar
                u = np.full_like(v, float(u[0]))
            elif u.size != v.size:
                raise ValueError(
                    f"Uncertainty length {u.size} does not match value length {v.size} "
                    f"for '{self.name}'."
                )

        n = len(v)
        print(f"\n{self.name} ({n} values):")
        print(f"{'Index':>6} | {'Value':>12} | {'± Uncertainty':>15}")
        print("-" * 38)
        for i, (val, err) in enumerate(zip(v, u)):
            print(f"{i:6d} | {val:12.6f} | {err:15.6f}")
        print("-" * 38)



    def get_description(self):
        vname = self.var.name if isinstance(self.var, sp.Symbol) else str(self.var)
        v, u = self.value, self.uncertainty
        if np.isscalar(v) and np.isscalar(u):
            print(f"{self.name}, {vname}: {v:.3e} ± {u:.3e}")
        else:
            print(f"{self.name}, {vname}: array(shape={np.asarray(v).shape})")

    def print_to_excel(self):
        print(f"{self.name} values:")
        for val in self.value:
            print(val)
        print(f"{self.name} uncertainties:")
        for uncert in self.uncertainty:
            print(uncert)

def create_dict(data_list):
    values = {}
    for d in data_list:
        key = d.var.name if isinstance(d.var, sp.Symbol) else str(d.var)
        values[key] = d.value
    return values


def RSS(name, function, data_list):
    """
    Root-sum-square uncertainty of f(...).
    Supports scalar or NumPy array inputs. Uncertainties may be scalar
    or arrays broadcastable to the input shape.

    function: string like "y = a*x + b" (spaces allowed)
    """
    function = function.replace(" ", "")
    if "=" not in function:
        print("Please enter a valid function: expected an '='.")
        return None

    lhs, rhs = function.split("=", 1)

    # Build symbol maps (string name -> sympy.Symbol) and value map (Symbol -> numeric/array)
    sym_map = {}
    subs_map = {}
    for d in data_list:
        if isinstance(d.var, sp.Symbol):
            sym = d.var
            var_name = sym.name
        else:
            var_name = str(d.var)
            sym = sp.Symbol(var_name)
        sym_map[var_name] = sym
        subs_map[sym] = d.value  # may be scalar or np.array

    lhs_sym = sp.Symbol(lhs)

    # Parse RHS into a SymPy expression
    try:
        expr = sp.sympify(rhs, locals=sym_map)
    except Exception as e:
        print(f"Could not parse RHS '{rhs}': {e}")
        return None

    # Check that all symbols in expr have values
    missing = [s for s in expr.free_symbols if s not in subs_map]
    if missing:
        names = ", ".join(sorted([s.name for s in missing]))
        print(f"Missing values for: {names}")
        return None

    # Stable argument order for lambdify (sorted by variable name)
    sym_order = [sym_map[name] for name in sorted(sym_map.keys(), key=lambda n: n)]
    arg_values = [subs_map[s] for s in sym_order]

    # Nominal value via lambdify (handles arrays)
    try:
        expr_func = sp.lambdify(tuple(sym_order), expr, modules="numpy")
        nominal_value = expr_func(*arg_values)
    except Exception as e:
        print(f"Could not evaluate expression '{rhs}': {e}")
        return None

    # RSS uncertainty: sum_i ( (∂f/∂x_i * u_i)^2 )
    rss_sq = 0.0
    for d in data_list:
        sym = d.var if isinstance(d.var, sp.Symbol) else sym_map[str(d.var)]
        dfdxi_expr = sp.diff(expr, sym)
        dfdxi_func = sp.lambdify(tuple(sym_order), dfdxi_expr, modules="numpy")
        dfdxi_val = dfdxi_func(*arg_values)  # scalar or array
        u = d.uncertainty                  # scalar or array
        rss_sq = rss_sq + (dfdxi_val * u) ** 2

    uncertainty = np.sqrt(rss_sq)

    # Convert 0-d arrays (numpy scalars) to Python scalars for clean printing
    if isinstance(nominal_value, np.ndarray) and nominal_value.shape == ():
        nominal_value = float(nominal_value)
    if isinstance(uncertainty, np.ndarray) and uncertainty.shape == ():
        uncertainty = float(uncertainty)

    return Data(str(name), lhs_sym, nominal_value, uncertainty)

def linear_monte_carlo(data_x: Data, data_y: Data, N=100_000):


    xv = np.asarray(data_x.value, dtype=float).ravel()
    yv = np.asarray(data_y.value, dtype=float).ravel()

    if xv.size != yv.size:
        raise ValueError("x and y must have the same number of points.")
    if xv.size < 2:
        raise ValueError("Need at least two (x, y) points to fit a line.")

    sx = np.asarray(data_x.uncertainty, dtype=float).ravel()
    sy = np.asarray(data_y.uncertainty, dtype=float).ravel()

    rand_data_x = np.random.normal(loc=xv, scale=sx/2, size=(N, xv.size))
    rand_data_y = np.random.normal(loc=yv, scale=sy/2, size=(N, yv.size))

    Xmean = rand_data_x.mean(axis=1, keepdims=True)
    Ymean = rand_data_y.mean(axis=1, keepdims=True)
    Xm = rand_data_x - Xmean
    Ym = rand_data_y - Ymean
    Sxx = np.sum(Xm**2, axis=1)
    Sxy = np.sum(Xm*Ym, axis=1)

    slopes = Sxy / np.maximum(Sxx, np.finfo(float).eps)
    intercepts = (Ymean[:, 0]) - slopes * (Xmean[:, 0])

    slope_std = np.std(slopes)
    intercept_std = np.std(intercepts)

    # Fit the nominal line from nominal data
    slope, intercept = np.polyfit(xv, yv, 1)

    # Return as Data objects
    data_slope = Data("Slope", "m", slope, slope_std * 2)
    data_intercept = Data("Intercept", "b", intercept, intercept_std * 2)

    return data_slope, data_intercept
        



def load_lab_dataframe(file_path):
    """
    Load lab instrument data exported as a fake .xls file containing
    tab-delimited text with metadata rows at the top.

    Steps:
    - Reads raw text
    - Splits rows by tabs
    - Automatically finds header row
    - Builds a clean pandas DataFrame
    - Makes duplicate column names unique
    - Converts numeric columns where possible
    - Drops empty columns

    Parameters
    ----------
    file_path : str or Path
        Path to exported data file.

    Returns
    -------
    pandas.DataFrame
        Cleaned data table.
    """

    file_path = Path(file_path)

    # ---------- Read raw file ----------
    raw = file_path.read_text(errors="ignore")
    lines = [ln for ln in raw.splitlines() if ln.strip()]
    rows = [ln.split("\t") for ln in lines]

    # ---------- Find header row ----------
    header_idx = None
    for i, r in enumerate(rows):
        joined = " ".join(r).lower()
        if "iteration" in joined and "time" in joined:
            header_idx = i
            break

    if header_idx is None:
        raise ValueError("Header row not found in file.")

    header = _make_unique_headers([h.strip() for h in rows[header_idx]])
    data_rows = rows[header_idx + 1:]
    ncol = len(header)

    # ---------- Normalize row widths ----------
    cleaned_rows = []
    for r in data_rows:
        r = [x.strip() for x in r]
        if len(r) < ncol:
            r += [""] * (ncol - len(r))
        else:
            r = r[:ncol]
        cleaned_rows.append(r)

    df = pd.DataFrame(cleaned_rows, columns=header)

    # ---------- Strip whitespace ----------
    df = df.map(lambda x: x.strip() if isinstance(x, str) else x)

    # ---------- Convert numerics ----------
    def try_numeric(col):
        try:
            return pd.to_numeric(col)
        except Exception:
            return col

    df = df.apply(try_numeric, axis=0)

    # ---------- Drop empty columns ----------
    df = df.dropna(axis=1, how="all")

    return df


def _make_unique_headers(names):
    """Ensure column names are unique."""
    seen = {}
    result = []

    for name in names:
        name = (name or "").strip()
        if name == "":
            name = "col"

        if name in seen:
            seen[name] += 1
            result.append(f"{name}_{seen[name]}")
        else:
            seen[name] = 0
            result.append(name)

    return result

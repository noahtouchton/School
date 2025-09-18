import math
import numpy as np
import sympy as sp

class Data:
    def __init__(self, name, var, value, uncertainty=0.0):
        self.name = name
        self.var = var
        self.value = value
        self.uncertainty = uncertainty

    def get_error(self):
        v, u = self.value, self.uncertainty
        if np.isscalar(v) and np.isscalar(u):
            print(f"{v:.3e} ± {u:.3e}")
        else:
            v_arr = np.asarray(v)
            u_arr = np.asarray(u)
            shape = v_arr.shape
            v_preview = v_arr.ravel()[:5]
            u_preview = u_arr.ravel()[:5] if u_arr.size == v_arr.size or np.isscalar(u) else u_arr.ravel()[:5]
            print(f"array(shape={shape}) value preview: {v_preview} ± {u_preview}")

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
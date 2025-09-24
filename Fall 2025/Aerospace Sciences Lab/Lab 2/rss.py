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

def drag_per_unit_span_from_pressure(name, deltaP: Data, theta_rad: np.ndarray, R_cyl: Data,
                                     theta_unc_rad: np.ndarray | float | None = None):
    import numpy as np, sympy as sp
    Dprime_sym = sp.Symbol("Dprime")

    dp  = np.asarray(deltaP.value, dtype=float)          # Pa
    sdp = np.asarray(deltaP.uncertainty, dtype=float)    # Pa
    Rv  = float(R_cyl.value)                             # m
    sR  = float(R_cyl.uncertainty)                       # m

    h = float(theta_rad[1] - theta_rad[0])               # rad
    coeff = np.ones_like(theta_rad); coeff[0]=0.5; coeff[-1]=0.5

    # Nominal D'
    integrand = dp * np.cos(theta_rad) * Rv
    Dprime_val = np.trapz(integrand, theta_rad)          # N/m

    # Pressure contribution: w_i = h c_i R cosθ_i
    w = h * coeff * Rv * np.cos(theta_rad)               # m
    var_dp = np.sum((w * sdp)**2)                        # (N/m)^2

    # Radius contribution: ∂D'/∂R = D'/R
    var_R = ((Dprime_val / Rv) * sR)**2 if sR > 0 else 0.0

    # Angle contribution (optional): ∂D'/∂θ_i = -h c_i R ΔP_i sinθ_i
    var_theta = 0.0
    if theta_unc_rad is not None:
        sθ = np.asarray(theta_unc_rad, dtype=float)
        if sθ.size == 1:
            sθ = np.full_like(theta_rad, float(sθ))
        var_theta = np.sum((h * coeff * Rv * dp * np.sin(theta_rad) * sθ)**2)

    sDprime = float(np.sqrt(var_dp + var_R + var_theta))
    return Data(name, Dprime_sym, float(Dprime_val), sDprime)



def cd_from_Dprime(name, Dprime: Data, q: Data, D: Data):
    """
    C_D = D' / (q * D)
    Uncertainty propagation (independent sources):
      var(C_D) = (∂C/∂D')^2 sD'^2 + (∂C/∂q)^2 s_q^2 + (∂C/∂D)^2 s_D^2
    ∂C/∂D' = 1/(qD),  ∂C/∂q = -D'/(q^2 D),  ∂C/∂D = -D'/(q D^2)
    """
    CD_sym = sp.Symbol("CD")

    Dp  = float(Dprime.value)
    sDp = float(Dprime.uncertainty)

    qv  = float(q.value)
    sq  = float(q.uncertainty)

    Dv  = float(D.value)
    sD  = float(D.uncertainty)

    CD_val = Dp / (qv * Dv)

    dC_dDp = 1.0 / (qv * Dv)
    dC_dq  = -Dp / (qv**2 * Dv)
    dC_dD  = -Dp / (qv * Dv**2)

    var_CD = (dC_dDp * sDp)**2 + (dC_dq * sq)**2 + (dC_dD * sD)**2
    sCD = float(np.sqrt(var_CD))

    return Data(name, CD_sym, float(CD_val), sCD)

def align_theta_for_integration(beta_deg, deltaP_data):
    """
    Rotate measured angles so that the tap with max ΔP becomes θ_flow = 0°,
    and sort to 0..360 for proper trapezoid integration. Reorders uncertainties too.
    Returns: theta_flow_rad (monotonic 0..2π), aligned Data (ΔP, same shape), beta_stag (deg).
    """
    beta_deg = np.asarray(beta_deg, dtype=float)
    dp  = np.asarray(deltaP_data.value, dtype=float)
    sdp = np.asarray(deltaP_data.uncertainty, dtype=float)

    i0 = int(np.argmax(dp))                # index of stagnation (max ΔP)
    beta_stag = beta_deg[i0]               # measured angle of stagnation (deg)

    theta_flow_deg = (beta_deg - beta_stag) % 360.0
    order = np.argsort(theta_flow_deg)

    theta_flow_rad = np.deg2rad(theta_flow_deg[order])  # 0..2π, monotonic
    dp_aligned  = dp[order]
    sdp_aligned = sdp[order]

    deltaP_aligned = Data(deltaP_data.name + " (aligned)", "deltaP",
                          dp_aligned, sdp_aligned)
    return theta_flow_rad, deltaP_aligned, float(beta_stag)

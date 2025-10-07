#rss.py
import math
import numpy as np
import sympy as sp
from numpy.polynomial import Polynomial

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


# === Polynomial calibration helpers =========================================
# Fits U(E) = a0 + a1 E + ... + aN E^N and returns coeffs and full covariance.
# Works with optional pointwise y-uncertainties (WLS) or weights.
import numpy as np

def fit_poly_with_cov(x, y, order: int, y_uncert: np.ndarray | float | None = None, weights: np.ndarray | None = None):
    """
    Fit polynomial y ≈ sum_{k=0..order} a_k x^k.
    Returns:
      coeffs: (order+1,) array [a0, a1, ..., a_order]
      cov:    (order+1, order+1) covariance matrix of coeffs
      sigma2: residual variance estimate
      yhat:   fitted values
      dof:    degrees of freedom (n - (order+1))

    If y_uncert is supplied (std dev per point) it does WLS with w_i = 1/σ_i^2.
    If 'weights' is supplied directly, it uses those. y_uncert takes precedence.
    """
    x = np.asarray(x, float).ravel()
    y = np.asarray(y, float).ravel()
    n = x.size
    p = order + 1
    if n < p:
        raise ValueError(f"Need at least {p} points for a degree-{order} polynomial; got {n}.")

    # Vandermonde: columns [1, x, x^2, ...]
    X = np.vander(x, N=p, increasing=True)

    if y_uncert is not None:
        y_uncert = np.asarray(y_uncert, float).ravel()
        if y_uncert.size not in (1, n):
            raise ValueError("y_uncert must be scalar or same length as x.")
        w = 1.0 / (y_uncert**2 if y_uncert.size == n else float(y_uncert)**2)
    elif weights is not None:
        w = np.asarray(weights, float).ravel()
        if w.size != n:
            raise ValueError("weights must be same length as x.")
    else:
        w = None

    if w is None:
        # OLS normal equations via QR for stability
        Q, R = np.linalg.qr(X, mode='reduced')
        coeffs = np.linalg.solve(R, Q.T @ y)
        # Residuals & variance
        yhat = X @ coeffs
        resid = y - yhat
        dof = n - p
        sigma2 = float((resid @ resid) / dof)
        # Covariance of coeffs: σ^2 (X^T X)^{-1}
        XtX_inv = np.linalg.inv(X.T @ X)
        cov = sigma2 * XtX_inv
    else:
        # WLS using sqrt weights
        Wsqrt = np.sqrt(w)
        Xw = X * Wsqrt[:, None]
        yw = y * Wsqrt
        Q, R = np.linalg.qr(Xw, mode='reduced')
        coeffs = np.linalg.solve(R, Q.T @ yw)
        yhat = X @ coeffs
        resid = y - yhat
        dof = n - p
        # Weighted residual variance estimate (per standard WLS practice)
        sigma2 = float((w * resid**2).sum() / dof)
        XtWX_inv = np.linalg.inv(X.T @ (w[:, None] * X))
        cov = sigma2 * XtWX_inv

    return coeffs, cov, sigma2, yhat, dof


def poly_value_uncert(x_eval, coeffs, cov, sigma2=0.0, include_prediction=False, dx_std: float | np.ndarray | None = None):
    """
    Uncertainty of y(x) = sum a_k x^k at x_eval, given coeff covariance.
    - If include_prediction=False -> returns standard error of the mean curve.
    - If include_prediction=True  -> adds residual variance (prediction interval base).
    - If dx_std provided (std dev of x), adds delta-method term (dy/dx * dx_std)^2.

    Returns:
      y_eval:  nominal y at x_eval
      u_total: standard uncertainty at x_eval (same shape as x_eval)
    """
    coeffs = np.asarray(coeffs, float).ravel()
    p = coeffs.size
    x_eval = np.asarray(x_eval, float)

    # Build row vector [1, x, x^2, ...] at each x_eval
    def _row(x):
        return np.power(x, np.arange(p, dtype=float))

    # Nominal value
    y_eval = np.polynomial.polynomial.polyval(x_eval, coeffs)

    # Mean-curve uncertainty from coefficient covariance: x* C * x^T
    # Handle scalar vs vector x_eval
    if x_eval.ndim == 0:
        r = _row(float(x_eval))
        u_mean2 = float(r @ cov @ r.T)
    else:
        R = np.stack([_row(x) for x in x_eval], axis=0)  # (n, p)
        u_mean2 = np.einsum('ni,ij,nj->n', R, cov, R)    # (n,)

    u_total2 = u_mean2

    # Add residual variance for prediction intervals if requested
    if include_prediction and sigma2 > 0.0:
        u_total2 = u_total2 + sigma2

    # Optional x-uncertainty via delta method
    if dx_std is not None:
        dx_std = np.asarray(dx_std, float)
        # dy/dx for polynomial: sum_{k=1..p-1} k * a_k x^{k-1}
        # np.polyder works for power-descending; we have power-ascending, so do manual:
        deriv_coeffs = np.arange(1, p, dtype=float) * coeffs[1:]
        # Evaluate derivative in power-ascending basis
        dy_dx = np.polynomial.polynomial.polyval(x_eval, deriv_coeffs) if deriv_coeffs.size else 0.0
        u_total2 = u_total2 + (dy_dx * dx_std)**2

    return y_eval, np.sqrt(u_total2)

import matplotlib.pyplot as plt

def plot_poly_fit(x, y, coeffs, cov=None, sigma2=None, x_label="Voltage (V)", y_label="Velocity (m/s)",
                  title="Calibration Fit", include_uncert=True, y_unc=None):
    """
    Plot data points and fitted polynomial curve.
    Optionally adds uncertainty bands if cov and sigma2 are provided.

    Parameters:
        x, y          : data arrays
        coeffs        : polynomial coefficients [a0, a1, ..., an]
        cov           : covariance matrix of coefficients (optional)
        sigma2        : residual variance (optional)
        x_label, y_label, title : labels for plot
        include_uncert : if True, plot ±1σ band
        y_unc         : optional uncertainties for data points
    """
    x = np.asarray(x)
    y = np.asarray(y)
    x_dense = np.linspace(np.min(x), np.max(x), 300)
    y_fit = np.polynomial.polynomial.polyval(x_dense, coeffs)

    plt.figure(figsize=(8, 5))
    plt.scatter(x, y, color='blue', label="Measured data", zorder=3)
    plt.plot(x_dense, y_fit, color='red', label="4th-order fit", zorder=2)

    if y_unc is not None:
        plt.errorbar(x, y, yerr=y_unc, fmt='none', ecolor='gray', alpha=0.6, label="Data uncertainty")

    if include_uncert and cov is not None:
        # Compute ±1σ uncertainty band
        R = np.stack([np.power(x_dense, i) for i in range(len(coeffs))], axis=1)
        u_mean = np.sqrt(np.einsum('ij,jk,ik->i', R, cov, R))
        if sigma2:
            u_mean = np.sqrt(u_mean**2 + sigma2)
        plt.fill_between(x_dense, y_fit - u_mean, y_fit + u_mean,
                         color='red', alpha=0.2, label="±1σ fit uncertainty")

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def find_voltage(coeffs, measured_vel):
    p = Polynomial(coeffs)

    roots = (p - measured_vel).roots()
    real_roots = roots[np.isreal(roots)].real

    if real_roots.size == 0:
        print("No real roots found.")
        return None
    
    valid = real_roots[(real_roots >= 0.0) & (real_roots <= 4.5)]
    if valid.size == 0:
        print(f"No roots in valid voltage range for U = {measured_vel}")
        return None
    
    return valid[np.argmin(np.abs(valid - 2.5))]


import numpy as np
from numpy.polynomial import Polynomial   # must be this one!

def z_to_v_coeffs(b, Vbar, Vs):
    """
    Convert standardized (Z-basis) polynomial coefficients b (ascending)
    to raw V-basis coefficients (ascending).
    """
    pZ = Polynomial(b)                           # b0 + b1*Z + b2*Z^2 + ...
    Z_poly = Polynomial([-Vbar / Vs, 1.0 / Vs])  # Z = (V - Vbar)/Vs
    pV = pZ(Z_poly)                              # compose the two polynomials
    return pV.coef
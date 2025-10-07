# calibration_helpers.py
import numpy as np
from numpy.polynomial import Polynomial

# ---------- Basis transform: Z = (V - Vbar)/Vs ----------
def _z_to_v_transform(p, Vbar, Vs):
    """Return T so that a = T @ b, where:
       v(V) = Σ a_i V^i  and  v(Z) = Σ b_k Z^k,  Z=(V-Vbar)/Vs."""
    Z_poly = Polynomial([-Vbar / Vs, 1.0 / Vs])  # Z = (V - Vbar)/Vs
    T = np.zeros((p + 1, p + 1))
    for k in range(p + 1):
        # expand Z^k in powers of V
        col = (Polynomial.basis(k))(Z_poly).coef
        if col.size < (p + 1):
            col = np.pad(col, (0, p + 1 - col.size))
        T[:, k] = col[: p + 1]
    return T

def z_to_v_coeffs_and_cov(b_asc, Cov_b, Vbar, Vs):
    """Map Z-basis coefficients & covariance to V-basis (ascending)."""
    p = len(b_asc) - 1
    T = _z_to_v_transform(p, Vbar, Vs)
    a_asc = T @ b_asc
    Cov_a = T @ Cov_b @ T.T
    return a_asc, Cov_a

# ---------- Fit in Z-basis, then map back ----------
def fit_poly_with_z_basis(x_V, y, deg=4):
    """
    Fit v = poly(V) by first scaling to Z, doing np.polyfit(Z,y,deg,cov=True),
    then mapping the covariance back to V-basis. Returns a dict.
    """
    x_V = np.asarray(x_V, dtype=float)
    y   = np.asarray(y, dtype=float)
    Vbar = np.mean(x_V)
    Vs   = np.std(x_V, ddof=1)
    Z    = (x_V - Vbar) / Vs

    # np.polyfit returns descending powers; convert to ascending
    b_desc, Cov_b_desc = np.polyfit(Z, y, deg, cov=True)
    b_asc  = b_desc[::-1]
    Cov_b  = Cov_b_desc[::-1, ::-1]

    # residuals & sigma on Z-scale (same values as V-scale)
    pZ = np.poly1d(b_desc)                # in Z
    resid = y - pZ(Z)
    dof   = len(x_V) - (deg + 1)
    sigma2 = float((resid @ resid) / dof)
    sigma  = np.sqrt(sigma2)

    # map to V-basis (ascending a0..a_deg) and also descending for convenience
    a_asc, Cov_a = z_to_v_coeffs_and_cov(b_asc, Cov_b, Vbar, Vs)
    a_desc = a_asc[::-1]

    out = {
        "Vbar": Vbar,
        "Vs": Vs,
        "b_asc": b_asc,
        "Cov_b": Cov_b,
        "a_asc": a_asc,
        "a_desc": a_desc,
        "Cov_a": Cov_a,
        "a_stderr_asc": np.sqrt(np.diag(Cov_a)),
        "sigma": sigma,
        "dof": dof,
    }
    return out

# ---------- Prediction & uncertainty ----------
def dv_dV(V, a_asc):
    """Derivative dv/dV for a = [a0,a1,a2,a3,a4]."""
    a = np.asarray(a_asc, dtype=float)
    p = Polynomial(a)
    return p.deriv()(V)

def predict_v_and_uncert(V, a_asc, Cov_a=None, sigma=0.0, sigma_V=0.0, include_param_cov=True):
    """
    Prediction at V with options:
      - include_param_cov: add r Cov_a r^T (parameter uncertainty)
      - sigma: residual std (lack-of-fit)
      - sigma_V: voltage noise std -> propagated via dv/dV
    Returns (v_hat, u_v).
    """
    V = float(V)
    a = np.asarray(a_asc, dtype=float)
    v_hat = Polynomial(a)(V)

    # components
    u2 = 0.0
    if include_param_cov and Cov_a is not None:
        r = np.array([V**k for k in range(len(a))], dtype=float)  # [1, V, V^2, ...]
        u2 += float(r @ Cov_a @ r)
    if sigma > 0:
        u2 += sigma**2
    if sigma_V > 0:
        u2 += (dv_dV(V, a) * sigma_V) ** 2

    return v_hat, np.sqrt(u2)

# ---------- Invert: velocity -> voltage ----------
def invert_voltage_from_velocity(a_asc, v_target, domain=(0.0, 5.0), tol=1e-12):
    """
    Solve a(V) - v_target = 0 on a domain; returns one root in range or None.
    """
    a = np.asarray(a_asc, dtype=float).copy()
    a[0] -= float(v_target)  # a0 := a0 - v_target
    roots = Polynomial(a).roots()
    roots = np.asarray(roots, dtype=complex)
    real_roots = roots[np.isreal(roots)].real
    lo, hi = domain
    in_range = real_roots[(real_roots >= lo - tol) & (real_roots <= hi + tol)]
    if in_range.size == 0:
        return None
    # if multiple valid, pick the one closest to mid-domain (or pick by slope sign if you prefer)
    mid = 0.5 * (lo + hi)
    return in_range[np.argmin(np.abs(in_range - mid))]

# ---------- Simple file helper for LVM-like plain text ----------
def load_single_column(path, usecol=1):
    """Load one 0-indexed column from a headerless text file."""
    return np.loadtxt(path, usecols=usecol)

# --- Drag per unit span from discrete velocity deficit -----------------------
# D' ≈ ρ Σ u_i (U∞ - u_i) Δy_i  (trapezoid-friendly if your Δy_i already embeds weights)

from rss import Data, RSS
import numpy as np
import sympy as sp

def Dprime_from_velocity_deficit(name, rho: Data, Uinf: Data,
                                 u_vals, dy_vals,
                                 u_unc=None, dy_unc=None):
    """
    Parameters
    ----------
    rho : Data           # ρ (with uncertainty)
    Uinf: Data           # U∞ (with uncertainty)
    u_vals : array-like  # u_i
    dy_vals: array-like  # Δy_i
    u_unc  : array-like or scalar or None (defaults to 0)
    dy_unc : array-like or scalar or None (defaults to 0)

    Returns
    -------
    Data("name", "Dprime", value, uncertainty)
    """
    u_vals = np.asarray(u_vals, float).ravel()
    dy_vals = np.asarray(dy_vals, float).ravel()
    n = u_vals.size
    if dy_vals.size != n:
        raise ValueError("u_vals and dy_vals must have same length")

    if u_unc is None:
        u_unc = np.zeros_like(u_vals)
    if dy_unc is None:
        dy_unc = np.zeros_like(dy_vals)
    u_unc = np.asarray(u_unc, float).ravel()
    dy_unc = np.asarray(dy_unc, float).ravel()

    # Symbol for T_i
    T_sym = sp.Symbol("T")

    # Accumulators
    T_values = []
    T_sigmas = []

    for i in range(n):
        u_i  = Data(f"u[{i}]", "u",  u_vals[i],  u_unc[i])
        dy_i = Data(f"dy[{i}]", "dy", dy_vals[i], dy_unc[i])

        # T_i = ρ * u_i * (Uinf - u_i) * dy_i
        Ti = RSS(f"T[{i}]", "T = ρ * u * (Uinf - u) * dy", [rho, u_i, Uinf, dy_i])
        T_values.append(float(Ti.value))
        T_sigmas.append(float(Ti.uncertainty))

    T_values = np.array(T_values)
    T_sigmas = np.array(T_sigmas)

    Dprime_val = float(np.sum(T_values))
    # independent T_i → variances add
    Dprime_unc = float(np.sqrt(np.sum(T_sigmas**2)))

    return Data(name, "Dprime", Dprime_val, Dprime_unc)

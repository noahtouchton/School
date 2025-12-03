# uncertainty.py
import os
import pandas as pd
import numpy as np
from rss import Data, RSS, linear_monte_carlo
import sympy as sp

in_to_m = 0.0254  # inches to meters
inH2o_to_Pa = 249.08891  # inches H2O to Pascals

# ------------------------------------------------------------
# Equations used with RSS
# ------------------------------------------------------------
equations = [
    "ρ = P / (R * T)",            # 0: density from ambient P,T
    "μ = MU_0 * (T / T_0)**(3/2) * (T_0 + C) / (T + C)",  # 1: Sutherland's law for dynamic viscosity
    "ν = μ / ρ",                     # 2: kinematic viscosity
    "y = yi + offset" ,              # 3: corrected position with ruler offset
    "V = (q - b) / m",               # 4: voltage from dynamic pressure calibration
    "q = (m*V+b) * 249.08891",
    "v = sqrt(2*q/ρ)"                # 5: velocity from dynamic pressure and density

]

q_cal_vals = np.array([2.8,-3.55,-0.8])
q_cal_uncert = np.array([0.05,0.05,0.05])
v_cal_vals = np.array([1.446,-1.774,-0.376])
v_cal_uncert = np.array([0.058,0.054,0.06])

q_cal = Data("q_cal", sp.Symbol("q"), q_cal_vals, q_cal_uncert)
v_cal = Data("v_cal", sp.Symbol("V"), v_cal_vals, v_cal_uncert)

slope, intercept = linear_monte_carlo(v_cal, q_cal, N=100_000)

slope.get_error()
intercept.get_error()

# Ambient conditions
T_amb = Data("Ambient Temperature", "T", 297.35, 0.4)        # K
P_amb = Data("Ambient Pressure",    "P", 101600.0, 400.0)    # Pa
# Gas & Sutherland constants
MU_0 = Data("mu not", "MU_0", 1.716e-5, 0.0)
T_0  = Data("T not",  "T_0",  273.0,   0.0)
C    = Data("Constant","C",   111.0,   0.0)
R    = Data("Ideal Gas Constant for Air", "R", 287.05, 0.0)   

# Calculate density ρ = P / (R * T)
ρ = RSS("Air Density", equations[0], [P_amb, R, T_amb])

μ = RSS("Dynamic Viscosity", equations[1], [MU_0, T_0, C, T_amb])
μ.get_error()
ν = RSS("Kinematic Viscosity", equations[2], [μ, ρ])
ν.get_error()

ruler_offset = Data("Ruler Offset", "offset", 0.45 * in_to_m, 1/32 * in_to_m)  # meters


# Load Data

# 30 Hz

y_vals_30hz = np.array([
    0.75,
    0.8125,
    0.875,
    0.9375,
    1.0,
    1.0625,
    1.125,
    1.1875,
    1.25,
    1.3125,
    1.375,
    1.4375,
    1.5,
    1.5625,
    2.0,
    2.5,
    3.0,
    3.5,
    4.0,
    4.5,
    5.0,
    5.5,
    6.0,
    6.5,
    7.0,
    7.5,
    8.0,
    8.5,
    9.0,
    9.0625,
    9.125,
    9.1875,
    9.25,
    9.3125,
    9.375,
    9.4375,
    9.5,
    9.5625,
    9.625,
    9.6875,
    9.75,
    9.8125,
    9.875,
    9.9375,
    10.0,
    10.0625,
    10.125,
    10.0625,
    10.125
])
y_unc_30hz = np.full_like(y_vals_30hz, 1/32) * in_to_m  # meters
y_raw_30hz = Data("Raw Position 30 Hz", "yi", y_vals_30hz * in_to_m, y_unc_30hz)
y_30hz = RSS("Position 30 Hz", equations[3], [y_raw_30hz, ruler_offset])
q_raw_vals_30hz = np.array([
    0.833,
    0.834,
    0.843,
    0.844,
    0.844,
    0.843,
    0.843,
    0.841,
    0.842,
    0.851,
    0.844,
    0.847,
    0.849,
    0.843,
    0.847,
    0.840,
    0.838,
    0.846,
    0.843,
    0.834,
    0.839,
    0.848,
    0.845,
    0.849,
    0.845,
    0.847,
    0.847,
    0.847,
    0.846,
    0.851,
    0.825,
    0.842,
    0.848,
    0.845,
    0.844,
    0.846,
    0.855,
    0.850,
    0.852,
    0.855,
    0.863,
    0.863,
    0.848,
    0.849,
    0.830,
    0.829,
    0.831,
    0.855,
    0.861
])
q_raw_uncer_vals_30hz = np.array([
    0.067,
    0.063,
    0.063,
    0.064,
    0.065,
    0.063,
    0.063,
    0.062,
    0.062,
    0.063,
    0.061,
    0.065,
    0.065,
    0.066,
    0.072,
    0.067,
    0.069,
    0.072,
    0.070,
    0.070,
    0.067,
    0.065,
    0.066,
    0.067,
    0.066,
    0.067,
    0.069,
    0.068,
    0.068,
    0.066,
    0.068,
    0.063,
    0.061,
    0.061,
    0.065,
    0.064,
    0.066,
    0.061,
    0.063,
    0.062,
    0.071,
    0.072,
    0.066,
    0.071,
    0.067,
    0.068,
    0.068,
    0.069,
    0.066
])
#convert calculated pressures back into measured voltages for both values and uncertainties
q_raw_vals_30hz = Data("Raw Dynamic Pressure 30 Hz", "q", q_raw_vals_30hz, np.full_like(q_raw_vals_30hz, 0.0))
q_raw_uncert_30hz = Data("Raw Dynamic Pressure Uncertainty 30 Hz", "q", q_raw_uncer_vals_30hz, np.full_like(q_raw_uncer_vals_30hz, 0.0))
#run the measured voltages through RSS to account for calibration uncertainty
V_raw_vals_30hz = RSS("Voltage 30 Hz", equations[4], [q_raw_vals_30hz, slope, intercept])
V_raw_uncert_30hz = RSS("Voltage Uncertainty 30 Hz", equations[4], [q_raw_uncert_30hz, slope, intercept])

V_30hz = Data("Voltage 30 Hz", "V", V_raw_vals_30hz.value, V_raw_uncert_30hz.value)
q_30hz = RSS("Dynamic Pressure 30 Hz", equations[5], [slope, V_30hz, intercept])
v_30hz = RSS("Velocity 30 Hz", equations[6], [q_30hz, ρ])

# 45 Hz
y_vals_45hz = np.array([
    0.75, 0.812, 0.875, 0.937, 1.0, 1.062, 1.125, 1.187, 1.25, 1.312,
    1.375, 1.437, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0,
    5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.062, 9.125,
    9.187, 9.25, 9.312, 9.375, 9.437, 9.5, 9.562, 9.625, 9.687, 9.75,
    9.812, 9.875, 9.937, 10.0, 10.062, 10.125
])
y_unc_45hz = np.full_like(y_vals_45hz, 1/32) * in_to_m  # meters
y_raw_45hz = Data("Raw Position 45 Hz", "yi", y_vals_45hz * in_to_m, y_unc_45hz)
y_45hz = RSS("Position 45 Hz", equations[3], [y_raw_45hz, ruler_offset])
q_raw_vals_45hz = np.array([
    2.231, 2.314, 2.34, 2.351, 2.359, 2.372, 2.37, 2.369, 2.375, 2.376,
    2.377, 2.382, 2.381, 2.382, 2.377, 2.374, 2.38, 2.378, 2.377, 2.385,
    2.387, 2.394, 2.39, 2.397, 2.387, 2.394, 2.394, 2.397, 2.363, 2.275,
    2.152, 2.101, 1.951, 1.817, 1.626, 1.501, 1.381, 1.256, 1.197, 1.161,
    1.113, 1.081, 1.087, 1.066, 1.052, 1.396
])
q_raw_uncer_vals_45hz = np.array([
    0.070, 0.066, 0.066, 0.064, 0.062, 0.075, 0.069, 0.059, 0.071, 0.068,
    0.064, 0.064, 0.062, 0.068, 0.071, 0.070, 0.067, 0.061, 0.069, 0.070,
    0.066, 0.069, 0.072, 0.067, 0.071, 0.065, 0.066, 0.066, 0.067, 0.076,
    0.070, 0.070, 0.071, 0.075, 0.070, 0.067, 0.070, 0.065, 0.067, 0.073,
    0.069, 0.063, 0.067, 0.062, 0.064, 0.068
])
q_raw_vals_45hz = Data("Raw Dynamic Pressure 45 Hz", "q", q_raw_vals_45hz, np.full_like(q_raw_vals_45hz, 0.0))
q_raw_uncert_45hz = Data("Raw Dynamic Pressure Uncertainty 45 Hz", "q", q_raw_uncer_vals_45hz, np.full_like(q_raw_uncer_vals_45hz, 0.0))
V_raw_vals_45hz = RSS("Voltage 45 Hz", equations[4], [q_raw_vals_45hz, slope, intercept])
V_raw_uncert_45hz = RSS("Voltage Uncertainty 45 Hz", equations[4], [q_raw_uncert_45hz, slope, intercept])
V_45hz = Data("Voltage 45 Hz", "V", V_raw_vals_45hz.value, V_raw_uncert_45hz.value)
q_45hz = RSS("Dynamic Pressure 45 Hz", equations[5], [slope, V_45hz, intercept])
v_45hz = RSS("Velocity 45 Hz", equations[6], [q_45hz, ρ])

#60 Hz
y_vals_60hz = np.array([
    0.75, 0.812, 0.875, 0.937, 1.0, 1.062, 1.125, 1.187, 1.25, 1.312,
    1.375, 1.437, 1.5, 1.562, 1.625, 1.687, 1.75, 1.812, 1.875, 1.937,
    2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5,
    7.0, 7.5, 8.0, 8.062, 8.125, 8.187, 8.25, 8.312, 8.375, 8.437,
    8.5, 8.562, 8.625, 8.687, 8.75, 8.812, 8.875, 8.937, 9.0, 9.062,
    9.125, 9.187, 9.25, 9.312, 9.375, 9.437, 9.5, 9.562, 9.625, 9.687,
    9.75, 9.812, 9.875, 9.937, 10.062, 10.0
])
y_unc_60hz = np.full_like(y_vals_60hz, 1/32) * in_to_m  # meters
y_raw_60hz = Data("Raw Position 60 Hz", "yi", y_vals_60hz * in_to_m, y_unc_60hz)
y_60hz = RSS("Position 60 Hz", equations[3], [y_raw_60hz, ruler_offset])
q_raw_vals_60hz = np.array([
    3.99, 4.198, 4.341, 4.413, 4.484, 4.515, 4.53, 4.556, 4.559, 4.565,
    4.568, 4.586, 4.574, 4.588, 4.584, 4.566, 4.568, 4.563, 4.553, 4.543,
    4.539, 4.539, 4.529, 4.527, 4.541, 4.544, 4.554, 4.57, 4.579, 4.586,
    4.579, 4.583, 4.608, 4.625, 4.651, 4.629, 4.63, 4.566, 4.533, 4.462,
    4.292, 4.098, 3.886, 3.686, 3.359, 3.171, 2.91, 2.712, 2.524, 2.36,
    2.237, 2.095, 2.073, 2.032, 2.021, 1.996, 1.972, 1.998, 1.956, 1.925,
    1.975, 1.909, 1.959, 1.948, 2.145, 1.931
])
q_raw_uncer_vals_60hz = np.array([
    0.070, 0.061, 0.064, 0.063, 0.065, 0.060, 0.058, 0.064, 0.065, 0.065,
    0.059, 0.061, 0.059, 0.060, 0.061, 0.070, 0.068, 0.068, 0.064, 0.064,
    0.062, 0.064, 0.069, 0.067, 0.064, 0.062, 0.063, 0.061, 0.062, 0.062,
    0.061, 0.058, 0.057, 0.059, 0.062, 0.059, 0.063, 0.063, 0.077, 0.075,
    0.073, 0.081, 0.079, 0.082, 0.082, 0.088, 0.089, 0.077, 0.076, 0.080,
    0.073, 0.071, 0.069, 0.069, 0.067, 0.074, 0.077, 0.079, 0.083, 0.079,
    0.073, 0.074, 0.082, 0.090, 0.085, 0.077
])
q_raw_vals_60hz = Data("Raw Dynamic Pressure 60 Hz", "q", q_raw_vals_60hz, np.full_like(q_raw_vals_60hz, 0.0))
q_raw_uncert_60hz = Data("Raw Dynamic Pressure Uncertainty 60 Hz", "q", q_raw_uncer_vals_60hz, np.full_like(q_raw_uncer_vals_60hz, 0.0))
V_raw_vals_60hz = RSS("Voltage 60 Hz", equations[4], [q_raw_vals_60hz, slope, intercept])
V_raw_uncert_60hz = RSS("Voltage Uncertainty 60 Hz", equations[4], [q_raw_uncert_60hz, slope, intercept])
V_60hz = Data("Voltage 60 Hz", "V", V_raw_vals_60hz.value, V_raw_uncert_60hz.value)
q_60hz = RSS("Dynamic Pressure 60 Hz", equations[5], [slope, V_60hz, intercept])
v_60hz = RSS("Velocity 60 Hz", equations[6], [q_60hz, ρ])

# ------------------------------------------------------------
# Boundary-layer analysis
# ------------------------------------------------------------

# Test-section height (given): H = 12 ± 1/32 in
H_in  = 12.0                      # inches
H_unc_in = 1.0 / 32.0             # inches

H     = H_in * in_to_m            # meters
H_unc = H_unc_in * in_to_m        # meters

H_data = Data("Test Section Height", "H", H, H_unc)

# Convenience: pull numpy arrays out of Data objects
y30 = np.asarray(y_30hz.value, dtype=float)
y45 = np.asarray(y_45hz.value, dtype=float)
y60 = np.asarray(y_60hz.value, dtype=float)

U30 = np.asarray(v_30hz.value, dtype=float)
U45 = np.asarray(v_45hz.value, dtype=float)
U60 = np.asarray(v_60hz.value, dtype=float)

# Non-dimensional vertical coordinate using the physical test-section height
# 0 at bottom wall, 1 at top wall (we assume y=0 corresponds to the bottom wall)
eta30 = y30 / H_data.value
eta45 = y45 / H_data.value
eta60 = y60 / H_data.value

# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------

def compute_U_inf(y, U, H, band=0.2):
    """
    Estimate free-stream/core velocity U_inf by averaging U
    over the central 'band' fraction of the test-section height.
    Default: central 40% (band = 0.2 -> 0.3H to 0.7H).
    """
    y_center = 0.5 * H
    mask = (y > (y_center - band * H)) & (y < (y_center + band * H))
    if not np.any(mask):
        raise RuntimeError("Core-mask for U_inf is empty; check band or y-range.")
    return U[mask].mean()


def delta_99_from_bottom(y, U, U_inf, tol=0.99):
    """
    Find 99% boundary-layer thickness from the bottom wall.

    y : array (m), measured from bottom wall, increasing upward
    U : array (m/s)
    U_inf : scalar free-stream velocity
    tol : threshold fraction (default 0.99)

    Returns:
      delta : estimated thickness (m)
      is_upper_bound : True if entire profile is already above tol*U_inf,
                       so delta is only an upper bound (delta <= y[0])
    """
    # Sort just in case (should already be ordered)
    idx = np.argsort(y)
    y_sorted = y[idx]
    Urel = U[idx] / U_inf

    # Case 1: first point is already at or above tol -> only an upper bound
    if Urel[0] >= tol:
        return y_sorted[0], True

    # Case 2: find bracket where Urel crosses tol and linearly interpolate
    for i in range(len(y_sorted) - 1):
        if (Urel[i] < tol) and (Urel[i+1] >= tol):
            y0, y1 = y_sorted[i],   y_sorted[i+1]
            u0, u1 = Urel[i],       Urel[i+1]
            delta = y0 + (tol - u0) * (y1 - y0) / (u1 - u0)
            return delta, False

    # If we never reach tol, we technically only know that δ > y_max
    # but that's non-physical for a wind-tunnel core; flag as NaN.
    return np.nan, False


def delta_99_from_top(y, U, U_inf, H, tol=0.99):
    """
    99% boundary-layer thickness from the TOP wall.

    We transform to a coordinate measured downward from the top wall,
    reuse delta_99_from_bottom, and return δ measured from the top.
    """
    y_from_top = H - y  # 0 at top wall, increasing downward
    delta_top, is_upper_bound = delta_99_from_bottom(y_from_top, U, U_inf, tol=tol)
    return delta_top, is_upper_bound


def summarize_bl(freq_label, y_data: Data, U_data: Data, H_data: Data, band=0.2):
    """
    Compute U_inf, bottom δ_99, and top δ_99 for a given frequency case
    using the Data objects (values + uncertainties).

    - U_inf uncertainty: from uncertainties of velocities in the core region.
    - δ_99 bottom: nominal from profile; uncertainty estimated from y-uncertainty
      near the crossing.
    - δ_99 top: nominal from profile; uncertainty from near-wall y-uncertainty
      and H uncertainty.
    """

    # Nominal arrays
    y = np.asarray(y_data.value, dtype=float).ravel()
    U = np.asarray(U_data.value, dtype=float).ravel()

    # Align lengths if something got mismatched upstream
    n = min(len(y), len(U))
    if len(y) != len(U):
        print(f"[WARN] {freq_label}: len(y)={len(y)}, len(U)={len(U)} -> trimming to {n}")
        y = y[:n]
        U = U[:n]

    # Position uncertainties
    y_unc_raw = y_data.uncertainty
    if np.isscalar(y_unc_raw):
        y_unc = np.full_like(y, y_unc_raw, dtype=float)
    else:
        y_unc_arr = np.asarray(y_unc_raw, dtype=float).ravel()
        if y_unc_arr.size < n:
            raise RuntimeError(f"{freq_label}: y uncertainty array too short.")
        y_unc = y_unc_arr[:n]

    # Velocity uncertainties
    U_unc_raw = U_data.uncertainty
    if np.isscalar(U_unc_raw):
        U_unc = np.full_like(U, U_unc_raw, dtype=float)
    else:
        U_unc_arr = np.asarray(U_unc_raw, dtype=float).ravel()
        if U_unc_arr.size < n:
            raise RuntimeError(f"{freq_label}: U uncertainty array too short.")
        U_unc = U_unc_arr[:n]

    # Test-section height
    H = H_data.value
    H_unc = H_data.uncertainty

    # --------------------------------------------------------
    # 1) Core (free-stream) velocity and its uncertainty
    # --------------------------------------------------------
    y_center = 0.5 * H
    mask_core = (y > (y_center - band * H)) & (y < (y_center + band * H))
    if not np.any(mask_core):
        raise RuntimeError(f"{freq_label}: core mask is empty; adjust 'band' or check y range.")

    U_core = U[mask_core]
    U_core_unc = U_unc[mask_core]

    U_inf_nom = U_core.mean()
    # Uncertainty in mean of independent measurements:
    U_inf_unc = np.sqrt(np.sum(U_core_unc**2)) / U_core.size

    # --------------------------------------------------------
    # 2) Boundary-layer thicknesses (nominal)
    # --------------------------------------------------------
    delta_b_nom, ub_b = delta_99_from_bottom(y, U, U_inf_nom)
    delta_t_nom, ub_t = delta_99_from_top(y, U, U_inf_nom, H)

    # --------------------------------------------------------
    # 3) Approximate uncertainties in δ
    #    (from position resolution and H uncertainty)
    # --------------------------------------------------------
    # Bottom δ: dominated by y-position uncertainty near the crossing
    if np.isfinite(delta_b_nom):
        idx_b = int(np.argmin(np.abs(y - delta_b_nom)))
        u_delta_b = y_unc[idx_b]
    else:
        u_delta_b = np.nan

    # Top δ: measured from top wall, so depends on a y near the top and on H
    if np.isfinite(delta_t_nom):
        # y_from_top = H - y; we want the point closest to where that equals δ_t
        y_from_top = H - y
        idx_t = int(np.argmin(np.abs(y_from_top - delta_t_nom)))
        u_y_top = y_unc[idx_t]
        # Combine H and y contribution in quadrature
        u_delta_t = np.sqrt(u_y_top**2 + H_unc**2)
    else:
        u_delta_t = np.nan

    # --------------------------------------------------------
    # 4) Wrap results in Data objects
    # --------------------------------------------------------
    U_inf_data   = Data(f"U_inf {freq_label}", "U_inf", U_inf_nom, U_inf_unc)
    delta_b_data = Data(f"δ_99 bottom {freq_label}", "δ_b", delta_b_nom, u_delta_b)
    delta_t_data = Data(f"δ_99 top {freq_label}", "δ_t", delta_t_nom, u_delta_t)

    # Non-dimensional thicknesses and their uncertainties
    delta_b_over_H = delta_b_nom / H if np.isfinite(delta_b_nom) else np.nan
    delta_t_over_H = delta_t_nom / H if np.isfinite(delta_t_nom) else np.nan

    # Propagate uncertainty in δ/H (simple ratio, assuming independence)
    def ratio_unc(val_num, u_num, val_den, u_den):
        if not np.isfinite(val_num) or val_num == 0 or val_den == 0:
            return np.nan
        rel2 = (u_num / val_num)**2 + (u_den / val_den)**2
        return (val_num / val_den) * np.sqrt(rel2)

    delta_b_over_H_unc = ratio_unc(delta_b_nom, u_delta_b, H, H_unc)
    delta_t_over_H_unc = ratio_unc(delta_t_nom, u_delta_t, H, H_unc)

    # Convert δ to inches for reporting
    delta_b_in = delta_b_nom / in_to_m if np.isfinite(delta_b_nom) else np.nan
    delta_t_in = delta_t_nom / in_to_m if np.isfinite(delta_t_nom) else np.nan
    u_delta_b_in = u_delta_b / in_to_m if np.isfinite(u_delta_b) else np.nan
    u_delta_t_in = u_delta_t / in_to_m if np.isfinite(u_delta_t) else np.nan

    print(f"\n--- {freq_label} ---")
    print(f"U_inf ≈ {U_inf_nom:.3f} ± {U_inf_unc:.3f} m/s")

    if np.isfinite(delta_b_nom):
        note_b = " (upper bound)" if ub_b else ""
        print(
            f"Bottom δ_99 ≈ {delta_b_nom*1000:.2f} ± {u_delta_b*1000:.2f} mm  "
            f"({delta_b_in:.3f} ± {u_delta_b_in:.3f} in)  "
            f"(δ/H ≈ {delta_b_over_H:.3f} ± {delta_b_over_H_unc:.3f}){note_b}"
        )
    else:
        print("Bottom δ_99 could not be determined from available data.")

    if np.isfinite(delta_t_nom):
        note_t = " (upper bound)" if ub_t else ""
        print(
            f"Top    δ_99 ≈ {delta_t_nom*1000:.2f} ± {u_delta_t*1000:.2f} mm  "
            f"({delta_t_in:.3f} ± {u_delta_t_in:.3f} in)  "
            f"(δ/H ≈ {delta_t_over_H:.3f} ± {delta_t_over_H_unc:.3f}){note_t}"
        )
    else:
        print("Top δ_99 could not be determined from available data.")

    return U_inf_data, delta_b_data, delta_t_data




# ------------------------------------------------------------
# Execute BL analysis for 30, 45, and 60 Hz
# ------------------------------------------------------------

print("\n========================================")
print(" Boundary-layer thickness (δ_99) summary")
print("========================================")

U_inf_30_data, delta_b_30_data, delta_t_30_data = summarize_bl("30 Hz", y_30hz, v_30hz, H_data)
U_inf_45_data, delta_b_45_data, delta_t_45_data = summarize_bl("45 Hz", y_45hz, v_45hz, H_data)
U_inf_60_data, delta_b_60_data, delta_t_60_data = summarize_bl("60 Hz", y_60hz, v_60hz, H_data)



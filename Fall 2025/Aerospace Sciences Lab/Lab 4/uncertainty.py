# uncertainty.py  — clean main
from rss import *  # uses your Data/RSS implementation
import numpy as np
import os
import os
import matplotlib.pyplot as plt


# ---------- constants ----------
inH2O_to_Pa = 249.0889

# Only keep what we need
equations = [
    "ρ = P / (R * T)",
    "mu = MU_0 * (T / T_0)**(3/2) * (T_0 + C) / (T + C)",
    "v = a0 + a1*V + a2*V**2 + a3*V**3 + a4*V**4",
    "Re = ρ * v * d / mu",
    "F = m*V + b",
    "v = sqrt(2*q/ρ)",
    "A = 3.14159 * d**2 / 4",
    "Cd = F / (q * A)",
]

# Gas & Sutherland constants
MU_0 = Data("mu not", "MU_0", 1.716e-5, 0.0)
T_0  = Data("T not",  "T_0",  273.0,   0.0)
C    = Data("Constant","C",   111.0,   0.0)
R    = Data("Ideal Gas Constant for Air", "R", 287.05, 0.0)

# Lab ambient & geometry (edit with your values/uncertainties)
T = Data("Temperature", "T", 296.65, 0.4)
P = Data("Pressure",    "P", 101300, 400.0)
d = Data("Diameter",    "d", 0.01905, 0.0)

# ---------- derived fluid props ----------
ρ  = RSS("Density",            equations[0], [P, R, T])
mu = RSS("Dynamic Viscosity",  equations[1], [T, MU_0, T_0, C])
mu.get_description()
ρ.get_description()




cal_weights_val = np.array([
    22.2687,
    21.582,
    19.62,
    9.81,
    4.905,
    1.962,
    0.981,
    0.4905,
    0.1962,
    0.0981,
    0.0
])

cal_voltages_val = np.array([
    -10.0054,
    -9.7002,
    -8.8093,
    -4.3909,
    -2.1936,
    -0.8781,
    -0.4412,
    -0.2224,
    -0.0909,
    -0.0466,
    -0.0028
])

cal_voltages_uncert = np.array([
    0.0021187,
    0.002820559,
    0.000948683,
    0.000567646,
    0.000516398,
    0.000316228,
    0.000632456,
    0.000516398,
    0.000567646,
    0.000516398,
    0.000421637
])*2.0

cal_weights = Data("Weights", "W", cal_weights_val, 0.0)
cal_voltages = Data("Voltages", "V", cal_voltages_val, cal_voltages_uncert)

cal_slope, cal_intercept = linear_monte_carlo(cal_voltages, cal_weights) 

cal_slope.get_description()
cal_intercept.get_description()


# =========================
# CASE ORDER 
# =========================
# 45 Hz
# 0: Circular Flat Disk (45)
# 1: Hollow Hemisphere Up (45)
# 2: Hollow Hemisphere Down (45)
# 3: Smooth Sphere (45)
# 4: Rough Sphere (45)
# 55 Hz
# 5: Circular Flat Disk (55)
# 6: Hollow Hemisphere Up (55)
# 7: Hollow Hemisphere Down (55)
# 8: Smooth Sphere (55)
# 9: Rough Sphere (55)

# ---- 45 Hz raw values ----
disk_45_vals   = np.array([1.171, 1.171, 1.178, 1.170, 1.179, 1.176, 1.176, 1.167, 1.170, 1.172])
hh_up_45_vals  = np.array([1.007, 0.994, 0.999, 1.006, 1.003, 1.010, 1.004, 1.010, 1.005, 1.003])
hh_dn_45_vals  = np.array([1.578, 1.580, 1.578, 1.579, 1.580, 1.580, 1.582, 1.578, 1.579, 1.580])
smooth_45_vals = np.array([1.562, 1.566, 1.564, 1.569, 1.566, 1.568, 1.563, 1.562, 1.560, 1.565])
rough_45_vals  = np.array([1.619, 1.620, 1.618, 1.617, 1.619, 1.620, 1.618, 1.617, 1.620, 1.618])

# ---- 55 Hz raw values ----
disk_55_vals   = np.array([0.801, 0.803, 0.806, 0.802, 0.800, 0.801, 0.794, 0.798, 0.800, 0.801])
hh_up_55_vals  = np.array([0.518, 0.523, 0.531, 0.525, 0.528, 0.520, 0.523, 0.525, 0.528, 0.535])
hh_dn_55_vals  = np.array([1.447, 1.445, 1.447, 1.447, 1.444, 1.448, 1.447, 1.445, 1.447, 1.445])
smooth_55_vals = np.array([1.423, 1.420, 1.421, 1.424, 1.422, 1.418, 1.420, 1.422, 1.418, 1.419])
rough_55_vals  = np.array([1.501, 1.490, 1.498, 1.490, 1.495, 1.496, 1.498, 1.494, 1.495, 1.500])

def make_data(var_name: str, arr: np.ndarray):
    mean = float(np.mean(arr))
    std  = float(np.std(arr, ddof=1))  # sample std
    # Name the Data with the *variable name* for printing clarity
    return Data(var_name, "V", mean, std)

# ---------- Build Data objects in the NEW 0..9 order ----------
circular_flat_disk_45 = make_data("circular_flat_disk_45", disk_45_vals)   # 0
hollow_hemi_up_45     = make_data("hollow_hemi_up_45",     hh_up_45_vals)  # 1
hollow_hemi_down_45   = make_data("hollow_hemi_down_45",   hh_dn_45_vals)  # 2
smooth_sphere_45      = make_data("smooth_sphere_45",      smooth_45_vals) # 3
rough_sphere_45       = make_data("rough_sphere_45",       rough_45_vals)  # 4
circular_flat_disk_55 = make_data("circular_flat_disk_55", disk_55_vals)   # 5
hollow_hemi_up_55     = make_data("hollow_hemi_up_55",     hh_up_55_vals)  # 6
hollow_hemi_down_55   = make_data("hollow_hemi_down_55",   hh_dn_55_vals)  # 7
smooth_sphere_55      = make_data("smooth_sphere_55",      smooth_55_vals) # 8
rough_sphere_55       = make_data("rough_sphere_55",       rough_55_vals)  # 9

# Optional: ordered list if you want to index by case id directly
volt_data_by_case = [
    circular_flat_disk_45,  # 0
    hollow_hemi_up_45,      # 1
    hollow_hemi_down_45,    # 2
    smooth_sphere_45,       # 3
    rough_sphere_45,        # 4
    circular_flat_disk_55,  # 5
    hollow_hemi_up_55,      # 6
    hollow_hemi_down_55,    # 7
    smooth_sphere_55,       # 8
    rough_sphere_55,        # 9
]

# ---------- Drag forces from voltages (consistent titles) ----------
drag_circular_flat_disk_45 = RSS("Circular Flat Disk Drag Force (45Hz)", equations[4], [circular_flat_disk_45, cal_slope, cal_intercept])  # 0
drag_hollow_hemi_up_45     = RSS("Hollow Hemisphere Up Drag Force (45Hz)", equations[4], [hollow_hemi_up_45,     cal_slope, cal_intercept])# 1
drag_hollow_hemi_down_45   = RSS("Hollow Hemisphere Down Drag Force (45Hz)", equations[4], [hollow_hemi_down_45, cal_slope, cal_intercept])# 2
drag_smooth_sphere_45      = RSS("Smooth Sphere Drag Force (45Hz)", equations[4], [smooth_sphere_45,      cal_slope, cal_intercept])       # 3
drag_rough_sphere_45       = RSS("Rough Sphere Drag Force (45Hz)", equations[4], [rough_sphere_45,       cal_slope, cal_intercept])        # 4
drag_circular_flat_disk_55 = RSS("Circular Flat Disk Drag Force (55Hz)", equations[4], [circular_flat_disk_55, cal_slope, cal_intercept])  # 5
drag_hollow_hemi_up_55     = RSS("Hollow Hemisphere Up Drag Force (55Hz)", equations[4], [hollow_hemi_up_55,     cal_slope, cal_intercept])# 6
drag_hollow_hemi_down_55   = RSS("Hollow Hemisphere Down Drag Force (55Hz)", equations[4], [hollow_hemi_down_55, cal_slope, cal_intercept])# 7
drag_smooth_sphere_55      = RSS("Smooth Sphere Drag Force (55Hz)", equations[4], [smooth_sphere_55,      cal_slope, cal_intercept])       # 8
drag_rough_sphere_55       = RSS("Rough Sphere Drag Force (55Hz)", equations[4], [rough_sphere_55,       cal_slope, cal_intercept])        # 9

drag_by_case = [
    drag_circular_flat_disk_45,  # 0
    drag_hollow_hemi_up_45,      # 1
    drag_hollow_hemi_down_45,    # 2
    drag_smooth_sphere_45,       # 3
    drag_rough_sphere_45,        # 4
    drag_circular_flat_disk_55,  # 5
    drag_hollow_hemi_up_55,      # 6
    drag_hollow_hemi_down_55,    # 7
    drag_smooth_sphere_55,       # 8
    drag_rough_sphere_55,        # 9
]


d_circular_flat_disk = Data("Circular Flat Disk Diameter", "d", 50.35e-3, 0.00002)  # meters
d_hollow_hemi_up     = Data("Hollow Hemisphere Up Diameter", "d", 53.32e-3, 0.00002)      # meters
d_hollow_hemi_down   = Data("Hollow Hemisphere Down Diameter", "d", 53.11e-3, 0.00002)    # meters
d_smooth_sphere      = Data("Smooth Sphere Diameter", "d", 50.83e-3, 0.00002)       # meters
d_rough_sphere       = Data("Rough Sphere Diameter", "d", 52.44e-3, 0.00002)        # meters

diameter_by_case = [
    d_circular_flat_disk,  # 0
    d_hollow_hemi_up,      # 1
    d_hollow_hemi_down,    # 2
    d_smooth_sphere,       # 3
    d_rough_sphere,        # 4
]
 

q_45 = Data("Dynamic Pressure (45Hz)", "q", 2.462 * inH2O_to_Pa, 0.132 * inH2O_to_Pa)
q_55 = Data("Dynamic Pressure (55Hz)", "q", 3.928 * inH2O_to_Pa, 0.132 * inH2O_to_Pa)

q_by_case = [
    q_45,  # 0
    q_55,  # 1
]

Res = []
Drag_Coeffs = []

for i, case in enumerate(drag_by_case):
    v = RSS(f"Velocity", equations[5], [q_by_case[i//5], ρ])  # i//5 gives 0 for first 5, 1 for last 5
    Re = RSS(f"Reynolds Number", equations[3], [ρ, v, diameter_by_case[i%5], mu])
    A = RSS(f"Reference Area", equations[6], [diameter_by_case[i%5]])
    Cd = RSS(f"Drag Coefficient", equations[7], [case, q_by_case[i//5], A])

    Res.append(Re)
    Drag_Coeffs.append(Cd)

    print(f"\n--- Case {i} Summary ---")
    diameter_by_case[i%5].get_description()
    q_by_case[i//5].get_description()
    Re.get_description()
    case.get_description()
    Cd.get_description()
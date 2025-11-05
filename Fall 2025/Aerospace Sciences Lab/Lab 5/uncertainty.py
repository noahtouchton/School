# uncertainty.py
import os
import pandas as pd
import numpy as np
from rss import Data, RSS, build_Cp_by_angle_P_over_q
import sympy as sp
from xfoil_driver import *


inH2O_to_Pa = 249.0889
# === CONFIGURATION ===
cwd = os.getcwd()

# Walk downward from the repo root (School) to find "Lab5AOA"
def find_lab5aoa(root):
    for dirpath, dirnames, filenames in os.walk(root):
        if "Lab5AOA" in dirnames:
            return os.path.join(dirpath, "Lab5AOA")
    raise FileNotFoundError("Couldn't find Lab5AOA folder under " + root)

folder = find_lab5aoa(cwd)
print(f"Found Lab5AOA folder at: {folder}")
angles = list(range(-22, 24, 2))  # -22, -20, ..., 0, ..., 22
header_row = 3  # Excel header row is 4 → pandas uses 0-index

# === STORAGE VARIABLE ===
all_data = {}  # all_data[angle]["Port 1"] → Data object

for angle in angles:
    # filename pattern
    if angle < 0:
        filename = f"Lab5neg{abs(angle)}.xlsx"
    else:
        filename = f"Lab5{angle}.xlsx"
    path = os.path.join(folder, filename)
    if not os.path.exists(path):
        print(f"Skipping {filename} (not found)")
        continue

    # read Excel, header row 4 (index 3)
    df = pd.read_excel(path, header=header_row)
    df.columns = [str(c).strip() for c in df.columns]

    # drop fully empty cols
    df = df.dropna(axis=1, how="all")

    # store each column as Data
    angle_data = {}
    for col in df.columns:
        vals = np.array(df[col], dtype=float)
        mean = np.mean(vals)
        u = np.std(vals, ddof=1) / np.sqrt(len(vals))
        sym = sp.Symbol(col.strip().replace(" ", "_").replace("(", "").replace(")", "").lower())
        if sym != 'q':
            sym = 'P'
        angle_data[col] = Data(name=col, var=sym, value=mean, uncertainty=u)

    all_data[angle] = angle_data
    print(f"Loaded {filename}: {len(df)} rows, {len(df.columns)} columns")

# Example usage after import:
# from uncertainty import all_data
#port1 = all_data[0]["Port 1"]



MU_0 = Data("mu not", "MU_0", 1.716e-5, 0.0)
T_0  = Data("T not",  "T_0",  273.0,   0.0)
C    = Data("Constant","C",   111.0,   0.0)
R    = Data("Ideal Gas Constant for Air", "R", 287.05, 0.0)

# Lab ambient & geometry (edit with your values/uncertainties)
T = Data("Temperature", "T", 23.4+273.15, 0.4)
P = Data("Pressure",    "P", 101700, 400.0)
gamma = Data("Ratio of Specific Heats", "gamma", 1.4, 0.0)

equations = [
    "ρ = P / (R * T)",
    "mu = MU_0 * (T / T_0)**(3/2) * (T_0 + C) / (T + C)",
    "v = sqrt(2*q/ρ)",
    "Re = ρ * v * L / mu",
    "Mach = v/sqrt(gamma*R*T)",
]

ρ = RSS("Density", equations[0], [P,R,T])
mu = RSS("Dynamic Viscosity",  equations[1], [T, MU_0, T_0, C])

xU_vals = np.array([0.2, 0.4, 0.8, 1.2, 1.6, 2.4, 3.2], dtype=float)   # Ports 2..8
xL_vals = np.array([0.0, 0.4, 0.8, 1.2, 1.6, 2.19, 2.785], dtype=float) # Port 1, Ports 9..14

c_val = 4

xU = Data("Upper X", "x", xU_vals, 0.0)
xL = Data("Lower X", "x", xL_vals, 0.0)

c = Data("Chord Length", "c", c_val, 0.0)

Cp_by_angle = build_Cp_by_angle_P_over_q(all_data)
print("\n=== Pressure Coefficient Summary ===")
for ang in sorted(Cp_by_angle.keys()):
    print(f"\nAngle of Attack: {ang:+d}°")
    for cp_name, cp_data in Cp_by_angle[ang].items():
        print(f"  {cp_name:<8} = {cp_data.value:8.5f} ± {cp_data.uncertainty:8.5f}")

q_vals = []
for ang in all_data:
    for col, data in all_data[ang].items():
        if data.var == 'q' or 'q' in data.name.lower():
            q_vals.append(data.value)

if q_vals:
    q_mean = np.mean(q_vals) * inH2O_to_Pa
    q_std  = np.std(q_vals, ddof=1) * inH2O_to_Pa

    q = Data("Dynamic Pressure", "q", q_mean, q_std * np.sqrt(1/len(q_vals)))
    v = RSS("Velocity", equations[2], [q, ρ])
    l = Data("Length", 'L', 4 / 39.37, 0.0)  
    Re = RSS("Reynolds Number", equations[3], [ρ, v, l, mu])
    Mach = RSS("Mach Number", equations[4], [v, gamma, R, T])

    print(f"q  = {q.value:.2f} ± {q.uncertainty:.2f} Pa  (n={len(q_vals)})")
    print(f"V  = {v.value:.3f} ± {v.uncertainty:.3f} m/s")
    print(f"Re = {Re.value:.3e} ± {Re.uncertainty:.3e}")
    print(f"Mach = {Mach.value:.4f} ± {Mach.uncertainty:.4f}")
else:
    print("No q values found in dataset.")




#find Cn

def compute_Cn(CpU, CpL, xU, xL, c):
    """
    Compute Cn = ∫(Cp_lower - Cp_upper) d(x/c) using trapezoidal integration.
    CpU, CpL are lists/arrays of Data objects (one per tap)
    """
    # Convert to numpy arrays of numeric values
    cpU_vals = np.array([cp.value for cp in CpU], dtype=float)
    cpL_vals = np.array([cp.value for cp in CpL], dtype=float)
    xU_vals  = np.array(xU.value, dtype=float)
    xL_vals  = np.array(xL.value, dtype=float)
    c_val    = c.value

    # Trapezoidal integration (normalized by chord)
    upper_int = np.trapz(cpU_vals, xU_vals / c_val)
    lower_int = np.trapz(cpL_vals, xL_vals / c_val)

    Cn_val = lower_int - upper_int

    # Estimate uncertainty by RSS of weighted tap uncertainties
    u_upper = np.array([cp.uncertainty for cp in CpU], dtype=float)
    u_lower = np.array([cp.uncertainty for cp in CpL], dtype=float)
    # For independent taps, use local trapezoid weights as coefficients
    wU = np.gradient(xU_vals / c_val)
    wL = np.gradient(xL_vals / c_val)
    u_Cn = np.sqrt(np.sum((wL * u_lower)**2 + (wU * u_upper)**2))

    # Return as a Data object
    Cn = Data("Normal Force Coefficient", sp.Symbol("C_N"), Cn_val, u_Cn)
    return Cn


def compute_Cn_all(Cp_by_angle, xU, xL, c):
    """
    Build Cn for every angle in Cp_by_angle using:
      Upper: Cp 2..8  (len=7, mapped to xU)
      Lower: Cp 1 and Cp 9..14 (len=7, mapped to xL)
    Returns: dict[angle] -> Data("Normal Force Coefficient", C_N, value, uncertainty)
    """
    results = {}
    for ang in sorted(Cp_by_angle.keys()):
        cols = Cp_by_angle[ang]
        # assemble upper & lower Cp Data lists in the correct order
        try:
            CpU = [cols[f"Cp {i}"] for i in range(2, 9)]                     # 2..8 (7 taps)
            CpL = [cols["Cp 1"]] + [cols[f"Cp {i}"] for i in range(9, 15)]    # 1, 9..14 (7 taps)
        except KeyError as e:
            print(f"[WARN] Missing {e} at angle {ang}; skipping.")
            continue

        # sanity check on lengths vs x arrays
        if len(CpU) != len(xU.value) or len(CpL) != len(xL.value):
            print(f"[WARN] Tap count mismatch at angle {ang}; skipping.")
            continue

        results[ang] = compute_Cn(CpU, CpL, xU, xL, c)
    return results

# --- run it and print nicely ---
Cn_by_angle = compute_Cn_all(Cp_by_angle, xU, xL, c)

print("\n=== Normal Coefficient Cn by angle ===")
for ang in sorted(Cn_by_angle.keys()):
    Cn = Cn_by_angle[ang]
    print(f"α={ang:+3}°  Cn = {Cn.value: .5f} ± {Cn.uncertainty:.5f}")

def compute_Cl_all(Cn_by_angle):
    Cl_by_angle = {}
    for ang, Cn in Cn_by_angle.items():
        alpha_rad = np.deg2rad(ang)
        Cl_val = Cn.value * np.cos(alpha_rad)
        u_Cl   = abs(np.cos(alpha_rad)) * Cn.uncertainty
        Cl_by_angle[ang] = Data("Lift Coefficient", sp.Symbol("C_L"), Cl_val, u_Cl)
    return Cl_by_angle

def compute_Cd_all(Cn_by_angle):
    Cd_by_angle = {}
    for ang, Cn in Cn_by_angle.items():
        alpha_rad = np.deg2rad(ang)
        Cd_val = Cn.value * np.sin(alpha_rad)
        u_Cd   = abs(np.sin(alpha_rad)) * Cn.uncertainty
        Cd_by_angle[ang] = Data("Drag Coefficient (pressure-only)", sp.Symbol("C_D"), Cd_val, u_Cd)
    return Cd_by_angle


Cl_by_angle = compute_Cl_all(Cn_by_angle)
Cd_by_angle = compute_Cd_all(Cn_by_angle)

print("\n=== Lift coefficient Cl by angle ===")
for ang in sorted(Cl_by_angle.keys()):
    d = Cl_by_angle[ang]
    print(f"α={ang:+3}°  Cl = {d.value: .5f} ± {d.uncertainty:.5f}")

def compute_Cm_c4(CpU, CpL, xU, xL, c, x_c_ref=0.25):
    cpU_vals = np.array([cp.value for cp in CpU], dtype=float)
    cpL_vals = np.array([cp.value for cp in CpL], dtype=float)
    uU       = np.array([cp.uncertainty for cp in CpU], dtype=float)
    uL       = np.array([cp.uncertainty for cp in CpL], dtype=float)
    xU_vals  = np.array(xU.value, dtype=float)
    xL_vals  = np.array(xL.value, dtype=float)
    c_val    = c.value

    # Weights for trapezoid in x/c
    xU_nc = xU_vals / c_val
    xL_nc = xL_vals / c_val
    wU = np.gradient(xU_nc)
    wL = np.gradient(xL_nc)

    # Lever arms
    armU = (xU_nc - x_c_ref)
    armL = (xL_nc - x_c_ref)

    # Moment value
    Cm_val = np.dot(wL, cpL_vals * armL) - np.dot(wU, cpU_vals * armU)

    # Uncertainty (linear RSS)
    u_Cm = np.sqrt(np.sum((wL * armL * uL)**2) + np.sum((wU * armU * uU)**2))

    return Data("Quarter-chord moment", sp.Symbol("C_m_c4"), float(Cm_val), float(u_Cm))

def compute_Cm_all(Cp_by_angle, xU, xL, c):
    res = {}
    for ang in sorted(Cp_by_angle.keys()):
        cols = Cp_by_angle[ang]
        try:
            CpU = [cols[f"Cp {i}"] for i in range(2, 9)]                     # 2..8
            CpL = [cols["Cp 1"]] + [cols[f"Cp {i}"] for i in range(9, 15)]    # 1,9..14
        except KeyError:
            continue
        if len(CpU) != len(xU.value) or len(CpL) != len(xL.value):
            continue
        res[ang] = compute_Cm_c4(CpU, CpL, xU, xL, c)
    return res


Cm_by_angle = compute_Cm_all(Cp_by_angle, xU, xL, c)
print("\n=== Quarter-chord moment Cm by angle ===")
for ang in sorted(Cm_by_angle.keys()):
    d = Cm_by_angle[ang]
    print(f"α={ang:+3}°  Cm(c/4) = {d.value: .5f} ± {d.uncertainty:.5f}")

# --- Drag coefficient (pressure-only) from tap data ---
Cd_by_angle = compute_Cd_all(Cn_by_angle)

print("\n=== Drag coefficient Cd (pressure-only) by angle ===")
for ang in sorted(Cd_by_angle.keys()):
    d = Cd_by_angle[ang]
    print(f"α={ang:+3}°  Cd = {d.value: .5f} ± {d.uncertainty:.5f}")


df_x = run_xfoil_polar(
    airfoil="NACA 4412",                 # or r"C:\path\to\foil.dat"
    Re=Re.value, Mach=Mach.value, Ncrit=9,
    a_start=-22, a_end=22, a_step=1,      # start modest; widen once it works
    iter_lim=400,
    xfoil_path=r"C:\Users\noaht\Downloads\xfoil6.99\xfoil.exe"
)
print(df_x.head())

import matplotlib.pyplot as plt
import pandas as pd
import os

# ---------- build experimental dataframe from your Cp-based results ----------
exp_rows = []
for ang in sorted(Cn_by_angle.keys()):
    exp_rows.append({
        "alpha": ang,
        "Cl": Cl_by_angle[ang].value,
        "Cd": Cd_by_angle[ang].value,
        "Cm": Cm_by_angle[ang].value,
    })
exp_df = pd.DataFrame(exp_rows).sort_values("alpha").reset_index(drop=True)

# ---------- try to get an inviscid reference ----------
def try_xfoil_inviscid():
    try:
        return run_xfoil_polar(
            airfoil="NACA 4412",
            Re=None,                # signals inviscid in our driver (<=0 or None)
            Mach=0.0,
            Ncrit=None,
            a_start=int(exp_df["alpha"].min()),
            a_end=int(exp_df["alpha"].max()),
            a_step=1,
            iter_lim=200,
            xfoil_path=r"C:\Users\noaht\Downloads\xfoil6.99\xfoil.exe",
            viscous=False           # if your driver supports this flag
        )
    except Exception:
        return None

df_inv = try_xfoil_inviscid()

# Fallback: thin-airfoil line if inviscid run not available
if df_inv is None:
    alphas = exp_df["alpha"].to_numpy()
    alpha0L_deg = -2.0
    a_per_rad = 2*np.pi                    # ≈ 6.283/rad
    Cl_tat = a_per_rad * np.deg2rad(alphas - alpha0L_deg)
    Cm_tat = np.full_like(Cl_tat, -0.10)   # NACA 4412 typical order; tweak if you prefer
    Cd_tat = np.zeros_like(Cl_tat)         # pressure-only theory
    df_inv = pd.DataFrame({
        "alpha": alphas,
        "Cl": Cl_tat,
        "Cd": Cd_tat,
        "Cm": Cm_tat,
    })
    inviscid_label = "Thin-airfoil theory"
else:
    inviscid_label = "XFOIL inviscid"

# ---------- output folder (relative to this file so it works on GitHub) ----------
outdir = os.path.join(os.path.dirname(__file__), "xfoil_outputs")
os.makedirs(outdir, exist_ok=True)
print(f"Saving plots to: {outdir}")

# also save the viscous polar you already computed
df_x.to_csv(os.path.join(outdir, "polar_viscous.csv"), index=False)
exp_df.to_csv(os.path.join(outdir, "polar_experimental.csv"), index=False)
df_inv.to_csv(os.path.join(outdir, "polar_inviscid_or_TAT.csv"), index=False)

# ---------- plots: overlay Experimental vs XFOIL (viscous) vs Inviscid/Theory ----------
def save_plot(xlabel, ylabel, title, x_exp, y_exp, x_v, y_v, x_i, y_i, filename):
    plt.figure()
    plt.plot(x_exp, y_exp, "o-", label="Experimental")
    plt.plot(x_v,   y_v,   "s-", label="XFOIL viscous")
    plt.plot(x_i,   y_i,   "^-", label=inviscid_label)
    plt.xlabel(xlabel); plt.ylabel(ylabel); plt.title(title)
    plt.grid(True, alpha=0.3); plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, filename), dpi=200)
    plt.close()

# Cl vs alpha
save_plot("Alpha (deg)", "Cl", "NACA 4412 — Cl vs Alpha",
          exp_df["alpha"], exp_df["Cl"],
          df_x["alpha"],   df_x["Cl"],
          df_inv["alpha"], df_inv["Cl"],
          "Cl_vs_alpha_all.png")

# Cd vs alpha
save_plot("Alpha (deg)", "Cd", "NACA 4412 — Cd vs Alpha",
          exp_df["alpha"], exp_df["Cd"],
          df_x["alpha"],   df_x["Cd"],
          df_inv["alpha"], df_inv["Cd"],
          "Cd_vs_alpha_all.png")

# Cm vs alpha
save_plot("Alpha (deg)", "Cm (c/4)", "NACA 4412 — Cm(c/4) vs Alpha",
          exp_df["alpha"], exp_df["Cm"],
          df_x["alpha"],   df_x["Cm"],
          df_inv["alpha"], df_inv["Cm"],
          "Cm_vs_alpha_all.png")

# Drag polar (Cl vs Cd)
save_plot("Cd", "Cl", "NACA 4412 — Drag Polar",
          exp_df["Cd"], exp_df["Cl"],
          df_x["Cd"],   df_x["Cl"],
          df_inv["Cd"], df_inv["Cl"],
          "Drag_Polar_all.png")
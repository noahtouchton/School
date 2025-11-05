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

        # robust q vs P detection
        name_clean = str(col).strip().lower()
        var = 'q' if name_clean == 'q' else 'P'

        angle_data[col] = Data(name=col, var=var, value=mean, uncertainty=u)

    all_data[angle] = angle_data
    print(f"Loaded {filename}: {len(df)} rows, {len(df.columns)} columns")

# Example usage after import:
# from uncertainty import all_data
# port1 = all_data[0]["Port 1"]

MU_0 = Data("mu not", "MU_0", 1.716e-5, 0.0)
T_0  = Data("T not",  "T_0",  273.0,   0.0)
C    = Data("Constant","C",   111.0,   0.0)
R    = Data("Ideal Gas Constant for Air", "R", 287.05, 0.0)

# Lab ambient & geometry (edit with your values/uncertainties)
T = Data("Temperature", "T", 23.4+273.15, 0.4)
P = Data("Pressure",    "P", 101700, 400.0)
gamma = Data("Ratio of Specific Heats", "gamma", 1.4, 0.0)

# NOTE: use ** for powers and pi, not ^
equations = [
    "ρ = P / (R * T)",
    "mu = MU_0 * (T / T_0)**(3/2) * (T_0 + C) / (T + C)",
    "v = sqrt(2*q/ρ)",
    "Re = ρ * v * L / mu",
    "Mach = v/sqrt(gamma*R*T)",
    "σ = (pi**2/48) * (c/h)**2",
    "ε_sb = Λ * (pi**2 / 48) * (c / h)**2",
    "ε_wb = (c / h)**2 * Cd_u",
    "ε = ε_sb + ε_wb",
    "V = V_u * (1 + ε)",
    "q = q_u * (1 + ε)",
    "Re_corr = Re_u * (1 + ε)",
    "α_corr = α_u + (57.3 * (pi**2 / 48) * (c / h)**2 / (2 * pi)) * (Cl_u + 4 * Cm_u)",
    "Cl_corr = Cl_u * (1 - (pi**2 / 48) * (c / h)**2 - 2 * ε)",
    "Cm_corr = Cm_u * (1 - 2 * (pi**2 / 48) * (c / h)**2) + 0.25 * (pi**2 / 48) * (c / h)**2 * Cl_corr",
    "Cd0_corr = Cd0_u * (1 - 3 * ε_sb - 2 * ε_wb)",
]

ρ = RSS("Density", equations[0], [P,R,T])
mu = RSS("Dynamic Viscosity",  equations[1], [T, MU_0, T_0, C])

xU_vals = np.array([0.2, 0.4, 0.8, 1.2, 1.6, 2.4, 3.2], dtype=float)   # Ports 2..8
xL_vals = np.array([0.0, 0.4, 0.8, 1.2, 1.6, 2.19, 2.785], dtype=float) # Port 1, Ports 9..14

c_val = 4

xU = Data("Upper X", "x", xU_vals, 0.0)
xL = Data("Lower X", "x", xL_vals, 0.0)

c = Data("Chord Length", "c", c_val, 0.0)
h = Data("Tunnel Height", "h", 24.0, 0.0)

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

# find Cn
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
    u_Cn = np.sqrt(np.sum((wL * u_lower)**2) + np.sum((wU * u_upper)**2))

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
    Cm_val = -(np.dot(wL, cpL_vals * armL)) + np.dot(wU, cpU_vals * armU)

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

# ---- Wind-tunnel corrections (solid + wake blockage) ----
# Body-shape factor Λ from your lab manual (set this to the right value for your model)
Lambda = Data("Body shape factor", "Λ", 1.0, 0.0)  # <-- update per lab

# Helpers and precompute
pi = np.pi
c_over_h = c.value / h.value
sigma = (pi**2 / 48.0) * (c_over_h**2)            # σ = (π^2/48)*(c/h)^2
eps_sb = Lambda.value * sigma                      # solid blockage

# We'll build angle-by-angle dicts:
alpha_corr_by_angle = {}
Cl_corr_by_angle    = {}
Cm_corr_by_angle    = {}
Re_corr_by_angle    = {}
q_corr_by_angle     = {}
eps_by_angle        = {}  # <— used later for corrected Re/Mach XFOIL run
Cd_corr_by_angle = {}     # NEW: corrected Cd by angle (Eq. 12 extended to all AoA)



for ang in sorted(Cl_by_angle.keys()):
    # Uncorrected inputs per angle
    Cl_u = Cl_by_angle[ang]      # Data
    Cm_u = Cm_by_angle[ang]      # Data
    Cd_u = Cd_by_angle.get(ang)  # Data (pressure-only drag from taps)
    if Cd_u is None:
        continue

    # Wake blockage with local Cd
    eps_wb = (c_over_h**2) * max(Cd_u.value, 0.0)
    eps    = eps_sb + eps_wb
    eps_by_angle[ang] = eps

    # --- Corrected quantities ---
    # Angle (deg): α_corr = α_u + (57.3 * σ / (2π)) * (Cl_u + 4 Cm_u)
    alpha_corr = ang + (57.3 * sigma / (2.0 * pi)) * (Cl_u.value + 4.0 * Cm_u.value)

    # Lift: Cl_corr = Cl_u * (1 - σ - 2 ε)
    Cl_scale     = (1.0 - sigma - 2.0 * eps)
    Cl_corr_val  = Cl_u.value * Cl_scale
    Cl_corr_unc  = abs(Cl_scale) * Cl_u.uncertainty

    # Moment: Cm_corr = Cm_u*(1 - 2σ) + 0.25*σ*Cl_corr
    Cm_a = (1.0 - 2.0 * sigma)
    Cm_b = 0.25 * sigma
    Cm_corr_val = Cm_a * Cm_u.value + Cm_b * Cl_corr_val
    Cm_corr_unc = np.sqrt((abs(Cm_a) * Cm_u.uncertainty)**2 + (abs(Cm_b) * Cl_corr_unc)**2)

    # Re, q corrections with same eps
    Re_corr_val = Re.value * (1.0 + eps)
    Re_corr_unc = abs(1.0 + eps) * Re.uncertainty

    q_corr_val  = q.value * (1.0 + eps)
    q_corr_unc  = abs(1.0 + eps) * q.uncertainty

        # Cd_corr = Cd_u * (1 - 3*ε_sb - 2*ε_wb)
    Cd_scale     = (1.0 - 3.0 * eps_sb - 2.0 * eps_wb)
    Cd_corr_val  = Cd_u.value * Cd_scale
    Cd_corr_unc  = abs(Cd_scale) * Cd_u.uncertainty

    # Store as Data
    alpha_corr_by_angle[ang] = alpha_corr  # plain float is fine for plotting
    Cl_corr_by_angle[ang] = Data("Lift Coefficient (corrected)", sp.Symbol("C_L_corr"),
                                 Cl_corr_val, Cl_corr_unc)
    Cm_corr_by_angle[ang] = Data("Moment Coefficient (c/4, corrected)", sp.Symbol("C_m_corr"),
                                 Cm_corr_val, Cm_corr_unc)
    Re_corr_by_angle[ang] = Data("Reynolds Number (corrected)", sp.Symbol("Re_corr"),
                                 Re_corr_val, Re_corr_unc)
    q_corr_by_angle[ang]  = Data("Dynamic Pressure (corrected)", sp.Symbol("q_corr"), q_corr_val, q_corr_unc)
    Cd_corr_by_angle[ang]    = Data("Drag Coefficient (corrected)",        sp.Symbol("C_D_corr"),  Cd_corr_val, Cd_corr_unc)  # NEW

# --- Quick sanity print (optional) ---
print("\n=== Corrected vs Uncorrected (samples) ===")
for ang in [-10, 0, 10]:
    if ang in Cl_by_angle and ang in Cl_corr_by_angle:
        print(f"α={ang:+2}°  α_corr={alpha_corr_by_angle[ang]:6.2f}°  "
              f"Cl_u={Cl_by_angle[ang].value: .3f} → Cl_corr={Cl_corr_by_angle[ang].value: .3f}  "
              f"Cm_u={Cm_by_angle[ang].value: .3f} → Cm_corr={Cm_corr_by_angle[ang].value: .3f}")

# --- Run XFOIL (uncorrected globals) ---
df_x = run_xfoil_polar(
    airfoil="NACA 4412",                 # or r"C:\path\to\foil.dat"
    Re=Re.value, Mach=Mach.value, Ncrit=9,
    a_start=-22, a_end=22, a_step=1,      # start modest; widen once it works
    iter_lim=400,
    xfoil_path=r"C:\Users\noaht\Downloads\xfoil6.99\xfoil.exe"
)
print(df_x.head())

# ---- Run XFOIL again with corrected global conditions (mean ε) ----
if len(eps_by_angle) > 0:
    eps_mean = float(np.nanmean(list(eps_by_angle.values())))
else:
    eps_mean = 0.0

Re_corr_global   = Re.value   * (1.0 + eps_mean)
Mach_corr_global = Mach.value * (1.0 + eps_mean)

print(f"\nRunning XFOIL with corrected globals: Re={Re_corr_global:.3e}, Mach={Mach_corr_global:.5f}, ε_mean={eps_mean:.3e}")
df_x_corr = run_xfoil_polar(
    airfoil="NACA 4412",
    Re=Re_corr_global, Mach=Mach_corr_global, Ncrit=9,
    a_start=-16, a_end=16, a_step=1,
    iter_lim=400,
    xfoil_path=r"C:\Users\noaht\Downloads\xfoil6.99\xfoil.exe"
)
print(df_x_corr.head())

import matplotlib.pyplot as plt
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

# also save the viscous polars you computed
df_x.to_csv(os.path.join(outdir, "polar_viscous.csv"), index=False)
df_x_corr.to_csv(os.path.join(outdir, "polar_viscous_corrected_globals.csv"), index=False)
exp_df.to_csv(os.path.join(outdir, "polar_experimental.csv"), index=False)
df_inv.to_csv(os.path.join(outdir, "polar_inviscid_or_TAT.csv"), index=False)

# ---------- build a corrected experimental dataframe for plotting ----------
corr_plot_rows = []
for ang in sorted(Cl_corr_by_angle.keys()):
    corr_plot_rows.append({
        "alpha": alpha_corr_by_angle[ang],             # corrected alpha
        "Cl": Cl_corr_by_angle[ang].value,
        "Cd":    Cd_corr_by_angle[ang].value,   # NEW
        "Cm": Cm_corr_by_angle[ang].value,
    })
corr_exp_df = pd.DataFrame(corr_plot_rows).sort_values("alpha").reset_index(drop=True)

# ---------- plotting helper ----------
def save_plot4(xlabel, ylabel, title, series, filename):
    plt.figure()
    for s in series:
        x, y = s.get("x"), s.get("y")
        if x is None or y is None:
            continue
        plt.plot(x, y, s.get("fmt", "-"), label=s.get("label", ""))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, filename), dpi=200)
    plt.close()


# ---------- Cl vs alpha ----------
save_plot4(
    xlabel="Alpha (deg)", ylabel="Cl",
    title="NACA 4412 — Cl vs Alpha",
    series=[
        {"x": df_inv["alpha"],      "y": df_inv["Cl"],      "fmt": "^-", "label": "Thin Airfoil Theory"},
        {"x": exp_df["alpha"],      "y": exp_df["Cl"],      "fmt": "o-", "label": "Experimental (uncorrected)"},
        {"x": corr_exp_df["alpha"], "y": corr_exp_df["Cl"], "fmt": "o-", "label": "Experimental (corrected)"},
        {"x": df_x["alpha"],        "y": df_x["Cl"],        "fmt": "s-", "label": "XFOIL viscous"},
    ],
    filename="Cl_vs_alpha_all.png"
)

# ---------- Cd vs alpha ----------
save_plot4(
    xlabel="Alpha (deg)", ylabel="Cd",
    title="NACA 4412 — Cd vs Alpha",
    series=[
        {"x": df_inv["alpha"],      "y": df_inv["Cd"],      "fmt": "^-", "label": "Thin Airfoil Theory"},
        {"x": exp_df["alpha"],      "y": exp_df["Cd"],      "fmt": "o-", "label": "Experimental (uncorrected)"},
        {"x": corr_exp_df["alpha"], "y": corr_exp_df["Cd"], "fmt": "o-", "label": "Experimental (corrected)"},
        {"x": df_x["alpha"],        "y": df_x["Cd"],        "fmt": "s-", "label": "XFOIL viscous"},
    ],
    filename="Cd_vs_alpha_all.png"
)

# ---------- Cm vs alpha ----------
save_plot4(
    xlabel="Alpha (deg)", ylabel="Cm (c/4)",
    title="NACA 4412 — Cm(c/4) vs Alpha",
    series=[
        {"x": df_inv["alpha"],      "y": df_inv["Cm"],      "fmt": "^-", "label": "Thin Airfoil Theory"},
        {"x": exp_df["alpha"],      "y": exp_df["Cm"],      "fmt": "o-", "label": "Experimental (uncorrected)"},
        {"x": corr_exp_df["alpha"], "y": corr_exp_df["Cm"], "fmt": "o-", "label": "Experimental (corrected)"},
        {"x": df_x["alpha"],        "y": df_x["Cm"],        "fmt": "s-", "label": "XFOIL viscous"},
    ],
    filename="Cm_vs_alpha_all.png"
)

# ---------- Drag polar (Cl vs Cd) ----------
save_plot4(
    xlabel="Cd", ylabel="Cl",
    title="NACA 4412 — Drag Polar",
    series=[
        {"x": df_inv["Cd"],         "y": df_inv["Cl"],         "fmt": "^-", "label": "Thin Airfoil Theory"},
        {"x": exp_df["Cd"],         "y": exp_df["Cl"],         "fmt": "o-", "label": "Experimental (uncorrected)"},
        {"x": corr_exp_df["Cd"],    "y": corr_exp_df["Cl"],    "fmt": "o-", "label": "Experimental (corrected)"},
        {"x": df_x["Cd"],           "y": df_x["Cl"],           "fmt": "s-", "label": "XFOIL viscous"},
    ],
    filename="Drag_Polar_all.png"
)


# === FINAL ORGANIZED SUMMARY PRINTS (keeps all prior functionality) ===
print("\n" + "="*78)
print("FINAL ORGANIZED SUMMARY")
print("="*78)

# 1) Scalars / environment
print("\n-- Environment / Flow scalars --")
print(f"ρ    = {ρ.value:.6f} ± {ρ.uncertainty:.6f} kg/m³")
print(f"μ    = {mu.value:.7f} ± {mu.uncertainty:.7f} Pa·s")
print(f"q    = {q.value:.2f} ± {q.uncertainty:.2f} Pa")
print(f"V    = {v.value:.3f} ± {v.uncertainty:.3f} m/s")
print(f"Re   = {Re.value:.3e} ± {Re.uncertainty:.3e}")
print(f"Mach = {Mach.value:.4f} ± {Mach.uncertainty:.4f}")

# 2) Blockage parameters summary (uses already-built eps_by_angle)
eps_vals = list(eps_by_angle.values())
eps_min = min(eps_vals) if eps_vals else float('nan')
eps_max = max(eps_vals) if eps_vals else float('nan')

print("\n-- Tunnel blockage/corrections --")
print(f"c/h = {c_over_h:.5f}")
print(f"σ   = (π^2/48)*(c/h)^2 = {sigma:.6e}")
print(f"Λ   = {Lambda.value:.3f}   (body shape factor)")
print(f"ε_sb (solid blockage) = {eps_sb:.6e}")
print(f"ε range (total)       = [{eps_min:.6e}, {eps_max:.6e}]")

# 3) Build per-angle tables (uncorrected vs corrected)
# Uncorrected experimental (from taps)
unc_rows = []
for ang in sorted(Cn_by_angle.keys()):
    unc_rows.append({
        "alpha_deg": ang,
        "Cl_u": Cl_by_angle[ang].value,
        "Cl_u_unc": Cl_by_angle[ang].uncertainty,
        "Cd_u(p-only)": Cd_by_angle[ang].value,
        "Cd_u_unc": Cd_by_angle[ang].uncertainty,
        "Cm_u(c/4)": Cm_by_angle[ang].value,
        "Cm_u_unc": Cm_by_angle[ang].uncertainty,
        "ε_total": eps_by_angle.get(ang, float('nan')),
    })
unc_df = pd.DataFrame(unc_rows)

# Corrected experimental
corr_rows = []
for ang in sorted(Cl_corr_by_angle.keys()):
    corr_rows.append({
        "alpha_u_deg": ang,
        "alpha_corr_deg": alpha_corr_by_angle.get(ang, np.nan),
        "Cl_corr": Cl_corr_by_angle[ang].value,
        "Cl_corr_unc": Cl_corr_by_angle[ang].uncertainty,
        "Cm_corr(c/4)": Cm_corr_by_angle[ang].value,
        "Cm_corr_unc": Cm_corr_by_angle[ang].uncertainty,
        "Re_corr": Re_corr_by_angle[ang].value if ang in Re_corr_by_angle else np.nan,
        "Re_corr_unc": Re_corr_by_angle[ang].uncertainty if ang in Re_corr_by_angle else np.nan,
        "q_corr": q_corr_by_angle[ang].value if ang in q_corr_by_angle else np.nan,
        "q_corr_unc": q_corr_by_angle[ang].uncertainty if ang in q_corr_by_angle else np.nan,
    })
corr_df = pd.DataFrame(corr_rows)

# XFOIL viscous polar (already computed as df_x)
xfoil_v_df = df_x[["alpha", "Cl", "Cd", "Cm"]].copy()
xfoil_v_df = xfoil_v_df.rename(columns={"alpha":"alpha_deg", "Cm":"Cm(c/4)"})

# Inviscid / TAT reference (df_inv already built)
xfoil_i_df = df_inv[["alpha", "Cl", "Cd", "Cm"]].copy()
xfoil_i_df = xfoil_i_df.rename(columns={"alpha":"alpha_deg", "Cm":"Cm(c/4)"})

# 4) Pretty prints
pd.set_option("display.width", 120)
pd.set_option("display.max_columns", None)

print("\n-- Uncorrected experimental (pressure-tap based) --")
print(unc_df.to_string(index=False))

print("\n-- Corrected experimental (wind-tunnel corrections applied) --")
print(corr_df.to_string(index=False))

print("\n-- XFOIL viscous polar --")
print(xfoil_v_df.to_string(index=False))

print(f"\n-- {inviscid_label} --")
print(xfoil_i_df.to_string(index=False))

# 5) Save organized tables to CSVs (alongside your plots)
summary_dir = outdir  # already created above
unc_df.to_csv(os.path.join(summary_dir, "summary_experimental_uncorrected.csv"), index=False)
corr_df.to_csv(os.path.join(summary_dir, "summary_experimental_corrected.csv"), index=False)
xfoil_v_df.to_csv(os.path.join(summary_dir, "summary_xfoil_viscous.csv"), index=False)
xfoil_i_df.to_csv(os.path.join(summary_dir, "summary_inviscid_or_TAT.csv"), index=False)

print("\nSaved summary CSVs to:", summary_dir)
print("="*78)

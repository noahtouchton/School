# uncertainty.py
import os
import pandas as pd
import numpy as np
from rss import Data, RSS, build_Cp_by_angle_P_over_q
import sympy as sp

# === CONFIGURATION ===
folder = os.path.join(os.getcwd(), "Lab5AOA")
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

equations = [
    "ρ = P / (R * T)",
    "mu = MU_0 * (T / T_0)**(3/2) * (T_0 + C) / (T + C)",
    "Re = ρ * v * d / mu",
]

ρ = RSS("Density", equations[0], [P,R,T])
mu = RSS("Dynamic Viscosity",  equations[1], [T, MU_0, T_0, C])

Cp_by_angle = build_Cp_by_angle_P_over_q(all_data)
print("\n=== Pressure Coefficient Summary ===")
for ang in sorted(Cp_by_angle.keys()):
    print(f"\nAngle of Attack: {ang:+d}°")
    for cp_name, cp_data in Cp_by_angle[ang].items():
        print(f"  {cp_name:<8} = {cp_data.value:8.5f} ± {cp_data.uncertainty:8.5f}")
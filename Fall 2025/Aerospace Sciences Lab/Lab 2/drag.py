import numpy as np
import pandas as pd
from pathlib import Path

# --- constants / geometry ---
INH2O_TO_PA = 249.0889
D  = 0.01905           # m
R  = D / 2             # m

# calibration to get q from scan Δp (inH2O)
m, b = 0.5713, -0.296
dp35_scan, dp55_scan = 2.701, 6.868  # inH2O
q35 = (m*dp35_scan + b) * INH2O_TO_PA
q55 = (m*dp55_scan + b) * INH2O_TO_PA

# --- load 37-point wrap blocks from your workbook ---
HERE = Path(__file__).resolve().parent
raw  = pd.read_excel(HERE / "Lab2Data.xlsx", sheet_name="Sheet1", header=None)

def run_block(row0, q):
    # use sheet columns: col 1 = Theta (Radians), col 8 = Cp
    theta = raw.iloc[row0:row0+37, 1].astype(float).to_numpy()
    Cp    = raw.iloc[row0:row0+37, 8].astype(float).to_numpy()

    # align so back = 0 (index of most negative Cp), wrap to [0, 2π), sort
    i_back = int(np.argmin(Cp))
    theta  = (theta - theta[i_back]) % (2*np.pi)
    order  = np.argsort(theta)
    theta, Cp = theta[order], Cp[order]

    # MATLAB-equivalent integrals (back-zero => plus sign)
    dp     = Cp * q
    Dprime = np.trapz(dp * np.cos(theta) * R, theta)     # N/m
    CD     = 0.5 * np.trapz(Cp * np.cos(theta), theta)   # dimensionless
    return Dprime, CD

# rows: 35-run starts at 26, 55-run at 77 (37 rows each)
Dprime35, CD35 = run_block(26, q35)
Dprime55, CD55 = run_block(77, q55)

print(f"q35 = {q35:.1f} Pa,  q55 = {q55:.1f} Pa")
print(f"D'(35) = {Dprime35:.3f} N/m,  C_D(35) = {CD35:.4f}")
print(f"D'(55) = {Dprime55:.3f} N/m,  C_D(55) = {CD55:.4f}")


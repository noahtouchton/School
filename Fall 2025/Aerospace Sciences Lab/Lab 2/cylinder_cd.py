# cylinder_cd.py
import numpy as np
import pandas as pd
import math

INH2O_TO_PA = 249.0889

# inputs
csv = "data.csv"          # columns: Angle_deg, dp_inH2O (no SD column)
D   = 0.0191              # m
q_inH2O = 4.36            # in. H2O

# load data
df = pd.read_csv(csv)

theta = np.deg2rad(df["beta"].to_numpy())     # radians
dp    = df["dp"].to_numpy()   # Pa

R = D/2
# Integrand: dp * cos(theta) * R, integrate over theta
integrand = dp * np.cos(theta) * R
Dprime = np.trapz(integrand, theta)                 # N/m (since Pa*m integrated over rad)

# dynamic pressure in Pa
q = q_inH2O * INH2O_TO_PA

CD = Dprime / (q * D)

print(f"D' = {Dprime:.3f} N/m")
print(f"q  = {q:.3f} Pa,  D = {D:.4f} m")
print(f"C_D = {CD:.4f}")
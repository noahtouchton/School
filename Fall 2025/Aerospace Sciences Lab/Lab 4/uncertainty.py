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




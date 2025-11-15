# uncertainty.py
import os
import pandas as pd
import numpy as np
from rss import Data, RSS
import sympy as sp


equations = [
    "ρ = P / (R * T)",
    "A = 3.14159 * d**2 / 4",
    "a = sqrt(gamma * R * T)",
    "tchar = V / (A * a)",
]

d_values = np.array([0.0017018,0.0023876,0.003175])
d_uncertainties = np.array([2.54e-05,2.54e-05,2.54e-05])


# Lab ambient & geometry (edit with your values/uncertainties)
T = Data("Temperature", "T", 297.05, 0.4)
P = Data("Pressure",    "P", 101500, 400.0)
R = Data("Ideal Gas Constant for Air", "R", 287.0, 0.0)
d = Data("Orifice Diameter",    "d", d_values, d_uncertainties)
A = RSS("Orifice Area", equations[1], [d])
V = Data("Tank Volume", "V", 0.125, 0.0)
gamma = Data("Specific Heat Ratio", "gamma", 1.4, 0.0)





# ---------- derived fluid props ----------
ρ  = RSS("Density",            equations[0], [P, R, T])


def load_lab6_data():
    # --- Locate Excel file next to this script ---
    here = os.path.dirname(os.path.abspath(__file__))
    excel_path = os.path.join(here, "Lab6Data.xlsx")

    # Helper to read a specific A:D range into a DataFrame
    def read_range(range_str, sheet_name=0):
        """
        range_str: e.g. 'A143:D206'
        Returns a DataFrame with columns [A, B, C, D] for that range.
        """
        # Use openpyxl engine so we can specify 'range' via usecols & skiprows logic
        # Easiest robust way is: read the whole sheet A:D and slice rows.
        # But here we parse the range directly.
        from openpyxl import load_workbook

        wb = load_workbook(excel_path, data_only=True)
        ws = wb[wb.sheetnames[sheet_name]]

        # Example range "A143:D206"
        start_cell, end_cell = range_str.split(":")
        start_col = start_cell[0]          # 'A'
        start_row = int(start_cell[1:])    # 143
        end_col   = end_cell[0]            # 'D'
        end_row   = int(end_cell[1:])      # 206

        # Collect values row by row
        data = []
        for r in range(start_row, end_row + 1):
            row_vals = []
            for c in ["A", "B", "C", "D"]:
                cell = f"{c}{r}"
                row_vals.append(ws[cell].value)
            data.append(row_vals)

        df = pd.DataFrame(data, columns=["A", "B", "C", "D"])
        return df

    # --- Read the three ranges by cell reference ---
    df_large  = read_range("A143:D206")   # large orifice
    df_medium = read_range("A302:D424")   # medium orifice
    df_small  = read_range("A734:D918")   # small orifice

    # --- Convert columns to physical quantities with correct units ---
    def convert_block(df):
        # Raw columns
        t   = np.array(df["A"], dtype=float)        # time [s]
        Pgk = np.array(df["B"], dtype=float)        # gauge pressure [kPa]
        Pabsk = np.array(df["C"], dtype=float)      # absolute pressure [kPa]
        Tc  = np.array(df["D"], dtype=float)        # temperature [°C]

        # Unit conversions
        Pg   = Pgk  * 1000.0            # Pa
        Pabs = Pabsk * 1000.0           # Pa
        T    = Tc + 273.15              # K

        return t, Pg, Pabs, T

    # --- Get arrays for each orifice size ---
    t_large, Pg_large, Pabs_large, T_large   = convert_block(df_large)
    t_med,   Pg_med,   Pabs_med,   T_med     = convert_block(df_medium)
    t_small, Pg_small, Pabs_small, T_small   = convert_block(df_small)

    # --- Build combined arrays (all data stacked) ---
    t_all    = np.concatenate([t_large, t_med, t_small])
    Pg_all   = np.concatenate([Pg_large, Pg_med, Pg_small])
    Pabs_all = np.concatenate([Pabs_large, Pabs_med, Pabs_small])
    T_all    = np.concatenate([T_large, T_med, T_small])

    # Return everything as a dict of 16 arrays
    return {
        # Large
        "t_large": t_large,
        "Pg_large": Pg_large,
        "Pabs_large": Pabs_large,
        "T_large": T_large,

        # Medium
        "t_med": t_med,
        "Pg_med": Pg_med,
        "Pabs_med": Pabs_med,
        "T_med": T_med,

        # Small
        "t_small": t_small,
        "Pg_small": Pg_small,
        "Pabs_small": Pabs_small,
        "T_small": T_small,

        # Combined (all)
        "t_all": t_all,
        "Pg_all": Pg_all,
        "Pabs_all": Pabs_all,
        "T_all": T_all,
    }

data = load_lab6_data()

# --- uncertainties from lab ---
P_unc = 1724.0   # Pa  (pressure transducer uncertainty)
T_unc = 0.5      # K   (thermocouple uncertainty)
t_unc = 0.0      # s   (LabVIEW time uncertainty)

# --- LARGE ORIFICE DATA ---
t_large   = Data("Time Large",            "t_large",   data["t_large"],   np.full_like(data["t_large"],   t_unc, dtype=float))
Pg_large  = Data("Gauge Pressure Large",  "Pg_large",  data["Pg_large"],  np.full_like(data["Pg_large"],  P_unc, dtype=float))
Pabs_large= Data("Abs Pressure Large",    "Pabs_large",data["Pabs_large"],np.full_like(data["Pabs_large"],P_unc, dtype=float))
T_large   = Data("Temperature Large",     "T_large",   data["T_large"],   np.full_like(data["T_large"],   T_unc, dtype=float))

# --- MEDIUM ORIFICE DATA ---
t_med   = Data("Time Medium",            "t_med",   data["t_med"],   np.full_like(data["t_med"],   t_unc, dtype=float))
Pg_med  = Data("Gauge Pressure Medium",  "Pg_med",  data["Pg_med"],  np.full_like(data["Pg_med"],  P_unc, dtype=float))
Pabs_med= Data("Abs Pressure Medium",    "Pabs_med",data["Pabs_med"],np.full_like(data["Pabs_med"],P_unc, dtype=float))
T_med   = Data("Temperature Medium",     "T_med",   data["T_med"],   np.full_like(data["T_med"],   T_unc, dtype=float))

# --- SMALL ORIFICE DATA ---
t_small   = Data("Time Small",            "t_small",   data["t_small"],   np.full_like(data["t_small"],   t_unc, dtype=float))
Pg_small  = Data("Gauge Pressure Small",  "Pg_small",  data["Pg_small"],  np.full_like(data["Pg_small"],  P_unc, dtype=float))
Pabs_small= Data("Abs Pressure Small",    "Pabs_small",data["Pabs_small"],np.full_like(data["Pabs_small"],P_unc, dtype=float))
T_small   = Data("Temperature Small",     "T_small",   data["T_small"],   np.full_like(data["T_small"],   T_unc, dtype=float))

T_large = Data("Temperature Large", "T_large", data["T_large"], np.full_like(data["T_large"], T_unc, dtype=float))
T_med   = Data("Temperature Medium","T_med",   data["T_med"],   np.full_like(data["T_med"],   T_unc, dtype=float))
T_small = Data("Temperature Small", "T_small", data["T_small"], np.full_like(data["T_small"], T_unc, dtype=float))



T0 = Data(
    "Initial Temperatures", 
    "T",
    np.array([
        T_small.value[0],
        T_med.value[0],
        T_large.value[0]
    ]),
    0.5   # scalar uncertainty – will broadcast to all three
)



ai = RSS("Speed of Sound", equations[2], [gamma, R, T0])

tchar = RSS("Characteristic Time", equations[3], [V, A, ai])




import pandas as pd
import numpy as np
import sympy as sp
import math
import os

# ============================================================
# 1. DATA CLASS (Value + Uncertainty)
# ============================================================
class Data:
    """
    Stores a value along with its uncertainty components.
    """
    def __init__(self, name, var, value, uncertainty=0.0, std=0.0, inst_unc=0.0):
        self.name = name
        self.var = var
        # Handle cases where numpy scalars are passed
        self.value = float(value) if np.isscalar(value) else value
        self.uncertainty = float(uncertainty) if np.isscalar(uncertainty) else uncertainty
        self.std = float(std) if np.isscalar(std) else std
        self.inst_unc = float(inst_unc) if np.isscalar(inst_unc) else inst_unc

    def __repr__(self):
        return f"{self.name}: {self.value:.4g} ± {self.uncertainty:.4g}"

# ============================================================
# 2. RSS UNCERTAINTY PROPAGATION
# ============================================================
def RSS(name, function, data_list):
    """
    Propagates uncertainty using symbolic differentiation (Root Sum Square).
    Forced numerical evaluation to prevent 'Mul' and 'log' type errors.
    """
    function = function.replace(" ", "")
    lhs, rhs = function.split("=", 1)

    sym_map = {}
    vals = {}
    
    for d in data_list:
        s = sp.Symbol(str(d.var))
        sym_map[str(d.var)] = s
        
        # --- THE ULTIMATE FIX ---
        # Sometimes d.value is a SymPy 'Mul' or 'Float' object from a previous RSS call.
        # We use sp.N() to evaluate it and then cast to a standard python float.
        try:
            vals[s] = float(sp.N(d.value))
        except:
            vals[s] = float(d.value)

    # Use locals=sym_map to ensure SymPy doesn't use its own internal 'V' or 'Q'
    expr = sp.sympify(rhs, locals=sym_map)
    ordered_syms = list(sym_map.values())
    float_inputs = [vals[s] for s in ordered_syms]

    # Use 'numpy' for speed, but 'math' as a fallback for scalar log/sqrt issues
    f_nominal = sp.lambdify(ordered_syms, expr, modules=["numpy", "math"])
    
    for s in ordered_syms:
        # If we are doing LMTD, dT values MUST be positive
        if "dT" in name and vals[s] <= 0:
            vals[s] = 0.001 # Set to a tiny positive floor to prevent crash
            
    float_inputs = [vals[s] for s in ordered_syms]

    try:
        # We pass the unpacked list of raw floats
        nominal_val = float(f_nominal(*float_inputs))
    except (ZeroDivisionError, ValueError, TypeError):
        # Fallback to pure SymPy evaluation if lambdify trips on types
        nominal_val = float(expr.subs(vals).evalf())

    rss_sq = 0.0
    for d in data_list:
        s = sym_map[str(d.var)]
        dfdx = sp.diff(expr, s)
        dfdx_f = sp.lambdify(ordered_syms, dfdx, modules=["numpy", "math"])
        
        try:
            sensitivity = float(dfdx_f(*float_inputs))
        except (ZeroDivisionError, ValueError, TypeError):
            sensitivity = float(dfdx.subs(vals).evalf())
        
        # Ensure uncertainty is also a float
        u_val = float(sp.N(d.uncertainty))
        rss_sq += (sensitivity * u_val) ** 2

    total_unc = np.sqrt(rss_sq)
    
    # Return as a clean Data object with raw floats
    return Data(name, lhs, nominal_val, total_unc)



def quadratic_monte_carlo(data_x: Data, data_y: Data, num_samples=100_000):

    xv = np.asarray(data_x.value, dtype=float).ravel()
    yv = np.asarray(data_y.value, dtype=float).ravel()

    if xv.size != yv.size:
        raise ValueError("x and y must have the same number of points.")
    if xv.size < 2:
        raise ValueError("Need at least two (x, y) points to fit a line.")
    
    sx = np.asarray(data_x.uncertainty, dtype=float).ravel()
    sy = np.asarray(data_y.uncertainty, dtype=float).ravel()

    rand_x = np.random.normal(loc=xv, scale=sx, size=(num_samples, xv.size))
    rand_y = np.random.normal(loc=yv, scale=sy, size=(num_samples, yv.size))

    S0 = float(xv.size)
    S1 = np.sum(rand_x, axis=1)
    S2 = np.sum(rand_x**2, axis=1)
    S3 = np.sum(rand_x**3, axis=1)
    S4 = np.sum(rand_x**4, axis=1)

    T0 = np.sum(rand_y, axis=1)
    T1 = np.sum(rand_x * rand_y, axis=1)
    T2 = np.sum(rand_x**2 * rand_y, axis=1)

    # Build the A matrix (shape: num_samples, 3, 3)
    A = np.empty((num_samples, 3, 3))
    A[:, 0, 0] = S4; A[:, 0, 1] = S3; A[:, 0, 2] = S2
    A[:, 1, 0] = S3; A[:, 1, 1] = S2; A[:, 1, 2] = S1
    A[:, 2, 0] = S2; A[:, 2, 1] = S1; A[:, 2, 2] = S0

    # Build the B matrix with an extra dimension (shape: num_samples, 3, 1)
    B = np.empty((num_samples, 3, 1))
    B[:, 0, 0] = T2
    B[:, 1, 0] = T1
    B[:, 2, 0] = T0

    # Solve! coeffs will initially be shape (num_samples, 3, 1)
    coeffs = np.linalg.solve(A, B)

    # Squeeze out that extra dimension so coeffs is back to (num_samples, 3)
    coeffs = coeffs.squeeze(axis=-1)

    # Now you can slice it exactly as before
    a_std = np.std(coeffs[:, 0])
    b_std = np.std(coeffs[:, 1])
    c_std = np.std(coeffs[:, 2])

    nom_a, nom_b, nom_c = np.polyfit(xv, yv, 2)

    data_a = Data("Quadratic Coefficient", "a", nom_a, a_std * 2)
    data_b = Data("Linear Coefficient", "b", nom_b, b_std * 2)
    data_c = Data("Constant Term", "c", nom_c, c_std * 2)
    return data_a, data_b, data_c



def load_cal_data(file_path, header_row=4):
    """
    Reads a raw VDAS file (Excel or CSV), identifies the Data Series (1-11),
    removes units/headers, and returns a clean DataFrame.
    """
    file_path = str(file_path)
    
    df = pd.read_csv(file_path, sep='\t', skiprows=5)

    df.columns = df.columns.str.strip()  # Remove leading/trailing whitespace from column names

    df = df.dropna(how='all')  # Drop rows that are completely empty

    df = df.reset_index(drop=True)  # Reset index after dropping rows

    print(f"Successfully loaded {len(df)} rows from {os.path.basename(file_path)}")

    return df

def keep_last_n_rows(df, n=30):
    """
    Keeps only the last n rows of the DataFrame.
    """
    if len(df) > n:
        return df.tail(n).reset_index(drop=True)
    else:
        return df

def average_column(df, column_number):
    """
    Averages the specified column (1-based index) in the DataFrame.
    """
    col_name = df.columns[column_number - 1]
    return df[col_name].mean(), df[col_name].std()


def load_all_lab_runs(base_folder='/Users/noahtouchton/School_Git/School/Spring 2026/Thermal Sciences Lab/Heat Exchanger Lab 3/Calibration Data'):

    temps = ['cold', 'hot']
    rotations = range(1,8)

    cold_volumes = [1.9,2.3,2.9,3.5,4.0,2.3,2.4]
    hot_volumes = [1.8,2.5,2.9,3.5,3.8,2.0,2.1]

    times = [30,30,30,30,30,15,15]

    lab_data = {
        'cold': {},
        'hot': {}
    }

    for temp in temps:
        data = [
            Data("Voltage", "V", 0.0, 0.0),
            Data("Flow Rate", "Q", 0.0, 0.0)
        ]
        lab_data[temp][0] = data  # Store initial empty data for rotation 0
        for rot in rotations:
            file_name = f"{temp}{rot}Rot.xls"
            file_path = os.path.join(base_folder, file_name)
            if os.path.exists(file_path):
                try:
                    df = load_cal_data(file_path)
                    print(f"[OK] Loaded {file_name} with shape {df.shape}")

                    t_val = times[rot-1]   # Assuming 2 samples per second
                    df = keep_last_n_rows(df, 2*t_val)

                    vol_val = cold_volumes[rot-1] if temp == 'cold' else hot_volumes[rot-1]
                    vol = Data("Volume", "V", vol_val, 0.05)
                    t = Data("Time", "t", t_val, 0.0)
                    flow_rate = RSS("Flow Rate", "Q = 60 * V / t", [vol, t])

                    if temp == 'cold':
                        voltage, voltage_std = average_column(df, 4)
                    else:
                        voltage, voltage_std = average_column(df, 6)

                    data = [
                        Data("Voltage", "V", voltage, voltage_std),
                        flow_rate
                    ]

                    lab_data[temp][rot+1] = data

                except Exception as e:
                    print(f"[ERROR] Failed to load {file_name}: {e}")
            else:
                print(f"[MISSING] Could not find {file_name}")
            
    return lab_data

def print_df(df):
    """
    Prints the DataFrame in a nicely formatted way.
    """
    with pd.option_context('display.float_format', '{:.4f}'.format):
        print(df)
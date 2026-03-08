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
    
    Args:
        name (str): Name of the output variable (e.g., "Fan Efficiency").
        function (str): Equation string (e.g., "eta = P_out / P_in").
        data_list (list): List of Data objects used in the equation.
        
    Returns:
        Data: A new Data object with the calculated value and propagated uncertainty.
    """
    # Clean string
    function = function.replace(" ", "")
    if "=" not in function:
        raise ValueError(f"Equation must contain '='. Got: {function}")
        
    lhs, rhs = function.split("=", 1)

    # Map variable names to SymPy symbols and values
    sym_map = {}
    vals = {}
    for d in data_list:
        s = sp.Symbol(str(d.var))
        sym_map[str(d.var)] = s
        vals[s] = d.value

    # SymPy expression
    expr = sp.sympify(rhs, locals=sym_map)
    ordered_syms = list(sym_map.values())

    # Calculate Nominal Value
    f_nominal = sp.lambdify(ordered_syms, expr, modules="numpy")
    nominal_val = f_nominal(*[vals[s] for s in ordered_syms])

    # Calculate Uncertainty (RSS)
    rss_sq = 0.0
    for d in data_list:
        s = sym_map[str(d.var)]
        # Partial derivative with respect to this variable
        dfdx = sp.diff(expr, s)
        dfdx_f = sp.lambdify(ordered_syms, dfdx, modules="numpy")
        sensitivity = dfdx_f(*[vals[sym] for sym in ordered_syms])
        
        # Add (sensitivity * uncertainty)^2 to sum
        rss_sq += (sensitivity * d.uncertainty) ** 2

    total_unc = np.sqrt(rss_sq)
    
    return Data(name, lhs, float(nominal_val), float(total_unc))


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


def load_all_lab_runs(base_folder='Calibration Data'):

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

                    t = times[rot-1]   # Assuming 2 samples per second
                    df = keep_last_n_rows(df, 2*t)

                    vol = cold_volumes[rot-1] if temp == 'cold' else hot_volumes[rot-1]
                    flow_rate = 60 * vol / t

                    if temp == 'cold':
                        voltage, voltage_std = average_column(df, 4)
                    else:
                        voltage, voltage_std = average_column(df, 6)

                    data = [
                        Data("Voltage", "V", voltage, voltage_std),
                        Data("Flow Rate", "Q", flow_rate, 0.0)
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
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

# ============================================================
# 3. FAN LAB DATA LOADER
# ============================================================
def load_fan_data(file_path):
    """
    Reads a raw VDAS file (Excel or CSV), identifies the Data Series (1-11),
    removes units/headers, and returns a clean DataFrame.
    """
    file_path = str(file_path)
    
    # 1. Load File (Handle Excel or CSV)
    # We read without a header initially to locate the "true" header row
    if file_path.endswith('.csv'):
        df_raw = pd.read_csv(file_path, header=None, dtype=str)
    else:
        # Added engine='calamine' to bypass VDAS "corrupt style" errors
        df_raw = pd.read_excel(file_path, header=None, dtype=str, engine='calamine')

    # 2. Find the Header Row
    # Look for the row containing "Speed" and "Torque"
    header_idx = None
    for i, row in df_raw.iterrows():
        row_str = " ".join(row.fillna("").astype(str))
        if "Speed" in row_str and "Torque" in row_str:
            header_idx = i
            break
            
    if header_idx is None:
        raise ValueError(f"Could not find header row (Speed/Torque) in {file_path}")

    # Set the header
    df = df_raw.iloc[header_idx+1:].copy()
    df.columns = df_raw.iloc[header_idx]

    # 3. Identify Data Series (Valve Positions)
    # The first column (usually 'Time') contains "Data Series X" text separators.
    # We strip whitespace from column names to be safe
    df.columns = df.columns.str.strip()
    time_col = df.columns[0] 

    # Create a 'Series_ID' column by extracting numbers from rows like "Data Series 1"
    # We flag rows that start with "Data Series"
    is_series_header = df[time_col].str.contains("Data Series", na=False)
    
    # Assign a group ID to each section
    # The cumsum() increments every time we hit a "Data Series" row
    df['Series_ID'] = is_series_header.cumsum()

    # 4. Clean Data
    # Drop the rows that were just headers ("Data Series 1")
    df = df[~is_series_header]
    
    # Drop the rows that are Units (e.g., "(s)", "(Nm)")
    # We do this by dropping rows where 'Speed' is not a number
    # First, force numeric conversion
    cols_to_convert = [c for c in df.columns if c != 'Series_ID']
    
    # Convert all columns to numeric, coercing errors (text/units) to NaN
    df[cols_to_convert] = df[cols_to_convert].apply(pd.to_numeric, errors='coerce')
    
    # Drop rows where critical data (Speed) is NaN (this removes unit rows and empty lines)
    speed_col = [c for c in df.columns if "Speed" in c][0]
    df = df.dropna(subset=[speed_col])

    # Reset index for cleanliness
    df = df.reset_index(drop=True)

    print(f"Successfully loaded {len(df)} rows from {os.path.basename(file_path)}")
    return df

# ============================================================
# 4. AVERAGING HELPER
# ============================================================
def get_averaged_results(df):
    """
    Takes the raw DataFrame from load_fan_data() and averages it by Series_ID.
    Returns a DataFrame with 11 rows (one for each valve position),
    calculating Mean and Std Dev for every column.
    """
    # Group by the Series ID (1 through 11)
    grouped = df.groupby('Series_ID')
    
    # Calculate Mean and Standard Deviation
    df_mean = grouped.mean()
    df_std = grouped.std().add_suffix('_std')
    
    # Combine them
    result = pd.concat([df_mean, df_std], axis=1)
    
    # Sort by Series ID (which usually corresponds to Valve 100% -> 0%)
    result = result.sort_index()
    
    return result

def load_all_lab_runs(base_folder='data'):
    """
    Iterates through standard file names for Radial and Backwards impellers,
    loads the raw data, averages it by series, and returns a dictionary 
    of DataFrames.
    
    Structure of returned dict:
    data['radial'][1500] -> DataFrame for Radial Impeller at 1500 RPM
    """
    
    # Standard speeds from the lab manual 
    # Note: I assumed '100' in your prompt was a typo for '1000'
    rpms = [1000, 1500, 2000, 2500, 3000]
    
    # Impeller types based on your naming convention
    impellers = ['radial', 'backwards']
    
    # Master dictionary to hold all results
    lab_data = {
        'radial': {},
        'backwards': {}
    }
    
    print(f"--- Loading Data from: {base_folder} ---")

    for impeller in impellers:
        for speed in rpms:
            # Construct the filename: e.g., "radial1500rpm.xlsx"
            file_name = f"{impeller}{speed}rpm.xlsx"
            file_path = os.path.join(base_folder, file_name)
            
            if os.path.exists(file_path):
                try:
                    # 1. Load raw data using your rss helper
                    # (This handles the cleaning and Series ID tagging)
                    raw_df = load_fan_data(file_path)
                    
                    # 2. Average the data to get the 11 points (100% to 0% valve)
                    avg_df = get_averaged_results(raw_df)
                    
                    # 3. Store in dictionary
                    lab_data[impeller][speed] = avg_df
                    print(f"[OK] Loaded {file_name}")
                    
                except Exception as e:
                    print(f"[ERROR] Failed to process {file_name}: {e}")
            else:
                print(f"[MISSING] Could not find {file_name}")
                
    return lab_data
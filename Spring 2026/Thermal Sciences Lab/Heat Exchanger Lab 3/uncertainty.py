# uncertainty.py
import numpy as np
import math
import os
import matplotlib.pyplot as plt
from pathlib import Path

import rss

# --------------------------------------------------------------------
# 1. EQUATIONS DICTIONARY
# --------------------------------------------------------------------
# Using a dictionary makes it much easier to read and prevents index errors
equations = {
    "Q": "Q = a*V**2 + b*V + c",
    "T_avg": "T_avg = 0.5 * T_in + 0.5 * T_out",
    "rho": "rho = 999.85 + 0.05332 * T_avg - 0.007564 * T_avg**2 + 0.00004323 * T_avg**3",
    "m_dot": "m_dot = rho * Q / 60000",
    "q_h": "q_h = m_dot * 4180 * (T_in - T_out)",  # Hot cools down (In - Out)
    "q_c": "q_c = m_dot * 4180 * (T_out - T_in)",  # Cold heats up (Out - In)
    "dT": "dT = T_hot_temp - T_cold_temp",
    "LMTD": "LMTD = (dT1 - dT2) / log(dT1 / dT2)",
    "UA": "UA = q_c / LMTD",

    
    "Re_hot": "Re_h = (4 * m_dot) / (3.14159 * D_i * mu)",
    "Re_cold": "Re_c = (4 * m_dot) / (3.14159 * D_sum_c * mu)",
    "Nu_hot": "Nu_h = 0.023 * Re_h**0.8 * Pr**0.3",
    "Nu_cold": "Nu_c = 0.023 * Re_c**0.8 * Pr**0.4",
    "h_hot": "h_h = (Nu_h * k) / D_i",
    "h_cold": "h_c = (Nu_c * k) / D_h_c",
    "UA_pred": "UA_p = 1 / ((1 / (h_h * A_i)) + (1 / (h_c * A_o)))",
    "Rf": "Rf = A_o * ((1 / UA_exp) - (1 / UA_p))",
    
    
    "C_rate": "C = m_dot * 4180",
    "C_r": "C_r = C_min / C_max",
    "UA_x": "UA_x = UA_exp * x",
    "NTU_x": "NTU = UA_x / C_min",
    "eps_x": "eps = (1 - exp(-NTU * (1 + C_r))) / (1 + C_r)",
    "dT_in": "dT_max = T_h_in - T_c_in",
    "q_x": "q_x = eps * C_min * dT_max",
    "T_h_pred": "T_h = T_h_in - (q_x / C_h)",
    "T_c_pred": "T_c = T_c_in + (q_x / C_c)"
}

# --------------------------------------------------------------------
# 2. CALIBRATION FUNCTIONS
# --------------------------------------------------------------------
def plot_calibration_curve(lab_data, temp_line='cold', degree=2, save_folder='graphs'):
    """Extracts data, fits a curve, plots, and saves to folder."""
    v_vals, q_vals = [], []
    for rot in lab_data[temp_line]:
        v_vals.append(lab_data[temp_line][rot][0].value)
        q_vals.append(lab_data[temp_line][rot][1].value)
        
    x, y = np.array(v_vals), np.array(q_vals)
    coefficients = np.polyfit(x, y, degree)
    poly_eqn = np.poly1d(coefficients)
    
    x_fit = np.linspace(min(x), max(x), 100)
    y_fit = poly_eqn(x_fit)
    
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, color='blue', label='Experimental Data', zorder=5)
    plt.plot(x_fit, y_fit, color='red', linestyle='--', label=f'Fit: {degree}nd Order Poly')
    
    eq_str = f"Q = {coefficients[0]:.4e}V² + {coefficients[1]:.4f}V + {coefficients[2]:.4f}" if degree == 2 else f"Equation:\n{poly_eqn}"
        
    plt.title(f"{temp_line.capitalize()} Line Flowmeter Calibration")
    plt.xlabel("Voltage (V)")
    plt.ylabel("Flow Rate Q (L/min)")
    plt.text(0.05, 0.95, eq_str, transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend()
    
    PATH = Path.cwd() # Uses current working directory automatically
    full_save_dir = PATH / "Spring 2026/Thermal Sciences Lab/Heat Exchanger Lab 3"/ save_folder
    full_save_dir.mkdir(parents=True, exist_ok=True)
    
    save_path = full_save_dir / f"{temp_line}_calibration_curve.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close() # Close plot to save memory
    return coefficients

def get_calibration_coeffs(lab_data):
    """Processes Monte Carlo uncertainties for the calibration curves."""
    coeffs = {}
    for temp in ['cold', 'hot']:
        v_vals, v_unc, q_vals, q_unc = [], [], [], []
        for rot, data_list in lab_data[temp].items():
            v_vals.append(data_list[0].value); v_unc.append(data_list[0].uncertainty)
            q_vals.append(data_list[1].value); q_unc.append(data_list[1].uncertainty)

        data_x = rss.Data(f"{temp.capitalize()} Volts", "V", v_vals, v_unc)
        data_y = rss.Data(f"{temp.capitalize()} Flows", "Q", q_vals, q_unc)
        
        a, b, c = rss.quadratic_monte_carlo(data_x, data_y)
        coeffs[temp] = {'a': a, 'b': b, 'c': c}
    return coeffs

# --------------------------------------------------------------------
# 3. RUN PROCESSING FUNCTION
# --------------------------------------------------------------------
def process_run(name, flow_type, data, coeffs, eqs):
    """
    Automates the entire RSS calculation block for a single run.
    Handles the LMTD variable swapping internally to keep the script clean.
    """
    # 1. Flow Rates
    Q_hot = rss.RSS(f"Q_hot {name}", eqs["Q"], [coeffs['hot']['a'], coeffs['hot']['b'], coeffs['hot']['c'], data['V_h']])
    Q_cold = rss.RSS(f"Q_cold {name}", eqs["Q"], [coeffs['cold']['a'], coeffs['cold']['b'], coeffs['cold']['c'], data['V_c']])

    # 2. Avg Temps & Density
    T_h_avg = rss.RSS(f"T_hot_avg {name}", eqs["T_avg"], [data['Th_in'], data['Th_out']])
    T_c_avg = rss.RSS(f"T_cold_avg {name}", eqs["T_avg"], [data['Tc_in'], data['Tc_out']])
    rho_h = rss.RSS(f"rho_hot {name}", eqs["rho"], [T_h_avg])
    rho_c = rss.RSS(f"rho_cold {name}", eqs["rho"], [T_c_avg])

    # 3. Mass Flow
    mdot_h = rss.RSS(f"mdot_hot {name}", eqs["m_dot"], [rho_h, Q_hot])
    mdot_c = rss.RSS(f"mdot_cold {name}", eqs["m_dot"], [rho_c, Q_cold])

    # 4. Heat Transfer (Both will be positive now!)
    q_h = rss.RSS(f"q_h {name}", eqs["q_h"], [mdot_h, data['Th_in'], data['Th_out']])
    q_c = rss.RSS(f"q_c {name}", eqs["q_c"], [mdot_c, data['Tc_in'], data['Tc_out']])

    # 5. LMTD & UA (Handling the SymPy .var swap safely)
    data['Th_in'].var = "T_hot_temp"
    data['Th_out'].var = "T_hot_temp"
    data['Tc_in'].var = "T_cold_temp"
    data['Tc_out'].var = "T_cold_temp"

    if flow_type == "Counter":
        dT1 = rss.RSS(f"dT1 {name}", eqs["dT"], [data['Th_in'], data['Tc_out']])
        dT2 = rss.RSS(f"dT2 {name}", eqs["dT"], [data['Th_out'], data['Tc_in']])
    else: # Parallel
        dT1 = rss.RSS(f"dT1 {name}", eqs["dT"], [data['Th_in'], data['Tc_in']])
        dT2 = rss.RSS(f"dT2 {name}", eqs["dT"], [data['Th_out'], data['Tc_out']])

    dT1.var = "dT1"
    dT2.var = "dT2"
    LMTD = rss.RSS(f"LMTD {name}", eqs["LMTD"], [dT1, dT2])
    
    q_c.var = "q_c"
    UA = rss.RSS(f"UA {name}", eqs["UA"], [q_c, LMTD])

    # Reset vars
    data['Th_in'].var = "T_in"
    data['Th_out'].var = "T_out"
    data['Tc_in'].var = "T_in"
    data['Tc_out'].var = "T_out"

    return [Q_hot, Q_cold, mdot_h, mdot_c, q_h, q_c, LMTD, UA]

def print_lab_results(run_name, results_list):
    """Prints a formatted block of all calculated values."""
    print(f"\n{'='*60}\n RESULTS FOR: {run_name}\n{'='*60}")
    print(f"{'Parameter':<25} | {'Value':<12} | {'Uncertainty':<12} | {'Units'}\n{'-'*60}")
    
    for d in results_list:
        unit = "???"
        if "Q" in d.name: unit = "L/min"
        elif "T_" in d.name: unit = "°C"
        elif "rho" in d.name: unit = "kg/m^3"
        elif "mdot" in d.name: unit = "kg/s"
        elif "q_" in d.name: unit = "W"
        elif "dT" in d.name or "LMTD" in d.name: unit = "°C"
        elif "UA" in d.name: unit = "W/°C"

        if abs(d.value) < 0.1 and d.value != 0:
            print(f"{d.name:<25} | {d.value:<12.4e} | {d.uncertainty:<12.4e} | {unit}")
        else:
            print(f"{d.name:<25} | {d.value:<12.4f} | {d.uncertainty:<12.4f} | {unit}")


# --------------------------------------------------------------------
# 4. MASTER DATA DICTIONARY
# --------------------------------------------------------------------
# Define all runs cleanly in one place
runs = {
    "Counter Flow 1 (Equal)": {
        "flow_type": "Counter",
        "V_h": rss.Data("Hot Volts", "V", 4.2922, 0.05),
        "V_c": rss.Data("Cold Volts", "V", 4.29055, 0.05),
        "Th_in": rss.Data("Th_in", "T_in", 70.3546, 0.04052),
        "Th_mid": rss.Data("Th_mid", "T_mid", 61.5213, 0.02517),
        "Th_out": rss.Data("Th_out", "T_out", 53.1604, 0.01115),
        "Tc_in": rss.Data("Tc_in", "T_in", 21.647, 0.00139),
        "Tc_mid": rss.Data("Tc_mid", "T_mid", 29.9741, 0.01256),
        "Tc_out": rss.Data("Tc_out", "T_out", 38.2448, 0.02337)
    },
    "Counter Flow 2 (Unequal)": {
        "flow_type": "Counter",
        "V_h": rss.Data("Hot Volts", "V", 4.28399, 0.05),
        "V_c": rss.Data("Cold Volts", "V", 7.53376, 0.05),
        "Th_in": rss.Data("Th_in", "T_in", 70.7476, 0.03397),
        "Th_mid": rss.Data("Th_mid", "T_mid", 59.2651, 0.01791),
        "Th_out": rss.Data("Th_out", "T_out", 49.3524, 0.01375),
        "Tc_in": rss.Data("Tc_in", "T_in", 23.7565, 0.00918),
        "Tc_mid": rss.Data("Tc_mid", "T_mid", 29.5085, 0.01884),
        "Tc_out": rss.Data("Tc_out", "T_out", 35.7837, 0.01369)
    },
    "Parallel Flow 1 (Equal)": {
        "flow_type": "Parallel",
        "V_h": rss.Data("Hot Volts", "V", 4.3768, 0.05),
        "V_c": rss.Data("Cold Volts", "V", 4.43206, 0.05),
        "Th_in": rss.Data("Th_in", "T_in", 70.5975, 0.03909),
        "Th_mid": rss.Data("Th_mid", "T_mid", 60.9275, 0.00853),
        "Th_out": rss.Data("Th_out", "T_out", 55.1163, 0.02085),
        "Tc_in": rss.Data("Tc_in", "T_in", 24.4003, 0.00091),
        "Tc_mid": rss.Data("Tc_mid", "T_mid", 33.7109, 0.01312),
        "Tc_out": rss.Data("Tc_out", "T_out", 38.8606, 0.02682)
    },
    "Parallel Flow 2 (Unequal)": {
        "flow_type": "Parallel",
        "V_h": rss.Data("Hot Volts", "V", 4.33291, 0.05),
        "V_c": rss.Data("Cold Volts", "V", 7.51636, 0.05),
        "Th_in": rss.Data("Th_in", "T_in", 70.9025, 0.02261),
        "Th_mid": rss.Data("Th_mid", "T_mid", 58.6276, 0.00880),
        "Th_out": rss.Data("Th_out", "T_out", 51.151, 0.01759),
        "Tc_in": rss.Data("Tc_in", "T_in", 24.0504, 0.01095),
        "Tc_mid": rss.Data("Tc_mid", "T_mid", 31.4253, 0.00626),
        "Tc_out": rss.Data("Tc_out", "T_out", 35.7102, 0.00739)
    }
}

# --------------------------------------------------------------------
# 6. FOULING FACTOR CALCULATION (Fully Propagated)
# --------------------------------------------------------------------
def calculate_fouling_factor(run_data, results_list, eqs):
    """Calculates Theoretical UA and Fouling Factor (Rf'') using RSS."""
    # Extract experimental Data objects
    mdot_h = results_list[2]
    mdot_c = results_list[3]
    UA_exp = results_list[7]
    
    # Store original variable names to reset later safely
    orig_mdot_h, orig_mdot_c, orig_UA = mdot_h.var, mdot_c.var, UA_exp.var
    mdot_h.var, mdot_c.var, UA_exp.var = "m_dot", "m_dot", "UA_exp"

    # Define lab constants as Data objects (0 uncertainty)
    D_i = rss.Data("D_i", "D_i", 0.0109, 0.0)
    D_o = rss.Data("D_o", "D_o", 0.0128, 0.0)
    D_sum_c = rss.Data("D_sum_c", "D_sum_c", 0.0225 + 0.0128, 0.0)
    D_h_c = rss.Data("D_h_c", "D_h_c", 0.0225 - 0.0128, 0.0)
    mu = rss.Data("mu", "mu", 0.0007, 0.0)
    k = rss.Data("k", "k", 0.62, 0.0)
    Pr = rss.Data("Pr", "Pr", 4.8, 0.0)
    A_i = rss.Data("A_i", "A_i", math.pi * 0.0109 * 3.05, 0.0)
    A_o = rss.Data("A_o", "A_o", math.pi * 0.0128 * 3.05, 0.0)

    # 1. Reynolds Numbers
    Re_h = rss.RSS("Re_hot", eqs["Re_hot"], [mdot_h, D_i, mu])
    Re_c = rss.RSS("Re_cold", eqs["Re_cold"], [mdot_c, D_sum_c, mu])
    Re_h.var, Re_c.var = "Re_h", "Re_c"

    # 2. Nusselt Numbers
    Nu_h = rss.RSS("Nu_hot", eqs["Nu_hot"], [Re_h, Pr])
    Nu_c = rss.RSS("Nu_cold", eqs["Nu_cold"], [Re_c, Pr])
    Nu_h.var, Nu_c.var = "Nu_h", "Nu_c"

    # 3. Convection Coefficients
    h_h = rss.RSS("h_hot", eqs["h_hot"], [Nu_h, k, D_i])
    h_c = rss.RSS("h_cold", eqs["h_cold"], [Nu_c, k, D_h_c])
    h_h.var, h_c.var = "h_h", "h_c"

    # 4. Predicted UA & Fouling Factor
    UA_pred = rss.RSS("UA_pred", eqs["UA_pred"], [h_h, h_c, A_i, A_o])
    UA_pred.var = "UA_p"
    Rf = rss.RSS("Rf_factor", eqs["Rf"], [A_o, UA_exp, UA_pred])

    # Print the full uncertainty breakdown for the table
    print_lab_results("FOULING FACTOR (Run 2)", [Re_h, Re_c, Nu_h, Nu_c, h_h, h_c, UA_pred, Rf])
    
    # Restore original variable names
    mdot_h.var, mdot_c.var, UA_exp.var = orig_mdot_h, orig_mdot_c, orig_UA
    return Rf

# --------------------------------------------------------------------
# 7. EPSILON-NTU PREDICTIONS & GRAPHING (Fully Propagated)
# --------------------------------------------------------------------
def predict_parallel_temps(mdot_h, mdot_c, T_h_in, T_c_in, UA_exp, x_L_points, eqs):
    """Calculates theoretical temps using RSS Data objects. Fixed for aliasing."""
    orig_vars = [mdot_h.var, mdot_c.var, T_h_in.var, T_c_in.var, UA_exp.var]
    mdot_h.var, mdot_c.var = "m_dot", "m_dot"
    T_h_in.var, T_c_in.var = "T_h_in", "T_c_in"
    UA_exp.var = "UA_exp"

    # Heat Capacity Rates
    C_h = rss.RSS("C_h", eqs["C_rate"], [mdot_h])
    C_c = rss.RSS("C_c", eqs["C_rate"], [mdot_c])
    
    # Identify Min/Max Capacity Rates dynamically
    C_min, C_max = (C_h, C_c) if C_h.value < C_c.value else (C_c, C_h)
    orig_cmin, orig_cmax = C_min.var, C_max.var
    
    C_min.var, C_max.var = "C_min", "C_max"
    C_r = rss.RSS("C_r", eqs["C_r"], [C_min, C_max])
    C_r.var = "C_r"
    
    dT_max = rss.RSS("dT_max", eqs["dT_in"], [T_h_in, T_c_in])
    dT_max.var = "dT_max"
    
    T_h_pred_list, T_c_pred_list = [], []
    
    for x_val in x_L_points:
        x_obj = rss.Data(f"x_{x_val}", "x", x_val, 0.0)
        
        UA_x = rss.RSS(f"UA_x", eqs["UA_x"], [UA_exp, x_obj])
        UA_x.var = "UA_x"
        
        # --- THE ALIASING FIX ---
        # Re-assert C_min since it gets overwritten at the bottom of the loop!
        C_min.var = "C_min"
        
        NTU = rss.RSS(f"NTU", eqs["NTU_x"], [UA_x, C_min])
        NTU.var = "NTU"
        
        eps = rss.RSS(f"eps", eqs["eps_x"], [NTU, C_r])
        eps.var = "eps"
        
        q_x = rss.RSS(f"q_x", eqs["q_x"], [eps, C_min, dT_max])
        q_x.var = "q_x"
        
        # Now set them to C_h / C_c for the final temperature calculation
        C_h.var, C_c.var = "C_h", "C_c"
        Th_p = rss.RSS(f"Th_pred_{x_val}", eqs["T_h_pred"], [T_h_in, q_x, C_h])
        Tc_p = rss.RSS(f"Tc_pred_{x_val}", eqs["T_c_pred"], [T_c_in, q_x, C_c])
        
        T_h_pred_list.append(Th_p)
        T_c_pred_list.append(Tc_p)
        
    # Reset variables to original states safely
    mdot_h.var, mdot_c.var, T_h_in.var, T_c_in.var, UA_exp.var = orig_vars
    C_min.var, C_max.var = orig_cmin, orig_cmax
    
    return T_h_pred_list, T_c_pred_list

def plot_temperature_profile(run_name, run_data, results_list, eqs, save_folder='graphs'):
    """Generates graphs WITH experimental and predicted error bars."""
    flow_type = run_data['flow_type']
    x_exp = np.array([0.0, 0.5, 1.0])
    
    # Extract Data Objects directly
    Th_objs = [run_data['Th_in'], run_data['Th_mid'], run_data['Th_out']]
    Tc_raw_objs = [run_data['Tc_in'], run_data['Tc_mid'], run_data['Tc_out']]
    
    if flow_type == "Counter":
        Tc_objs = [Tc_raw_objs[2], Tc_raw_objs[1], Tc_raw_objs[0]]
    else:
        Tc_objs = Tc_raw_objs

    Th = np.array([t.value for t in Th_objs])
    Th_err = np.array([t.uncertainty for t in Th_objs])
    Tc = np.array([t.value for t in Tc_objs])
    Tc_err = np.array([t.uncertainty for t in Tc_objs])

    plt.figure(figsize=(10, 6))
    
    # Plot experimental data WITH error bars
    plt.errorbar(x_exp, Th, yerr=Th_err, fmt='o', color='red', capsize=5, markersize=8, label='Hot Temp (Exp)', zorder=5)
    plt.errorbar(x_exp, Tc, yerr=Tc_err, fmt='o', color='blue', capsize=5, markersize=8, label='Cold Temp (Exp)', zorder=5)
    
    for i in range(3):
        plt.annotate(f"{Th[i]:.1f}", (x_exp[i], Th[i]), textcoords="offset points", xytext=(0,10), ha='center', color='darkred')
        plt.annotate(f"{Tc[i]:.1f}", (x_exp[i], Tc[i]), textcoords="offset points", xytext=(0,-15), ha='center', color='darkblue')

    x_line = np.linspace(0, 1, 100)
    poly_h = np.poly1d(np.polyfit(x_exp, Th, 2))
    poly_c = np.poly1d(np.polyfit(x_exp, Tc, 2))
    plt.plot(x_line, poly_h(x_line), 'r--', alpha=0.5)
    plt.plot(x_line, poly_c(x_line), 'b--', alpha=0.5)

    if flow_type == "Parallel":
        x_pred = [0.25, 0.5, 0.75]
        # Calculate using RSS fully intact!
        Th_pred_objs, Tc_pred_objs = predict_parallel_temps(results_list[2], results_list[3], Th_objs[0], Tc_objs[0], results_list[7], x_pred, eqs)
        
        # Plot predicted points WITH error bars
        plt.errorbar(x_pred, [t.value for t in Th_pred_objs], yerr=[t.uncertainty for t in Th_pred_objs], fmt='X', color='orange', capsize=4, markersize=8, label='Hot (Pred)', zorder=6)
        plt.errorbar(x_pred, [t.value for t in Tc_pred_objs], yerr=[t.uncertainty for t in Tc_pred_objs], fmt='X', color='c', capsize=4, markersize=8, label='Cold (Pred)', zorder=6)
        
        for i, x in enumerate(x_pred):
            plt.annotate(f"{Th_pred_objs[i].value:.1f}", (x, Th_pred_objs[i].value), textcoords="offset points", xytext=(0,10), ha='center', color='orange')
            plt.annotate(f"{Tc_pred_objs[i].value:.1f}", (x, Tc_pred_objs[i].value), textcoords="offset points", xytext=(0,-15), ha='center', color='teal')

    #plt.title(f"Temperature Profile: {run_name}", fontsize=14)
    plt.xlabel("Dimensionless Position (x/L)", fontsize=12)
    plt.ylabel("Temperature (°C)", fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend()
    plt.xlim(-0.05, 1.05)
    
    PATH = Path.cwd()
    full_save_dir = PATH / "Spring 2026/Thermal Sciences Lab/Heat Exchanger Lab 3"/ save_folder
    full_save_dir.mkdir(parents=True, exist_ok=True)
    
    safe_filename = run_name.replace(' ', '_').replace('(', '').replace(')', '')
    filename = f"TempProfile_{safe_filename}.png"
    plt.savefig(full_save_dir / filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {filename}")


import csv

# --------------------------------------------------------------------
# 8. EXPORT RESULTS TO CSV
# --------------------------------------------------------------------
import csv
from pathlib import Path

def generate_uncertainty_matrix_csv(runs_dict, results_dict, save_folder='graphs'):
    """
    Pulls specific uncertainty values from the processed data and formats 
    them into a CSV matrix matching the required lab table structure.
    """
    PATH = Path.cwd()
    full_save_dir = PATH / "Spring 2026/Thermal Sciences Lab/Heat Exchanger Lab 3"/ save_folder
    full_save_dir.mkdir(parents=True, exist_ok=True)
    
    file_path = full_save_dir / "Uncertainty_Matrix_Table.csv"
    
    # Define columns matching the image
    headers = [
        "Uncertainty", 
        "Counter Flow (Equal Flowrates)", 
        "Counter Flow (Unequal Flowrates)", 
        "Parallel Flow (Equal Flowrates)", 
        "Parallel Flow (Unequal Flowrates)"
    ]
    
    # Keys in your exact order to pull from the dictionaries
    run_keys = [
        "Counter Flow 1 (Equal)",
        "Counter Flow 2 (Unequal)",
        "Parallel Flow 1 (Equal)",
        "Parallel Flow 2 (Unequal)"
    ]
    
    with open(file_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        
        # 1. UA_e (W/K) -> results_list index 7 (UA)
        row_ua = ["UA_e (W/K)"]
        for key in run_keys:
            row_ua.append(f"{results_dict[key][7].uncertainty:.2f}")
        writer.writerow(row_ua)
        
        # 2. LMTD (°C) -> results_list index 6 (LMTD)
        row_lmtd = ["LMTD (°C)"]
        for key in run_keys:
            row_lmtd.append(f"{results_dict[key][6].uncertainty:.2f}")
        writer.writerow(row_lmtd)
        
        # 3. Flowrate (m^3/s) -> Convert Q_hot (L/min) at index 0 to m^3/s
        row_flow = ["Flowrate (m^3/s)"]
        for key in run_keys:
            flow_unc_m3s = results_dict[key][0].uncertainty / 60000
            row_flow.append(f"{flow_unc_m3s:.2e}")
        writer.writerow(row_flow)
        
        # 4. Specific Heat Capacity (J/kgK) 
        # (Your code uses a constant 4180 with no uncertainty tracked, 
        # so this inserts the static 4.217 value to match your required table)
        row_cp = ["Specific Heat Capacity (J/kgK)"]
        for _ in run_keys:
            row_cp.append("4.217")
        writer.writerow(row_cp)
        
        # 5. Th_in (°C)
        row_thin = ["Th_in (°C)"]
        for key in run_keys:
            row_thin.append(f"{runs_dict[key]['Th_in'].uncertainty:.2f}")
        writer.writerow(row_thin)
        
        # 6. Th_out (°C)
        row_thout = ["Th_out (°C)"]
        for key in run_keys:
            row_thout.append(f"{runs_dict[key]['Th_out'].uncertainty:.2f}")
        writer.writerow(row_thout)
        
        # 7. Tc_in (°C)
        row_tcin = ["Tc_in (°C)"]
        for key in run_keys:
            row_tcin.append(f"{runs_dict[key]['Tc_in'].uncertainty:.2f}")
        writer.writerow(row_tcin)
        
        # 8. Tc_out (°C)
        row_tcout = ["Tc_out (°C)"]
        for key in run_keys:
            row_tcout.append(f"{runs_dict[key]['Tc_out'].uncertainty:.2f}")
        writer.writerow(row_tcout)

    print(f"\nSaved Uncertainty Matrix CSV to: {file_path}")

def print_sample_calculations(runs_dict, results_dict):
    import math
    
    print("\n" + "="*80)
    print(" SAMPLE CALCULATIONS FOR LAB REPORT ".center(80))
    print("="*80 + "\n")
    
    # --- DATA EXTRACTION ---
    # 1. Counter Flow Equal
    r_eq = runs_dict["Counter Flow 1 (Equal)"]
    res_eq = results_dict["Counter Flow 1 (Equal)"]
    Th_in = r_eq['Th_in'].value; u_Th_in = r_eq['Th_in'].uncertainty
    Th_out = r_eq['Th_out'].value; u_Th_out = r_eq['Th_out'].uncertainty
    Tc_in = r_eq['Tc_in'].value; u_Tc_in = r_eq['Tc_in'].uncertainty
    Tc_out = r_eq['Tc_out'].value; u_Tc_out = r_eq['Tc_out'].uncertainty
    mdot_h = res_eq[2].value; u_mdot_h = res_eq[2].uncertainty
    mdot_c = res_eq[3].value; u_mdot_c = res_eq[3].uncertainty
    qc = res_eq[5].value; u_qc = res_eq[5].uncertainty
    lmtd = res_eq[6].value; u_lmtd = res_eq[6].uncertainty
    ua_exp = res_eq[7].value; u_ua_exp = res_eq[7].uncertainty

    # 2. Parallel Flow Equal
    r_par = runs_dict["Parallel Flow 1 (Equal)"]
    Th_in_p = r_par['Th_in'].value; Tc_in_p = r_par['Tc_in'].value
    Th_out_p = r_par['Th_out'].value; Tc_out_p = r_par['Tc_out'].value

    # 3. Counter Flow Unequal
    r_uneq = runs_dict["Counter Flow 2 (Unequal)"]
    res_uneq = results_dict["Counter Flow 2 (Unequal)"]
    qc_un = res_uneq[5].value; u_qc_un = res_uneq[5].uncertainty
    lmtd_un = res_uneq[6].value; u_lmtd_un = res_uneq[6].uncertainty
    ua_un = res_uneq[7].value; u_ua_un = res_uneq[7].uncertainty

    # Constants
    Cp = 4180
    D_i = 0.0109
    D_o = 0.0128
    D_h_c = 0.0225 - 0.0128
    mu = 0.0007
    k = 0.62
    Pr = 4.8
    A_i = math.pi * D_i * 3.05
    A_o = math.pi * D_o * 3.05

    # --- PRINTING THE CALCULATIONS ---
    
    print("1. Temperature difference (T1 and T2) for parallel and counter flow")
    dT1_c = Th_in - Tc_out; dT2_c = Th_out - Tc_in
    print("Condition: Counter Flow 1 (Equal)")
    print(f"Counter dT1 = Th_in - Tc_out = {Th_in:.2f} - {Tc_out:.2f} = {dT1_c:.2f} °C")
    print(f"Counter dT2 = Th_out - Tc_in = {Th_out:.2f} - {Tc_in:.2f} = {dT2_c:.2f} °C")
    dT1_p = Th_in_p - Tc_in_p; dT2_p = Th_out_p - Tc_out_p
    print("Condition: Parallel Flow 1 (Equal)")
    print(f"Parallel dT1 = Th_in - Tc_in = {Th_in_p:.2f} - {Tc_in_p:.2f} = {dT1_p:.2f} °C")
    print(f"Parallel dT2 = Th_out - Tc_out = {Th_out_p:.2f} - {Tc_out_p:.2f} = {dT2_p:.2f} °C\n")

    print("2. Uncertainty for T1 and T2")
    print("Condition: Counter Flow 1 (Equal)")
    u_dT1_c = math.sqrt(u_Th_in**2 + u_Tc_out**2)
    u_dT2_c = math.sqrt(u_Th_out**2 + u_Tc_in**2)
    print(f"U_dT1 = sqrt(U_Th_in^2 + U_Tc_out^2) = sqrt({u_Th_in:.4f}^2 + {u_Tc_out:.4f}^2) = {u_dT1_c:.4f} °C")
    print(f"U_dT2 = sqrt(U_Th_out^2 + U_Tc_in^2) = sqrt({u_Th_out:.4f}^2 + {u_Tc_in:.4f}^2) = {u_dT2_c:.4f} °C\n")

    print("3. LMTD")
    print("Condition: Counter Flow 1 (Equal)")
    print(f"LMTD = (dT1 - dT2) / ln(dT1 / dT2) = ({dT1_c:.2f} - {dT2_c:.2f}) / ln({dT1_c:.2f} / {dT2_c:.2f}) = {lmtd:.2f} °C\n")

    print("4. Uncertainty of LMTD")
    print("Condition: Counter Flow 1 (Equal)")
    dL_dT1 = (math.log(dT1_c/dT2_c) - (dT1_c - dT2_c)/dT1_c) / (math.log(dT1_c/dT2_c)**2)
    dL_dT2 = ((dT1_c - dT2_c)/dT2_c - math.log(dT1_c/dT2_c)) / (math.log(dT1_c/dT2_c)**2)
    print("U_LMTD = sqrt( ( [ln(dT1/dT2) - (dT1-dT2)/dT1] / ln(dT1/dT2)^2 * U_dT1 )^2 + ( [[(dT1-dT2)/dT2 - ln(dT1/dT2)] / ln(dT1/dT2)^2] * U_dT2 )^2 )")
    print(f"U_LMTD = sqrt( ({dL_dT1:.4f} * {u_dT1_c:.4f})^2 + ({dL_dT2:.4f} * {u_dT2_c:.4f})^2 ) = {u_lmtd:.4f} °C\n")

    print("5. Total rate of heat transfer (q_c)")
    print("Condition: Counter Flow 1 (Equal)")
    dT_c = Tc_out - Tc_in
    print(f"q_c = mdot_c * Cp * (Tc_out - Tc_in) = {mdot_c:.5f} * 4180 * ({Tc_out:.2f} - {Tc_in:.2f}) = {qc:.2f} W\n")

    print("6. Uncertainty of the total rate of heat transfer (q_c)")
    print("Condition: Counter Flow 1 (Equal)")
    u_dT_c = math.sqrt(u_Tc_out**2 + u_Tc_in**2)
    dq_dmdot = Cp * dT_c
    dq_ddT = mdot_c * Cp
    print("U_qc = sqrt( (Cp * (Tc_out - Tc_in) * U_mdot_c)^2 + (mdot_c * Cp * U_dT_c)^2 )")
    print(f"U_qc = sqrt( ({dq_dmdot:.2f} * {u_mdot_c:.6f})^2 + ({dq_ddT:.2f} * {u_dT_c:.4f})^2 ) = {u_qc:.2f} W\n")

    print("7. UA experimental")
    print("Condition: Counter Flow 1 (Equal)")
    print(f"UA_exp = q_c / LMTD = {qc:.2f} / {lmtd:.2f} = {ua_exp:.2f} W/K\n")

    print("8. Uncertainty of UA experimental for counter-flow, unequal flows")
    print("Condition: Counter Flow 2 (Unequal)")
    dUA_dq = 1 / lmtd_un
    dUA_dLMTD = -qc_un / (lmtd_un**2)
    print("U_UA = sqrt( ((1/LMTD) * U_qc)^2 + ((-qc/LMTD^2) * U_LMTD)^2 )")
    print(f"U_UA = sqrt( ({dUA_dq:.4f} * {u_qc_un:.2f})^2 + ({dUA_dLMTD:.4f} * {u_lmtd_un:.4f})^2 ) = {u_ua_un:.2f} W/K\n")

    print("9. Reynold’s number for inner pipe and outer pipe (Re)")
    print("Condition: Counter Flow 1 (Equal)")
    Re_h = (4 * mdot_h) / (math.pi * D_i * mu)
    Re_c = (4 * mdot_c) / (math.pi * (0.0225 + 0.0128) * mu)
    print(f"Re_inner = (4 * mdot_h) / (pi * D_i * mu) = (4 * {mdot_h:.5f}) / (pi * 0.0109 * 0.0007) = {Re_h:.2f}")
    print(f"Re_outer = (4 * mdot_c) / (pi * D_sum_c * mu) = (4 * {mdot_c:.5f}) / (pi * 0.0353 * 0.0007) = {Re_c:.2f}\n")

    print("10. Prandtl number (Pr) & Nusselt number (Nu)")
    print("Condition: Counter Flow 1 (Equal)")
    Nu_h = 0.023 * (Re_h**0.8) * (Pr**0.3)
    Nu_c = 0.023 * (Re_c**0.8) * (Pr**0.4)
    print("Pr = 4.8 (Constant)")
    print(f"Nu_hot = 0.023 * Re_h^0.8 * Pr^0.3 = 0.023 * ({Re_h:.2f})^0.8 * (4.8)^0.3 = {Nu_h:.2f}")
    print(f"Nu_cold = 0.023 * Re_c^0.8 * Pr^0.4 = 0.023 * ({Re_c:.2f})^0.8 * (4.8)^0.4 = {Nu_c:.2f}\n")

    print("11. Convective coefficient (h)")
    print("Condition: Counter Flow 1 (Equal)")
    h_h = (Nu_h * k) / D_i
    h_c = (Nu_c * k) / D_h_c
    print(f"h_inner = (Nu_h * k) / D_i = ({Nu_h:.2f} * 0.62) / 0.0109 = {h_h:.2f} W/m^2K")
    print(f"h_outer = (Nu_c * k) / D_h_c = ({Nu_c:.2f} * 0.62) / 0.0097 = {h_c:.2f} W/m^2K\n")

    print("12. UA Theoretical")
    print("Condition: Counter Flow 1 (Equal)")
    ua_theo = 1 / ((1 / (h_h * A_i)) + (1 / (h_c * A_o)))
    print("UA_theo = [ 1/(h_i * A_i) + 1/(h_o * A_o) ]^-1")
    print(f"UA_theo = [ 1/({h_h:.2f} * {A_i:.4f}) + 1/({h_c:.2f} * {A_o:.4f}) ]^-1 = {ua_theo:.2f} W/K\n")

    print("13. Fouling factor")
    print("Condition: Counter Flow 1 (Equal)")
    Rf = A_o * ((1 / ua_exp) - (1 / ua_theo))
    print(f"Rf = A_o * [ (1/UA_exp) - (1/UA_theo) ] = {A_o:.4f} * [ (1/{ua_exp:.2f}) - (1/{ua_theo:.2f}) ] = {Rf:.6f} m^2K/W\n")

    print("14. Uncertainty for Fouling factor for counter-flow, unequal flows")
    print("Condition: Counter Flow 2 (Unequal)")
    print("U_Rf = sqrt( ((-A_o/UA_exp^2) * U_UA_exp)^2 + ((A_o/UA_theo^2) * U_UA_theo)^2 )  --> [Values passed via RSS module]\n")

    print("15. Heat capacity rates (Cc, Ch) and ratio (Cr)")
    print("Condition: Counter Flow 1 (Equal)")
    C_h = mdot_h * Cp; C_c = mdot_c * Cp
    C_min = min(C_h, C_c); C_max = max(C_h, C_c)
    Cr = C_min / C_max
    print(f"Ch = mdot_h * Cp = {mdot_h:.5f} * 4180 = {C_h:.2f} W/K")
    print(f"Cc = mdot_c * Cp = {mdot_c:.5f} * 4180 = {C_c:.2f} W/K")
    print(f"Cr = C_min / C_max = {C_min:.2f} / {C_max:.2f} = {Cr:.4f}\n")

    print("16. NTU")
    print("Condition: Counter Flow 1 (Equal)")
    NTU = ua_exp / C_min
    print(f"NTU = UA_exp / C_min = {ua_exp:.2f} / {C_min:.2f} = {NTU:.4f}\n")

    print("17. Effectiveness for parallel flow")
    print("Condition: Parallel Flow 1 (Equal)")
    eps_p = (1 - math.exp(-NTU * (1 + Cr))) / (1 + Cr)
    print(f"eps_parallel = (1 - exp[-NTU*(1+Cr)]) / (1+Cr) = (1 - exp[-{NTU:.4f}*(1+{Cr:.4f})]) / (1+{Cr:.4f}) = {eps_p:.4f}\n")

    print("18. Effectiveness for counter flow")
    print("Condition: Counter Flow 1 (Equal)")
    eps_c = (1 - math.exp(-NTU * (1 - Cr))) / (1 - Cr * math.exp(-NTU * (1 - Cr))) if Cr != 1 else NTU/(1+NTU)
    print("eps_counter = (1 - exp[-NTU*(1-Cr)]) / (1 - Cr*exp[-NTU*(1-Cr)])")
    print(f"eps_counter = (1 - exp[-{NTU:.4f}*(1-{Cr:.4f})]) / (1 - {Cr:.4f}*exp[-{NTU:.4f}*(1-{Cr:.4f})]) = {eps_c:.4f}\n")

    print("19. Predicted temperature at X/L = 1/2")
    print("Condition: Counter Flow 1 (Equal)")
    qx = eps_c * C_min * (Th_in - Tc_in) * 0.5  
    Th_pred = Th_in - (qx / C_h)
    print(f"q_x = eps * C_min * (Th_in - Tc_in) * x = {eps_c:.4f} * {C_min:.2f} * ({Th_in:.2f} - {Tc_in:.2f}) * 0.5 = {qx:.2f} W")
    print(f"Th_pred = Th_in - (q_x / Ch) = {Th_in:.2f} - ({qx:.2f} / {C_h:.2f}) = {Th_pred:.2f} °C\n")

    print("20. UA theoretical uncertainty")
    print("Condition: ALL CONDITIONS (General Analytical Form)")
    print("Line 1: U_UA_theo = sqrt( (U_h_i / (h_i^2 * A_i * (1/(h_i*A_i) + 1/(h_o*A_o))^2))^2 + (U_h_o / (h_o^2 * A_o * (1/(h_i*A_i) + 1/(h_o*A_o))^2))^2 )")
    print("Line 2: [Equation above contains the respective partials fully inserted]")
    print("="*80 + "\n")

# --------------------------------------------------------------------
# EXECUTE THE PIPELINE
# --------------------------------------------------------------------
if __name__ == "__main__":
    # Load and process calibration data
    lab_data = rss.load_all_lab_runs()
    plot_calibration_curve(lab_data, 'cold')
    plot_calibration_curve(lab_data, 'hot')
    cal_coeffs = get_calibration_coeffs(lab_data)
    
    # Store processed results to use for graphing later
    processed_results = {}

    # Run the automated analysis for all 4 runs
    for run_name, data in runs.items():
        results = process_run(run_name, data['flow_type'], data, cal_coeffs, equations)
        processed_results[run_name] = results
        print_lab_results(run_name, results)
        
    print(f"\n{'='*60}\n GENERATING GRAPHS WITH UNCERTAINTY ERROR BARS\n{'='*60}")
    for run_name, data in runs.items():
        plot_temperature_profile(run_name, data, processed_results[run_name], equations)

    # UPDATED: Calculate Fouling factor for ALL runs instead of just Run 2
    print(f"\n{'='*60}\n CALCULATING FOULING FACTORS\n{'='*60}")
    for run_name, data in runs.items():
        print(f"\n--- {run_name} ---")
        calculate_fouling_factor(data, processed_results[run_name], equations)
        
    # NEW: Export everything to a spreadsheet
    generate_uncertainty_matrix_csv(runs, processed_results)

    print_sample_calculations(runs, processed_results)
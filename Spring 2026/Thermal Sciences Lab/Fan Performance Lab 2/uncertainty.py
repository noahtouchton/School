# uncertainty.py
import numpy as np
import math
import os
import matplotlib.pyplot as plt

import rss

# Create the graphs directory if it doesn't exist
os.makedirs("graphs", exist_ok=True)

# Impeller diameter measurements (in mm)
backwards_impeller_vals = [154.84, 154.90, 154.42, 154.67, 154.40] #keep
forwards_impeller_vals = [153.42, 153.67, 153.76, 153.54, 153.75] #keep

backwards_impeller_vals = [250]
forwards_impeller_vals = [250]

# Average the measurements and convert mm to meters
D_BACKWARDS = np.mean(backwards_impeller_vals) / 1000.0
# Assuming 'forwards' measurements belong to the Radial impeller used in the lab
D_RADIAL = np.mean(forwards_impeller_vals) / 1000.0 

RHO_AIR = 1.2

MECH_LOSS_MAP = {
    1000: 4,
    1500: 5,
    2000: 9,
    2500: 12,
    3000: 14
}

# Styling lists for unique colors and markers
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
MARKERS = ['o', 's', '^', 'D', 'v']

lab_data = rss.load_all_lab_runs()

def get_trendline(x, y, degree=2):
    """Helper function to generate a smooth trendline."""
    z = np.polyfit(x, y, degree)
    p = np.poly1d(z)
    x_smooth = np.linspace(x.min(), x.max(), 100)
    return x_smooth, p(x_smooth)

def plot_head_vs_flow(lab_data):
    impellers = [('radial', 'Radial'), ('backwards', 'Backward')]
    
    for key, title in impellers:
        if key not in lab_data:
            print(f"No data found for {key}")
            continue
            
        plt.figure(figsize=(8, 6))
        speeds = sorted(lab_data[key].keys())
        
        for idx, speed in enumerate(speeds):
            df = lab_data[key][speed]
            
            q_cols = [c for c in df.columns if "Volumetric Flow" in c]
            if not q_cols: continue
            Q = df[q_cols[0]]

            h_cols = [c for c in df.columns if "Total Pressure" in c]
            if h_cols:
                H = df[h_cols[0]]
            else:
                try:
                    p3_col = [c for c in df.columns if "P3" in c or "Outlet" in c][0]
                    p2_col = [c for c in df.columns if "P2" in c or "Inlet" in c and "Nozzle" not in c][0]
                    H = df[p3_col] - df[p2_col]
                except IndexError:
                    continue

            sort_idx = np.argsort(Q)
            Q_sorted = Q.iloc[sort_idx]
            H_sorted = H.iloc[sort_idx]

            c = COLORS[idx % len(COLORS)]
            m = MARKERS[idx % len(MARKERS)]

            # Scatter points
            plt.plot(Q_sorted, H_sorted, linestyle='', marker=m, color=c, label=f"{speed} RPM")
            # Trendline
            x_line, y_line = get_trendline(Q_sorted, H_sorted)
            plt.plot(x_line, y_line, linestyle='-', color=c)

        plt.xlabel("Volumetric Flow Rate ($m^3/s$)")
        plt.ylabel("Head Rise / Total Pressure (Pa)")
        plt.grid(True, which='both', linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"graphs/Page_2_Head_vs_Flow_{title}.png", bbox_inches='tight')
        plt.close()

def augment_data(df, rpm, impeller_type):
    p_loss = MECH_LOSS_MAP.get(rpm, 0)
    d_impeller = D_BACKWARDS if impeller_type == 'backwards' else D_RADIAL
    
    try:
        Q = df[[c for c in df.columns if "Volumetric Flow" in c][0]]
        dP = df[[c for c in df.columns if "Total Pressure" in c][0]]
        P_total = df[[c for c in df.columns if "Power" in c and "Mechanical" not in c][0]]
    except IndexError:
        return df

    df['BHP_W'] = P_total - p_loss
    df['WHP_W'] = Q * dP
    df['Efficiency_Calc'] = (df['WHP_W'] / df['BHP_W']) * 100
    
    n = rpm / 60.0 
    df['Cq'] = Q / (n * d_impeller**3)
    df['Ch'] = dP / (RHO_AIR * (n**2) * (d_impeller**2))
    df['Cp'] = df['BHP_W'] / (RHO_AIR * (n**3) * (d_impeller**5))
    df['Mech_Loss_W'] = p_loss
    
    return df

# ==========================================
# PAGE 3-5: BHP & WHP vs Flow (5 Graphs)
# ==========================================
def plot_bhp_whp_vs_flow(lab_data):
    speeds = [1000, 1500, 2000, 2500, 3000]
    
    for rpm in speeds:
        plt.figure(figsize=(8, 6))
        
        # Plot configurations: (Impeller, Value_Col, Label, Color, Marker)
        configs = []
        if 'radial' in lab_data and rpm in lab_data['radial']:
            df_rad = augment_data(lab_data['radial'][rpm].copy(), rpm, 'radial')
            configs.extend([
                (df_rad, 'BHP_W', 'Radial BHP', COLORS[0], MARKERS[0]),
                (df_rad, 'WHP_W', 'Radial WHP', COLORS[1], MARKERS[1])
            ])
            
        if 'backwards' in lab_data and rpm in lab_data['backwards']:
            df_back = augment_data(lab_data['backwards'][rpm].copy(), rpm, 'backwards')
            configs.extend([
                (df_back, 'BHP_W', 'Backward BHP', COLORS[2], MARKERS[2]),
                (df_back, 'WHP_W', 'Backward WHP', COLORS[3], MARKERS[3])
            ])

        for df, col, label, c, m in configs:
            Q = df[[c for c in df.columns if "Volumetric" in c][0]]
            sort_idx = np.argsort(Q)
            Q_sorted = Q.iloc[sort_idx]
            Y_sorted = df[col].iloc[sort_idx]
            
            # Scatter
            plt.plot(Q_sorted, Y_sorted, linestyle='', marker=m, color=c, label=label)
            # Trendline
            x_line, y_line = get_trendline(Q_sorted, Y_sorted)
            plt.plot(x_line, y_line, linestyle='-', color=c)

        plt.xlabel("Volumetric Flow Rate ($m^3/s$)")
        plt.ylabel("Power (Watts)")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"graphs/Pages_3_to_5_Power_vs_Flow_{rpm}RPM.png", bbox_inches='tight')
        plt.close()

# ==========================================
# PAGE 6: Efficiency vs Flow (2 Graphs Separated)
# ==========================================
def plot_efficiency(lab_data):
    impellers = [('radial', 'Radial'), ('backwards', 'Backward')]
    
    for key, title in impellers:
        if key not in lab_data: continue
        
        plt.figure(figsize=(8, 6))
        for idx, rpm in enumerate(sorted(lab_data[key].keys())):
            df = augment_data(lab_data[key][rpm].copy(), rpm, key)
            Q = df[[c for c in df.columns if "Volumetric" in c][0]]
            
            sort_idx = np.argsort(Q)
            Q_sorted = Q.iloc[sort_idx]
            E_sorted = df['Efficiency_Calc'].iloc[sort_idx]
            
            c = COLORS[idx % len(COLORS)]
            m = MARKERS[idx % len(MARKERS)]
            
            # Scatter
            plt.plot(Q_sorted, E_sorted, linestyle='', marker=m, color=c, label=f'{rpm} RPM')
            # Trendline
            x_line, y_line = get_trendline(Q_sorted, E_sorted)
            plt.plot(x_line, y_line, linestyle='-', color=c)
            
        plt.xlabel("Volumetric Flow Rate ($m^3/s$)")
        plt.ylabel("Efficiency (%)")
        plt.ylim(0, 100)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"graphs/Page_6_Efficiency_vs_Flow_{title}.png", bbox_inches='tight')
        plt.close()

# ==========================================
# PAGE 7: Dimensionless Curves (2 Graphs Separated)
# ==========================================
def plot_nondimensional_curves(lab_data):
    impellers = [('radial', 'Radial'), ('backwards', 'Backward')]
    
    for key, title in impellers:
        if key not in lab_data: continue
        
        plt.figure(figsize=(8, 6))
        all_Cq, all_Ch, all_Cp = [], [], []
        
        for rpm in lab_data[key]:
            df = augment_data(lab_data[key][rpm].copy(), rpm, key)
            all_Cq.extend(df['Cq'])
            all_Ch.extend(df['Ch'])
            all_Cp.extend(df['Cp'])
            
        # Convert to arrays for sorting/fitting
        all_Cq = np.array(all_Cq)
        all_Ch = np.array(all_Ch)
        all_Cp = np.array(all_Cp)
        
        sort_idx = np.argsort(all_Cq)
        all_Cq = all_Cq[sort_idx]
        all_Ch = all_Ch[sort_idx]
        all_Cp = all_Cp[sort_idx]

        # Scatter Ch
        plt.plot(all_Cq, all_Ch, linestyle='', marker=MARKERS[0], color=COLORS[0], label='$C_H$ (Head Coeff)')
        x_line_h, y_line_h = get_trendline(all_Cq, all_Ch)
        plt.plot(x_line_h, y_line_h, linestyle='-', color=COLORS[0])
        
        # Scatter Cp
        plt.plot(all_Cq, all_Cp, linestyle='', marker=MARKERS[1], color=COLORS[1], label='$C_P$ (Power Coeff)')
        x_line_p, y_line_p = get_trendline(all_Cq, all_Cp)
        plt.plot(x_line_p, y_line_p, linestyle='-', color=COLORS[1])
        
        plt.xlabel("Flow Coefficient ($C_Q$)")
        plt.ylabel("Coefficient Value")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"graphs/Page_7_Nondimensional_Curves_{title}.png", bbox_inches='tight')
        plt.close()

# ==========================================
# PAGE 8: Annual Cost Calculation
# ==========================================
def calculate_annual_cost(lab_data):
    HOURS_PER_YEAR = 2800
    RATE_PER_KWH = 0.15
    MOTOR_EFF = 0.80
    
    print("=== PAGE 8: ANNUAL OPERATING COST (Backward Impeller, 100% Open) ===")
    print(f"{'RPM':<10} | {'BHP (W)':<10} | {'Grid Power (kW)':<15} | {'Annual Cost ($)':<15}")
    print("-" * 60)
    
    if 'backwards' not in lab_data:
        print("No Backward data found.")
        return

    for rpm in sorted(lab_data['backwards'].keys()):
        df = augment_data(lab_data['backwards'][rpm].copy(), rpm, 'backwards')
        Q_col = [c for c in df.columns if "Volumetric" in c][0]
        max_flow_idx = df[Q_col].idxmax()
        
        bhp_watts = df.loc[max_flow_idx, 'BHP_W']
        grid_power_kw = (bhp_watts / MOTOR_EFF) / 1000.0
        annual_cost = grid_power_kw * HOURS_PER_YEAR * RATE_PER_KWH
        
        print(f"{rpm:<10} | {bhp_watts:<10.2f} | {grid_power_kw:<15.4f} | ${annual_cost:<15.2f}")
    print("=" * 60 + "\n")

# ==========================================
# PAGE 9: Data Tables
# ==========================================
def generate_data_tables(lab_data):
    print("=== PAGE 9: EXPERIMENTAL DATA TABLES ===")
    
    target_cols = [
        'Speed (rev.min-1)', 
        'Slide Valve Position (%)',
        'Torque (Nm)',
        'Power (W)', 
        'Nozzle Total Pressure (Pa)', 
        'Inlet Pressure (Pa)', 
        'Outlet Pressure (Pa)', 
        'Volumetric Flow Rate (m3.s-1)',
        'Mech_Loss_W',
        'Fan Total Pressure (Pa)', 
        'Efficiency_Calc'
    ]
    
    for impeller in lab_data:
        for rpm in lab_data[impeller]:
            df = augment_data(lab_data[impeller][rpm].copy(), rpm, impeller)
            cols_to_show = [c for c in target_cols if c in df.columns or c in ['Mech_Loss_W', 'Efficiency_Calc']]
            print(f"\nTable: {impeller.capitalize()} Impeller - {rpm} RPM")
            print(df[cols_to_show].round(3).to_string()) 

# ==========================================
# PAGE 10: Sample Calculations
# ==========================================
def generate_sample_calculations(lab_data):
    print("\n=== PAGE 10: SAMPLE CALCULATIONS ===")
    
    try:
        df = lab_data['radial'][2000]
        
        # Find the row where the valve is fully open (max volumetric flow)
        Q_col = [c for c in df.columns if "Volumetric" in c][0]
        max_flow_idx = df[Q_col].idxmax()
        row = df.loc[max_flow_idx]
        
        N = row[[c for c in df.columns if "Speed" in c][0]]
        T = row[[c for c in df.columns if "Torque" in c][0]]
        P_tot = row[[c for c in df.columns if "Power" in c and "Mech" not in c][0]]
        Q = row[Q_col]
        dP = row[[c for c in df.columns if "Total Pressure" in c][0]]
        P_loss = MECH_LOSS_MAP[2000]
        
        print(f"Data Point: Radial Impeller, Nominal 2000 RPM, Fully Open Valve (100%)")
        print(f"Measured Speed (N): {N:.1f} RPM")
        print(f"Measured Torque (T): {T:.3f} Nm")
        print(f"Measured Flow (Q): {Q:.4f} m^3/s")
        print(f"Measured Head (dP): {dP:.1f} Pa")
        print(f"Total Power Displayed: {P_tot:.1f} W")
        print(f"Mechanical Loss @ 2000 RPM: {P_loss} W")
        print(f"Impeller Diameter (D): {D_RADIAL:.3f} m")
        print(f"Air Density (rho): {RHO_AIR} kg/m^3")
        
        print("\n1. Brake Horsepower (BHP):")
        print(f"   BHP = P_total - P_mech_loss")
        bhp = P_tot - P_loss
        print(f"   BHP = {P_tot:.2f} - {P_loss} = {bhp:.2f} W")
        
        print("\n2. Water Horsepower (WHP):")
        print(f"   WHP = Q * dP_total")
        whp = Q * dP
        print(f"   WHP = {Q:.4f} * {dP:.1f} = {whp:.2f} W")
        
        print("\n3. Fan Efficiency:")
        print(f"   Eta = (WHP / BHP) * 100")
        eta = (whp / bhp) * 100
        print(f"   Eta = ({whp:.2f} / {bhp:.2f}) * 100 = {eta:.1f} %")

        # --- COEFFICIENT CALCULATIONS ---
        n = N / 60.0
        cq = Q / (n * D_RADIAL**3)
        ch = dP / (RHO_AIR * (n**2) * (D_RADIAL**2))
        cp = bhp / (RHO_AIR * (n**3) * (D_RADIAL**5))

        print("\n4. Flow Coefficient (C_Q):")
        print(f"   n = N / 60 = {N:.1f} / 60 = {n:.2f} rev/s")
        print(f"   C_Q = Q / (n * D^3)")
        print(f"   C_Q = {Q:.4f} / ({n:.2f} * {D_RADIAL:.3f}^3) = {cq:.4f}")

        print("\n5. Head Coefficient (C_H):")
        print(f"   C_H = dP / (rho * n^2 * D^2)")
        print(f"   C_H = {dP:.1f} / ({RHO_AIR} * {n:.2f}^2 * {D_RADIAL:.3f}^2) = {ch:.4f}")

        print("\n6. Power Coefficient (C_P):")
        print(f"   C_P = BHP / (rho * n^3 * D^5)")
        print(f"   C_P = {bhp:.2f} / ({RHO_AIR} * {n:.2f}^3 * {D_RADIAL:.3f}^5) = {cp:.4f}")
        
    except Exception as e:
        print(f"Could not generate sample calc: {e}")

plot_head_vs_flow(lab_data)
plot_bhp_whp_vs_flow(lab_data)
plot_efficiency(lab_data)
plot_nondimensional_curves(lab_data)
calculate_annual_cost(lab_data)
generate_sample_calculations(lab_data)
# generate_data_tables(lab_data) # Uncomment to print large tables

# ==========================================
# DEBUGGING: Cp Scatter Analysis
# ==========================================
def debug_cp_scatter(lab_data):
    print("\n=== DEBUGGING Cp SCATTER (Radial Impeller, 100% Valve) ===")
    print(f"{'RPM':<6} | {'n (rev/s)':<10} | {'Raw P (W)':<10} | {'Loss (W)':<9} | {'BHP (W)':<8} | {'Cp':<10} | {'n^3 Factor':<10}")
    print("-" * 75)
    
    if 'radial' not in lab_data:
        print("No radial data found.")
        return

    for rpm in sorted(lab_data['radial'].keys()):
        df = augment_data(lab_data['radial'][rpm].copy(), rpm, 'radial')
        
        # Grab the row with the maximum flow rate (100% open valve)
        Q_col = [c for c in df.columns if "Volumetric" in c][0]
        max_flow_idx = df[Q_col].idxmax()
        row = df.loc[max_flow_idx]
        
        # Extract variables used in Cp
        n = rpm / 60.0
        n3 = n**3
        raw_p = row[[c for c in df.columns if "Power" in c and "Mechanical" not in c][0]]
        p_loss = MECH_LOSS_MAP.get(rpm, 0)
        bhp = row['BHP_W']
        cp = row['Cp']
        
        print(f"{rpm:<6} | {n:<10.2f} | {raw_p:<10.2f} | {p_loss:<9} | {bhp:<8.2f} | {cp:<10.4f} | {n3:<10.2f}")
    print("=" * 75 + "\n")

# Run the debug function
debug_cp_scatter(lab_data)
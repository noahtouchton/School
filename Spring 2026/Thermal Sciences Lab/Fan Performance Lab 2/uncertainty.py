# uncertainty.py
import numpy as np
import math
import os
import matplotlib.pyplot as plt

import rss

# Create the graphs directory if it doesn't exist
os.makedirs("graphs", exist_ok=True)

D_IMPELLER = 0.15 
RHO_AIR = 1.2

MECH_LOSS_MAP = {
    1000: 4,
    1500: 5,
    2000: 9,
    2500: 12,
    3000: 14
}

lab_data = rss.load_all_lab_runs()

# Accessing specific data:
# df_1500 = lab_data['radial'][1500]
# print(df_1500)

def plot_head_vs_flow(lab_data):
    """
    Plots Head Rise vs. Flow Rate for Radial and Backward impellers.
    Assumes lab_data is the dictionary structure: 
    lab_data['radial'][1000] -> DataFrame
    """
    
    # Setup the two plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Helper to process one impeller's data
    def process_impeller(impeller_name, ax, color_map):
        if impeller_name not in lab_data:
            print(f"No data found for {impeller_name}")
            return

        speeds = sorted(lab_data[impeller_name].keys())
        
        for speed in speeds:
            df = lab_data[impeller_name][speed]
            
            # --- 1. GET FLOW RATE (Q) ---
            q_cols = [c for c in df.columns if "Volumetric Flow" in c]
            if q_cols:
                Q = df[q_cols[0]]
            else:
                print(f"Warning: Volumetric Flow column missing for {impeller_name} {speed}")
                continue

            # --- 2. GET HEAD RISE (H) ---
            h_cols = [c for c in df.columns if "Total Pressure" in c]
            
            if h_cols:
                H = df[h_cols[0]]
            else:
                try:
                    p3_col = [c for c in df.columns if "P3" in c or "Outlet" in c][0]
                    p2_col = [c for c in df.columns if "P2" in c or "Inlet" in c and "Nozzle" not in c][0]
                    H = df[p3_col] - df[p2_col]
                except IndexError:
                    print(f"Warning: Pressure columns missing for {impeller_name} {speed}")
                    continue

            # Plot the curve
            sort_idx = np.argsort(Q)
            ax.plot(Q.iloc[sort_idx], H.iloc[sort_idx], marker='o', label=f"{speed} RPM")

        # Styling
        #ax.set_title(f"{impeller_name.capitalize()} Impeller: Head Rise vs Flow Rate")
        ax.set_xlabel("Volumetric Flow Rate ($m^3/s$)")
        ax.set_ylabel("Head Rise / Total Pressure (Pa)")
        ax.grid(True, which='both', linestyle='--', alpha=0.7)
        ax.legend(title="Fan Speed")

    # Generate the two plots
    process_impeller('radial', ax1, None)
    process_impeller('backwards', ax2, None)

    plt.tight_layout()
    # Save the figure to the graphs folder
    plt.savefig("graphs/Page_2_Head_vs_Flow.png", bbox_inches='tight')
    plt.show()


def augment_data(df, rpm):
    """
    Adds calculated columns (BHP, WHP, Efficiency, Coeffs) to the dataframe.
    """
    p_loss = MECH_LOSS_MAP.get(rpm, 0)
    
    try:
        Q = df[[c for c in df.columns if "Volumetric Flow" in c][0]]
        dP = df[[c for c in df.columns if "Total Pressure" in c][0]]
        P_total = df[[c for c in df.columns if "Power" in c and "Mechanical" not in c][0]]
        Torque = df[[c for c in df.columns if "Torque" in c][0]]
    except IndexError:
        return df

    df['BHP_W'] = P_total - p_loss
    df['WHP_W'] = Q * dP
    df['Efficiency_Calc'] = (df['WHP_W'] / df['BHP_W']) * 100
    
    n = rpm / 60.0 
    df['Cq'] = Q / (n * D_IMPELLER**3)
    df['Ch'] = dP / (RHO_AIR * (n**2) * (D_IMPELLER**2))
    df['Cp'] = df['BHP_W'] / (RHO_AIR * (n**3) * (D_IMPELLER**5))
    df['Mech_Loss_W'] = p_loss
    
    return df

# ==========================================
# PAGE 3-5: BHP & WHP vs Flow (5 Graphs)
# ==========================================
def plot_bhp_whp_vs_flow(lab_data):
    """
    Generates 5 separate figures (one per RPM).
    Each figure compares Radial vs Backward for both BHP and WHP.
    """
    speeds = [1000, 1500, 2000, 2500, 3000]
    
    for rpm in speeds:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot Radial
        if 'radial' in lab_data and rpm in lab_data['radial']:
            df = augment_data(lab_data['radial'][rpm].copy(), rpm)
            Q = df[[c for c in df.columns if "Volumetric" in c][0]]
            
            ax.plot(Q, df['BHP_W'], 'r-o', label=f'Radial BHP')
            ax.plot(Q, df['WHP_W'], 'r--s', label=f'Radial WHP')
            
        # Plot Backward
        if 'backwards' in lab_data and rpm in lab_data['backwards']:
            df = augment_data(lab_data['backwards'][rpm].copy(), rpm)
            Q = df[[c for c in df.columns if "Volumetric" in c][0]]
            
            ax.plot(Q, df['BHP_W'], 'b-o', label=f'Backward BHP')
            ax.plot(Q, df['WHP_W'], 'b--s', label=f'Backward WHP')

        #ax.set_title(f"Power Analysis at {rpm} RPM")
        ax.set_xlabel("Volumetric Flow Rate ($m^3/s$)")
        ax.set_ylabel("Power (Watts)")
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()
        plt.tight_layout()
        
        # Save each figure uniquely by RPM
        plt.savefig(f"graphs/Pages_3_to_5_Power_vs_Flow_{rpm}RPM.png", bbox_inches='tight')
        plt.show()

# ==========================================
# PAGE 6: Efficiency vs Flow (2 Graphs)
# ==========================================
def plot_efficiency(lab_data):
    """
    Generates 2 plots: One for Radial, One for Backward.
    Each plot contains curves for all 5 speeds.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    impellers = [('radial', ax1, 'Radial'), ('backwards', ax2, 'Backward')]
    
    for key, ax, title in impellers:
        if key not in lab_data: continue
        
        for rpm in sorted(lab_data[key].keys()):
            df = augment_data(lab_data[key][rpm].copy(), rpm)
            Q = df[[c for c in df.columns if "Volumetric" in c][0]]
            
            ax.plot(Q, df['Efficiency_Calc'], marker='.', label=f'{rpm} RPM')
            
        #ax.set_title(f"{title} Impeller Efficiency")
        ax.set_xlabel("Volumetric Flow Rate ($m^3/s$)")
        ax.set_ylabel("Efficiency (%)")
        ax.set_ylim(0, 100)
        ax.grid(True)
        ax.legend()
        
    plt.tight_layout()
    # Save the figure to the graphs folder
    plt.savefig("graphs/Page_6_Efficiency_vs_Flow.png", bbox_inches='tight')
    plt.show()

# ==========================================
# PAGE 7: Dimensionless Curves (2 Graphs)
# ==========================================
def plot_nondimensional_curves(lab_data):
    """
    Generates 2 plots: Cp and Ch vs Cq.
    Collapses all speeds onto single universal curves.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    impellers = [('radial', ax1, 'Radial'), ('backwards', ax2, 'Backward')]
    
    for key, ax, title in impellers:
        if key not in lab_data: continue
        
        all_Cq, all_Ch, all_Cp = [], [], []
        
        for rpm in lab_data[key]:
            df = augment_data(lab_data[key][rpm].copy(), rpm)
            all_Cq.extend(df['Cq'])
            all_Ch.extend(df['Ch'])
            all_Cp.extend(df['Cp'])
            
        ax.scatter(all_Cq, all_Ch, c='blue', label='$C_H$ (Head Coeff)')
        ax.scatter(all_Cq, all_Cp, c='red', marker='x', label='$C_P$ (Power Coeff)')
        
        #ax.set_title(f"{title} Non-Dimensional Performance")
        ax.set_xlabel("Flow Coefficient ($C_Q$)")
        ax.set_ylabel("Coefficient Value")
        ax.grid(True)
        ax.legend()
        
    plt.tight_layout()
    # Save the figure to the graphs folder
    plt.savefig("graphs/Page_7_Nondimensional_Curves.png", bbox_inches='tight')
    plt.show()

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
        df = augment_data(lab_data['backwards'][rpm].copy(), rpm)
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
            df = augment_data(lab_data[impeller][rpm].copy(), rpm)
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
        row = df.iloc[len(df)//2] 
        
        N = row[[c for c in df.columns if "Speed" in c][0]]
        T = row[[c for c in df.columns if "Torque" in c][0]]
        P_tot = row[[c for c in df.columns if "Power" in c and "Mech" not in c][0]]
        Q = row[[c for c in df.columns if "Volumetric" in c][0]]
        dP = row[[c for c in df.columns if "Total Pressure" in c][0]]
        P_loss = MECH_LOSS_MAP[2000]
        
        print(f"Data Point: Radial Impeller, Nominal 2000 RPM, Mid-Valve Position")
        print(f"Measured Speed (N): {N:.1f} RPM")
        print(f"Measured Torque (T): {T:.3f} Nm")
        print(f"Measured Flow (Q): {Q:.4f} m^3/s")
        print(f"Measured Head (dP): {dP:.1f} Pa")
        print(f"Total Power Displayed: {P_tot:.1f} W")
        print(f"Mechanical Loss @ 2000 RPM: {P_loss} W")
        
        print("\n1. Brake Horsepower (BHP):")
        print(f"   BHP = P_total - P_mech_loss")
        print(f"   BHP = {P_tot} - {P_loss} = {P_tot - P_loss:.2f} W")
        
        print("\n2. Water Horsepower (WHP):")
        print(f"   WHP = Q * dP_total")
        print(f"   WHP = {Q:.4f} * {dP:.1f} = {Q*dP:.2f} W")
        
        print("\n3. Fan Efficiency:")
        print(f"   Eta = (WHP / BHP) * 100")
        print(f"   Eta = ({Q*dP:.2f} / {P_tot - P_loss:.2f}) * 100 = {(Q*dP)/(P_tot-P_loss)*100:.1f} %")
        
    except Exception as e:
        print(f"Could not generate sample calc: {e}")

plot_head_vs_flow(lab_data)
plot_bhp_whp_vs_flow(lab_data)
plot_efficiency(lab_data)
plot_nondimensional_curves(lab_data)
calculate_annual_cost(lab_data)
generate_sample_calculations(lab_data)
# generate_data_tables(lab_data) # Uncomment to print large tables
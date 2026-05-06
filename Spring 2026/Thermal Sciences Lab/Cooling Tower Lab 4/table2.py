import math

# --- Constants ---
P_atm = 101.325   # Atmospheric pressure (kPa)
Cp_water = 4.186  # Specific heat of water (kJ/kg.K)
R_a = 0.28705     # Gas constant for dry air (kJ/kg.K)
delta_P = 10      # Measured pressure drop (mmH2O)

# --- Raw Data ---
# Format: [Water Flow (g/s), T1(db_in), T2(wb_in), T3(db_out), T4(wb_out), T5(water_in), T6(water_out)]
tests = [
    [20, 22.8, 18.5, 20.8, 19.9, 24.7, 19.7],
    [30, 23.3, 18.9, 21.5, 20.8, 24.7, 20.5],
    [40, 23.2, 19.1, 22.3, 21.0, 24.4, 21.1],
    [20, 24.1, 20.0, 24.5, 23.6, 35.2, 22.3],
    [30, 24.3, 20.1, 25.6, 25.0, 34.9, 24.3],
    [40, 24.5, 20.5, 26.6, 25.3, 33.3, 25.3]
]

def get_psychrometrics(t_db, t_wb):
    # Vapor pressures
    p_ws_wb = 0.6108 * math.exp((17.27 * t_wb) / (t_wb + 237.3))
    p_w = p_ws_wb - (6.66e-4 * P_atm * (t_db - t_wb))
    
    # Humidity ratio
    w = 0.62198 * (p_w / (P_atm - p_w))
    
    # Air enthalpy equation (kJ/kg)
    h = 1.006 * t_db + w * (2501 + 1.86 * t_db)
    return h, w, p_w

# --- Table Header ---
print(f"{'Test':<5} | {'m_a':<7} | {'T_pred':<7} | {'T_act':<6} | {'del_T':<6} | {'Q_pred':<7} | {'Q_act':<7} | {'del_Q':<7}")
print("-" * 72)

# --- First Law Analysis Loop ---
for i, test in enumerate(tests, 1):
    m_w = test[0] / 1000  
    t_db_in, t_wb_in = test[1], test[2]
    t_db_out, t_wb_out = test[3], test[4]
    t5_water_in, t6_water_out = test[5], test[6]
    
    # 1. Calculate Air Properties
    h_in, w_in, pw_in = get_psychrometrics(t_db_in, t_wb_in)
    h_out, w_out, pw_out = get_psychrometrics(t_db_out, t_wb_out)
    
    # 2. Calculate Air Mass Flow Rate (Using manual equations)
    p_a_out = P_atm - pw_out # Partial pressure of dry air at exit
    rho_a = p_a_out / (R_a * (t_db_out + 273.15)) # Density of dry air at exit
   
    rho_mix = (1 + w_out) * rho_a # Mixture density
    m_a = 0.0137 * math.sqrt(rho_mix * delta_P) # Corrected mass flow rate
    
    # 3. Predicted Heat Dissipation (Air Side Energy Balance)
    Q_pred = m_a * (h_out - h_in)
    
    # 4. Actual Heat Dissipation (Water Side Energy Balance)
    Q_act = m_w * Cp_water * (t5_water_in - t6_water_out)
    
    # 5. Predicted Exit Temperature
    T_pred = t5_water_in - (Q_pred / (m_w * Cp_water))
    
    # 6. Deltas
    del_T = t6_water_out - T_pred
    del_Q = Q_pred - Q_act
    
    print(f"{i:<5} | {m_a:<7.4f} | {T_pred:<7.1f} | {t6_water_out:<6.1f} | {del_T:<6.1f} | {Q_pred:<7.3f} | {Q_act:<7.3f} | {del_Q:<7.3f}")

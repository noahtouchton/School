import math

# Atmospheric pressure in kPa (Standard sea level)
P = 101.325 

# Your raw data: [T1(db_in), T2(wb_in), T3(db_out), T4(wb_out)]
tests = [
    [22.8, 18.5, 20.8, 19.9],
    [23.3, 18.9, 21.5, 20.8],
    [23.2, 19.1, 22.3, 21.0],
    [24.1, 20.0, 24.5, 23.6],
    [24.3, 20.1, 25.6, 25.0],
    [24.5, 20.5, 26.6, 25.3]
]

def calc_psychrometrics(t_db, t_wb):
    # Saturation vapor pressure (Tetens equation) in kPa
    p_ws_db = 0.6108 * math.exp((17.27 * t_db) / (t_db + 237.3))
    p_ws_wb = 0.6108 * math.exp((17.27 * t_wb) / (t_wb + 237.3))
    
    # Actual vapor pressure (Carrier equation)
    p_w = p_ws_wb - (6.66e-4 * P * (t_db - t_wb))
    
    # Relative humidity (%)
    rh = (p_w / p_ws_db) * 100
    
    # Humidity ratio (kg/kg dry air)
    w = 0.62198 * (p_w / (P - p_w))
    
    return w, rh

print(f"{'Test':<6} | {'w_in':<8} | {'w_out':<8} | {'phi_in (%)':<10} | {'phi_out (%)':<10}")
print("-" * 55)

for i, test in enumerate(tests, 1):
    w_in, phi_in = calc_psychrometrics(test[0], test[1])
    w_out, phi_out = calc_psychrometrics(test[2], test[3])
    print(f"{i:<6} | {w_in:<8.4f} | {w_out:<8.4f} | {round(phi_in):<10} | {round(phi_out):<10}")
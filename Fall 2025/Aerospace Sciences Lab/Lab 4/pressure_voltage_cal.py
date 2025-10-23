from rss import *  # uses your Data/RSS implementation
import numpy as np

# Pressure (in H2O)
P_1 = Data("Pressure_1", "inH2O",  2.00, 0.05)
P_2 = Data("Pressure_2", "inH2O",  0.65, 0.05)
P_3 = Data("Pressure_3", "inH2O", -1.90, 0.05)

# Corresponding Voltage readings
V_1 = Data("Voltage_1", "V",  1.066, 0.060)
V_2 = Data("Voltage_2", "V",  0.385, 0.058)
V_3 = Data("Voltage_3", "V", -0.951, 0.054)


P_vals = np.array([P_1.value, P_2.value, P_3.value])
P_uncs = np.array([P_1.uncertainty, P_2.uncertainty, P_3.uncertainty])

V_vals = np.array([V_1.value, V_2.value, V_3.value])
V_uncs = np.array([V_1.uncertainty, V_2.uncertainty, V_3.uncertainty])

pressures = Data("Pressures", "inH2O", P_vals, P_uncs)
voltages  = Data("Voltages",  "V",      V_vals, V_uncs)

cal_slope, cal_intercept = linear_monte_carlo(voltages, pressures)

cal_slope.get_description()
cal_intercept.get_description()
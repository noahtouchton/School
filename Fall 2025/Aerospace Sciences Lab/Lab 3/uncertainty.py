#uncertainty.py
from rss import *
import numpy

inH2O_to_Pa = 249.0889

equations = [
    "q = (m*dp+b) * 249.0889",
    "mu = MU_0 * (T / T_0)**(3/2) * (T_0 + C) / (T + C)",
    "Re = (d/mu) * sqrt( (2*q*P) / (R*T) )",
    "Cp = deltaP/q",
    "ρ = P / (R * T)",
    "v = sqrt(2*q/ρ)",
    "v = a0 + a1*V + a2*V**2 + a3*V**3 + a4*V**4"

]

MU_0 = Data("mu not", "MU_0", 1.716 * 10**-5, 0.0)
T_0 = Data("T not", "T_0", 273, 0.0)
C = Data("Constant", "C",111.0, 0.0)
T = Data("Temperature", "T", 296.15, 0.4)
P = Data("Pressure", "P", 101207.54, 400.0)
d = Data("Diameter", "d", 0.01905, 0.0)
R = Data("Ideal Gas Constant for Air", "R", 287.05, 0.0)

m = Data("Calibration Slope", "m", 0.5713, 0.025106)
b = Data("Calibration Intercept", "b", -0.296, 0.121)

ρ = RSS(
    "Density",
    equations[4],
    [P,R,T]
)

ρ.get_description()

q_cal_vals = np.array([
    -0.006, -0.047, -0.148, -0.316, -0.560,
    -0.881, -1.302, -1.836, -2.462, -3.176,
    -3.928, -4.684
]) * inH2O_to_Pa * -1.0

q_cal_uncert = np.array([
    0.060, 0.065, 0.070, 0.076, 0.066,
    0.062, 0.068, 0.065, 0.066, 0.061,
    0.066, 0.064
]) * 2.0 * inH2O_to_Pa

q_cal = Data("Q Calibration", "q", q_cal_vals, q_cal_uncert)

v_cal = RSS(
    "Calibration Velocity",
    equations[5],
    [q_cal, ρ]
)

voltage_cal_vals = np.array([
    2.11012, 2.50558, 2.76121, 2.94751, 3.10566, 3.24794,
    3.39233, 3.51891, 3.64815, 3.76203, 3.86064, 3.93145
])

voltage_cal_uncert = np.array([
    0.0128176, 0.0131985, 0.0137291, 0.0128479, 0.0126538, 0.0163259,
    0.0154925, 0.0157290, 0.0122171, 0.0122837, 0.0107712, 0.0108837
]) * 2.0

voltage_cal = Data("Calibration Voltages", "V", voltage_cal_vals, voltage_cal_uncert)


x = voltage_cal_vals
y = v_cal.value

y_unc = v_cal.uncertainty  # same shape as y


# --- Excel-style OLS (unweighted), with covariance ---
deg = 4
coef_desc, cov_desc = np.polyfit(x, y, deg, cov=True)   # descending powers [a4..a0]
p = np.poly1d(coef_desc)

resid = y - p(x)
dof = len(x) - (deg + 1)
sigma2 = float((resid @ resid) / dof)
sigma = np.sqrt(sigma2)

# Coefficient 1σ (same ordering as coef_desc)
coef_stderr_desc = np.sqrt(np.diag(cov_desc))
print("Excel-style coeffs (desc):", coef_desc)
print("coef 1σ (desc):", coef_stderr_desc)
print("residual σ:", sigma, "  dof:", dof)

coeffs_asc = coef_desc[::-1]         # [a0..a4]
cov_asc    = cov_desc[::-1, ::-1]    # flip rows & cols

# plot_poly_fit(x, y, coeffs_asc, cov=cov_asc, sigma2=sigma2,
#               y_label="Velocity (m/s)", x_label="Voltage (V)",
#               title="Hot-Wire Calibration: Velocity vs. Voltage (Excel-style OLS)",
#               y_unc=None)   # Excel is unweighted; leave None to mirror Excel

coef_stderr_asc = coef_stderr_desc[::-1]

a0 = Data("Polynomial coefficient a0", "a0", coeffs_asc[0], coef_stderr_asc[0])
a1 = Data("Polynomial coefficient a1", "a1", coeffs_asc[1], coef_stderr_asc[1])
a2 = Data("Polynomial coefficient a2", "a2", coeffs_asc[2], coef_stderr_asc[2])
a3 = Data("Polynomial coefficient a3", "a3", coeffs_asc[3], coef_stderr_asc[3])
a4 = Data("Polynomial coefficient a4", "a4", coeffs_asc[4], coef_stderr_asc[4])

for a in [a0, a1, a2, a3, a4]:
    a.get_description()
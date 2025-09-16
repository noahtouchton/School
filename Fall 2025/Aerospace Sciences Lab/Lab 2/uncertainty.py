from rss import RSS, Data

MU_0 = Data("mu not", "MU_0", 1.716 * 10**-5, 0.0)
T_0 = Data("T not", "T_0", 273, 0.0)
C = Data("Constant", "C",111.0, 0.0)
T = Data("Temperature", "T", 23 + 273.15, 0.4)
P = Data("Pressure", "P", 101300, 400.0)
d = Data("Diameter", "d", 0.01905, 0.0)

m = Data("Calibration Slope", "m", 0.5713, 0.025106)
b = Data("Calibration Intercept", "b", -0.296, 0.121)
dp = Data("Pressure change", "dp", 2.701, 0.146778745)

q_val = m.value*dp.value + b.value

q = RSS(
    "Dynamic Pressure",
    q_val,
    "q = m*dp+b",
    [m,b,dp]
)

q.get_description()

mu_val = (
    MU_0.value *
    (T.value / T_0.value)**(3/2) *
    (T_0.value + C.value) / (T.value + C.value)
)

mu = RSS(
    "Dynamic Viscosity",
    mu_val,
    "mu = MU_0 * (T / T_0)**(3/2) * (T_0 + C) / (T + C)",
    [T, MU_0, T_0, C]   # include any variables you want propagated
)
mu.get_description()
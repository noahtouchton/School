from rss import RSS, Data

MU_0 = Data("mu not", "MU_0", 1.716 * 10**-5, 0.0)
T_0 = Data("T not", "T_0", 273, 0.0)
C = Data("Constant", "C",111.0, 0.0)
T = Data("Temperature", "T", 23 + 273.15, 0.4)

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
mu.get_error()
from rss import RSS, Data

inH2O_to_Pa = 249.0889

equations = [
    "q = (m*dp+b) * 249.0889",
    "mu = MU_0 * (T / T_0)**(3/2) * (T_0 + C) / (T + C)",
    "Re = (d/mu) * sqrt( (2*q*P) / (R*T) )"

]



MU_0 = Data("mu not", "MU_0", 1.716 * 10**-5, 0.0)
T_0 = Data("T not", "T_0", 273, 0.0)
C = Data("Constant", "C",111.0, 0.0)
T = Data("Temperature", "T", 23 + 273.15, 0.4)
P = Data("Pressure", "P", 101300, 400.0)
d = Data("Diameter", "d", 0.01905, 0.0)
R = Data("Ideal Gas Constant for Air", "R", 287.05, 0.0)

m = Data("Calibration Slope", "m", 0.5713, 0.025106)
b = Data("Calibration Intercept", "b", -0.296, 0.121)
dp = Data("Pressure change", "dp", 2.701, 0.146778745)


q = RSS(
    "Dynamic Pressure",
    equations[0],
    [m,b,dp]
)

q.get_description()

mu = RSS(
    "Dynamic Viscosity",
    equations[1],
    [T, MU_0, T_0, C] 
)

mu.get_description()

Re = RSS(
    "Reynolds Number",
    equations[2],
    [d,mu,q,P,R,T]
)

Re.get_description()
import nunpy as np

PO = 5

zeta = np.sqrt( (np.log(PO/100))**2 / (np.pi**2 + (np.log(PO/100))**2) )

print(zeta)
import numpy as np
import control as ct

PO = 5
Kdc = 1.13
tau = 0.03

zeta = np.sqrt((np.log(PO/100))**2 / (np.pi**2 + (np.log(PO/100))**2))
wn = 4 / (zeta * 0.3)
wd = wn * np.sqrt(1 - zeta**2)

s1 = complex(-zeta * wn,  wd)
s2 = complex(-zeta * wn, -wd)

A = np.array([[0, 1],
              [0, -1/tau]], dtype=float)
B = np.array([[0],
              [Kdc/tau]], dtype=float)

poles = [s1, s2]

K = ct.acker(A, B, poles)
K = np.atleast_2d(K)           # <- ensure shape (1, 2)

Ctrb = np.hstack([B, A @ B])
print("rank(Ctrb) =", np.linalg.matrix_rank(Ctrb))

Acl = A - B @ K                # (2,2) - (2,1)@(1,2) -> (2,2)
print("eig(A-BK) =", np.linalg.eigvals(Acl))
print("Desired   =", poles)

# Optional: prefilter for step tracking with state-feedback
C = np.array([[1., 0.]])
Acl = A - B @ K
Nbar = -1.0 / (C @ np.linalg.inv(Acl) @ B)
Nbar = np.asarray(Nbar).item()   # <— scalar extract (no deprecation)
print("Nbar =", Nbar)
print("K   =", K)

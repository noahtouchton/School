import numpy as np
import matplotlib.pyplot as plt

# Constants
c1 = 3.742e8      # W·µm^4 / m^2
c2 = 1.439e4      # µm·K

# Wavelength range (µm), log-spaced
lam = np.logspace(-1, 2, 500)   # 0.1 to 100 µm

def planck_E_lambda(lam, T):
    # lam in µm, T in K
    return c1 / (lam**5 * (np.exp(c2 / (lam * T)) - 1.0))

temps = [300, 2900, 5800]
labels = [r"T = 300 K", r"T = 2900 K", r"T = 5800 K"]

plt.figure()
for T, lab in zip(temps, labels):
    E = planck_E_lambda(lam, T)
    plt.loglog(lam, E, label=lab)

plt.xlabel(r"Wavelength $\lambda$ [$\mu$m]")
plt.ylabel(r"$E_{\lambda,b}$ [W/m$^2\cdot\mu$m]")
plt.xlim(0.1, 100)
plt.ylim(1e-4, 1e9)
plt.grid(True, which="both", ls=":")
plt.legend()
plt.title("Blackbody Spectral Hemispherical Emissive Power")
plt.show()
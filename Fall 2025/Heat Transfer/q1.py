import numpy as np
import matplotlib.pyplot as plt

# Constants
c1 = 3.742e8      # W * μm^4 / m^2
c2 = 1.439e4      # μm * K

# Wavelength array in μm (log-spaced from 0.1 to 100)
lam = np.logspace(-1, 2, 500)   # 10^-1 to 10^2 μm

# Temperatures in K
temps = [300, 2900, 5800]
labels = [r"T = 300 K", r"T = 2900 K", r"T = 5800 K"]

def E_lambda_b(lam_um, T):
    # lam_um in μm, T in K
    return c1 / (lam_um**5 * (np.exp(c2 / (lam_um * T)) - 1.0))

plt.figure()

for T, lab in zip(temps, labels):
    E = E_lambda_b(lam, T)
    plt.loglog(lam, E, label=lab)

plt.xlabel(r"Wavelength $\lambda$ [$\mu$m]")
plt.ylabel(r"$E_{\lambda,b}$ [W/m$^2\cdot\mu$m]")
plt.xlim(0.1, 100)
plt.ylim(1e-4, 1e9)
plt.grid(True, which='both', ls=':')
plt.legend()
plt.title("Blackbody Spectral Hemispherical Emissive Power")
plt.show()
#!/usr/bin/env python3
"""
Impulse response (small-angle) for a motor-driven base + pendulum (pendulum down).
No SciPy required — custom RK4 integrator.

State: x = [theta, phi, theta_dot, phi_dot]
- theta: base/platen angle (rad)
- phi:   pendulum angle relative to base (rad), phi=0 is DOWNWARD small-angle eq.
Input: motor torque tau_m (N·m). We approximate a unit-area impulse as a short, high pulse.

Lagrange linearized model (about down):
Let J0 = base inertia, J2 = pendulum inertia about pivot, b1 = base viscous friction,
b2 = pendulum joint viscous friction, k = m_total * g * l_COM.

Derived continuous-time equations:
    (J0+J2)*theta_dd + J2*phi_dd + b1*theta_dot = tau_m
     J2*theta_dd + J2*phi_dd + b2*phi_dot + k*phi = 0

In first-order form with x = [theta, phi, w1, w2] = [theta, phi, theta_dot, phi_dot]:
    [theta_dot] = w1
    [  phi_dot] = w2
    [ w1_dot ]   = [a31]*phi + [a33]*w1 + [a34]*w2 + [b3]*tau_m
    [ w2_dot ]   = [a41]*phi + [a43]*w1 + [a44]*w2 + [b4]*tau_m
where the coefficients come from M^{-1} * rhs with
    M = [[J0+J2, J2],
         [J2,     J2]],
    rhs = [ -b1*w1 + tau_m,
            -b2*w2 - k*phi ]

This script plots theta(t), phi(t), and (theta+phi)(t), and saves CSVs.
"""

import numpy as np
import matplotlib.pyplot as plt
import math
from dataclasses import dataclass, asdict
from pathlib import Path
import csv

# ----------------- Parameters (edit here) -----------------
g = 9.81

# Base (platen) side
J0 = 0.04          # kg·m^2
b1 = 0.0146        # N·m·s

# Pendulum: choose either the assembled inertia (recommended) or the smaller one
# Assembled inertia about pivot using your parts: ~1.376e-4 kg·m^2 (recommended)
J2_recommended = 1.376e-4
# Your earlier single value (likely just one component): 2.929e-5 kg·m^2
J2_alt = 2.929e-5

# Choose which J2 to use:
J2 = J2_recommended

# Pendulum damping from your free-swing estimate
b2 = 4.72e-5       # N·m·s

# Masses and geometry to compute gravity stiffness k = m_tot*g*l_COM
m_pend = 0.00491   # kg
m_nuts = 0.00377   # kg
d2 = 0.0736        # m (pend COM distance from pivot)
d3 = 0.1472        # m (nuts distance from pivot)
m_tot = m_pend + m_nuts
l_COM = (m_pend*d2 + m_nuts*d3) / m_tot
k = m_tot * g * l_COM

# Simulation
t_final = 5.0      # seconds
dt = 0.001         # seconds
# Impulse approximation: rectangular pulse with area = 1 N·m·s
impulse_width = 0.001   # s
impulse_area  = 1.0     # N·m·s
tau_amp = impulse_area / impulse_width
# ---------------------------------------------------------

@dataclass
class Coeffs:
    a31: float; a33: float; a34: float; b3: float
    a41: float; a43: float; a44: float; b4: float

def build_coeffs(J0, J2, b1, b2, k):
    # Invert M = [[J0+J2, J2],[J2, J2]]
    M = np.array([[J0+J2, J2],
                  [J2,     J2]], dtype=float)
    Minv = np.linalg.inv(M)
    # RHS decomposition
    # w1 term -> rhs = [-b1*w1, 0]
    v_w1 = Minv @ np.array([-b1, 0.0])
    # w2 term -> rhs = [0, -b2*w2]
    v_w2 = Minv @ np.array([0.0, -b2])
    # phi term -> rhs = [0, -k*phi]
    v_phi = Minv @ np.array([0.0, -k])
    # input term -> rhs = [1, 0]*tau
    v_u = Minv @ np.array([1.0, 0.0])
    return Coeffs(
        a31 = v_phi[0], a33 = v_w1[0], a34 = v_w2[0], b3 = v_u[0],
        a41 = v_phi[1], a43 = v_w1[1], a44 = v_w2[1], b4 = v_u[1]
    )

def rhs(x, t, coeffs):
    theta, phi, w1, w2 = x
    # input torque: rectangular impulse starting at t=0
    tau = tau_amp if (0.0 <= t < impulse_width) else 0.0
    # First-order dynamics
    dtheta = w1
    dphi   = w2
    dw1 = coeffs.a31*phi + coeffs.a33*w1 + coeffs.a34*w2 + coeffs.b3*tau
    dw2 = coeffs.a41*phi + coeffs.a43*w1 + coeffs.a44*w2 + coeffs.b4*tau
    return np.array([dtheta, dphi, dw1, dw2])

def rk4_step(x, t, h, coeffs):
    k1 = rhs(x, t, coeffs)
    k2 = rhs(x + 0.5*h*k1, t + 0.5*h, coeffs)
    k3 = rhs(x + 0.5*h*k2, t + 0.5*h, coeffs)
    k4 = rhs(x + h*k3, t + h, coeffs)
    return x + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)

def simulate(J0, J2, b1, b2, k, t_final, dt):
    coeffs = build_coeffs(J0, J2, b1, b2, k)
    N = int(t_final/dt) + 1
    t = np.linspace(0.0, t_final, N)
    x = np.zeros(4)  # start at rest
    X = np.zeros((N, 4))
    U = np.zeros(N)
    for i in range(N):
        X[i] = x
        U[i] = (tau_amp if (t[i] < impulse_width) else 0.0)
        if i < N-1:
            x = rk4_step(x, t[i], dt, coeffs)
    return t, X, U, coeffs

def save_csv(t, X, U, path):
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time_s","theta_rad","phi_rad","theta_dot","phi_dot","tau_Nm","theta_plus_phi_rad"])
        for ti, xi, ui in zip(t, X, U):
            w.writerow([ti, xi[0], xi[1], xi[2], xi[3], ui, xi[0]+xi[1]])

def main():
    t, X, U, coeffs = simulate(J0, J2, b1, b2, k, t_final, dt)

    # Plots
    theta = X[:,0]
    phi   = X[:,1]
    abs_pend = theta + phi

    plt.figure()
    plt.plot(t, theta, label="θ (base)")
    plt.plot(t, phi,   label="φ (pend rel base)")
    plt.plot(t, abs_pend, label="θ+φ (pend absolute)")
    plt.xlabel("Time (s)")
    plt.ylabel("Angle (rad)")
    plt.title("Impulse response: angles")
    plt.legend()
    plt.tight_layout()
    plt.savefig("impulse_angles.png", dpi=150)

    plt.figure()
    plt.plot(t, U, label="motor torque τ (N·m)")
    plt.xlabel("Time (s)")
    plt.ylabel("τ (N·m)")
    plt.title("Input torque (impulse approx)")
    plt.tight_layout()
    plt.savefig("impulse_input.png", dpi=150)

    # Save CSV of the trajectory
    save_csv(t, X, U, "impulse_response.csv")

    # Print key numbers
    print("Model coefficients (from M^{-1}):", coeffs)
    print("Files saved: impulse_angles.png, impulse_input.png, impulse_response.csv")

if __name__ == "__main__":
    main()

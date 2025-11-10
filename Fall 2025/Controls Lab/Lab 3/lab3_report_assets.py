#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import StateSpace, lsim

# --- numeric system matrices (from previous derivation) ---
A = np.array([
    [0, 0, 1, 0],
    [0, 0, 0, 1],
    [0, 0.22472748, -0.365, 0.00118],
    [0, -65.56623996, 0.365, -0.34427549]
])
B = np.array([[0], [0], [25], [-25]])
C = np.array([
    [1, 0, 0, 0],     # platen rotation angle θ
    [1, 1, 0, 0]      # pendulum absolute angle θ+φ
])
D = np.zeros((2,1))

# --- simulate system dynamics ---
sys = StateSpace(A, B, C, D)
t = np.linspace(0, 5, 2000)
u = np.ones_like(t)  # step torque input (1 N·m)

tout, y, x = lsim(sys, u, t)

theta = y[:,0]
pendulum = y[:,1]

# --- plot ---
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8,6))
ax1.plot(tout, theta, label='Platen rotation (θ)', color='tab:blue')
ax1.set_ylabel('Angle [rad]')
ax1.legend(loc='upper right')
ax1.grid(True)

ax2.plot(tout, pendulum, label='Pendulum absolute rotation (θ+φ)', color='tab:orange')
ax2.set_xlabel('Time [s]')
ax2.set_ylabel('Angle [rad]')
ax2.legend(loc='upper right')
ax2.grid(True)

fig.suptitle('Pendulum System Step Response')
plt.tight_layout()
plt.show()

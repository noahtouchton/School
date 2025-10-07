# lab3_helpers.py
import os
import numpy as np
from numpy.polynomial import Polynomial

def load_lvm_second_col(path):
    """
    Load a LabVIEW .lvm file that has no header and return the 2nd column (0-based).
    """
    return np.loadtxt(path, usecols=1)

def invert_voltage_from_velocity(a_asc, v_target, v_lo=0.0, v_hi=5.0):
    """
    Solve a0 + a1 V + a2 V^2 + a3 V^3 + a4 V^4 = v_target.
    Returns one real root in [v_lo, v_hi] if available, otherwise np.nan.
    a_asc should be [a0..a4].
    """
    p = Polynomial(a_asc)              # v(V)
    roots = (p - v_target).roots()     # solve v(V)-v_target=0
    real = roots[np.isreal(roots)].real
    mask = (real >= v_lo) & (real <= v_hi)
    if not np.any(mask):
        return np.nan
    # pick the root closest to mid of expected HW range
    guess = 3.0
    cand = real[mask]
    return cand[np.argmin(np.abs(cand - guess))]

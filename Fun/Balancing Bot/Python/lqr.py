"""
LQR gain design for the two-wheeled balancing bot.

Derived from the equations of motion in Dynamics.pdf:

    Frames      G (ground): X up, Y left, Z out of page
                B (chassis): rotated +theta about  Y
                A (axle):    rotated -phi   about  Y

    Sign convention   +theta -> chassis top tips toward -Z
                      +phi   -> bot drives toward +Z
                      +V     -> drives wheels in the +phi sense

    Chassis, moments about the axle Q:
        J_th*thdd - J_c*phidd = Gam*theta + T
    Wheels, after eliminating ground friction:
        J_ph*phidd - J_c*thdd = T
    Motor:
        T = beta*V - gam*(thd + phid)

    Mass matrix is symmetric (J_c on both off-diagonals) -- that is the
    correctness check on the derivation.

Run this on a laptop, not the Arduino. It prints four gains to hardcode.
"""

import numpy as np
from scipy.linalg import solve_continuous_are
from scipy.integrate import solve_ivp

# ============================================================
# 1. PHYSICAL CONSTANTS   (SI throughout: m, kg, s, rad, V, A, ohm)
# ============================================================

g = 9.81            # gravity                                (m/s^2)

# --- geometry -----------------------------------------------------------
Rw = 0.03382        # wheel radius                           (m)
rcx = 0.098         # chassis COM height above floor         (m)
h = 0.154           # accelerometer height above floor       (m)  [sensor model only]

dx = rcx - Rw       # axle -> COM lever arm                  (m)

# --- motor --------------------------------------------------------------
R = 5.5             # armature resistance                    (ohm)
kt = 0.28           # torque constant                        (N*m/A)
ke = 0.28           # back-EMF constant                      (V*s/rad)
# NOTE: kt and ke are the SAME physical constant in SI (tau*w = V_emf*I).
# The old file had ke = 0.56, which is a factor-of-2 inconsistency.
# 0.28 is corroborated by TT gearmotor no-load speed (~200 RPM @ 6 V).
# Confirm by spinning a motor at a known rate and measuring open-circuit V.

n_motors = 2
N_gear = 1.0        # set to 1.0 if kt/ke are referred to the OUTPUT shaft

V_bat = 7.4         # battery voltage -> control saturation limit (V)

# --- masses and inertias ------------------------------------------------
# >>> THESE THREE ARE THE WEAK LINK. Measure them. <<<
m = 0.26311             # chassis mass                           (kg)  MEASURE
M = 0.02532*2             # total wheel mass (both)                (kg)  MEASURE

# Chassis inertia about its OWN center of mass.
# The inertiaCalc.ino trial measures inertia about the AXLE, so if you have
# that number instead, use:  I_C = I_axle_measured - m*dx**2
#
# The stored 0.869964 kg*m^2 is physically impossible for this bot: with
# dx = 0.064 m and m ~ 0.5 kg you expect ~2e-3. Reaching 0.87 would need the
# mass 1.3 m from the axle; the whole robot is 0.154 m tall. Re-run the trial.
I_C = 0.002         # chassis inertia about COM              (kg*m^2)  MEASURE

I_w = 0.5 * M * Rw**2   # wheel pair inertia about axle, solid-disc estimate


# ============================================================
# 2. DERIVED COEFFICIENTS
# ============================================================

J_th = I_C + m * dx**2              # chassis inertia about the axle
J_c = m * dx * Rw                   # theta <-> phi coupling
J_ph = I_w + (m + M) * Rw**2        # effective wheel inertia
Gam = m * g * dx                    # gravitational (destabilizing) term

beta = n_motors * N_gear * kt / R                   # torque per volt
gam = n_motors * (N_gear**2) * kt * ke / R          # back-EMF damping

Delta = J_th * J_ph - J_c**2        # mass-matrix determinant
assert Delta > 0, "Delta <= 0: mass matrix not positive definite, check J's"

p = (J_ph + J_c) / Delta            # T coefficient in thetadd
q = (J_th + J_c) / Delta            # T coefficient in phidd


# ============================================================
# 3. STATE SPACE     x = [theta, thetadot, phi, phidot],  u = V
# ============================================================

A = np.array([
    [0.0,            1.0,       0.0,   0.0     ],
    [J_ph * Gam / Delta, -p * gam, 0.0, -p * gam],
    [0.0,            0.0,       0.0,   1.0     ],
    [J_c * Gam / Delta,  -q * gam, 0.0, -q * gam],
])

B = np.array([[0.0],
              [p * beta],
              [0.0],
              [q * beta]])


def ctrb(A, B):
    n = A.shape[0]
    return np.hstack([np.linalg.matrix_power(A, i) @ B for i in range(n)])


def check_structure(A):
    """Two properties that must hold by construction.

    Note the damping check compares only the DYNAMICS rows (1 and 3).
    Rows 0 and 2 are the trivial kinematic rows (d/dt theta = thetadot),
    which carry a 1 in one column and a 0 in the other by definition.
    """
    ok_col = np.allclose(A[:, 2], 0.0)
    ok_damp = np.allclose(A[[1, 3], 1], A[[1, 3], 3])
    print(f"  phi column is zero        : {ok_col}   (position drives nothing)")
    print(f"  damping cols match (r1,r3): {ok_damp}  (damping sees only thd+phid)")
    return ok_col and ok_damp


# ============================================================
# 4. COST WEIGHTS   (Bryson: penalty = 1 / tolerance^2)
# ============================================================

tol_theta = 0.10    # rad     (~6 deg)
tol_thdot = 1.0     # rad/s
tol_phi = 2.0       # rad
tol_phidot = 10.0   # rad/s

rho = 1.0           # <<< the single tuning dial: sweep 0.1 / 1 / 10 / 100

Q = np.diag([1 / tol_theta**2,
             1 / tol_thdot**2,
             1 / tol_phi**2,
             1 / tol_phidot**2])

Rc = np.array([[rho / V_bat**2]])


# ============================================================
# 5. SOLVE + VERIFY
# ============================================================

def design(A, B, Q, Rc, label):
    print(f"\n=== {label} ===")

    ol = np.linalg.eigvals(A)
    print(f"  open-loop eigenvalues     : {np.round(ol, 3)}")
    unstable = ol[np.real(ol) > 0]
    if len(unstable) == 0:
        print("  !! NO unstable pole. An inverted pendulum must have one.")
        print("     A sign is flipped somewhere in the derivation.")
    else:
        tau = 1.0 / np.max(np.real(unstable))
        print(f"  fall time constant        : {tau*1000:.0f} ms")

    rank = np.linalg.matrix_rank(ctrb(A, B))
    print(f"  controllability rank      : {rank} / {A.shape[0]}")
    if rank < A.shape[0]:
        print("  !! not controllable -- LQR cannot stabilize this")
        return None

    P = solve_continuous_are(A, B, Q, Rc)
    K = np.linalg.inv(Rc) @ B.T @ P

    cl = np.linalg.eigvals(A - B @ K)
    print(f"  closed-loop eigenvalues   : {np.round(cl, 2)}")
    print(f"  stable                    : {bool(np.all(np.real(cl) < 0))}")

    fastest = np.max(np.abs(cl))
    print(f"  fastest pole              : {fastest:.1f} rad/s"
          f"   (keep well under {2*np.pi/0.005:.0f} at 5 ms loop)")

    # saturation: what does a 6 deg lean actually command?
    x_test = np.zeros(A.shape[0])
    x_test[0] = tol_theta
    v = (K @ x_test).item()
    print(f"  command at {np.degrees(tol_theta):.0f} deg lean   : {v:+.2f} V"
          f"   (rail {V_bat} V) {'*** SATURATES ***' if abs(v) > V_bat else 'ok'}")

    return K


def sweep_rho(A, B, Q, label, dt=0.005, rhos=(0.1, 1, 10, 100, 1000, 1e4, 1e5)):
    """Find rho values that neither saturate the rail nor outrun the loop rate.

    Two hard limits:
      - |V| at the tolerance lean must fit inside V_bat, else the LQR
        optimality guarantee is void and you are running bang-bang.
      - The fastest closed-loop pole must be well below the Nyquist rate of
        the discrete loop. Rule of thumb: under 1/5 of 2*pi/dt.
    """
    pole_cap = 0.2 * 2 * np.pi / dt
    x_test = np.zeros(A.shape[0])
    x_test[0] = tol_theta

    print(f"\n=== rho sweep: {label} ===")
    print(f"  {'rho':>8}  {'V @ lean':>9}  {'fastest pole':>13}   verdict")
    feasible = []
    for r in rhos:
        Rc_r = np.array([[r / V_bat**2]])
        P = solve_continuous_are(A, B, Q, Rc_r)
        K = np.linalg.inv(Rc_r) @ B.T @ P
        v = abs((K @ x_test).item())
        fast = np.max(np.abs(np.linalg.eigvals(A - B @ K)))

        bad = []
        if v > V_bat:
            bad.append("saturates")
        if fast > pole_cap:
            bad.append("too fast for loop")
        verdict = "OK" if not bad else " + ".join(bad)
        if not bad:
            feasible.append((r, K))
        print(f"  {r:>8.4g}  {v:>8.2f} V  {fast:>10.0f} r/s   {verdict}")

    print(f"  (pole cap {pole_cap:.0f} rad/s at dt = {dt*1000:.0f} ms)")
    return feasible


def simulate(A, B, K, label, theta0=0.10, t_end=3.0):
    """Integrate the closed loop from an initial lean."""
    Acl = A - B @ K
    x0 = np.zeros(A.shape[0])
    x0[0] = theta0
    sol = solve_ivp(lambda t, x: Acl @ x, [0, t_end], x0, max_step=0.005)

    v = -(K @ sol.y).ravel()
    theta = sol.y[0]
    settled = np.where(np.abs(theta) < np.radians(1.0))[0]
    t_settle = sol.t[settled[0]] if len(settled) else float("nan")

    print(f"\n  --- sim ({label}) from {np.degrees(theta0):.0f} deg ---")
    print(f"  peak |theta|              : {np.degrees(np.max(np.abs(theta))):.1f} deg")
    print(f"  settle to 1 deg           : {t_settle*1000:.0f} ms")
    print(f"  peak |V|                  : {np.max(np.abs(v)):.2f} V"
          f" {'*** over rail ***' if np.max(np.abs(v)) > V_bat else ''}")


if __name__ == "__main__":
    np.set_printoptions(suppress=True)

    print("derived coefficients")
    print(f"  dx    = {dx*1000:.1f} mm      (axle -> COM)")
    print(f"  J_th  = {J_th:.3e}   J_c = {J_c:.3e}   J_ph = {J_ph:.3e}")
    print(f"  Gam   = {Gam:.4f}    Delta = {Delta:.3e}")
    print(f"  beta  = {beta:.4f} N*m/V   gam = {gam:.4f} N*m*s/rad")

    print("\nstructural checks")
    check_structure(A)

    # ---- full 4-state (needs encoders) ----
    K4 = design(A, B, Q, Rc, "4-STATE  [theta, thd, phi, phid]  -- requires encoders")
    if K4 is not None:
        simulate(A, B, K4, "4-state")
        print(f"\n  K = {np.round(K4.ravel(), 4)}")

    # ---- reduced 2-state (IMU only) ----
    # Dropping phi is exact (zero column). Dropping phidot loses the back-EMF
    # damping, but J_c is already folded into p and Delta by the 2x2 solve.
    A2 = A[:2, :2]
    B2 = B[:2]
    Q2 = Q[:2, :2]

    K2 = design(A2, B2, Q2, Rc, "2-STATE  [theta, thd]  -- IMU only, drifts")
    if K2 is not None:
        simulate(A2, B2, K2, "2-state")
        print(f"\n  K = {np.round(K2.ravel(), 4)}")

    # ---- pick a usable rho ----
    feasible = sweep_rho(A2, B2, Q2, "2-state")
    if feasible:
        r, K = feasible[0]
        k = K.ravel()
        print(f"\n=== RECOMMENDED (rho = {r:g}) ===")
        simulate(A2, B2, K, f"2-state, rho={r:g}")
        print("\n  --- paste into the Arduino sketch ---")
        print(f"  const float K_THETA = {k[0]:.4f}f;   // V per rad")
        print(f"  const float K_RATE  = {k[1]:.4f}f;   // V per rad/s")
        print("  // V = -(K_THETA*theta + K_RATE*thetaDot);   theta in RADIANS")
    else:
        print("\n  !! no rho in the sweep satisfies both limits.")
        print("     Widen tol_theta, slow the loop, or re-check the constants.")

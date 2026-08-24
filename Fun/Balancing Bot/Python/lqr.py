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
from scipy.linalg import solve_continuous_are, solve_discrete_are
from scipy.integrate import solve_ivp
from scipy.signal import cont2discrete

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

V_bat = 5.4         # battery voltage -> control saturation limit (V)

# --- masses and inertias ------------------------------------------------
# >>> THESE THREE ARE THE WEAK LINK. Measure them. <<<
m = 0.26311             # chassis mass                           (kg)  MEASURE
M = 0.02532*2             # total wheel mass (both)                (kg)  MEASURE

# --- chassis inertia, from the physical-pendulum swing test ---------------
# Rod through the axle, wheels locked to the rod (so they neither rotate nor
# load the pivot). The swinging body is the chassis alone, pivoting about the
# axle -- which measures J_th DIRECTLY, no parallel-axis subtraction:
#
#     T = 2*pi*sqrt(J_th / (m*g*dx))   ->   J_th = m*g*dx*(T/2pi)^2
#
# This replaces the old inertiaCalc.ino motor-pulse trial, which returned
# 0.869964 kg*m^2 -- off by a factor of ~1800 (that value implies the mass
# sits 1.3 m from the axle; the whole robot is 0.154 m tall).
swing_count = 7         # number of full swings timed
swing_time = 4.28       # total elapsed for those swings          (s)

T_swing = swing_time / swing_count
J_th_meas = m * g * dx * (T_swing / (2 * np.pi))**2

I_C = J_th_meas - m * dx**2     # back out COM inertia, for reference only

I_w = 0.5 * M * Rw**2   # wheel pair inertia about axle, solid-disc estimate


# ============================================================
# 2. DERIVED COEFFICIENTS
# ============================================================

J_th = J_th_meas                    # chassis inertia about the axle (measured)
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


def sweep_rho(A, B, Q, label, dt=0.005, rhos=(0.1, 1, 10, 100, 1000, 1e4, 1e5),
              A_full=None, B_full=None):
    """Find rho values that pass every hard limit.

    Three checks:
      - |V| at the tolerance lean must fit inside V_bat, else the LQR
        optimality guarantee is void and you are running bang-bang.
      - The fastest closed-loop pole must be well below the Nyquist rate of
        the discrete loop. Rule of thumb: under 1/5 of 2*pi/dt.
      - If designing on the reduced model, the resulting gains must still
        stabilize the FULL plant. The 2-state model overestimates damping
        (it applies all of gam*(thd+phid) to thd alone), so it thinks the bot
        falls ~3x slower than it really does. Gains that look fine against
        the reduced model can be too weak for the real system -- this is the
        check that catches it.

    Pass A_full / B_full as the BALANCE subsystem [theta, thd, phid], not the
    raw 4-state. The phi column of A is identically zero, so the full closed
    loop always carries an eigenvalue at exactly 0 -- that is wheel position
    integrating freely (the bot drifts), which no static feedback on theta
    alone can remove and which is not a balancing instability. Deleting phi
    is exact, so the 3-state subsystem is the right thing to test.
    """
    pole_cap = 0.2 * 2 * np.pi / dt
    x_test = np.zeros(A.shape[0])
    x_test[0] = tol_theta
    cross = A_full is not None and B_full is not None

    print(f"\n=== rho sweep: {label} ===")
    hdr = f"  {'rho':>8}  {'V @ lean':>9}  {'fastest pole':>13}"
    print(hdr + ("   full-plant   verdict" if cross else "   verdict"))
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

        full_txt = ""
        if cross:
            # embed the reduced gains into the full state vector (zeros on
            # the states this controller cannot measure)
            K_emb = np.zeros((1, A_full.shape[0]))
            K_emb[0, :K.shape[1]] = K
            ok_full = bool(np.all(np.real(np.linalg.eigvals(
                A_full - B_full @ K_emb)) < 0))
            full_txt = f"   {'stable' if ok_full else 'UNSTABLE':>9}"
            if not ok_full:
                bad.append("unstable on real plant")

        verdict = "OK" if not bad else " + ".join(bad)
        if not bad:
            feasible.append((r, K))
        print(f"  {r:>8.4g}  {v:>8.2f} V  {fast:>10.0f} r/s{full_txt}   {verdict}")

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


def discrete_ok(Ap, Bp, K, dt=0.005):
    """Honest sampled-data check: ZOH-discretize the plant, close the loop
    with the same K, and require every discrete pole inside the unit circle.

    This replaces the "fastest pole << 2*pi/dt" rule of thumb, which is too
    conservative for real, heavily damped modes. This plant has an open-loop
    pole near -313 rad/s (the back-EMF mode); at dt = 5 ms that maps to
    z = 0.21, perfectly well behaved, even though the rule of thumb rejects it.
    """
    n = Ap.shape[0]
    Ad, Bd, *_ = cont2discrete((Ap, Bp, np.eye(n), np.zeros((n, 1))), dt)
    return float(np.max(np.abs(np.linalg.eigvals(Ad - Bd @ K))))


# ============================================================
# 6. OBSERVER  --  estimate phidot from the IMU alone
# ============================================================
# The balance subsystem is observable from theta (rank 3/3), so a Luenberger
# / Kalman observer can in principle reconstruct wheel speed without an
# encoder, and u = -K*xhat then stabilizes. That is LQG.
#
# The catch is that the observer INFERS phidot from the lean dynamics, so it
# is only as good as beta, gam, and everything unmodeled (gearbox Coulomb
# friction, wheel slip). Those errors accumulate on the same multi-second
# timescale as the momentum mode the phidot feedback exists to arrest.
# The robustness sweep below quantifies how much mismatch it survives.

def build_bal(kt_s=1.0, Jth_s=1.0):
    """Rebuild the 3-state balance plant [theta, thetadot, phidot] with the
    motor constant and/or chassis inertia scaled. Used to make a 'true' plant
    that differs from the one the observer believes in."""
    kt_, ke_ = kt * kt_s, ke * kt_s
    J_th_ = J_th * Jth_s
    beta_ = n_motors * N_gear * kt_ / R
    gam_ = n_motors * (N_gear**2) * kt_ * ke_ / R
    Delta_ = J_th_ * J_ph - J_c**2
    p_ = (J_ph + J_c) / Delta_
    q_ = (J_th_ + J_c) / Delta_
    Ab = np.array([
        [0.0,                 1.0,        0.0     ],
        [J_ph * Gam / Delta_, -p_ * gam_, -p_ * gam_],
        [J_c * Gam / Delta_,  -q_ * gam_, -q_ * gam_],
    ])
    Bb = np.array([[0.0], [p_ * beta_], [q_ * beta_]])
    return Ab, Bb


def design_observer(Ap, Bp, C, W, V):
    """Kalman gain from the dual Riccati equation.

    scipy solves   a'X + Xa - Xb r^-1 b'X + q = 0.
    Feeding a = A', b = C' turns that into the filter ARE
        AP + PA' - PC'V^-1 CP + W = 0,   L = PC'V^-1
    """
    P = solve_continuous_are(Ap.T, C.T, W, V)
    return P @ C.T @ np.linalg.inv(V)


def augmented(A_true, B_true, A_est, B_est, K, L, C):
    """Closed loop when the observer's model != the real plant.

        xdot    = A_true x - B_true K xhat
        xhatdot = LC x + (A_est - B_est K - LC) xhat

    With a perfect model this block-triangularizes and the separation
    principle applies (poles = eig(A-BK) U eig(A-LC)). With mismatch it
    does not, so the full 6x6 has to be checked directly.
    """
    return np.block([
        [A_true,  -B_true @ K],
        [L @ C,    A_est - B_est @ K - L @ C],
    ])


if __name__ == "__main__":
    np.set_printoptions(suppress=True)

    print("derived coefficients")
    print(f"  dx    = {dx*1000:.1f} mm      (axle -> COM)")
    print(f"  T_swing = {T_swing:.4f} s  ->  J_th = {J_th_meas:.4e} kg*m^2")
    print(f"  (implied I_C = {I_C:.3e}; thin-plate estimate ~5e-4)")
    print(f"  J_th  = {J_th:.3e}   J_c = {J_c:.3e}   J_ph = {J_ph:.3e}")
    print(f"  Gam   = {Gam:.4f}    Delta = {Delta:.3e}")
    print(f"  beta  = {beta:.4f} N*m/V   gam = {gam:.4f} N*m*s/rad")

    print("\nstructural checks")
    check_structure(A)

    # ------------------------------------------------------------------
    # The balance subsystem [theta, thetadot, phidot].
    # phi is deleted exactly (its column in A is zero -- wheel POSITION
    # drives no dynamics). What remains is what actually has to be
    # stabilized: lean, lean rate, and wheel speed.
    # ------------------------------------------------------------------
    bal = [0, 1, 3]
    A_bal = A[np.ix_(bal, bal)]
    B_bal = B[bal]
    Q_bal = np.diag([1 / tol_theta**2, 1 / tol_thdot**2, 1 / tol_phidot**2])

    # ==================================================================
    # CASE 1: IMU only  -- feedback on [theta, thetadot], K_phidot = 0
    # ==================================================================
    print("\n" + "=" * 62)
    print("CASE 1: IMU ONLY  (no encoders)  --  NOT STABILIZABLE")
    print("=" * 62)
    print("""
  The motor torque T is INTERNAL: it acts between chassis and wheels, so
  it cannot change the system's angular momentum about the contact point.
  Only gravity can, and gravity's moment goes as theta.

  Concretely, w = q*thetadot - p*phidot has BOTH the input and the back-EMF
  cancel exactly (q*p*beta - p*q*beta = 0), leaving  wdot = (Gam/Delta)*theta.
  V has zero direct authority over w.

  Consequence: the constant term of the closed-loop characteristic
  polynomial is independent of k1 and k2, so one root always sits in the
  right half plane. Raising the gains only slows the runaway.""")

    print(f"\n  {'k1':>7}{'k2':>7}{'unstable pole':>15}{'runaway tau':>13}"
          f"{'V @ 1.5deg':>12}")
    for k1 in [25, 50, 100, 200, 400, 800]:
        k2 = k1 / 10.0
        Kt = np.array([[k1, k2, 0.0]])
        mre = float(np.max(np.real(np.linalg.eigvals(A_bal - B_bal @ Kt))))
        rail = "  <-- over rail" if k1 * 0.026 > V_bat else ""
        print(f"  {k1:>7}{k2:>7.1f}{mre:>15.3f}{1/mre:>12.1f}s"
              f"{k1*0.026:>11.1f}V{rail}")
    print(f"\n  Within the {V_bat} V rail you can reach a runaway time constant of only")
    print("  a few seconds. That is exactly how encoder-less balancers behave:")
    print("  they hold for a moment, then accelerate away. Useful for a first")
    print("  bring-up and for checking signs, but it is not a working robot.")

    # ==================================================================
    # CASE 2: add wheel-SPEED feedback -- the real design
    # ==================================================================
    print("\n" + "=" * 62)
    print("CASE 2: + WHEEL SPEED  [theta, thetadot, phidot]  --  STABILIZABLE")
    print("=" * 62)
    print("  Note this needs wheel VELOCITY only, not absolute position.")

    rank = np.linalg.matrix_rank(ctrb(A_bal, B_bal))
    print(f"\n  controllability rank : {rank} / 3")

    print(f"\n  {'rho':>8}{'K_theta':>10}{'K_rate':>9}{'K_wheel':>9}"
          f"{'slowest pole':>14}{'V@5.7deg':>10}{'max|z|':>9}")
    best = None
    for r in [1, 10, 100, 1000, 1e4]:
        Rc_r = np.array([[r / V_bat**2]])
        P = solve_continuous_are(A_bal, B_bal, Q_bal, Rc_r)
        K = np.linalg.inv(Rc_r) @ B_bal.T @ P
        ev = np.linalg.eigvals(A_bal - B_bal @ K)
        k = K.ravel()
        v = abs(k[0] * tol_theta)
        z = discrete_ok(A_bal, B_bal, K)
        flag = ""
        if v <= V_bat and z < 1.0 and best is None:
            best = (r, K)
            flag = "  <-- recommended"
        print(f"  {r:>8.4g}{k[0]:>10.2f}{k[1]:>9.2f}{k[2]:>9.3f}"
              f"{max(np.real(ev)):>14.2f}{v:>9.2f}V{z:>9.3f}{flag}")

    print(f"\n  (max|z| is the discrete closed-loop spectral radius at "
          f"{1/0.005:.0f} Hz; must be < 1)")

    if best is not None:
        r, K = best
        k = K.ravel()
        simulate(A_bal, B_bal, K, f"3-state, rho={r:g}")
        print("\n  --- paste into the Arduino sketch ---")
        print(f"  const float K_THETA = {k[0]:.4f}f;   // V per rad")
        print(f"  const float K_RATE  = {k[1]:.4f}f;   // V per rad/s")
        print(f"  const float K_WHEEL = {k[2]:.4f}f;   // V per rad/s of wheel")
        print("  // V = -(K_THETA*theta + K_RATE*thetaDot + K_WHEEL*phiDot);")
        print("  // theta in RADIANS, rates in RAD/S")

    # ==================================================================
    # CASE 3: observer -- estimate phidot from the IMU, no encoder
    # ==================================================================
    print("\n" + "=" * 62)
    print("CASE 3: OBSERVER  (LQG)  --  phidot ESTIMATED, no encoder")
    print("=" * 62)

    C = np.array([[1.0, 0.0, 0.0],       # theta   (complementary filter)
                  [0.0, 1.0, 0.0]])      # thetadot (gyro, direct)

    # Measurement noise: the accel-derived theta is the bad one -- it is
    # corrupted by linear acceleration exactly when the bot is correcting.
    sig_theta = np.radians(2.0)          # rad
    sig_rate = np.radians(0.5)           # rad/s
    Vn = np.diag([sig_theta**2, sig_rate**2])

    # Process noise: put most of it on phidot, since that is where the
    # unmodeled physics (friction, slip) actually enters.
    ctrl_pole = float(np.max(np.real(np.linalg.eigvals(A_bal - B_bal @ K))))
    print(f"\n  {'w_phid':>8}{'slowest obs':>13}{'slowest CL':>12}   note")
    for w_phid in [0.03, 0.1, 0.3, 1.0, 3.0, 10.0]:
        W = np.diag([1e-6, 1e-4, w_phid])
        L = design_observer(A_bal, B_bal, C, W, Vn)
        obs = float(np.max(np.real(np.linalg.eigvals(A_bal - L @ C))))
        M = augmented(*build_bal(), A_bal, B_bal, K, L, C)
        cl = float(np.max(np.real(np.linalg.eigvals(M))))
        note = "observer limits" if obs > ctrl_pole else "controller limits (good)"
        print(f"  {w_phid:>8.2f}{obs:>13.2f}{cl:>12.2f}   {note}")
    print(f"\n  control poles sit at {ctrl_pole:.2f}. Low process noise makes the")
    print("  observer fast enough that the CONTROLLER sets the closed-loop speed --")
    print("  i.e. estimating phidot costs nothing nominally. High process noise")
    print("  slows the weakly-observable mode and the observer becomes the limit.")

    # ---- robustness to model error ----
    W_PHID = 0.1        # <<< observer tuning dial
    DT_LOOP = 0.005     # Arduino loop period -- discrete design is tied to it
    W = np.diag([1e-6, 1e-4, W_PHID])
    L = design_observer(A_bal, B_bal, C, W, Vn)
    print(f"\n  using w_phid = {W_PHID}")
    print(f"  L =\n{np.round(L, 4)}")

    print("\n  --- tolerance to model error (observer keeps the nominal model) ---")
    print(f"  {'kt error':>10}{'J_th error':>12}{'max Re':>10}   verdict")
    for kt_s, Jth_s in [(1.0, 1.0), (1.2, 1.0), (0.8, 1.0), (1.5, 1.0),
                        (0.5, 1.0), (1.0, 1.2), (1.0, 0.8), (1.3, 1.2)]:
        At, Bt = build_bal(kt_s, Jth_s)
        M = augmented(At, Bt, A_bal, B_bal, K, L, C)
        mre = float(np.max(np.real(np.linalg.eigvals(M))))
        print(f"  {(kt_s-1)*100:>+9.0f}%{(Jth_s-1)*100:>+11.0f}%{mre:>10.2f}"
              f"   {'stable' if mre < 0 else 'UNSTABLE'}")

    print("\n  Compare against CASE 2: with a real encoder, none of this")
    print("  matters -- the wheel speed is measured, not inferred.")

    # ==================================================================
    # CASE 4: DISCRETE design -- what actually ships to the Arduino
    # ==================================================================
    print("\n" + "=" * 62)
    print(f"CASE 4: DISCRETE-TIME DESIGN at dt = {DT_LOOP*1000:.0f} ms")
    print("=" * 62)
    print("""
  The continuous gains above CANNOT be forward-Euler integrated on the
  Arduino. The plant has a back-EMF pole near -313 rad/s; at 5 ms that
  gives lambda*dt = -1.57, and the Euler-integrated closed loop comes out
  at |z| = 2.58 -- violently unstable, despite a perfectly good continuous
  design. Discretize first, then design. No integration error at all.""")

    Ad, Bd, *_ = cont2discrete(
        (A_bal, B_bal, np.eye(3), np.zeros((3, 1))), DT_LOOP)

    print(f"\n  {'rho':>6}{'K_theta':>10}{'K_rate':>9}{'K_wheel':>10}"
          f"{'|z| ctrl':>10}{'V@5.7deg':>10}")
    for r in [1, 10, 100]:
        Rd = np.array([[r / V_bat**2]])
        Pd = solve_discrete_are(Ad, Bd, Q_bal, Rd)
        Kd = np.linalg.inv(Rd + Bd.T @ Pd @ Bd) @ (Bd.T @ Pd @ Ad)
        zc = float(np.max(np.abs(np.linalg.eigvals(Ad - Bd @ Kd))))
        print(f"  {r:>6}{Kd[0,0]:>10.4f}{Kd[0,1]:>9.4f}{Kd[0,2]:>10.4f}"
              f"{zc:>10.4f}{abs(Kd[0,0]*tol_theta):>9.2f}V")

    RHO_D = 10
    Rd = np.array([[RHO_D / V_bat**2]])
    Pd = solve_discrete_are(Ad, Bd, Q_bal, Rd)
    Kd = np.linalg.inv(Rd + Bd.T @ Pd @ Bd) @ (Bd.T @ Pd @ Ad)

    # discrete Kalman PREDICTOR gain, matching
    #   xhat[k+1] = Ad xhat + Bd u + L(y - C xhat)
    Wd = np.diag([1e-6, 1e-4, W_PHID]) * DT_LOOP
    Sd = solve_discrete_are(Ad.T, C.T, Wd, Vn)
    Ld = Ad @ Sd @ C.T @ np.linalg.inv(C @ Sd @ C.T + Vn)

    Mfull = np.block([[Ad, -Bd @ Kd], [Ld @ C, Ad - Bd @ Kd - Ld @ C]])
    print(f"\n  rho = {RHO_D}, w_phid = {W_PHID}")
    print(f"  observer |z|          : "
          f"{np.max(np.abs(np.linalg.eigvals(Ad - Ld @ C))):.4f}")
    print(f"  FULL sampled loop |z| : "
          f"{np.max(np.abs(np.linalg.eigvals(Mfull))):.4f}   (must be < 1)")

    print("\n  robustness (observer keeps the nominal model):")
    for s in [1.5, 1.2, 0.8, 0.5]:
        At, Bt = build_bal(s, 1.0)
        Adt, Bdt, *_ = cont2discrete(
            (At, Bt, np.eye(3), np.zeros((3, 1))), DT_LOOP)
        Mt = np.block([[Adt, -Bdt @ Kd], [Ld @ C, Ad - Bd @ Kd - Ld @ C]])
        z = float(np.max(np.abs(np.linalg.eigvals(Mt))))
        print(f"    kt x{s:<4} -> |z| = {z:.4f}  "
              f"{'stable' if z < 1 else 'UNSTABLE'}")

    print("\n  ---------- paste into Arduino/balance/balance.ino ----------")
    print(f"  const float DT = {DT_LOOP}f;")
    print(f"  const float K_THETA = {Kd[0,0]:.4f}f;")
    print(f"  const float K_RATE  = {Kd[0,1]:.4f}f;")
    print(f"  const float K_WHEEL = {Kd[0,2]:.4f}f;")
    for i in range(3):
        print(f"  const float Ad{i}0 = {Ad[i,0]:.6f}f, "
              f"Ad{i}1 = {Ad[i,1]:.6f}f, Ad{i}2 = {Ad[i,2]:.6f}f;")
    print(f"  const float Bd0  = {Bd[0,0]:.6f}f, "
          f"Bd1  = {Bd[1,0]:.6f}f, Bd2  = {Bd[2,0]:.6f}f;")
    for i in range(3):
        print(f"  const float L{i}0 = {Ld[i,0]:.6f}f, L{i}1 = {Ld[i,1]:.6f}f;")

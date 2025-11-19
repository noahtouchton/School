import numpy as np
from numpy.linalg import eig
from scipy.linalg import solve_continuous_are
from scipy.integrate import solve_ivp

# =========================
# 1. System definition
# =========================

def get_AB():
    # Your parameters
    Rm = 16.0
    Km = 0.525
    L1 = 0.1231
    m2 = 0.00488
    J2 = 0.0001374
    l2 = 0.1472

    b1 = 0.0146
    J0 = 0.00224
    b2 = 4.72e-5

    tau = 1.263e-3
    Lm = Rm * tau
    g  = -9.8

    delta = J0*J2 - (m2**2)*(L1**2)*(l2**2)

    B31_t =  J2/delta
    B41_t =  m2*L1*l2/delta

    A31 = 0.0
    A32 = g*m2**2*l2**2*L1/delta
    A33 = -b1*J2/delta   - (Km**2/Rm)*B31_t
    A34 = -b2*m2*l2*L1/delta

    A41 = 0.0
    A42 = g*m2*l2*J0/delta
    A43 = -b1*m2*l2*L1/delta - (Km**2/Rm)*B41_t
    A44 = -b2*J0/delta

    A = np.array([
        [0,   0,   1,      0,           0],
        [0,   0,   0,      1,           0],
        [A31, A32, A33,    A34,    B31_t*Km],
        [A41, A42, A43,    A44,    B41_t*Km],
        [0,   0,  -Km/Lm,  0,     -Rm/Lm   ]
    ], dtype=float)

    B = np.array([
        [0],
        [0],
        [0],
        [0],
        [1/Lm]
    ], dtype=float)

    return A, B, Km, Lm, Rm

# =========================
# 2. LQR helper
# =========================

def lqr(A, B, Q, R):
    """Continuous-time LQR."""
    P = solve_continuous_are(A, B, Q, R)
    K = np.linalg.inv(R) @ B.T @ P
    return K

# =========================
# 3. Closed-loop simulation
# =========================

def base_reference(t, A_deg=10.0, freq=0.1):
    """Square-wave reference for theta1 in radians."""
    # simple square wave: sign(sin(2π f t))
    s = np.sign(np.sin(2*np.pi*freq*t))
    theta1_ref = np.deg2rad(A_deg) * s
    return theta1_ref

def closed_loop_dynamics(t, x, A, B, K, A_deg, freq, phase):
    """
    Continuous-time closed-loop dynamics:
    u = -K (x - x_ref(t)), with x_ref = [theta1_ref, 0, 0, 0, 0]^T
    phase = 1: balancing only → theta1_ref = 0
    phase = 2: tracking phase   → square wave
    """
    theta1 = x[0]
    theta2 = x[1]
    dtheta1 = x[2]
    dtheta2 = x[3]
    i = x[4]

    if phase == 1:
        theta1_ref = 0.0
    else:
        theta1_ref = base_reference(t, A_deg=A_deg, freq=freq)

    x_ref = np.array([theta1_ref, 0, 0, 0, 0])
    e = x - x_ref

    u = float(-K @ e)   # voltage

    # Saturate to ±12 V (hardware constraint)
    UMAX = 12.0
    u = max(min(u, UMAX), -UMAX)

    dx = A @ x + B.flatten() * u
    return dx

# =========================
# 4. Cost function
# =========================

def simulate_and_cost(A, B, K,
                      A_deg_track=10.0,
                      freq_track=0.1,
                      T1=2.0, T2=8.0):
    """
    Simulate two phases:
    - Phase 1: 0..T1   (balancing from small perturbation)
    - Phase 2: T1..T1+T2  (base tracking a square wave)
    Return scalar cost J to minimize.
    """

    # Phase 1 initial condition: small perturbation
    x0 = np.array([0.0, np.deg2rad(5.0), 0.0, 0.0, 0.0])  # 5° pendulum offset

    # Integrate phase 1
    def f1(t, x):
        return closed_loop_dynamics(t, x, A, B, K, A_deg_track, freq_track, phase=1)

    sol1 = solve_ivp(f1, [0, T1], x0, max_step=0.001, rtol=1e-6, atol=1e-8)
    t1 = sol1.t
    X1 = sol1.y.T

    # Phase 2 initial condition = last state of phase 1
    x1_end = X1[-1, :]

    def f2(t, x):
        return closed_loop_dynamics(t, x, A, B, K, A_deg_track, freq_track, phase=2)

    sol2 = solve_ivp(f2, [T1, T1+T2], x1_end, max_step=0.001, rtol=1e-6, atol=1e-8)
    t2 = sol2.t
    X2 = sol2.y.T

    # Combine
    t = np.concatenate([t1, t2])
    X = np.vstack([X1, X2])

    # Compute cost
    theta1 = X[:, 0]
    theta2 = X[:, 1]
    dtheta1 = X[:, 2]
    dtheta2 = X[:, 3]
    i = X[:, 4]

    # Reconstruct u(t) for cost:
    U = []
    for k in range(len(t)):
        tt = t[k]
        xx = X[k, :]
        if tt <= T1:
            theta1_ref = 0.0
        else:
            theta1_ref = base_reference(tt, A_deg=A_deg_track, freq=freq_track)
        x_ref = np.array([theta1_ref, 0, 0, 0, 0])
        e = xx - x_ref
        u = float(-K @ e)
        u = max(min(u, 12.0), -12.0)
        U.append(u)
    U = np.array(U)

    # Weights for performance cost (not the same as Q!)
    w_theta2 = 50.0     # upright priority
    w_theta1_err = 5.0  # base tracking
    w_vel = 0.5
    w_i = 0.5
    w_u = 0.1

    # base reference signal over full time
    theta1_ref_full = np.zeros_like(theta1)
    for k, tt in enumerate(t):
        if tt > T1:
            theta1_ref_full[k] = base_reference(tt, A_deg=A_deg_track, freq=freq_track)

    theta1_err = theta1 - theta1_ref_full

    # Fall penalty: if |theta2| ever exceeds 45°, huge cost
    if np.any(np.abs(theta2) > np.deg2rad(45)):
        return 1e9

    # Time integration via Riemann sum
    dt = np.mean(np.diff(t))
    J = dt * np.sum(
        w_theta2 * theta2**2 +
        w_theta1_err * theta1_err**2 +
        w_vel * (dtheta1**2 + dtheta2**2) +
        w_i * i**2 +
        w_u * (U**2)
    )
    return float(J)

# =========================
# 5. Parameterization of Q,R
# =========================

def qr_from_params(params):
    """
    params: array of 6 values = [log10(q1), log10(q2), log10(q3), log10(q4), log10(q5), log10(R)]
    returns: Q (5x5), R (1x1)
    """
    q_logs = params[:5]
    r_log = params[5]
    q_vals = 10.0**q_logs
    r_val = 10.0**r_log

    # Avoid zeros / extreme tiny
    q_vals = np.clip(q_vals, 1e-6, 1e6)
    r_val  = np.clip(r_val,  1e-3, 1e6)

    Q = np.diag(q_vals)
    R = np.array([[r_val]])
    return Q, R

# =========================
# 6. Random-search "ML" optimizer
# =========================

def optimize_QR(num_iters=100, refine_top_k=5, verbose=True):
    """
    Simple black-box optimizer:
    - Sample QR in log-space
    - Evaluate cost via simulation
    - Keep best few and locally perturb them
    """
    A, B, Km, Lm, Rm = get_AB()

    best_params = None
    best_cost = np.inf
    history = []

    # Initial bounds for log10(qi) and log10(R)
    # You can tighten these over time
    q_low, q_high = -2, 4        # 10^-2 .. 10^4
    r_low, r_high = 0, 3         # 10^0 .. 10^3

    # store candidates for refinement
    candidates = []

    for it in range(num_iters):
        if it < num_iters//2 or best_params is None:
            # global random search
            q_logs = np.random.uniform(q_low, q_high, size=5)
            r_log  = np.random.uniform(r_low, r_high)
            params = np.concatenate([q_logs, [r_log]])
        else:
            # local refinement around best few
            # pick a previous good candidate
            parent = candidates[np.random.randint(0, len(candidates))]
            params = parent + np.random.normal(0, 0.25, size=6)

        try:
            Q, R = qr_from_params(params)
            K = lqr(A, B, Q, R)
            # Quick stability check, if wanted:
            lam = eig(A - B@K)[0]
            if np.max(np.real(lam)) >= 0:
                # unstable closed loop -> huge cost
                cost = 1e8
            else:
                cost = simulate_and_cost(A, B, K)
        except Exception as e:
            cost = 1e9  # penalize failures

        history.append((cost, params))

        if cost < best_cost:
            best_cost = cost
            best_params = params.copy()
            if verbose:
                q_vals = 10**best_params[:5]
                r_val  = 10**best_params[5]
                print(f"Iter {it:3d}: New best cost = {best_cost:.3e}")
                print(f"    q = {q_vals}, R = {r_val:.3e}")

        # maintain top-k candidates
        history_sorted = sorted(history, key=lambda x: x[0])
        candidates = [h[1] for h in history_sorted[:refine_top_k]]

    # Final best Q,R
    Q_best, R_best = qr_from_params(best_params)
    return Q_best, R_best, best_cost, best_params, history

# =========================
# 7. Main
# =========================

if __name__ == "__main__":
    np.random.seed(0)

    Q_best, R_best, best_cost, best_params, history = optimize_QR(
        num_iters=80,   # bump this higher if you want a better search
        refine_top_k=5,
        verbose=True
    )

print("\n========== BEST FOUND ==========")
print(f"Best cost J = {best_cost:.3e}")
print("Best log10 params [q1..q5, R] =", best_params)

print("\nQ_best (rounded to 2 decimals):")
Q_round = np.round(Q_best, 2)
for row in Q_round:
    print("  ", [float(f"{x:.2f}") for x in row])

print("\nR_best (rounded to 2 decimals):")
R_round = np.round(R_best, 2)
print("  ", [float(f"{x:.2f}") for x in R_round.flatten()])


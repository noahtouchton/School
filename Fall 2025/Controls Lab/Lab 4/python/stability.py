# stability_check.py
import numpy as np
from numpy.linalg import eig, matrix_rank
from scipy.linalg import solve_continuous_lyapunov as lyap
from scipy.linalg import block_diag
from scipy.signal import cont2discrete

def is_stable_continuous(A, tol=1e-9):
    """Continuous-time stability: all eigenvalues must have Re(λ) < 0."""
    lam = eig(A)[0]
    max_real = np.max(np.real(lam))
    stable = max_real < -tol
    return stable, lam, max_real

def is_stable_discrete(Ad, tol=1e-9):
    """Discrete-time stability: all eigenvalues must be inside unit circle."""
    lam = eig(Ad)[0]
    max_mod = np.max(np.abs(lam))
    stable = max_mod < 1.0 - tol
    return stable, lam, max_mod

def controllability_rank(A, B):
    n = A.shape[0]
    Ctrb = B
    AB = np.eye(n)
    for _ in range(1, n):
        AB = AB @ A
        Ctrb = np.hstack((Ctrb, AB @ B))
    return matrix_rank(Ctrb), Ctrb

def observability_rank(A, C):
    n = A.shape[0]
    Obsv = C
    CA = C
    for _ in range(1, n):
        CA = CA @ A
        Obsv = np.vstack((Obsv, CA))
    return matrix_rank(Obsv), Obsv

def modal_info(A):
    """Damping ratio & natural frequency for continuous-time modes."""
    lam = eig(A)[0]
    # For each eigenvalue λ = σ + jω, natural freq = |λ|, damping = -σ/|λ|
    info = []
    for l in lam:
        sigma = np.real(l)
        omega = np.imag(l)
        wn = np.sqrt(sigma**2 + omega**2)
        zeta = (-sigma/wn) if wn > 0 else np.inf
        info.append(dict(lambda_=l, sigma=sigma, omega=omega, wn=wn, zeta=zeta))
    return info

def lyapunov_cert(A):
    """Try Lyapunov certificate: P>0 s.t. A^T P + P A = -I. Returns True if P is PD."""
    Q = -np.eye(A.shape[0])
    try:
        P = lyap(A.T, Q)
        # Check positive definiteness (numerically)
        eigP = np.linalg.eigvalsh(P)
        return np.all(eigP > 1e-9), P, eigP
    except Exception as e:
        return False, None, None

def print_modes(info):
    print("Modes (continuous):")
    for k, m in enumerate(info, 1):
        print(f"  {k:>2}: λ = {m['lambda_']:.6g} | σ={m['sigma']:.6g}, jω={m['omega']:.6g}, "
              f"ω_n={m['wn']:.6g}, ζ={m['zeta']:.6g}")

def report_continuous(A, name="A (open-loop)"):
    print(f"\n=== Continuous-time Stability Report: {name} ===")
    stable, lam, max_real = is_stable_continuous(A)
    print(f"Eigenvalues:\n{lam}\nMax Re(λ) = {max_real:.6g}")
    print(f"Stable? {'YES' if stable else 'NO'}")
    print_modes(modal_info(A))
    lyap_ok, P, eigP = lyapunov_cert(A)
    print(f"Lyapunov certificate (A^T P + P A = -I): {'OK' if lyap_ok else 'FAIL'}")
    if eigP is not None:
        print(f"  min eig(P) = {np.min(eigP):.3e}, max eig(P) = {np.max(eigP):.3e}")

def report_discrete(Ad, name="Ad"):
    print(f"\n=== Discrete-time Stability Report: {name} ===")
    stable, lam, max_mod = is_stable_discrete(Ad)
    print(f"Eigenvalues (z-plane):\n{lam}\nMax |λ| = {max_mod:.6g}")
    print(f"Stable? {'YES' if stable else 'NO'}")

def make_closed_loop(A, B, K):
    """u = -K x"""
    return A - B @ K

def augment_LQI(A, B, Cc):
    """Augment with integral of error for output y_c = Cc x (single-output)."""
    Aa = np.block([
        [A,            np.zeros((A.shape[0], 1))],
        [-Cc,          np.zeros((1, 1))]
    ])
    Ba = np.vstack([B, np.zeros((1, B.shape[1]))])
    return Aa, Ba

if __name__ == "__main__":
    # ======= 1) PUT YOUR MATRICES HERE =======
    # Example placeholders (replace with your real ones)
    # A, B are continuous-time (upright linearization)
    A = np.array([
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0.03070, -2.9119, -0.0007105, 31.9277],
        [0, 17.7590, -0.05187, -0.007721, 17.4219],
        [0, 0, -23.8333, 0, -133.3333]
    ])

    B = np.array([
        [0],
        [0],
        [0],
        [0],
        [83.3333]
    ])

    # Optional: your LQR/LQI gain (u = -K x or u = -Kx - Ki*xi for augmented)
    # K must be (1x5) for the 5-state case
    K = None  # e.g., K = np.array([[k1, k2, k3, k4, k5]])

    # Output matrix for base-angle tracking (if doing LQI)
    Cc = np.array([[1, 0, 0, 0, 0]])  # y_c = theta1

    # ======= 2) BASIC CHECKS =======
    print("\nRanks:")
    ctrb_rank, _ = controllability_rank(A, B)
    print(f"  Controllability rank = {ctrb_rank} / {A.shape[0]}")
    # If you have a C matrix for full-state sensing, you can check observability too:
    # C = np.eye(5)
    # obsv_rank, _ = observability_rank(A, C)
    # print(f"  Observability rank    = {obsv_rank} / {A.shape[0]}")

    # Open-loop continuous stability
    report_continuous(A, "A (open-loop)")

    # ======= 3) CLOSED-LOOP (STATE FEEDBACK) =======
    if K is not None:
        Acl = make_closed_loop(A, B, K)
        report_continuous(Acl, "A - B K (closed-loop)")

    # ======= 4) OPTIONAL: DISCRETE CHECK =======
    # If you run a digital controller with sample time Ts (seconds):
    Ts = None  # e.g., Ts = 0.001
    if Ts is not None:
        Ad, Bd, Cd, Dd, _ = cont2discrete((A, B, np.eye(A.shape[0]), np.zeros((A.shape[0], B.shape[1]))), Ts, method='zoh')
        report_discrete(Ad, f"Ad (open-loop, Ts={Ts}s)")
        if K is not None:
            # For discrete u[k] = -K x[k], closed-loop Ad_cl = Ad - Bd*K
            Ad_cl = Ad - Bd @ K
            report_discrete(Ad_cl, f"Ad - Bd K (closed-loop, Ts={Ts}s)")

    # ======= 5) OPTIONAL: LQI AUGMENTED CHECK =======
    # If you designed an LQI on augmented (A_a, B_a) with K_a = [Kx  Ki], you can check A_a - B_a K_a as well:
    # Aa, Ba = augment_LQI(A, B, Cc)
    # Ka = np.zeros((1, Aa.shape[0]))  # <-- your augmented gain [1 x (5+1)]
    # Acl_a = Aa - Ba @ Ka
    # report_continuous(Acl_a, "Augmented (A_a - B_a K_a)")
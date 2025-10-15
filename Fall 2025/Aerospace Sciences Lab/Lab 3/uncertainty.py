# uncertainty.py  — clean main
from rss import *  # uses your Data/RSS implementation
import numpy as np
import os
from calibration_helpers import fit_poly_with_z_basis   # to get a, Cov_a, sigma
from mc_uncert import simulate_v_from_V                 # Monte Carlo for v
import os
from drag import Dprime_from_velocity_deficit  # your function from drag.py
import matplotlib.pyplot as plt


# ---------- constants ----------
inH2O_to_Pa = 249.0889

# Only keep what we need
equations = [
    "ρ = P / (R * T)",
    "mu = MU_0 * (T / T_0)**(3/2) * (T_0 + C) / (T + C)",
    "v = a0 + a1*V + a2*V**2 + a3*V**3 + a4*V**4",
    "Re = ρ * v * d / mu",
]

# Gas & Sutherland constants
MU_0 = Data("mu not", "MU_0", 1.716e-5, 0.0)
T_0  = Data("T not",  "T_0",  273.0,   0.0)
C    = Data("Constant","C",   111.0,   0.0)
R    = Data("Ideal Gas Constant for Air", "R", 287.05, 0.0)

# Lab ambient & geometry (edit with your values/uncertainties)
T = Data("Temperature", "T", 296.15, 0.4)
P = Data("Pressure",    "P", 101207.54, 400.0)
d = Data("Diameter",    "d", 0.01905, 0.0)

# ---------- derived fluid props ----------
ρ  = RSS("Density",            equations[0], [P, R, T])
mu = RSS("Dynamic Viscosity",  equations[1], [T, MU_0, T_0, C])
mu.get_description()
ρ.get_description()
mu.get_description()

# ---------- calibration data (your 12 points) ----------
q_cal_vals = np.array([
    -0.006, -0.047, -0.148, -0.316, -0.560,
    -0.881, -1.302, -1.836, -2.462, -3.176,
    -3.928, -4.684
]) * -1.0  # positive q

q_cal_uncert = np.array([
    0.060, 0.065, 0.070, 0.076, 0.066,
    0.062, 0.068, 0.065, 0.066, 0.061,
    0.066, 0.064
]) * 2.0

q_cal = Data("Q Calibration", "q", q_cal_vals, q_cal_uncert)

v_cal = RSS("Calibration Velocity", "v = sqrt(2*q/ρ)", [q_cal, ρ])

V_cal = np.array([  # your 12 voltages
    2.11012, 2.50558, 2.76121, 2.94751, 3.10566, 3.24794,
    3.39233, 3.51891, 3.64815, 3.76203, 3.86064, 3.93145
])

v_cal_pa = Data("Calibration Velocity (Pa)", "v", v_cal.value * 15.78, v_cal.uncertainty * 15.78)

v_cal_pa.print_to_excel()
Re = RSS("Calibration Reynolds", equations[3], [ρ, v_cal_pa, d, mu])
Re.print_to_excel()


# ---------- fit U(E) with Z-basis then map to V-basis ----------
fit = fit_poly_with_z_basis(V_cal, v_cal.value, deg=4)
a_asc  = fit["a_asc"]       # [a0..a4]
Cov_a  = fit["Cov_a"]
sigma  = fit["sigma"]       # residual std (velocity units)
dof    = fit["dof"]

print("V-basis coeffs (asc):", a_asc)
print("Coeff 1σ (asc):", np.sqrt(np.diag(Cov_a)))
print("residual σ:", sigma, " dof:", dof)

# Package coefficients as Data for RSS when computing v = f(V)
a0 = Data("a0","a0", a_asc[0], np.sqrt(Cov_a[0,0]))
a1 = Data("a1","a1", a_asc[1], np.sqrt(Cov_a[1,1]))
a2 = Data("a2","a2", a_asc[2], np.sqrt(Cov_a[2,2]))
a3 = Data("a3","a3", a_asc[3], np.sqrt(Cov_a[3,3]))
a4 = Data("a4","a4", a_asc[4], np.sqrt(Cov_a[4,4]))

for a in [a0,a1,a2,a3,a4]:
    a.get_description()



def _dominant_frequency_from_voltage(t, V, freq_range=None):
    """
    Returns (f_peak, u_f, freqs, P) from the one-sided FFT of V(t).
    If freq_range=(fmin, fmax) is given, finds the peak only within that band.
    """
    t = np.asarray(t, float).ravel()
    V = np.asarray(V, float).ravel()
    N = V.size
    if N < 8:
        raise ValueError("Not enough samples to compute FFT")

    dt = float(np.median(np.diff(t)))
    fs = 1.0 / dt
    T_total = N * dt
    df = 1.0 / T_total

    # detrend + window
    s = V - V.mean()
    w = np.hanning(N)
    s_win = s * w

    Y = np.fft.rfft(s_win)
    freqs = np.fft.rfftfreq(N, d=dt)
    P = (np.abs(Y)**2) / (np.sum(w**2))

    # ignore DC
    i_lo = 1
    i_hi = len(freqs) - 1

    if freq_range is not None:
        fmin, fmax = freq_range
        mask = (freqs >= fmin) & (freqs <= fmax)
        # ensure we don't include DC even if fmin≈0
        mask &= (freqs > 0)
        if not np.any(mask):
            raise ValueError("freq_range excludes all bins")
        # peak within band
        idx = np.argmax(P[mask])
        # map back to absolute index
        i_candidates = np.flatnonzero(mask)
        i_peak = int(i_candidates[idx])
    else:
        i_peak = i_lo + int(np.argmax(P[i_lo:i_hi+1]))

    f_peak = float(freqs[i_peak])
    u_f = 0.5 * df  # conservative 1σ ~ half-bin
    return f_peak, u_f, freqs, P

def _plot_spectrum(name, t, V, freqs, P, f_peak, f_range=None, save=True, show=False):
    """
    Plot the one-sided FFT power spectrum, restricted to the given frequency range.
    """
    import matplotlib.pyplot as plt

    dt = float(np.median(np.diff(t)))
    T_total = len(t) * dt
    P_plot = P / T_total

    # Apply the designated frequency range (e.g., (80, 220))
    if f_range is not None:
        fmin, fmax = f_range
        mask = (freqs >= fmin) & (freqs <= fmax)
        freqs = freqs[mask]
        P_plot = P_plot[mask]

    plt.figure(figsize=(8, 4.5))
    plt.plot(freqs, P_plot, lw=1.2, color="C0")
    plt.axvline(f_peak, ls="--", lw=1.2, color="C3", label=f"peak ≈ {f_peak:.2f} Hz")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power (V²/Hz)")
    plt.title(f"{name} — One-Sided Spectrum ({f_range[0]}–{f_range[1]} Hz)" if f_range else f"{name} — One-Sided Spectrum")
    plt.grid(alpha=0.3)
    plt.legend()

    if save:
        os.makedirs("figs", exist_ok=True)
        out = os.path.join("figs", f"{name.replace(' ', '_')}_spectrum.png")
        plt.tight_layout()
        plt.savefig(out, dpi=160)
        print(f"[saved] {out}")

    if show:
        plt.show()
    else:
        plt.close()



# ---------- helper to process one position ----------
def process_point(name, lvm_filename, include_param_cov=True, use_mean=True, plot = True):
    base_dir = os.path.dirname(__file__)
    path = os.path.join(base_dir, "LVM", lvm_filename)

    # LVM has only numeric rows, second column = voltage samples
    try:
        tV = np.loadtxt(path, usecols=(0, 1))
    except OSError as e:
        raise FileNotFoundError(f"Cannot read LVM file: {path}") from e
    t = tV[:, 0]
    V_series = tV[:, 1]
    N = V_series.size
    V_mean = float(np.mean(V_series))
    V_std  = float(np.std(V_series, ddof=1))

    # choose to propagate instantaneous σV or mean σV
    sigma_V = V_std / np.sqrt(N) if use_mean else V_std

    # Monte Carlo through v = poly(V) with coeff covariance + lack-of-fit
    stats, _ = simulate_v_from_V(
        V_mean=V_mean,
        a_asc=a_asc,
        sigma_V=sigma_V,
        sigma_lackofit=sigma,
        Cov_a=Cov_a,
        include_param_cov=include_param_cov,
        N=100_000,
    )

    vel_scaler_to_m_per_s = 15.78

    U_mean = stats["mean"] * vel_scaler_to_m_per_s
    U_std  = stats["std"]  * vel_scaler_to_m_per_s


    V_data = Data(
        f"{name} Voltage (mean)" if use_mean else f"{name} Voltage (inst)",
        "V",
        V_mean,
        sigma_V
    )
    v_data = Data(f"{name} Velocity (MC)", "v", U_mean, U_std)

    # Reynolds using RSS with Re = ρ*v*d/mu
    Re_data = RSS(f"{name} Reynolds", equations[3], [ρ, v_data, d, mu])

    # expected shedding band for your U and D: roughly 80–250 Hz
    f_peak, u_f, freqs, P = _dominant_frequency_from_voltage(
        t, V_series, freq_range=(80.0, 220.0)
    )

    # --- Strouhal: St = f D / U  (U = mean velocity)
    # Propagate 1σ: var(St) = (D/U)^2 u_f^2 + (f/U)^2 u_D^2 + (-fD/U^2)^2 u_U^2
    D_val = float(d.value)
    u_D   = float(d.uncertainty)
    U_val = float(v_data.value)
    u_U   = float(v_data.uncertainty)

    St_val = f_peak * D_val / U_val
    var_St = ( (D_val / U_val)**2 ) * (u_f**2) \
           + ( (f_peak / U_val)**2 ) * (u_D**2) \
           + ( (f_peak * D_val / (U_val**2))**2 ) * (u_U**2)
    St_std = float(np.sqrt(var_St))

    St_data = Data(f"{name} Strouhal", "St", St_val, St_std)

    # Optional: quick prints
    # print(f"{name} Voltage (mean), V: {V_mean:.3e} ± {sigma_V:.3e}")
    # print(f"{name} Velocity (MC), v: {U_mean:.3e} ± {U_std:.3e}")
    # print(f"{name} Reynolds, Re: {Re_data.value:.3e} ± {Re_data.uncertainty:.3e}")
    # print(f"{name} f_peak: {f_peak:.3f} Hz  (±{u_f:.3f} Hz)")
    # print(f"{name} Strouhal, St: {St_val:.4f} ± {St_std:.4f}")

    if plot:
        _plot_spectrum(name, t, V_series, freqs, P, f_peak, f_range=(80.0, 220.0), save=True, show=False)


    return V_data, v_data, Re_data, St_data

# ---------- run for the four stations ----------
pts = [
    ("Centerline", "Lab3VelocityCenter.lvm"),
    ("Top",        "Lab3VelocityTop.lvm"),
    ("Edge",       "Lab3VelocityEdge.lvm"),
    ("Outside",    "Lab3VelocityOut.lvm"),
]

for name, fname in pts:
    Vd, vd, Red, St = process_point(name, fname, include_param_cov=True, use_mean=True)
    Vd.get_description()
    vd.get_description()
    Red.get_description()
    St.get_description()


points = ["Centerline", "Top", "Edge", "Outside"]

# you already returned these from process_point, but we’ll collect them
# assuming the loop variables still hold the final iteration, we rebuild lists manually
results = [process_point(name, fname, include_param_cov=True, use_mean=True) for name, fname in pts]
Vd_list, vd_list, Re_list, St_list = zip(*results)

# Extract numerical values
velocities = [v.value for v in vd_list]
vel_unc = [v.uncertainty for v in vd_list]
strouhal = [s.value for s in St_list]
strouhal_unc = [s.uncertainty for s in St_list]

# Pretty print
print("\n" + "=" * 70)
print(" FINAL SUMMARY (All values in regular notation)")
print("=" * 70)
print(f"{'Point':<12}{'Velocity (m/s)':>18}{'±σv':>12}{'Strouhal':>16}{'±σSt':>10}")
print("-" * 70)

for i in range(4):
    print(f"{points[i]:<12}{velocities[i]:>18.3f}{vel_unc[i]:>12.3f}{strouhal[i]:>16.4f}{strouhal_unc[i]:>10.4f}")

print("=" * 70)



q_cal_25 = Data("Q Calibration at 25hz", "q", q_cal_vals[4], q_cal_uncert[4])
u_inf = Data("Velocity at 25hz", "v", v_cal.value[4]*15.78, v_cal.uncertainty[4]*15.78)

# ui, dy and their 1σ uncertainties from your table (m/s, m)
ui_vals = np.array([11.946, 12.343, 14.338, 15.062])
ui_unc  = np.array([0.245,  0.244,  0.245,  0.243])
dy_vals = np.array([0.007874, 0.0433705, 0.046482, 0.014097])

# position uncertainty: 1/32" for each y (so Δy has √2 times that, applied to each strip)
u_y     = (1/32)*0.0254
dy_unc  = np.full_like(dy_vals, np.sqrt(2)*u_y)

# U∞ from your calibration (m/s)
Uinf = Data("Freestream speed", "Uinf", u_inf.value, u_inf.uncertainty)

# Build D′ with your helper (uses D' = ρ Σ u_i (U∞–u_i) Δy_i )
Dprime = Dprime_from_velocity_deficit(
    "Cylinder drag/length", ρ, Uinf,
    ui_vals, dy_vals, ui_unc, dy_unc
)

# q∞ and C_D with RSS
q_inf = RSS("Dynamic pressure", "q = 0.5 * ρ * Uinf**2", [ρ, Uinf])
CD    = RSS("Drag coefficient", "CD = Dprime / (q * d)", [Dprime, q_inf, d])

print(f"D' = {Dprime.value:.3f} ± {Dprime.uncertainty:.3f} N/m")
print(f"C_D = {CD.value:.3f} ± {CD.uncertainty:.3f}")

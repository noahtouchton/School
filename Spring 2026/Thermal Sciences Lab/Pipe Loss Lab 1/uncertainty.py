# uncertainty.py
import numpy as np
import math
import os
import matplotlib.pyplot as plt

from rss import Data, load_large_run, load_small_run1, RSS

equations = [
    "rho = 1000 * (1 - ((T + 288.9414) / (508929.2 * (T + 68.12963))) * (T - 3.9863)**2)",
    "hl  = 6894.76 * 39.37007874 * dp / (rho * g)",  # psi->Pa and m->in
    "A = 3.14159265358979 * d**2 / 4",
    "Qms = Qv / 60000.0",  # LPM -> m³/s
    "v = Qms / A",
    "Re = rho * v * d / mu",
    "f = 2 * g * (hl/39.37007874) * d / (L * v**2)",
    "h_el_m = 6894.76 * dp / (rho * g)",            # meters (for KL)
    "Kl = 2 * g * h_el_m / (v**2)",
    "Le = 4.4 * d * Re**(1/6)",

]

# -------------------------------------------------
# Pipe diameters (kept, even if not used yet)
# -------------------------------------------------
big_vals = [18.86, 18.84, 18.89, 18.82, 18.85]   # mm
small_vals = [9.42, 9.40, 9.41, 9.42, 9.42]      # mm

caliper_unc_mm = 0.02
caliper_unc_m = caliper_unc_mm / 1000.0

big_std_m = np.std(big_vals, ddof=1) / 1000.0
small_std_m = np.std(small_vals, ddof=1) / 1000.0


g = Data("Gravity", "g", 9.81, 0.0)
mu = Data("Dynamic Viscosity", "mu", 0.000653, 0.0)  # Pa·s (kg/(m·s))
L = Data("Pipe Length", "L", 1.0, 0.0)  # m 

d_big = Data("Large Pipe Diameter", "d",
             np.mean(big_vals) / 1000.0,
             math.sqrt(caliper_unc_m**2 + big_std_m**2))

d_small = Data("Small Pipe Diameter", "d",
               np.mean(small_vals) / 1000.0,
               math.sqrt(caliper_unc_m**2 + small_std_m**2))


# -------------------------------------------------
# Instrument uncertainties (from your equipment)
# -------------------------------------------------
DP_LARGE_INST = 0.0008   # 0–1 psid PX409 (±0.08% BSL -> 0.0008 psi)
DP_ELBOW_INST = 0.0008   # 0–1 psid PX409
DP_SMALL_INST = 0.0120   # 0–15 psid PX409 (±0.08% BSL -> 0.012 psi)

TEMP_INST_UNC = 2.2      # Type J standard limit (°C)

dp_tuple = (DP_LARGE_INST, DP_SMALL_INST, DP_ELBOW_INST)

# -------------------------------------------------
# Load runs
# -------------------------------------------------
large_run1 = load_large_run(1, base_dir="data", dp_inst_unc=dp_tuple, temp_inst_unc=TEMP_INST_UNC)
large_run2 = load_large_run(2, base_dir="data", dp_inst_unc=dp_tuple, temp_inst_unc=TEMP_INST_UNC)
large_run3 = load_large_run(3, base_dir="data", dp_inst_unc=dp_tuple, temp_inst_unc=TEMP_INST_UNC)
small_run1 = load_small_run1(base_dir="data", dp_inst_unc=dp_tuple, temp_inst_unc=TEMP_INST_UNC)


# -------------------------------------------------
# Compute rho + head loss (MATCH POINT-BY-POINT)
# -------------------------------------------------
def compute_rho_and_hl(run, dp_list, size):
    run.rho = []
    run.head_loss = []
    run.d = []
    run.A = []
    run.v = []
    run.Re = []
    run.f = []
    
    

    # Density per point
    for T in run.T_C:
        rho = RSS("density", equations[0], [T])
        run.rho.append(rho)

    # Head loss per point (pair dp with rho at same i)
    for dp, rho in zip(dp_list, run.rho):
        hl = RSS("head_loss", equations[1], [dp, rho, g])
        run.head_loss.append(hl)

    if size == "large":
        run.d = [d_big] * len(run.Q_lpm)
    elif size == "small":
        run.d = [d_small] * len(run.Q_lpm)
    else:
        raise ValueError("size must be 'large' or 'small'")
    
    # Velocity and Reynolds number per point
    for Q, d, rho in zip(run.Q_lpm, run.d, run.rho):
        A = RSS("area", equations[2], [d])
        Qms = RSS("flow_m3s", equations[3], [Q])
        v = RSS("velocity", equations[4], [Qms, A])
        Re = RSS("Reynolds number", equations[5], [rho, v, d, mu])

        run.A.append(A)
        run.v.append(v)
        run.Re.append(Re)

    for hl, d, v in zip(run.head_loss, run.d, run.v):
        f = RSS("friction factor", equations[6], [hl, d, v, g, L])
        run.f.append(f)
    
def compute_kl(run, elbow_dp_list):
    run.kl = []
    run.hl_elbow = []

    for rho, v, dp in zip(run.rho, run.v, elbow_dp_list):
        elbow_hl = RSS("elbow_head_loss", equations[7], [dp, rho, g])
        run.hl_elbow.append(elbow_hl)
        kl = RSS("Kl", equations[8], [elbow_hl, v, g])
        run.kl.append(kl)

def compute_le(run):
    run.Le = []
    for d, Re in zip(run.d, run.Re):
        # If Re is non-physical, store NaN so tables/plots can skip it
        if Re.value <= 0 or not np.isfinite(Re.value):
            run.Le.append(Data("Le", "Le", float("nan"), float("nan")))
            continue

        Le = RSS("Le", equations[9], [d, Re])
        run.Le.append(Le)



for run in (large_run1, large_run2, large_run3):
    compute_rho_and_hl(run, run.dp_large_psid, size="large")
    compute_kl(run, run.dp_elbow_psid)
    compute_le(run)

compute_rho_and_hl(small_run1, small_run1.dp_small_psid, size="small")

compute_kl(small_run1, small_run1.dp_elbow_psid)
compute_le(small_run1)

# -------------------------------------------------
# Plot: one combined graph (hl vs Q^2) with uncertainty bars
# -------------------------------------------------
def get_arrays(run):
    Q = np.array([d.value for d in run.Q_lpm], dtype=float)
    Q_unc = np.array([d.uncertainty for d in run.Q_lpm], dtype=float)

    Q2 = Q**2
    Q2_unc = 2 * np.abs(Q) * Q_unc  # must be nonnegative

    hl = np.array([d.value for d in run.head_loss], dtype=float)
    hl_unc = np.array([d.uncertainty for d in run.head_loss], dtype=float)

    return Q2, Q2_unc, hl, hl_unc

plt.figure()

runs = [
    (large_run1, "Large Run 1", 'o'),
    (large_run2, "Large Run 2", 's'),
    (large_run3, "Large Run 3", '^'),
    (small_run1, "Small Run",   'D'),
]

for run, label, marker in runs:
    Q2, Q2_unc, hl, hl_unc = get_arrays(run)
    plt.errorbar(Q2, hl, xerr=Q2_unc, yerr=hl_unc, fmt=marker, capsize=3, label=label)

plt.xlabel("Flowrate² (LPM²)")
plt.ylabel("Head loss (inches of water)")
#plt.title("Head Loss vs Flowrate²")
plt.legend()
plt.grid(True, linestyle="--", linewidth=0.5)
plt.tight_layout()

out_dir = os.path.dirname(os.path.abspath(__file__))
fname = os.path.join(out_dir, "head_loss_all_runs.png")
plt.savefig(fname, dpi=200)
plt.close()
print(f"Saved combined plot: {fname}")


# -------------------------------------------------
# Print run (debug)
# -------------------------------------------------
def print_run(run):
    print(f"\n===== {run.name} =====")
    for i, pct in enumerate(run.pcts):
        print(f"\n--- {pct}% ---")
        run.Q_lpm[i].get_error()
        run.dp_large_psid[i].get_error()
        run.dp_small_psid[i].get_error()
        run.dp_elbow_psid[i].get_error()
        run.T_C[i].get_error()
        run.rho[i].get_error()
        run.head_loss[i].get_error()
        run.Re[i].get_error()
        run.f[i].get_error()
        run.hl_elbow[i].get_error()
        run.kl[i].get_error()


# -------------------------------------------------
# CSV table for Excel (Std Dev vs Total Uncertainty DIFFERENT)
# -------------------------------------------------
def print_head_loss_table(run, pipe_type="small"):
    """
    CSV for Excel. Rows in reverse order (100% -> 0%).
    Columns:
      Actual Flow, Flow Unc (total), Head Loss, Std Dev Head Loss, Total Head Loss Unc, Temp, Temp Unc (total)
    """
    if pipe_type == "large":
        dp_list = run.dp_large_psid
    elif pipe_type == "small":
        dp_list = run.dp_small_psid
    else:
        raise ValueError("pipe_type must be 'large' or 'small'")

    header = (
        "Actual Flow (LPM),"
        "Flow Uncertainty (± LPM),"
        "Head Loss Δh (in),"
        "Std Dev Head Loss (in),"
        "Total Head Loss Uncertainty (± in),"
        "Temperature (°C),"
        "Temperature Uncertainty (± °C)"
    )
    print(header)

    for i in reversed(range(len(run.Q_lpm))):
        Q_act = run.Q_lpm[i].value
        Q_unc_total = run.Q_lpm[i].uncertainty

        hl = run.head_loss[i].value
        hl_unc_total = run.head_loss[i].uncertainty

        # IMPORTANT: Std Dev Head Loss should come from sample std of DP ONLY
        dp_std_only = dp_list[i].std  # <-- this is sample standard deviation (not total uncertainty)
        rho_nom = run.rho[i].value

        hl_std = (dp_std_only * 6894.76) / (rho_nom * 9.81) * 39.37007874

        T = run.T_C[i].value
        T_unc_total = run.T_C[i].uncertainty

        print(
            f"{Q_act:.3f},"
            f"{Q_unc_total:.4f},"
            f"{hl:.3f},"
            f"{hl_std:.3f},"
            f"{hl_unc_total:.4f},"
            f"{T:.2f},"
            f"{T_unc_total:.4f}"
        )



def get_moody_arrays(run, skip_first=True):
    start = 1 if skip_first else 0

    Re = np.array([d.value for d in run.Re[start:]], dtype=float)
    Re_unc = np.array([d.uncertainty for d in run.Re[start:]], dtype=float)

    f = np.array([d.value for d in run.f[start:]], dtype=float)
    f_unc = np.array([d.uncertainty for d in run.f[start:]], dtype=float)

    # Safety: only keep positive Reynolds and positive f for log plots
    mask = (Re > 0) & (f > 0)
    return Re[mask], Re_unc[mask], f[mask], f_unc[mask]


plt.figure()

runs_moody = [
    (large_run1, "Large Pipe (Run 1)", 'o'),
    (large_run2, "Large Pipe (Run 2)", 's'),
    (large_run3, "Large Pipe (Run 3)", '^'),
    (small_run1, "Small Pipe", 'D'),
]

for run, label, marker in runs_moody:
    Re, Re_unc, fvals, f_unc = get_moody_arrays(run, skip_first=True)

    plt.errorbar(
        Re, fvals,
        xerr=Re_unc,
        yerr=f_unc,
        fmt=marker,
        capsize=3,
        label=label
    )

plt.xscale("log")
plt.yscale("log")  # optional, but most Moody-style plots use log-log

plt.xlabel("Reynolds number, Re")
plt.ylabel("Darcy friction factor, f")
#plt.title("Friction Factor vs Reynolds Number")
plt.legend()
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.tight_layout()

out_dir = os.path.dirname(os.path.abspath(__file__))
fname = os.path.join(out_dir, "moody_f_vs_Re.png")
plt.savefig(fname, dpi=200)
plt.close()
print(f"Saved Moody-style plot: {fname}")



def plot_elbow_hL_vs_Q2(run, label, marker):
    # Use elbow head loss in meters (run.elbow_h_m) and Q in m^3/s
    Qms = np.array([RSS("Qms", equations[3], [Q]).value for Q in run.Q_lpm], dtype=float)
    Qms_unc = np.array([RSS("Qms", equations[3], [Q]).uncertainty for Q in run.Q_lpm], dtype=float)

    Q2 = Qms**2
    Q2_unc = 2 * np.abs(Qms) * Qms_unc

    h = np.array([d.value for d in run.hl_elbow], dtype=float)          # meters
    h_unc = np.array([d.uncertainty for d in run.hl_elbow], dtype=float)

    # Skip the first point (0%) because it is noisy / near-zero flow
    Q2, Q2_unc, h, h_unc = Q2[1:], Q2_unc[1:], h[1:], h_unc[1:]

    plt.errorbar(Q2, h, xerr=Q2_unc, yerr=h_unc, fmt=marker, capsize=3, label=label)


plt.figure()

plot_elbow_hL_vs_Q2(large_run1, "Elbow (Large Run 1)", 'o')
plot_elbow_hL_vs_Q2(large_run2, "Elbow (Large Run 2)", 's')
plot_elbow_hL_vs_Q2(large_run3, "Elbow (Large Run 3)", '^')
#plot_elbow_hL_vs_Q2(small_run1, "Elbow (Small Run)", 'D')

plt.xlabel("Flow rate squared, $Q^2$ (m$^6$/s$^2$)")
plt.ylabel("Elbow head loss, $h_L$ (m)")
#plt.title("Elbow Loss: $h_L$ vs $Q^2$")
plt.legend()
plt.grid(True, linestyle="--", linewidth=0.5)
plt.tight_layout()

out_dir = os.path.dirname(os.path.abspath(__file__))
fname = os.path.join(out_dir, "elbow_hL_vs_Q2.png")
plt.savefig(fname, dpi=200)
plt.close()
print(f"Saved elbow plot: {fname}")

def print_measurement_uncertainty_table(
    runs,
    d_big,
    d_small,
    L,
    mu,
    temp_inst_unc,
    dp_inst_tuple,
    include_components=True
):
    """
    Prints a measurement-only uncertainty table in CSV form for Excel copy/paste.

    - Includes: pressure (dp), flow rate (Q), temperature (T), diameters (d), length (L)
    - Also lists thermodynamic inputs (mu) as assumed, and density as computed-from-T (not a direct measurement)

    include_components=True prints instrument-only and sample-std components in separate columns.
    """

    # Header
    if include_components:
        header = (
            "Run,Point(%),Parameter,Value,Units,"
            "Instrument Unc,Instrument Units,"
            "Sample Std,Sample Std Units,"
            "Total Unc,Total Unc Units"
        )
    else:
        header = "Run,Point(%),Parameter,Value,Units,Total Unc,Total Unc Units"

    print(header)

    def row(run_name, pct, param, value, units, inst, inst_units, std, std_units, tot, tot_units):
        if include_components:
            print(
                f"{run_name},{pct},{param},"
                f"{value},{units},"
                f"{inst},{inst_units},"
                f"{std},{std_units},"
                f"{tot},{tot_units}"
            )
        else:
            print(f"{run_name},{pct},{param},{value},{units},{tot},{tot_units}")

    # ---- Global geometry measurements (not per-point) ----
    # Diameters and length are measurements, but not time-series. Their "std" is from repeated caliper readings.
    # inst_unc is your caliper resolution / stated accuracy.
    row("GLOBAL","", "Large pipe diameter d_big",  f"{d_big.value:.6g}", "m",
        f"{getattr(d_big,'inst_unc',0.0):.6g}", "m",
        f"{getattr(d_big,'std',0.0):.6g}", "m",
        f"{d_big.uncertainty:.6g}", "m"
    )
    row("GLOBAL","", "Small pipe diameter d_small", f"{d_small.value:.6g}", "m",
        f"{getattr(d_small,'inst_unc',0.0):.6g}", "m",
        f"{getattr(d_small,'std',0.0):.6g}", "m",
        f"{d_small.uncertainty:.6g}", "m"
    )
    row("GLOBAL","", "Pipe length L", f"{L.value:.6g}", "m",
        f"{getattr(L,'inst_unc',0.0):.6g}", "m",
        f"{getattr(L,'std',0.0):.6g}", "m",
        f"{L.uncertainty:.6g}", "m"
    )

    # ---- Instrument specs summary (optional but nice for the report) ----
    dpL_inst, dpS_inst, dpE_inst = dp_inst_tuple
    row("GLOBAL","", "DP inst unc (large channel)", f"{dpL_inst:.6g}", "psi", f"{dpL_inst:.6g}", "psi", "0", "psi", f"{dpL_inst:.6g}", "psi")
    row("GLOBAL","", "DP inst unc (small channel)", f"{dpS_inst:.6g}", "psi", f"{dpS_inst:.6g}", "psi", "0", "psi", f"{dpS_inst:.6g}", "psi")
    row("GLOBAL","", "DP inst unc (elbow channel)", f"{dpE_inst:.6g}", "psi", f"{dpE_inst:.6g}", "psi", "0", "psi", f"{dpE_inst:.6g}", "psi")
    row("GLOBAL","", "Temp inst unc", f"{temp_inst_unc:.6g}", "degC", f"{temp_inst_unc:.6g}", "degC", "0", "degC", f"{temp_inst_unc:.6g}", "degC")

    # ---- Thermodynamic inputs ----
    # mu is assumed unless you have a viscometer measurement. We list it but keep the uncertainty 0 (as you set).
    row("GLOBAL","", "Dynamic viscosity mu (assumed)", f"{mu.value:.6g}", "Pa*s",
        f"{getattr(mu,'inst_unc',0.0):.6g}", "Pa*s",
        f"{getattr(mu,'std',0.0):.6g}", "Pa*s",
        f"{mu.uncertainty:.6g}", "Pa*s"
    )

    # ---- Per-run per-point measured parameters ----
    for run in runs:
        for i, pct in enumerate(run.pcts):
            Q = run.Q_lpm[i]
            dpL = run.dp_large_psid[i]
            dpS = run.dp_small_psid[i]
            dpE = run.dp_elbow_psid[i]
            T  = run.T_C[i]

            # Flow
            row(run.name, pct, "Flow rate Q", f"{Q.value:.6g}", "LPM",
                f"{Q.inst_unc:.6g}", "LPM",
                f"{Q.std:.6g}", "LPM",
                f"{Q.uncertainty:.6g}", "LPM"
            )

            # Pressures
            row(run.name, pct, "DP large channel", f"{dpL.value:.6g}", "psi",
                f"{dpL.inst_unc:.6g}", "psi",
                f"{dpL.std:.6g}", "psi",
                f"{dpL.uncertainty:.6g}", "psi"
            )
            row(run.name, pct, "DP small channel", f"{dpS.value:.6g}", "psi",
                f"{dpS.inst_unc:.6g}", "psi",
                f"{dpS.std:.6g}", "psi",
                f"{dpS.uncertainty:.6g}", "psi"
            )
            row(run.name, pct, "DP elbow channel", f"{dpE.value:.6g}", "psi",
                f"{dpE.inst_unc:.6g}", "psi",
                f"{dpE.std:.6g}", "psi",
                f"{dpE.uncertainty:.6g}", "psi"
            )

            # Temperature
            row(run.name, pct, "Water temperature T", f"{T.value:.6g}", "degC",
                f"{T.inst_unc:.6g}", "degC",
                f"{T.std:.6g}", "degC",
                f"{T.uncertainty:.6g}", "degC"
            )

            # Density is NOT a direct measurement here—computed from T.
            # We include it but label it explicitly so your “measurement related uncertainty” claim stays clean.
            if hasattr(run, "rho") and len(run.rho) == len(run.pcts):
                rho = run.rho[i]
                row(run.name, pct, "Density rho (computed from T)", f"{rho.value:.6g}", "kg/m^3",
                    "N/A", "kg/m^3", "N/A", "kg/m^3",
                    f"{rho.uncertainty:.6g}", "kg/m^3"
                )


def print_kl_table(run, pipe_type="small"):
    """
    Prints table rows for Excel paste.
    Nominal flow column is ignored.
    Rows printed 100% -> 0%.
    """

    if pipe_type == "large":
        dp_list = run.dp_large_psid
    elif pipe_type == "small":
        dp_list = run.dp_small_psid
    else:
        raise ValueError("pipe_type must be 'large' or 'small'")

    header = (
        "Actual Flow (LPM),"
        "Flow Uncertainty (± LPM),"
        "Head Loss Δh (in),"
        "Std Dev Head Loss (in),"
        "Total Head Loss Uncertainty (± in),"
        "Loss Coefficient KL,"
        "KL Uncertainty"
    )
    print(header)

    for i in reversed(range(len(run.Q_lpm))):

        Q_act = run.Q_lpm[i].value
        Q_unc = run.Q_lpm[i].uncertainty

        hl = run.head_loss[i].value
        hl_unc_total = run.head_loss[i].uncertainty

        # std dev from DP only
        dp_std = dp_list[i].std
        rho = run.rho[i].value

        hl_std = (dp_std * 6894.76) / (rho * 9.81) * 39.37007874

        KL = run.kl[i].value
        KL_unc = run.kl[i].uncertainty

        print(
            f"{Q_act:.3f},"
            f"{Q_unc:.4f},"
            f"{hl:.3f},"
            f"{hl_std:.3f},"
            f"{hl_unc_total:.4f},"
            f"{KL:.4f},"
            f"{KL_unc:.4f}"
        )

def print_re_le_table(run):
    """
    Prints table rows for Excel paste.
    Nominal flow column is ignored.
    Rows printed 100% -> 0%.
    """

    

    header = (
        "Re +- Uncertainty,"
        "Le +- Uncertainty,"
    )
    print(header)

    for i in reversed(range(len(run.Q_lpm))):

        Re_act = run.Re[i].value
        Re_unc = run.Re[i].uncertainty

        Le = run.Le[i].value
        Le_unc = run.Le[i].uncertainty

        print(
            f"{Re_act:.3f} ± {Re_unc:.4f},"
            f"{Le:.4f} ± {Le_unc:.4f}"
        )


# Example outputs:
print_run(large_run2)
print_head_loss_table(small_run1, pipe_type="small")

print_kl_table(large_run2, pipe_type="large")

print_re_le_table(small_run1)
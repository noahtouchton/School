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
    
    


for run in (large_run1, large_run2, large_run3):
    compute_rho_and_hl(run, run.dp_large_psid, size="large")

compute_rho_and_hl(small_run1, small_run1.dp_small_psid, size="small")


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
plt.title("Head Loss vs Flowrate²")
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
plt.title("Friction Factor vs Reynolds Number")
plt.legend()
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.tight_layout()

out_dir = os.path.dirname(os.path.abspath(__file__))
fname = os.path.join(out_dir, "moody_f_vs_Re.png")
plt.savefig(fname, dpi=200)
plt.close()
print(f"Saved Moody-style plot: {fname}")

# Example outputs:
print_run(large_run1)
print_head_loss_table(small_run1, pipe_type="small")

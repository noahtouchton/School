# uncertainty.py
import numpy as np
import math

from rss import Data, load_large_run, load_small_run1, RSS
import matplotlib.pyplot as plt
import os
equations = [
    "rho = 1000 * (1 - ((T + 288.9414) / (508929.2 * (T + 68.12963))) * (T - 3.9863)**2)",
    "hl = 6894.76 * 39.37007874*dp/(rho*g)", # convert dp from psi to Pa and m to in

]

# -------------------------------------------------
# Pipe diameters
# -------------------------------------------------
big_vals = [18.86, 18.84, 18.89, 18.82, 18.85]   # mm
small_vals = [9.42, 9.40, 9.41, 9.42, 9.42]      # mm

caliper_unc_mm = 0.02  # mm (reasonable digital caliper)
caliper_unc_m = caliper_unc_mm / 1000.0

big_std_m = np.std(big_vals, ddof=1) / 1000.0
small_std_m = np.std(small_vals, ddof=1) / 1000.0

g = Data("Gravity", "g", 9.81, 0.0)  # m/s², assume exact

d_big = Data(
    "Large Pipe Diameter",
    "d",
    np.mean(big_vals) / 1000.0,
    math.sqrt(caliper_unc_m**2 + big_std_m**2),
)

d_small = Data(
    "Small Pipe Diameter",
    "d",
    np.mean(small_vals) / 1000.0,
    math.sqrt(caliper_unc_m**2 + small_std_m**2),
)


# -------------------------------------------------
# Instrument uncertainties (set these from your equipment)
# dp_inst_unc: uncertainty in PSID for the DP channels you use
# temp_inst_unc: uncertainty in deg C (thermocouple)
#
# NOTE: If you want different DP uncertainties per channel (1 psi vs 15 psi),
# we can split dp_inst_unc into dp_large_unc, dp_small_unc, dp_elbow_unc later.
# -------------------------------------------------
DP_INST_UNC = 0.0008   # psi (example: PX409 1 psi range -> ±0.0008 psi)
TEMP_INST_UNC = 2.2    # °C (Type J standard limits, conservative)


# -------------------------------------------------
# Load runs (arrays)
# -------------------------------------------------
large_run1 = load_large_run(1, base_dir="data", dp_inst_unc=DP_INST_UNC, temp_inst_unc=TEMP_INST_UNC)
large_run2 = load_large_run(2, base_dir="data", dp_inst_unc=DP_INST_UNC, temp_inst_unc=TEMP_INST_UNC)
large_run3 = load_large_run(3, base_dir="data", dp_inst_unc=DP_INST_UNC, temp_inst_unc=TEMP_INST_UNC)

small_run1 = load_small_run1(base_dir="data", dp_inst_unc=DP_INST_UNC, temp_inst_unc=TEMP_INST_UNC)


# -------------------------------------------------
# Print uncertainties (all points)
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


for run in (large_run1, large_run2, large_run3):
    run.rho = []
    run.head_loss = []

    for T in run.T_C:
        rho = RSS("density", equations[0], [T])
        run.rho.append(rho)
    for dp in run.dp_large_psid:
        hl = RSS("head_loss", equations[1], [dp, run.rho[0], g])
        run.head_loss.append(hl)

for run in (small_run1,):
    run.rho = []
    run.head_loss = []

    for T in run.T_C:
        rho = RSS("density", equations[0], [T])
        run.rho.append(rho)
    for dp in run.dp_small_psid:
        hl = RSS("head_loss", equations[1], [dp, run.rho[0], g])
        run.head_loss.append(hl)


def get_arrays(run):
    Q = np.array([d.value for d in run.Q_lpm], dtype=float)
    Q_unc = np.array([d.uncertainty for d in run.Q_lpm], dtype=float)

    Q2 = Q**2
    Q2_unc = 2 * np.abs(Q) * Q_unc

    hl = np.array([d.value for d in run.head_loss], dtype=float)
    hl_unc = np.array([d.uncertainty for d in run.head_loss], dtype=float)

    return Q2, Q2_unc, hl, hl_unc

plt.figure()

runs = [
    (large_run1, "Large Run 1"),
    (large_run2, "Large Run 2"),
    (large_run3, "Large Run 3"),
    (small_run1, "Small Run"),
]

for run, label in runs:
    Q2, Q2_unc, hl, hl_unc = get_arrays(run)

    plt.errorbar(
        Q2, hl,
        xerr=Q2_unc,
        yerr=hl_unc,
        fmt='o',
        capsize=3,
        label=label
    )

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

#print_run(large_run1)
#print_run(large_run2)
#print_run(large_run3)
print_run(small_run1)
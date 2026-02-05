# uncertainty.py
import numpy as np
import math

from rss import Data, load_large_run, load_small_run1


# -------------------------------------------------
# Pipe diameters
# -------------------------------------------------
big_vals = [18.86, 18.84, 18.89, 18.82, 18.85]   # mm
small_vals = [9.42, 9.40, 9.41, 9.42, 9.42]      # mm

caliper_unc_mm = 0.02  # mm (reasonable digital caliper)
caliper_unc_m = caliper_unc_mm / 1000.0

big_std_m = np.std(big_vals, ddof=1) / 1000.0
small_std_m = np.std(small_vals, ddof=1) / 1000.0

d_big = Data(
    "Large Pipe Diameter",
    "d_big",
    np.mean(big_vals) / 1000.0,
    math.sqrt(caliper_unc_m**2 + big_std_m**2),
)

d_small = Data(
    "Small Pipe Diameter",
    "d_small",
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


print_run(large_run1)
print_run(large_run2)
print_run(large_run3)
print_run(small_run1)
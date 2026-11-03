import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

# extract all data from cvs into data frames

# Part 1
# model 1 zeta = 0.1, model 2 zeta = 0.5, model 3 zeta = 0.5
# each model has one length of L = 1.0m

part1_modelA_files = [Path("Data/GT_BridgeCrane_Data_0.1.csv")]
part1_modelB_files = [Path("Data/GT_BridgeCrane_Data_0.5.csv")]
part1_modelC_files = [Path("Data/GT_BridgeCrane_Data_0.8.csv")]

part1_data = [
    [pd.read_csv(f) for f in part1_modelA_files],
    [pd.read_csv(f) for f in part1_modelB_files],
    [pd.read_csv(f) for f in part1_modelC_files],
]

# Part 2 (i)
# fixed Wn and move zeta_m

part2_files = [
    "Data/GT_BridgeCrane_Data_i_0.6.csv",  #
    "Data/GT_BridgeCrane_Data_i_0.7.csv",  #
    "Data/GT_BridgeCrane_Data_i_0.9.csv",  #
    "Data/GT_BridgeCrane_Data_i_1.0.csv",  #
    "Data/GT_BridgeCrane_Data_i_1.1.csv",  #
    "Data/GT_BridgeCrane_Data_i_1.2.csv",  #
    "Data/GT_BridgeCrane_Data_i_1.5.csv",  #
]

part2_data = [pd.read_csv(f) for f in part2_files]

# Part 3 (ii)
# fixed zeta_m and move Wn

part3_files = [
    "Data/GT_BridgeCrane_Data_ii_5%.csv",  #
    "Data/GT_BridgeCrane_Data_ii_10%.csv",  #
    "Data/GT_BridgeCrane_Data_ii_25%.csv",  #
    "Data/GT_BridgeCrane_Data_ii_35%.csv",  #
    "Data/GT_BridgeCrane_Data_ii_50%.csv",  #
]

part3_data = [pd.read_csv(f) for f in part3_files]



# Make Plots

plt.figure(figsize=(10,6))

labels_part1 = [r"$\zeta_m = 0.1$", r"$\zeta_m = 0.5$", r"$\zeta_m = 0.5$"]

for model_runs, label in zip(part1_data, labels_part1):
    # trim the data frame first
    df = model_runs[0].copy()
    df.columns = df.columns.str.strip()

    df_clean = df[df["Time Y-dir (sec)"] < 100.0]
    df_clean = df_clean.sort_values(by="Time Y-dir (sec)")


    plt.plot(
        df_clean["Time Y-dir (sec)"],
        df_clean["Y Payload Deflection (mm)"],
        label = label,
        linewidth = 1.5,
    )

plt.xlabel("Time (s)")
plt.ylabel("Payload Deflection (mm)")
plt.title("Models Deflection at Fixed Length")
plt.grid(True, linestyle=":", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()
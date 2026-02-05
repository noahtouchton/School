# uncertainty.py
import os
import pandas as pd
import numpy as np
from rss import Data, RSS, linear_monte_carlo
import sympy as sp

from pathlib import Path
import re
from rss import load_lab_dataframe

equations = [
  rho = 1000 * (1 - ((T + 288.9414) / (508929.2 * (T + 68.12963))) * (T - 3.9863)**2),

]




PCTS = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]


def load_pipe_loss_data(base_dir="data"):
    """
    Loads all pipe loss lab files into dictionaries of DataFrames.

    Expected structure:
      data/
        Large Pipe/
          Run 1/
            largepipe_run1_0.xls ... largepipe_run1_100.xls
          Run 2/
            largepipe_run2_0.xls ... largepipe_run2_100.xls
          Run 3/
            largepipe_run3_0.xls ... largepipe_run3_100.xls
        Small Pipe/
          smallpipe_run1_0.xls ... smallpipe_run1_100.xls

    Returns:
      large: dict[int run][int pct] -> DataFrame
      small: dict[int pct] -> DataFrame
    """
    base_dir = Path(base_dir)

    # ---------------- Large Pipe ----------------
    large = {}
    large_root = base_dir / "Large Pipe"
    for run in [1, 2, 3]:
        run_dir = large_root / f"Run {run}"
        if not run_dir.exists():
            raise FileNotFoundError(f"Missing folder: {run_dir}")

        large[run] = {}
        for pct in PCTS:
            f = run_dir / f"largepipe_run{run}_{pct}.xls"
            if not f.exists():
                raise FileNotFoundError(f"Missing file: {f}")

            large[run][pct] = load_lab_dataframe(f)

    # ---------------- Small Pipe ----------------
    small = {}
    small_root = base_dir / "Small Pipe"
    if not small_root.exists():
        raise FileNotFoundError(f"Missing folder: {small_root}")

    for pct in PCTS:
        f = small_root / f"smallpipe_run1_{pct}.xls"
        if not f.exists():
            raise FileNotFoundError(f"Missing file: {f}")

        small[pct] = load_lab_dataframe(f)

    return large, small



# loads the data
large, small = load_pipe_loss_data("data")

# quick sanity prints
print("Loaded large runs:", list(large.keys()))
print("Loaded large pcts for Run 1:", list(large[1].keys()))
print("Loaded small pcts:", list(small.keys()))

print("\nExample Large Run2 40% head:")
print(large[2][40].head())

print("\nExample Small 70% columns:")
print(small[70].columns.tolist())
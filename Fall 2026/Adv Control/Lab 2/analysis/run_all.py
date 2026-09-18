#!/usr/bin/env python3
"""
ME6404 Lab 2 - run the whole reduction.

    python3 analysis/run_all.py

Order matters: each step reads the tables written by the previous one.

  s1_inventory   scan every raw file, catch duplicates and mislabelled files,
                 read the tower shaper identities out of the command profiles
  s2_bridge      per-trial bridge residual metrics + applied-command read-back
  s3_tower       per-trial tower residual metrics, split by mode
  s4_shapers     frequencies, shaper designs, sensitivity curves, predictions
  s5_figures     report-ready figures
  s6_report_data assemble results/REPORT_DATA.md

Everything lands in results/.  Nothing under data/ is ever modified.
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import s1_inventory
import s2_bridge
import s3_tower
import s4_shapers
import s5_figures
import s6_report_data

STEPS = [
    ("inventory + data-quality screen", s1_inventory),
    ("bridge crane (Part 1)", s2_bridge),
    ("tower crane (Part 2)", s3_tower),
    ("shaper design + sensitivity", s4_shapers),
    ("figures", s5_figures),
    ("REPORT_DATA.md", s6_report_data),
]


def main():
    t0 = time.time()
    for i, (name, mod) in enumerate(STEPS, start=1):
        print(f"\n{'#' * 78}\n# step {i}/{len(STEPS)}: {name}\n{'#' * 78}")
        mod.main()
    print(f"\ndone in {time.time() - t0:.1f} s")
    print("\nStart here:")
    print("  results/REPORT_DATA.md   numbers and argument, organised for the write-up")
    print("  results/DATA_QUALITY.md  what was thrown out and why - read before citing")
    print("  results/figures/         17 report-ready figures")
    print("  results/tables/          every number as CSV")


if __name__ == "__main__":
    main()

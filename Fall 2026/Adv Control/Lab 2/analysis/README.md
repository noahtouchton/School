# ME6404 Lab 2 - data reduction

```bash
python3 analysis/run_all.py
```

Reads everything under `data/`, writes everything under `results/`. Never modifies
`data/`. Needs numpy, pandas, matplotlib, openpyxl.

## Where to start

| file | what it is |
|---|---|
| `results/REPORT_DATA.md` | the numbers and the argument, in the order the handout asks for the write-up. **Start here.** |
| `results/DATA_QUALITY.md` | which files were set aside and why. Read before citing any trial by name. |
| `results/figures/` | 12 report-ready figures, each with one message |
| `results/tables/` | every number as CSV |

## Pipeline

| step | does |
|---|---|
| `s1_inventory.py` | scans every raw file, decodes the test condition, then **checks that claim against the file contents**: MD5 duplicates, filename vs the in-file cable-length tag, tower shaper identity read out of the command profile |
| `s2_bridge.py` | bridge residual metrics per trial, plus the applied impulse sequence reconstructed from the measured velocity |
| `s3_tower.py` | tower residual metrics per trial, split into the two pendulum modes, plus command-execution diagnostics |
| `s4_shapers.py` | frequencies, shaper designs and design rationale, sensitivity curves, predicted residuals |
| `s5_figures.py` | the figures |
| `s6_report_data.py` | assembles `REPORT_DATA.md` |
| `lab2lib.py` | shared loaders, segmentation, residual metrics and shaper mathematics |

## The three things that were not obvious in the raw data

1. **The bridge CSVs only have live data in the Y-direction columns.** `Time X-dir`
   and `X Command Velocity` are neither a time nor a velocity - both sit at a
   constant equal to the **cable length in mm**. That is used as ground truth to
   catch two mislabelled files. `Y Command Velocity` is never logged, so the
   applied shaper has to be reconstructed from the measured velocity staircase.

2. **Each tower workbook is two acquisition blocks on different clocks** - a 30 ms
   motion stream with the vision columns blank, then a 20 ms vision stream with the
   motion columns frozen. The vision block starts ~15 s after the move ends, so all
   tower swing amplitudes are post-decay.

3. **The "ZVD" trials on both rigs were run with amplitudes [0.25, 0.50, 0.50]**,
   summing to 1.25 instead of 1.00, read back independently from each rig's data.
   On the bridge that makes the move 27 % longer and removes the notch from the
   sensitivity curve; on the tower it saturated the slew rate limit. This is why
   the robust shaper did not beat the ZV, and it is the single most important
   thing to get right in the write-up.

## Metric definitions

- **Bridge residual** `amp_pp_mm`: largest peak-to-peak payload deflection within
  the first 4 measured periods after the trolley stops, between the end of the
  outbound move and the start of the return move. The window is capped at a fixed
  number of periods so trials with different record lengths stay comparable.
- **Tower residual** `tan_/rad_amp_pp_rad`: same idea over 5 mode-1 periods of the
  vision record, for the tangential (direction of travel) and radial (along the
  jib) swing channels, each also band-split per pendulum mode.
- **`V_rel`**: residual vibration of the applied command relative to the unshaped
  command at the same condition. On these rigs the command is the shaper convolved
  with a velocity pulse whose impulses sum to zero, so the usual normalisation by
  the impulse sum is undefined - `lab2lib.pulse_residual_ratio` normalises by the
  unshaped pulse instead.

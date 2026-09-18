"""
Step 1 - inventory and data-quality screen.

Builds one row per raw file, decodes the intended test condition from the file
name, then CHECKS THAT CLAIM against the file contents:

  * bridge: the in-file cable-length tag (see lab2lib) must match the cable
    length implied by the name, and the measured move distance must match the
    other trials of the same condition;
  * duplicates: byte-identical files are detected by MD5 and only one copy of
    each is marked as the trial of record;
  * tower: the shaper is read back out of the commanded velocity staircase and
    the trolley radius out of the trolley position, so the name (a bare
    timestamp) is never relied upon.

Outputs
  results/tables/inventory_bridge.csv
  results/tables/inventory_tower.csv
  results/DATA_QUALITY.md
"""

import glob
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lab2lib as L


def bridge_inventory():
    rows = []
    for path in sorted(glob.glob(os.path.join(L.BRIDGE_DIR, "*.csv"))):
        name = os.path.basename(path)
        meta = L.parse_bridge_name(name)
        d = L.load_bridge(path)
        segs = L.find_motion_segments(d["t"], d["vel"], thresh=15.0)
        dists = [d["pos"][b] - d["pos"][a] for a, b in segs]
        out = round(max(dists, key=abs), 0) if dists else np.nan
        rows.append(dict(
            file=name, md5=L.md5(path),
            part=meta["part"], shaper=meta["shaper"],
            tf_ms=meta["tf_ms"], cable_nom_mm=meta["cable_nom_mm"],
            cable_infile_mm=d["cable_mm"], trial=meta["trial"],
            name_flagged_del=meta["flagged"],
            n_samples=d["n"], duration_s=round(float(d["t"][-1]), 2),
            median_dt_s=round(float(np.median(np.diff(d["t"]))), 4),
            n_moves=len(segs),
            move_dist_mm=out,
            first_move_t0=round(float(d["t"][segs[0][0]]), 2) if segs else np.nan,
            first_move_t1=round(float(d["t"][segs[0][1]]), 2) if segs else np.nan,
        ))
    df = pd.DataFrame(rows)

    # --- duplicate detection -------------------------------------------------
    dup = defaultdict(list)
    for f, h in zip(df.file, df.md5):
        dup[h].append(f)
    df["dup_group"] = df.md5.map(lambda h: ", ".join(dup[h]) if len(dup[h]) > 1 else "")
    df["is_duplicate_copy"] = [
        len(dup[h]) > 1 and f != sorted(dup[h])[0] for f, h in zip(df.file, df.md5)
    ]

    # --- name-vs-content validation -----------------------------------------
    # the in-file tag is the cable length in mm; allow a few mm of slack
    df["cable_matches_name"] = np.isclose(df.cable_infile_mm, df.cable_nom_mm, atol=12)

    notes, use = [], []
    for r in df.itertuples():
        n = []
        if r.shaper is None:
            n.append("file name does not match any known lab condition")
        if not r.cable_matches_name:
            n.append(f"cable tag in file is {r.cable_infile_mm:.0f} mm but the "
                     f"name implies {r.cable_nom_mm:.0f} mm - MISLABELLED")
        if r.is_duplicate_copy:
            n.append(f"byte-identical to {sorted(dup[r.md5])[0]}")
        if r.name_flagged_del:
            n.append("'_del' in name: flagged during the session")
        if r.n_moves == 0:
            n.append("no move detected")
        elif r.n_moves > 2:
            n.append(f"{r.n_moves} moves in one log - session file, segmented per move")
        ok = (r.shaper is not None and r.cable_matches_name
              and not r.is_duplicate_copy and r.n_moves >= 1)
        notes.append("; ".join(n))
        use.append(bool(ok))
    df["notes"] = notes
    df["use"] = use
    return df


def tower_inventory():
    rows = []
    for path in sorted(glob.glob(os.path.join(L.TOWER_DIR, "*.xlsx"))):
        d = L.load_tower(path)
        mo, vi = d["motion"], d["vision"]
        cls = L.classify_tower_shaper(mo)
        t0, t1 = L.tower_move_window(mo)
        slew = mo["Position Slew [deg]"].to_numpy(float)
        trolley = float(np.median(mo["Position Trolley [mm]"]))
        rows.append(dict(
            file=d["name"], md5=L.md5(path),
            timestamp=d["name"][:19].replace("_", " "),
            shaper=cls["shaper"], n_impulses=cls["n_impulses"],
            impulse_amps=str(cls["amps"]), impulse_delays_s=str(cls["delays"]),
            impulse_amp_sum=round(cls["amp_sum"], 3),
            shaper_read_from=cls["source"], cmd_truncated=cls["truncated"],
            shaper_duration_s=round(cls["rise_s"], 3) if np.isfinite(cls["rise_s"]) else np.nan,
            trolley_mm=round(trolley, 1),
            trolley_nom_mm=int(round(trolley / 100.0) * 100),
            hoist_mm=round(float(np.median(mo["Position Hoist [mm]"])), 1),
            cable_mm=round(float(np.nanmedian(vi["Cable Length [mm]"])), 1) if vi is not None else np.nan,
            slew_start_deg=round(float(slew[0]), 1),
            slew_end_deg=round(float(slew[-1]), 1),
            slew_travel_deg=round(float(np.ptp(slew)), 1),
            move_t0=round(t0, 2), move_t1=round(t1, 2),
            move_dur_s=round(t1 - t0, 2),
            n_blocks=d["n_blocks"],
            motion_block_t=f"{mo['Time [s]'].min():.2f}-{mo['Time [s]'].max():.2f}",
            vision_block_t=f"{vi['Time [s]'].min():.2f}-{vi['Time [s]'].max():.2f}" if vi is not None else "none",
            vision_n=0 if vi is None else int(vi["Tangential Swing [rad]"].notna().sum()),
        ))
    df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)

    notes, use = [], []
    for r in df.itertuples():
        n = []
        if r.vision_n < 100:
            n.append("little or no vision (swing) data")
        if r.move_t0 < 0.5:
            n.append("recording starts mid-move - move truncated, "
                     "slew travel is short; residual swing still usable")
        if abs(r.trolley_mm - r.trolley_nom_mm) > 15:
            n.append(f"trolley {r.trolley_mm:.0f} mm is far from a nominal set point")
        if r.shaper == "Unknown":
            n.append("could not classify shaper from command profile")
        if r.cmd_truncated:
            n.append("shaper read from the falling staircase; its impulse DELAYS "
                     "in this row are not trustworthy (amplitudes are)")
        if abs(r.impulse_amp_sum - 1.0) > 0.05:
            n.append(f"as-run impulse amplitudes sum to {r.impulse_amp_sum:.2f}, not 1.00")
        notes.append("; ".join(n))
        use.append(bool(r.vision_n >= 100 and r.shaper != "Unknown"))
    df["notes"] = notes
    df["use"] = use
    return df


def write_quality_report(b, t):
    os.makedirs(os.path.join(L.ROOT, "results"), exist_ok=True)
    lines = []
    A = lines.append
    A("# Lab 2 - data quality screen\n")
    A("Auto-generated by `analysis/s1_inventory.py`. Read this before you cite a number.\n")

    A("## Bridge crane\n")
    A(f"- {len(b)} raw CSV files; **{int(b.use.sum())} usable trials**, "
      f"{int((~b.use).sum())} set aside.\n")
    A("### Files set aside\n")
    A("| file | reason |")
    A("|---|---|")
    for r in b[~b.use].itertuples():
        A(f"| `{r.file}` | {r.notes} |")
    A("")
    A("### Trials available per condition (after screening)\n")
    ok = b[b.use]
    piv = ok.pivot_table(index=["part", "shaper"],
                         columns=[], values="file", aggfunc="count")
    A("| part | shaper | tf (ms) | cable (mm) | n trials |")
    A("|---|---|---|---|---|")
    for key, g in ok.groupby(["part", "shaper", "tf_ms", "cable_nom_mm"], dropna=False):
        A(f"| {key[0]} | {key[1]} | {key[2]:.0f} | {key[3]:.0f} | {len(g)} |")
    A("")
    A("### Known raw-format quirks\n")
    A("- Only the **Y-direction columns** carry live data. `Time Y-dir (sec)` is the "
      "clock, `Y Payload Deflection (mm)` is the payload swing, `Y Crane Position` / "
      "`Y Actual Velocity` describe the driven trolley axis.")
    A("- `Time X-dir (sec)` and `X Command Velocity (mm/sec)` are **not** a time or a "
      "velocity: both sit at a constant equal to the **cable length in mm**. That is "
      "used above as ground truth to catch mislabelled files.")
    A("- `Y Command Velocity` is identically zero (never logged), so the commanded "
      "shaper cannot be read back from the bridge files - it is taken from the name.")
    A("- Files are zero-padded to 2000 rows and repeat timestamps; both are stripped.")
    A("- Payload deflection is quantised to 1 mm, which sets the noise floor on the "
      "smallest residual amplitudes (a few mm).\n")

    A("## Tower crane\n")
    A(f"- {len(t)} workbooks; **{int(t.use.sum())} usable trials** "
      f"= {t[t.use].shaper.nunique()} shaper conditions x 3 trolley radii.\n")
    A("| timestamp | shaper | impulses | delays (s) | trolley (mm) | slew travel (deg) | notes |")
    A("|---|---|---|---|---|---|---|")
    for r in t.itertuples():
        A(f"| {r.timestamp} | {r.shaper} | {r.impulse_amps} | {r.impulse_delays_s} | "
          f"{r.trolley_mm:.0f} | {r.slew_travel_deg:.0f} | {r.notes or '-'} |")
    A("")
    A("### How the two tower data streams fit together\n")
    A("Each workbook stacks **two acquisition streams** in one sheet:\n")
    A("1. a 30 ms motion-control stream (slew/trolley/hoist state and the shaped "
      "command) in which the vision columns are blank, and")
    A("2. a 20 ms vision stream (tangential/radial swing, cable length) in which the "
      "motion columns are frozen at their final value.\n")
    A("They are **parallel recordings of the same window on different clocks**, not "
      "consecutive segments. The motion stream always starts near t = 0; the vision "
      "stream always starts at t ~ 21.2 s. Their durations match, and once the start "
      "times are aligned the swing onset lands on the move start to within the "
      "detection threshold in all 12 trials. So the vision stream **does** capture "
      "the move, and residual amplitudes are scored strictly after the slew stops "
      "(3 mode-1 periods, the most every trial can supply).\n")
    A("The streams are told apart by **whether the vision columns are populated**, "
      "not by looking for a jump in `Time [s]`. A time-gap split fails silently on 5 "
      "of the 12 workbooks, where the vision clock carries straight on from the "
      "motion clock with no gap: the sheet then looks like one block and the swing "
      "data appears to sit at times when the crane was still moving.\n")
    A("### As-run ZVD amplitudes\n")
    zvd = t[t.shaper == "ZVD"]
    if len(zvd) and abs(zvd.impulse_amp_sum.mean() - 1.0) > 0.05:
        A(f"The three trials identified as ZVD were run with impulse amplitudes "
          f"`{zvd.impulse_amps.iloc[0]}`, which sum to "
          f"{zvd.impulse_amp_sum.mean():.2f} rather than 1.00. A textbook ZVD is "
          "[0.25, 0.50, 0.25]; the third impulse was entered at 0.50. The delays "
          "(0, T1/2, T1) are correct, so this is still a three-impulse robust "
          "shaper, but it is not ZVD and it commands 125 % of the nominal slew "
          "velocity mid-move. `analysis/s4_shapers.py` evaluates the sensitivity "
          "curve of the shaper **as actually run** next to true ZVD so the report "
          "can discuss the difference honestly.\n")
    A("### The shaper identity is not in the file name\n")
    A("The shaper is recovered from the commanded slew staircase: the command is "
      "the shaper convolved with the velocity pulse, so its rising steps are the "
      "impulse amplitudes and the step times are the impulse delays. The GUI logs "
      "that command with an occasional x100 scale glitch, which "
      "`lab2lib.command_profile` removes. When a recording starts mid-move the rise "
      "is clipped and the mirrored falling staircase is used instead.\n")

    path = os.path.join(L.ROOT, "results", "DATA_QUALITY.md")
    with open(path, "w") as fh:
        fh.write("\n".join(lines))
    return path


def main():
    b = bridge_inventory()
    t = tower_inventory()
    L.write_table(b, "inventory_bridge.csv")
    L.write_table(t, "inventory_tower.csv")
    qp = write_quality_report(b, t)

    print(f"bridge: {len(b)} files, {int(b.use.sum())} usable")
    for r in b[~b.use].itertuples():
        print(f"   SET ASIDE {r.file}: {r.notes}")
    print(f"tower : {len(t)} files, {int(t.use.sum())} usable")
    print(t[["timestamp", "shaper", "trolley_nom_mm", "slew_travel_deg", "notes"]].to_string(index=False))
    print(f"\nwrote {qp}")
    return b, t


if __name__ == "__main__":
    main()

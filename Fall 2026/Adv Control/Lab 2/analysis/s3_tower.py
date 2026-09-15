"""
Step 3 - tower crane (Part 2) reduction.

Shaper identity and move geometry come from the motion block; residual hook
swing comes from the vision block.  Three things make the tower analysis
different from the bridge:

1.  TANGENTIAL vs RADIAL.  Tangential swing is along the direction of travel
    and is what the shaper is designed to cancel.  Radial swing is along the
    jib; it is driven by the centripetal/Coriolis terms of the rotating frame
    and is not a target of a translational shaper design, so it is the
    signature of slewing rather than straight-line motion.

2.  THE PAYLOAD IS A DOUBLE PENDULUM, so zero-crossing periods are
    meaningless - they count both modes at once.  Each swing channel is
    instead band-split about the two predicted mode frequencies and scored per
    mode (`lab2lib.mode_amplitudes`).  That split is what separates a
    single-mode shaper (cancels mode 1, ignores mode 2) from a two-mode shaper.

3.  COMMAND EXECUTION.  The shaped command was not always executed as
    designed, so commanded vs actual slew velocity, travel vs commanded area,
    and rate-limit saturation are measured for every trial.  Two findings come
    out of it and both matter for the report: the ZVD command asked for 125 %
    of the rate limit and saturated, and in the ZV trials the logged actual
    velocity never follows the logged half-amplitude steps.

    A residual prediction from each trial's own logged velocity is also
    reported (`pred_m*_from_velocity_rad`), but for the TOWER it is only
    indicative, not a test.  The tower runs at tf/T1 ~ 2.1-2.5, where the
    pulse response |sin(pi*tf/T)| turns over rapidly, so a 10 % error in the
    assumed mode frequency - and the measured frequencies are 10-16 % off the
    model - changes the predicted residual by a factor of several.  It is
    therefore computed at BOTH the model and the measured frequencies so the
    spread is visible, and no conclusion is drawn from a single value.  (On the
    bridge, where tf/T ~ 0.6-1.1, the same prediction is well conditioned and
    is used quantitatively.)

The vision stream comes online ~15 s after the move ends, so all tower
amplitudes are post-decay; see results/DATA_QUALITY.md.

Outputs
  results/tables/tower_trials.csv    one row per trial, everything
  results/tables/tower_summary.csv   by shaper, averaged over the 3 radii
  results/tables/tower_modes.csv     per-mode amplitudes and frequencies
  results/tables/tower_tracking.csv  command-execution diagnostics
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lab2lib as L

SWING_QUANT = 0.002      # rad, vision resolution
MIN_SWING = 2 * SWING_QUANT
# Score a fixed span of MODE-1 periods.  A zero-crossing-based cycle cap is
# useless on a double pendulum: the crossings count both modes, so the apparent
# "period" is a fraction of a second and the window collapses to a sliver.
N_MODE1_PERIODS = 5
FULL_SLEW_CMD = 32.4     # deg/s at 100 % speed
TF_MS = 4000             # commanded pulse length for every tower trial
CHANNELS = (("Tangential Swing [rad]", "tan"), ("Radial Swing [rad]", "rad"))


def tracking_diagnostics(mo):
    """How faithfully did the crane execute the shaped command?"""
    t = mo["Time [s]"].to_numpy(float)
    c = mo["Command Velo Slew [deg/s]"].to_numpy(float)
    if np.nanmax(np.abs(c)) > 200:      # the x100 logging glitch
        c = c / 100.0
    v = mo["Actual Velo Slew [deg/s]"].to_numpy(float)
    p = mo["Position Slew [deg]"].to_numpy(float)

    out = dict(
        cmd_area_deg=round(float(np.trapezoid(c, t)), 1),
        actual_travel_deg=round(float(np.trapezoid(v, t)), 1),
        cmd_peak_deg_s=round(float(np.max(c)), 1),
        actual_peak_deg_s=round(float(np.max(v)), 1),
    )
    out["travel_over_cmd_area"] = round(out["actual_travel_deg"] / out["cmd_area_deg"], 3) \
        if out["cmd_area_deg"] else np.nan
    # the rate limit clipped the command if the command asked for more than the
    # crane ever delivered anywhere in the record
    out["rate_saturated"] = bool(out["cmd_peak_deg_s"] > 1.05 * out["actual_peak_deg_s"])

    # overshoot on the FIRST commanded step (breakaway from rest)
    nz = np.nonzero(c > 0.01)[0]
    over = np.nan
    if nz.size:
        i0 = nz[0]
        lvl = c[i0]
        hold = np.nonzero(np.abs(c[i0:] - lvl) > 1e-6)[0]
        i1 = i0 + (hold[0] if hold.size else c.size - i0)
        if i1 > i0 + 2 and lvl > 0:
            over = float(np.max(v[i0:i1]) / lvl - 1.0)
    out["first_step_cmd_deg_s"] = round(float(c[nz[0]]), 1) if nz.size else np.nan
    out["first_step_overshoot_pct"] = round(100 * over, 1) if np.isfinite(over) else np.nan
    return out


def analyse(path):
    d = L.load_tower(path)
    mo, vi = d["motion"], d["vision"]
    cls = L.classify_tower_shaper(mo)
    t0, t1 = L.tower_move_window(mo)

    trolley = float(np.median(mo["Position Trolley [mm]"]))
    cable_mm = float(np.nanmedian(vi["Cable Length [mm]"])) if vi is not None else np.nan
    L1 = cable_mm / 1000.0 if np.isfinite(cable_mm) else 0.6
    w1, w2, R, beta = L.double_pendulum_freqs(L1, L.L2_RIG)
    f1, f2 = w1 / (2 * np.pi), w2 / (2 * np.pi)

    row = dict(
        file=d["name"], timestamp=d["name"][:19].replace("_", " "),
        shaper=cls["shaper"], n_impulses=cls["n_impulses"],
        impulse_amps=str(cls["amps"]), impulse_delays_s=str(cls["delays"]),
        impulse_amp_sum=round(cls["amp_sum"], 3),
        shaper_duration_s=round(cls["rise_s"], 3),
        cmd_truncated=cls["truncated"],
        trolley_mm=round(trolley, 1),
        trolley_nom_mm=int(round(trolley / 100.0) * 100),
        cable_mm=round(cable_mm, 1),
        move_t0=round(t0, 2), move_t1=round(t1, 2), move_dur_s=round(t1 - t0, 2),
        R_mass_ratio=round(R, 4),
        w1_theory=round(w1, 4), w2_theory=round(w2, 4),
        f1_theory_hz=round(f1, 4), f2_theory_hz=round(f2, 4),
        T1_theory_s=round(2 * np.pi / w1, 4), T2_theory_s=round(2 * np.pi / w2, 4),
    )
    row.update(tracking_diagnostics(mo))

    # --- model-free residual predicted by the logged velocity --------------
    tm = mo["Time [s]"].to_numpy(float)
    v_tan = np.deg2rad(mo["Actual Velo Slew [deg/s]"].to_numpy(float)) * (trolley / 1000.0)
    row["tf_effective_s"] = round(
        float(np.trapezoid(mo["Actual Velo Slew [deg/s]"].to_numpy(float), tm))
        / max(float(np.max(mo["Actual Velo Slew [deg/s]"])), 1e-9), 3)
    for tag, wmode in (("m1", w1), ("m2", w2)):
        amp = L.predicted_residual_from_velocity(tm, v_tan, wmode, t0 - 1.0, t1 + 1.0)
        row[f"pred_{tag}_from_velocity_rad"] = round(
            2 * amp / (trolley / 1000.0), 5) if np.isfinite(amp) else np.nan

    # --- residual swing, overall and per mode -------------------------------
    if vi is not None:
        tv = vi["Time [s]"].to_numpy(float)
        T1 = 2 * np.pi / w1
        for chan, tag in CHANNELS:
            y = vi[chan].to_numpy(float)
            ok = np.isfinite(y)
            tvv, yy = tv[ok], y[ok]
            t_hi = min(tvv[-1], tvv[0] + N_MODE1_PERIODS * T1)
            r = L.residual_metrics(tvv, yy, tvv[0], t_hi, min_swing=MIN_SWING,
                                   detrend="linear")
            row[f"{tag}_amp_pp_rad"] = round(r.amp_pp_max, 5)
            row[f"{tag}_rms_rad"] = round(r.rms, 5)
            row[f"{tag}_amp_pp_mm"] = round(r.amp_pp_max * cable_mm, 1)

            modes = L.mode_amplitudes(tvv, yy, [f1, f2], min_swing=MIN_SWING,
                                      t_score_hi=t_hi)
            for m in modes:
                i = m["mode"]
                row[f"{tag}_m{i}_amp_pp_rad"] = round(m["amp_pp"], 5)
                row[f"{tag}_m{i}_amp_pp_mm"] = round(m["amp_pp"] * cable_mm, 1)
                row[f"{tag}_m{i}_rms_rad"] = round(m["rms"], 5)
                row[f"{tag}_m{i}_f_meas_hz"] = round(m["f_peak_hz"], 4)
                row[f"{tag}_m{i}_f_err_pct"] = round(
                    100 * (m["f_peak_hz"] - m["f_center_hz"]) / m["f_center_hz"], 1)
                row[f"{tag}_m{i}_f_unreliable"] = bool(m["f_at_band_edge"])
        row["scoring_window_s"] = round(float(t_hi - tv[0]), 2)
        row["vision_t0"] = round(float(tv[0]), 2)
        row["vision_t1"] = round(float(tv[-1]), 2)
        row["vision_delay_after_move_s"] = round(float(tv[0]) - t1, 2)
        row["total_amp_pp_rad"] = round(
            float(np.hypot(row["tan_amp_pp_rad"], row["rad_amp_pp_rad"])), 5)

    # --- what the as-run shaper should do, at theory and at measured freqs --
    A = np.array(cls["amps"], dtype=float)
    T = np.array(cls["delays"], dtype=float)
    if A.size and not cls["truncated"]:
        tf = TF_MS / 1000.0
        row["pred_V1_theory_freq"] = round(float(L.pulse_residual_ratio(A, T, tf, w1)), 4)
        row["pred_V2_theory_freq"] = round(float(L.pulse_residual_ratio(A, T, tf, w2)), 4)
        fm1 = row.get("tan_m1_f_meas_hz", np.nan)
        fm2 = row.get("tan_m2_f_meas_hz", np.nan)
        if np.isfinite(fm1):
            row["pred_V1_meas_freq"] = round(
                float(L.pulse_residual_ratio(A, T, tf, 2 * np.pi * fm1)), 4)
        if np.isfinite(fm2):
            row["pred_V2_meas_freq"] = round(
                float(L.pulse_residual_ratio(A, T, tf, 2 * np.pi * fm2)), 4)
    return row


def main():
    inv = pd.read_csv(os.path.join(L.TABLES, "inventory_tower.csv"))
    tr = pd.DataFrame([analyse(os.path.join(L.TOWER_DIR, f)) for f in inv[inv.use].file])

    order = {"Unshaped": 0, "ZV": 1, "ZVD": 2, "Two-mode ZV": 3}
    tr["_o"] = tr.shaper.map(lambda s: order.get(s, 9))
    tr = tr.sort_values(["_o", "trolley_nom_mm"]).drop(columns="_o").reset_index(drop=True)

    # --- normalise to the unshaped trial at the same trolley radius ---------
    base = tr[tr.shaper == "Unshaped"].set_index("trolley_nom_mm")
    for col in [c for c in tr.columns if c.endswith("_amp_pp_rad")]:
        tr["pct_" + col.replace("_amp_pp_rad", "")] = [
            round(100 * v / base.loc[k, col], 1)
            if k in base.index and base.loc[k, col] else np.nan
            for v, k in zip(tr[col], tr.trolley_nom_mm)]

    # --- how frequency-sensitive is that prediction? -----------------------
    # Recompute it at the MEASURED mode frequencies and show both, so the
    # report never leans on a number that a 10 % frequency error can move by a
    # factor of several.
    for tag, fcol in (("m1", "tan_m1_f_meas_hz"), ("m2", "tan_m2_f_meas_hz")):
        vals = []
        for r in tr.itertuples():
            d = L.load_tower(os.path.join(L.TOWER_DIR, r.file))
            mo = d["motion"]
            tm = mo["Time [s]"].to_numpy(float)
            rad = r.trolley_mm / 1000.0
            v_tan = np.deg2rad(mo["Actual Velo Slew [deg/s]"].to_numpy(float)) * rad
            f = getattr(r, fcol)
            if not np.isfinite(f):
                vals.append(np.nan)
                continue
            amp = L.predicted_residual_from_velocity(
                tm, v_tan, 2 * np.pi * f, r.move_t0 - 1.0, r.move_t1 + 1.0)
            vals.append(2 * amp / rad)
        tr[f"pred_{tag}_from_velocity_measfreq_rad"] = np.round(vals, 5)
        tr[f"pred_{tag}_freq_sensitivity"] = np.round(
            tr[f"pred_{tag}_from_velocity_measfreq_rad"]
            / tr[f"pred_{tag}_from_velocity_rad"].replace(0, np.nan), 2)

    # --- quality / interpretation notes ------------------------------------
    notes = []
    for r in tr.itertuples():
        n = []
        if r.cmd_truncated:
            n.append("recording starts mid-move: move is short and the impulse "
                     "delays could not be read back - amplitudes only")
        if r.rate_saturated:
            n.append(f"command peaked at {r.cmd_peak_deg_s:.0f} deg/s but the crane "
                     f"only reached {r.actual_peak_deg_s:.0f} deg/s - SLEW RATE "
                     "SATURATED, so the shaper was not executed as designed")
        if np.isfinite(r.first_step_overshoot_pct) and r.first_step_overshoot_pct > 40:
            n.append(f"logged actual velocity sits at {r.actual_peak_deg_s:.0f} deg/s "
                     f"while the logged command asks for {r.first_step_cmd_deg_s:.0f} "
                     "deg/s, i.e. the log shows the half-amplitude steps not being "
                     "applied at all")
        if np.isfinite(r.pred_m1_freq_sensitivity) and (
                r.pred_m1_freq_sensitivity > 2.5 or r.pred_m1_freq_sensitivity < 0.4):
            n.append(f"the velocity-based residual prediction changes by "
                     f"{r.pred_m1_freq_sensitivity:.1f}x between the model and the "
                     "measured mode-1 frequency - do not quote it as a number for "
                     "this trial, only as a trend")
        if getattr(r, "tan_m1_f_unreliable", False):
            n.append("mode-1 band peak sits on a band edge: there is essentially no "
                     "mode-1 content left to measure, so that frequency is not a "
                     "real reading")
        if np.isfinite(r.travel_over_cmd_area) and abs(r.travel_over_cmd_area - 1) > 0.08:
            n.append(f"crane travelled {r.actual_travel_deg:.0f} deg against a commanded "
                     f"{r.cmd_area_deg:.0f} deg ({r.travel_over_cmd_area:.2f}x)")
        notes.append("; ".join(n))
    tr["notes"] = notes

    L.write_table(tr.round(5), "tower_trials.csv")

    summ = tr.groupby("shaper", sort=False).agg(
        n=("file", "count"),
        shaper_dur_s=("shaper_duration_s", "mean"),
        cmd_area_deg=("cmd_area_deg", "mean"),
        actual_travel_deg=("actual_travel_deg", "mean"),
        tan_amp_pp_rad=("tan_amp_pp_rad", "mean"),
        tan_amp_pp_sd=("tan_amp_pp_rad", "std"),
        tan_m1_rad=("tan_m1_amp_pp_rad", "mean"),
        tan_m2_rad=("tan_m2_amp_pp_rad", "mean"),
        rad_amp_pp_rad=("rad_amp_pp_rad", "mean"),
        rad_m1_rad=("rad_m1_amp_pp_rad", "mean"),
        rad_m2_rad=("rad_m2_amp_pp_rad", "mean"),
        pct_tan=("pct_tan", "mean"),
        pct_tan_m1=("pct_tan_m1", "mean"),
        pct_tan_m2=("pct_tan_m2", "mean"),
        pct_rad=("pct_rad", "mean"),
        pred_V1=("pred_V1_theory_freq", "mean"),
        pred_V2=("pred_V2_theory_freq", "mean"),
        pred_V1_meas=("pred_V1_meas_freq", "mean"),
        pred_V2_meas=("pred_V2_meas_freq", "mean"),
        tf_effective_s=("tf_effective_s", "mean"),
    ).reset_index()
    L.write_table(summ.round(4), "tower_summary.csv")

    mode_cols = ["file", "shaper", "trolley_nom_mm", "cable_mm",
                 "f1_theory_hz", "f2_theory_hz"]
    mode_cols += [c for c in tr.columns if "_m1_" in c or "_m2_" in c]
    L.write_table(tr[mode_cols].round(5), "tower_modes.csv")

    track_cols = ["file", "shaper", "trolley_nom_mm", "impulse_amps", "impulse_amp_sum",
                  "impulse_delays_s", "shaper_duration_s", "cmd_area_deg",
                  "actual_travel_deg", "travel_over_cmd_area", "cmd_peak_deg_s",
                  "actual_peak_deg_s", "rate_saturated", "first_step_cmd_deg_s",
                  "first_step_overshoot_pct", "tf_effective_s",
                  "pred_m1_from_velocity_rad",
                  "pred_m1_from_velocity_measfreq_rad", "pred_m1_freq_sensitivity",
                  "tan_m1_amp_pp_rad", "notes"]
    L.write_table(tr[track_cols], "tower_tracking.csv")

    pd.set_option("display.width", 260)
    print("=" * 120)
    print("TOWER - residual hook swing (vision block, ~%.0f s after the move ends)"
          % tr.vision_delay_after_move_s.mean())
    print("cable L1 = %.0f mm, L2 = %.0f mm  ->  f1 = %.3f Hz (T1 = %.2f s), f2 = %.3f Hz"
          % (tr.cable_mm.mean(), L.L2_RIG * 1000, tr.f1_theory_hz.mean(),
             tr.T1_theory_s.mean(), tr.f2_theory_hz.mean()))
    print("=" * 120)
    print(tr[["shaper", "trolley_nom_mm", "actual_travel_deg",
              "tan_amp_pp_rad", "pct_tan", "tan_m1_amp_pp_rad", "pct_tan_m1",
              "tan_m2_amp_pp_rad", "pct_tan_m2",
              "rad_amp_pp_rad", "pct_rad"]].to_string(index=False))
    print()
    print("=" * 120)
    print("BY SHAPER (mean over the three trolley radii).  pct_* = % of the unshaped "
          "trial at the SAME radius")
    print("=" * 120)
    print(summ[["shaper", "n", "shaper_dur_s", "actual_travel_deg",
                "tan_amp_pp_rad", "pct_tan", "tan_m1_rad", "pct_tan_m1",
                "tan_m2_rad", "pct_tan_m2", "rad_amp_pp_rad", "pct_rad"]]
          .round(4).to_string(index=False))
    print()
    print("measured mode frequencies vs the double-pendulum model:")
    print(tr[["shaper", "trolley_nom_mm", "f1_theory_hz", "tan_m1_f_meas_hz",
              "tan_m1_f_err_pct", "f2_theory_hz", "tan_m2_f_meas_hz",
              "tan_m2_f_err_pct"]].to_string(index=False))
    print()
    print("mode-1 residual predicted from each trial's own logged velocity,")
    print("at the model frequency vs at the measured frequency.  The spread is")
    print("why this prediction is only a trend indicator on the tower:")
    print(tr[["shaper", "trolley_nom_mm", "tf_effective_s",
              "pred_m1_from_velocity_rad", "pred_m1_from_velocity_measfreq_rad",
              "pred_m1_freq_sensitivity", "tan_m1_amp_pp_rad"]].to_string(index=False))
    print()
    print("command-execution problems found:")
    for r in tr[tr.notes != ""].itertuples():
        print(f"   {r.timestamp}  {r.shaper:12s} {r.notes}")
    return tr, summ


if __name__ == "__main__":
    main()

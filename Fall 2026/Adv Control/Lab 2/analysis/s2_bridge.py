"""
Step 2 - bridge crane (Part 1) reduction.

For every usable trial we locate the outbound move from the trolley velocity,
then score the PAYLOAD RESIDUAL OSCILLATION over the window between the end of
the outbound move and the start of the return move.

Headline metric is `amp_pp_max`, the largest peak-to-peak payload swing within
the first four measured periods after the trolley stops.  `amp_pp_first` (the
first swing) is also reported but is NOT used as the headline: the residual
window necessarily opens while the trolley is still finishing its deceleration
ramp, so the first detected swing is sometimes a partial one and reads low.
A long-window mean is biased the other way, low, by damping.  Capping the
window at four periods keeps every trial comparable.

Each trial also carries a PREDICTED residual, obtained by evaluating the
Fourier transform of that trial's own logged trolley velocity at the pendulum
frequency (see `lab2lib.predicted_residual_from_velocity`).  That is a real
theory-vs-experiment check which needs no assumption that the commanded
profile was an ideal pulse.  It is approximate here because the logged velocity
is coarsely quantised, so treat it as a trend check rather than an absolute.

Because the bridge GUI logs no command (`Y Command Velocity` is all zeros),
the impulse sequence that was actually applied is reconstructed from the
measured velocity staircase (`lab2lib.reconstruct_staircase`).  That read-back
is not a formality: it is what shows that the trials named "ZVD" were run with
amplitudes [0.25, 0.50, 0.50] summing to 1.25 instead of the textbook
[0.25, 0.50, 0.25], which is why they move 27 % further than the ZV trials and
why they do not beat ZV on residual vibration.

Outputs
  results/tables/bridge_commands.csv  applied impulse sequence, read back
  results/tables/bridge_moves.csv     every move found in every file
  results/tables/bridge_trials.csv    one row per trial of record
  results/tables/bridge_partA.csv     Part A summary (move distance sweep)
  results/tables/bridge_partB.csv     Part B summary (cable length sweep)
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lab2lib as L

VEL_THRESH = 15.0        # mm/s, above the encoder noise floor
DEFL_QUANT = 1.0         # mm, deflection is logged to the nearest mm
MIN_SWING = 3.0 * DEFL_QUANT
MAX_WINDOW = 12.0        # s, never score more than this after a move
CYCLE_CAP = 4            # score at most 4 measured periods
SETTLE_FRAC = 0.10       # residual threshold for the settling-time metric


def analyse_file(path, meta):
    """Return one row per move found in `path`."""
    d = L.load_bridge(path)
    t, defl, pos, vel = d["t"], d["defl"], d["pos"], d["vel"]
    segs = L.find_motion_segments(t, vel, thresh=VEL_THRESH)
    rows = []
    for k, (a, b) in enumerate(segs):
        t_end = t[b]
        t_next = t[segs[k + 1][0]] if k + 1 < len(segs) else t[-1] + 1.0
        w1 = min(t_next - 0.25, t_end + MAX_WINDOW, t[-1])
        r = L.residual_metrics(t, defl, t_end, w1,
                               min_swing=MIN_SWING, n_cycles_cap=CYCLE_CAP)
        r_full = L.residual_metrics(t, defl, t_end, w1, min_swing=MIN_SWING)

        rec = L.reconstruct_staircase(t, vel, t[a] - 0.6, t_end + 0.8,
                                      v_max=float(np.max(np.abs(vel))))

        w_th, _ = L.pendulum_freq(d["cable_mm"] / 1000.0)
        pred_pp = 2 * L.predicted_residual_from_velocity(
            t, vel, w_th, t[a] - 0.5, t_end + 0.5)

        dist = float(pos[b] - pos[a])
        vpeak = float(np.max(np.abs(vel[a:b + 1])))
        # rise time: trolley within 2 % of its final position
        seg_t, seg_p = t[a:b + 1], pos[a:b + 1]
        tgt = pos[b]
        within = np.abs(seg_p - tgt) <= 0.02 * abs(dist) if dist else np.array([False])
        t_rise = float(seg_t[np.argmax(within)] - seg_t[0]) if within.any() else np.nan

        row = dict(
            file=d["name"], move_index=k,
            direction="outbound" if k == 0 else ("return" if k == 1 else f"extra{k}"),
            cable_infile_mm=d["cable_mm"],
            move_t0=round(float(t[a]), 3), move_t1=round(t_end, 3),
            move_dur_s=round(t_end - float(t[a]), 3),
            move_dist_mm=round(dist, 1), peak_vel_mm_s=round(vpeak, 1),
            rise_time_s=round(t_rise, 3) if np.isfinite(t_rise) else np.nan,
            resid_window_s=round(r_full.window_s, 2),
            cycles_available=r_full.n_cycles,
            pred_amp_pp_mm=round(pred_pp, 1),
            applied_amps=str(rec.get("shaper_amps")),
            applied_delays_s=str(rec.get("shaper_delays")),
            applied_amp_sum=rec.get("shaper_amp_sum", np.nan),
            applied_pattern=L.name_impulse_pattern(rec.get("shaper_amps", [])),
            mirror_amps=str(rec.get("mirror_amps")),
            mirror_matches_shaper=rec.get("mirror_matches_shaper", None),
            tf_inferred_s=rec.get("tf_inferred_s", np.nan),
            v_top_mm_s=rec.get("v_top", np.nan),
            plateau_fracs=str(rec.get("plateau_fracs")),
        )
        for k2, v2 in vars(r).items():
            row[k2] = v2
        row["amp_pp_mean_full"] = r_full.amp_pp_mean
        row["rms_full"] = r_full.rms
        row.update(meta)
        rows.append(row)
    return rows


def main():
    inv = pd.read_csv(os.path.join(L.TABLES, "inventory_bridge.csv"))
    use = inv[inv.use]

    all_rows = []
    for r in use.itertuples():
        meta = dict(part=r.part, shaper=r.shaper, tf_ms=r.tf_ms,
                    cable_nom_mm=r.cable_nom_mm, trial=r.trial)
        all_rows += analyse_file(os.path.join(L.BRIDGE_DIR, r.file), meta)
    moves = pd.DataFrame(all_rows)
    L.write_table(moves, "bridge_moves.csv")

    # ---- trial of record = the outbound move -------------------------------
    tr = moves[moves.move_index == 0].copy()
    tr["condition"] = np.where(
        tr.part == "A",
        tr.shaper + " | tf=" + tr.tf_ms.astype(int).astype(str) + " ms",
        tr.shaper + " | L=" + (tr.cable_nom_mm / 1000).map("{:.1f}".format) + " m")
    # theoretical frequency for the cable length actually used
    w, T = zip(*[L.pendulum_freq(c / 1000.0) for c in tr.cable_infile_mm])
    tr["w_theory_rad_s"] = np.round(w, 4)
    tr["T_theory_s"] = np.round(T, 4)
    tr["T_meas_s"] = tr.period_zc
    tr["T_err_pct"] = 100 * (tr.T_meas_s - tr.T_theory_s) / tr.T_theory_s
    tr["amp_pp_mm"] = tr.amp_pp_max          # headline metric
    tr["pred_over_meas"] = tr.pred_amp_pp_mm / tr.amp_pp_mm

    # ---- per-trial quality flags ------------------------------------------
    flags = []
    for r in tr.itertuples():
        f = []
        if r.cycles_available < 3:
            f.append(f"only {r.cycles_available:.1f} residual cycles recorded "
                     "(handout asks for 4-5) - weakest trials, avoid for plots")
        if r.amp_pp_mm <= 6 * DEFL_QUANT:
            f.append("amplitude within a few counts of the 1 mm sensor "
                     "quantisation - period/damping estimates unreliable")
        if abs(r.T_err_pct) > 8:
            f.append(f"measured period is {r.T_err_pct:+.0f} % off the pendulum "
                     "value (small-amplitude quantisation, or wrong cable tag)")
        if r.amp_pp_first < 0.5 * r.amp_pp_max:
            f.append("first swing is a partial one; amp_pp_max is the number to cite")
        exp_n = {"Unshaped": 1, "ZV": 2, "ZVD": 3}.get(r.shaper)
        n_app = len(eval(r.applied_amps)) if r.applied_amps != "None" else 0
        if exp_n and n_app and n_app < exp_n:
            f.append(f"only {n_app} of the {exp_n} commanded impulses are visible: "
                     "the hold time tf is shorter than the shaper, so the steps "
                     "overlap and the trolley never reaches top speed")
        if np.isfinite(r.applied_amp_sum) and abs(r.applied_amp_sum - 1.0) > 0.1 \
                and n_app >= exp_n if exp_n else False:
            f.append(f"applied impulse amplitudes sum to {r.applied_amp_sum:.2f}, "
                     "not 1.00 - this is not the textbook shaper")
        flags.append("; ".join(f))
    tr["quality_notes"] = flags
    tr["good_for_plots"] = (tr.cycles_available >= 3)

    cmd_cols = ["file", "part", "shaper", "tf_ms", "cable_nom_mm", "trial",
                "applied_pattern", "applied_amps", "applied_delays_s",
                "applied_amp_sum", "mirror_amps", "mirror_matches_shaper",
                "tf_inferred_s", "v_top_mm_s", "plateau_fracs",
                "move_dist_mm", "rise_time_s"]
    L.write_table(tr[cmd_cols], "bridge_commands.csv")

    cols = ["part", "shaper", "tf_ms", "cable_nom_mm", "cable_infile_mm", "trial",
            "file", "condition", "applied_pattern", "applied_amps",
            "applied_delays_s", "applied_amp_sum", "tf_inferred_s",
            "move_dist_mm", "move_dur_s", "rise_time_s",
            "peak_vel_mm_s", "amp_pp_mm", "amp_pp_first", "amp_pp_mean",
            "amp_pp_max", "rms", "pred_amp_pp_mm", "pred_over_meas",
            "resid_window_s", "cycles_available", "n_cycles",
            "T_meas_s", "T_theory_s", "T_err_pct", "freq_fft_hz", "zeta",
            "move_t0", "move_t1", "good_for_plots", "quality_notes"]
    tr = tr[cols].sort_values(["part", "shaper", "tf_ms", "cable_nom_mm", "trial"])
    L.write_table(tr.round(4), "bridge_trials.csv")

    # ---- condition summaries ----------------------------------------------
    def summarise(sub, group_keys):
        g = sub.groupby(group_keys, dropna=False)
        s = g.agg(
            n=("file", "count"),
            dist_mm=("move_dist_mm", "mean"),
            rise_s=("rise_time_s", "mean"),
            amp_pp_mm=("amp_pp_mm", "mean"),
            amp_pp_sd=("amp_pp_mm", "std"),
            amp_pp_min=("amp_pp_mm", "min"),
            amp_pp_max_of_trials=("amp_pp_mm", "max"),
            amp_pp_first_mm=("amp_pp_first", "mean"),
            amp_pp_mean_mm=("amp_pp_mean", "mean"),
            rms_mm=("rms", "mean"),
            pred_amp_pp_mm=("pred_amp_pp_mm", "mean"),
            applied_pattern=("applied_pattern", lambda v: v.mode().iat[0] if len(v.mode()) else ""),
            applied_amp_sum=("applied_amp_sum", "mean"),
            tf_inferred_s=("tf_inferred_s", "mean"),
            n_short_window=("good_for_plots", lambda v: int((~v).sum())),
            T_meas_s=("T_meas_s", "mean"),
            T_theory_s=("T_theory_s", "mean"),
            zeta=("zeta", "mean"),
            window_s=("resid_window_s", "mean"),
        ).reset_index()
        return s

    A = tr[tr.part == "A"]
    B = tr[tr.part == "B"]
    sA = summarise(A, ["shaper", "tf_ms"])
    sB = summarise(B, ["shaper", "cable_nom_mm"])

    # percent reduction vs the unshaped trial at the same condition
    def add_reduction(s, key):
        base = s[s.shaper == "Unshaped"].set_index(key)
        for col, out in (("amp_pp_mm", "pct_of_unshaped"),
                         ("rms_mm", "pct_of_unshaped_rms"),
                         ("pred_amp_pp_mm", "pct_of_unshaped_predicted")):
            s[out] = [
                round(100 * v / base.loc[k, col], 1) if k in base.index and base.loc[k, col] else np.nan
                for v, k in zip(s[col], s[key])
            ]
        s["pct_reduction"] = (100 - s.pct_of_unshaped).round(1)
        return s

    sA = add_reduction(sA, "tf_ms")
    sB = add_reduction(sB, "cable_nom_mm")
    L.write_table(sA.round(3), "bridge_partA.csv")
    L.write_table(sB.round(3), "bridge_partB.csv")

    pd.set_option("display.width", 220)
    print("=" * 100)
    print("PART A - variation in move distance (0.8 m cable, 100 % speed)")
    print("=" * 100)
    print("applied impulse sequences, read back from the measured velocity:")
    ap = tr.groupby(["part", "shaper"]).agg(
        pattern=("applied_pattern", lambda v: v.mode().iat[0]),
        amps=("applied_amps", lambda v: v.mode().iat[0]),
        amp_sum=("applied_amp_sum", "mean"),
        delay_s=("applied_delays_s", lambda v: v.mode().iat[0]),
        tf_inferred=("tf_inferred_s", "mean")).reset_index()
    print(ap.round(3).to_string(index=False))
    print()
    print(sA[["shaper", "tf_ms", "n", "dist_mm", "amp_pp_mm", "amp_pp_sd", "rms_mm",
              "pred_amp_pp_mm", "pct_of_unshaped", "pct_reduction",
              "T_meas_s", "T_theory_s", "n_short_window"]].round(2).to_string(index=False))
    print()
    print("=" * 100)
    print("PART B - variation in system frequency (tf = 1500 ms)")
    print("=" * 100)
    print(sB[["shaper", "cable_nom_mm", "n", "dist_mm", "rise_s", "amp_pp_mm",
              "amp_pp_sd", "rms_mm", "pred_amp_pp_mm", "pct_of_unshaped",
              "pct_reduction", "T_meas_s", "T_theory_s", "n_short_window"]].round(2).to_string(index=False))
    print()
    flagged = tr[tr.quality_notes != ""]
    if len(flagged):
        print(f"{len(flagged)} of {len(tr)} trials carry a quality note:")
        for r in flagged.itertuples():
            print(f"   {r.file:38s} {r.quality_notes}")
    print()
    print("per-trial detail -> results/tables/bridge_trials.csv")
    return tr, sA, sB


if __name__ == "__main__":
    main()

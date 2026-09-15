"""
Step 4 - shaper design, and the robustness argument the report has to make.

The handout asks you to "justify and explain" your ZV and robust shaper
designs.  This script produces the numbers behind that justification:

  * the pendulum / double-pendulum frequencies for every configuration tested;
  * the shapers that were actually run, reverse-engineered from the data
    (tower) or from the file names (bridge), with their design frequency;
  * SENSITIVITY CURVES - percent residual vibration versus actual system
    frequency - for each shaper, evaluated for the command the rigs really
    apply (shaper convolved with the velocity pulse, whose impulses sum to
    zero rather than one);
  * the predicted residual at each tested condition, so measured reductions
    can be compared against what the design promised;
  * what a shaper designed at the MEASURED frequency would have achieved,
    which is the honest answer to "why was the cancellation imperfect".

Outputs
  results/tables/design_frequencies.csv
  results/tables/design_shapers.csv
  results/tables/design_predictions.csv
  results/tables/sensitivity_curves.csv
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lab2lib as L

# what was actually run, from the file names and the tower command read-back
BRIDGE_CABLES = [0.6, 0.8, 0.9, 1.2]
PART_A_TF = [1.1, 1.6, 1.8, 2.0]
PART_B_TF = 1.5
ZV_DT_A = 0.90      # s, Part A
ZV_DT_B = 0.91      # s, Part B
ZVD_DT_B = 0.91     # s, Part B
TOWER_TF = 4.0
TOWER_L1 = 0.598
TOWER_L1_TABLE = 1.000   # the value printed in Table 1 of the handout


def frequency_table():
    rows = []
    for Lc in BRIDGE_CABLES:
        w, T = L.pendulum_freq(Lc)
        rows.append(dict(rig="Bridge", config=f"cable {Lc*1000:.0f} mm",
                         model="single pendulum", L1_m=Lc, L2_m=np.nan,
                         w_rad_s=round(w, 4), f_hz=round(w / 2 / np.pi, 4),
                         T_s=round(T, 4), half_T_s=round(T / 2, 4)))
    for L1, tag in ((TOWER_L1, f"hoist {TOWER_L1*1000:.0f} mm (as run)"),
                    (TOWER_L1_TABLE, f"hoist {TOWER_L1_TABLE*1000:.0f} mm (Table 1)")):
        w1, w2, R, beta = L.double_pendulum_freqs(L1, L.L2_RIG)
        for i, w in ((1, w1), (2, w2)):
            rows.append(dict(rig="Tower", config=f"{tag}, mode {i}",
                             model="double pendulum", L1_m=L1, L2_m=L.L2_RIG,
                             w_rad_s=round(w, 4), f_hz=round(w / 2 / np.pi, 4),
                             T_s=round(2 * np.pi / w, 4),
                             half_T_s=round(np.pi / w, 4), R=round(R, 4),
                             beta=round(beta, 4)))
    return pd.DataFrame(rows)


def shaper_table():
    """Every shaper used, plus the design frequency each one implies."""
    rows = []

    def add(rig, part, name, A, T, note, tf):
        w_design = np.pi / T[1] if len(T) > 1 and T[1] > 0 else np.nan
        rows.append(dict(
            rig=rig, part=part, shaper=name,
            amps=np.round(A, 4).tolist(), delays_s=np.round(T, 4).tolist(),
            amp_sum=round(float(np.sum(A)), 4),
            duration_s=round(float(T[-1]), 4),
            implied_design_w_rad_s=round(w_design, 4) if np.isfinite(w_design) else np.nan,
            implied_design_T_s=round(2 * np.pi / w_design, 4) if np.isfinite(w_design) else np.nan,
            implied_design_L_m=round(L.G / w_design ** 2, 4) if np.isfinite(w_design) else np.nan,
            tf_s=tf, note=note))

    add("Bridge", "A", "Unshaped", [1.0], [0.0], "trapezoidal velocity, 100 % speed", "1.1/1.6/2.0")
    add("Bridge", "A", "ZV (as run)", [0.5, 0.5], [0.0, ZV_DT_A],
        f"dt = {ZV_DT_A} s = T/2 for the 0.8 m cable actually used "
        f"(T/2 = {L.pendulum_freq(0.8)[1]/2:.3f} s)", "1.1/1.6/1.8/2.0")

    w_lo, _ = L.pendulum_freq(1.2)
    w_hi, _ = L.pendulum_freq(0.6)
    w_mid = 0.5 * (w_lo + w_hi)
    add("Bridge", "B", "Unshaped", [1.0], [0.0], "baseline at each cable length", PART_B_TF)
    add("Bridge", "B", "ZV (as run)", [0.5, 0.5], [0.0, ZV_DT_B],
        "dt = 0.91 s = pi / mean(w at 0.6 m, w at 1.2 m) = "
        f"pi/{w_mid:.3f} = {np.pi/w_mid:.3f} s - centred on the MEAN of the two "
        "extreme frequencies, not on the mid cable length, which is the standard "
        "way to centre a shaper over a known frequency range", PART_B_TF)
    add("Bridge", "B", "ZVD (as run, 0.25/0.5/0.5)", [0.25, 0.5, 0.5],
        [0.0, ZVD_DT_B, 2 * ZVD_DT_B],
        "READ BACK FROM THE MEASURED VELOCITY of all nine trials: the third "
        "impulse was entered as 0.50 instead of 0.25, so the amplitudes sum to "
        "1.25.  Consequences: the move is 27 % longer (371 mm vs 291 mm) and, "
        "critically, the residual at the design frequency is 0.25/1.25 = 20 % "
        "instead of 0 - which is why these trials do not beat the plain ZV.  "
        "The same mis-entry appears in the tower ZVD trials.", PART_B_TF)
    add("Bridge", "B", "ZVD (textbook, for comparison)", [0.25, 0.5, 0.25],
        [0.0, ZVD_DT_B, 2 * ZVD_DT_B],
        "what the same delays would have delivered with correct amplitudes - "
        "the design the report should describe as intended", PART_B_TF)

    # reference designs for comparison
    for Lc in (0.6, 0.9, 1.2):
        w, T = L.pendulum_freq(Lc)
        A, Tt = L.zv_shaper(w)
        add("Bridge", "B-ref", f"ZV tuned to {Lc*1000:.0f} mm", A, Tt,
            "reference: a ZV designed for this one cable length only", PART_B_TF)
    A, Tt = L.ei_shaper(w_mid, V=0.05)
    add("Bridge", "B-ref", "EI (5 %) at the mean frequency", A, Tt,
        "reference: an extra-insensitive shaper, same duration as ZVD", PART_B_TF)

    w1, w2, _, _ = L.double_pendulum_freqs(TOWER_L1, L.L2_RIG)
    add("Tower", "2", "Unshaped", [1.0], [0.0], "100 % slew, tf = 4000 ms", TOWER_TF)
    add("Tower", "2", "ZV (as run)", [0.5, 0.5], [0.0, 0.84],
        f"dt = 0.84 s ~ T1/2 = {np.pi/w1:.3f} s: single-mode ZV on mode 1", TOWER_TF)
    add("Tower", "2", "ZVD (as run, 0.25/0.5/0.5)", [0.25, 0.5, 0.5], [0.0, 0.84, 1.71],
        "READ BACK FROM THE DATA: the third impulse was entered as 0.50 instead "
        "of 0.25, so the amplitudes sum to 1.25 and the mid-move command asked "
        "for 125 % of the slew rate limit and saturated", TOWER_TF)
    add("Tower", "2", "ZVD (textbook, for comparison)", [0.25, 0.5, 0.25], [0.0, 0.84, 1.71],
        "what the same delays would have done with correct amplitudes", TOWER_TF)
    A2, T2 = L.convolve_shapers(*L.zv_shaper(w1), *L.zv_shaper(w2))
    add("Tower", "2", "Two-mode ZV (as run)", [0.25, 0.25, 0.25, 0.25], [0.0, 0.27, 0.84, 1.11],
        f"convolution of a ZV on mode 1 (T1/2 = {np.pi/w1:.3f} s) with a ZV on "
        f"mode 2 (T2/2 = {np.pi/w2:.3f} s); read back from the data", TOWER_TF)
    add("Tower", "2", "Two-mode ZV (exact theory)", A2, T2,
        "the same design evaluated at the exact model frequencies", TOWER_TF)
    return pd.DataFrame(rows)


def sensitivity_curves():
    """V_rel(w) for each shaper, as applied (convolved with the velocity pulse)."""
    w = np.linspace(0.5, 16.0, 1200)
    cols = {"w_rad_s": w, "f_hz": w / (2 * np.pi), "cable_len_equiv_m": L.G / w ** 2}
    designs = [
        ("bridge_A_unshaped_tf1.1", [1.0], [0.0], 1.1),
        ("bridge_A_unshaped_tf1.6", [1.0], [0.0], 1.6),
        ("bridge_A_unshaped_tf2.0", [1.0], [0.0], 2.0),
        ("bridge_A_ZV_tf1.1", [0.5, 0.5], [0.0, ZV_DT_A], 1.1),
        ("bridge_A_ZV_tf1.6", [0.5, 0.5], [0.0, ZV_DT_A], 1.6),
        ("bridge_A_ZV_tf2.0", [0.5, 0.5], [0.0, ZV_DT_A], 2.0),
        ("bridge_B_unshaped", [1.0], [0.0], PART_B_TF),
        ("bridge_B_ZV_0.91", [0.5, 0.5], [0.0, ZV_DT_B], PART_B_TF),
        ("bridge_B_ZVD_asrun_0.91", [0.25, 0.5, 0.5], [0.0, ZVD_DT_B, 2 * ZVD_DT_B], PART_B_TF),
        ("bridge_B_ZVD_textbook_0.91", [0.25, 0.5, 0.25], [0.0, ZVD_DT_B, 2 * ZVD_DT_B], PART_B_TF),
        ("bridge_B_EI5_0.91", *[np.array(x) for x in
                                (L.ei_shaper(np.pi / ZVD_DT_B, 0.05))], PART_B_TF),
        ("tower_ZV_0.84", [0.5, 0.5], [0.0, 0.84], TOWER_TF),
        ("tower_ZVD_asrun", [0.25, 0.5, 0.5], [0.0, 0.84, 1.71], TOWER_TF),
        ("tower_ZVD_textbook", [0.25, 0.5, 0.25], [0.0, 0.84, 1.71], TOWER_TF),
        ("tower_2modeZV", [0.25, 0.25, 0.25, 0.25], [0.0, 0.27, 0.84, 1.11], TOWER_TF),
    ]
    for name, A, T, tf in designs:
        cols[name] = L.pulse_residual_ratio(A, T, tf, w)
    # shaper-only sensitivity (the textbook curve, no pulse)
    for name, A, T in (
            ("shaperonly_ZV_0.91", [0.5, 0.5], [0.0, ZV_DT_B]),
            ("shaperonly_ZVD_textbook_0.91", [0.25, 0.5, 0.25], [0.0, ZVD_DT_B, 2 * ZVD_DT_B]),
            ("shaperonly_ZVD_asrun_0.91", [0.25, 0.5, 0.5], [0.0, ZVD_DT_B, 2 * ZVD_DT_B]),
            ("shaperonly_ZV_0.84", [0.5, 0.5], [0.0, 0.84]),
            ("shaperonly_2modeZV", [0.25, 0.25, 0.25, 0.25], [0.0, 0.27, 0.84, 1.11])):
        cols[name] = L.residual_vibration(A, T, w)
    return pd.DataFrame(cols)


def predictions():
    """Predicted V_rel at every condition that was actually tested."""
    rows = []
    for tf in PART_A_TF:
        w, _ = L.pendulum_freq(0.8)
        for name, A, T in (("Unshaped", [1.0], [0.0]),
                           ("ZV", [0.5, 0.5], [0.0, ZV_DT_A])):
            rows.append(dict(rig="Bridge", part="A", shaper=name, tf_s=tf,
                             cable_m=0.8, w_rad_s=round(w, 4),
                             tf_over_T=round(tf / (2 * np.pi / w), 4),
                             V_rel_vs_unshaped=round(float(
                                 L.pulse_residual_ratio(A, T, tf, w)), 4)))
    for Lc in (0.6, 0.9, 1.2):
        w, _ = L.pendulum_freq(Lc)
        for name, A, T in (("Unshaped", [1.0], [0.0]),
                           ("ZV", [0.5, 0.5], [0.0, ZV_DT_B]),
                           ("ZVD (as run)", [0.25, 0.5, 0.5], [0.0, ZVD_DT_B, 2 * ZVD_DT_B]),
                           ("ZVD (textbook)", [0.25, 0.5, 0.25], [0.0, ZVD_DT_B, 2 * ZVD_DT_B])):
            rows.append(dict(rig="Bridge", part="B", shaper=name, tf_s=PART_B_TF,
                             cable_m=Lc, w_rad_s=round(w, 4),
                             tf_over_T=round(PART_B_TF / (2 * np.pi / w), 4),
                             V_rel_vs_unshaped=round(float(
                                 L.pulse_residual_ratio(A, T, PART_B_TF, w)), 4)))
    w1, w2, _, _ = L.double_pendulum_freqs(TOWER_L1, L.L2_RIG)
    # measured frequencies, from s3
    f1_meas, f2_meas = 0.527, 1.600
    for name, A, T in (("Unshaped", [1.0], [0.0]),
                       ("ZV", [0.5, 0.5], [0.0, 0.84]),
                       ("ZVD (as run)", [0.25, 0.5, 0.5], [0.0, 0.84, 1.71]),
                       ("ZVD (textbook)", [0.25, 0.5, 0.25], [0.0, 0.84, 1.71]),
                       ("Two-mode ZV", [0.25, 0.25, 0.25, 0.25], [0.0, 0.27, 0.84, 1.11])):
        for tag, wa, wb in (("model", w1, w2),
                            ("measured", 2 * np.pi * f1_meas, 2 * np.pi * f2_meas)):
            rows.append(dict(rig="Tower", part="2", shaper=name, tf_s=TOWER_TF,
                             freqs=tag,
                             w1_rad_s=round(wa, 4), w2_rad_s=round(wb, 4),
                             V_rel_mode1=round(float(L.pulse_residual_ratio(A, T, TOWER_TF, wa)), 4),
                             V_rel_mode2=round(float(L.pulse_residual_ratio(A, T, TOWER_TF, wb)), 4)))
    return pd.DataFrame(rows)


def main():
    fq = frequency_table()
    sh = shaper_table()
    sc = sensitivity_curves()
    pr = predictions()
    L.write_table(fq, "design_frequencies.csv")
    L.write_table(sh, "design_shapers.csv")
    L.write_table(sc.round(6), "sensitivity_curves.csv")
    L.write_table(pr, "design_predictions.csv")

    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 90)
    print("=" * 110)
    print("NATURAL FREQUENCIES")
    print("=" * 110)
    print(fq[["rig", "config", "w_rad_s", "f_hz", "T_s", "half_T_s"]].to_string(index=False))
    print()
    print("=" * 110)
    print("SHAPERS USED (and reference designs)")
    print("=" * 110)
    print(sh[["rig", "part", "shaper", "amps", "delays_s", "amp_sum", "duration_s",
              "implied_design_T_s", "implied_design_L_m"]].to_string(index=False))
    print()
    print("design rationale:")
    for r in sh[sh.note.str.len() > 0].itertuples():
        print(f"  [{r.rig} {r.part}] {r.shaper}: {r.note}")
    print()
    print("=" * 110)
    print("PREDICTED RESIDUAL VIBRATION, relative to the unshaped move at the")
    print("same condition (0 = perfect cancellation, 1 = no benefit)")
    print("=" * 110)
    pa = pr[(pr.part == "A")]
    print("Part A (0.8 m cable) - the unshaped column is the pulse's own")
    print("frequency response, which is why move distance matters so much:")
    print(pa[["shaper", "tf_s", "tf_over_T", "V_rel_vs_unshaped"]].to_string(index=False))
    print()
    print("Part B (tf = 1500 ms):")
    print(pr[pr.part == "B"][["shaper", "cable_m", "w_rad_s", "tf_over_T",
                              "V_rel_vs_unshaped"]].to_string(index=False))
    print()
    print("Tower (tf = 4000 ms), at the model frequencies and at the frequencies")
    print("actually measured (f1 = 0.527 Hz, f2 = 1.600 Hz):")
    print(pr[pr.part == "2"][["shaper", "freqs", "V_rel_mode1", "V_rel_mode2"]].to_string(index=False))
    return fq, sh, sc, pr


if __name__ == "__main__":
    main()

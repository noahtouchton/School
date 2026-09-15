"""
Step 5 - report-ready figures.

Everything lands in results/figures at 200 dpi.  The set is deliberately
non-redundant: each figure carries one message, which is named in its title so
it is obvious which figure supports which paragraph of the report.

  fig01  Part A time histories, unshaped vs ZV at each move distance
  fig02  Part A residual vs tf, measured against the pulse's own frequency
         response - this is the figure that explains the move-distance trend
  fig03  Part B time histories, unshaped vs ZV vs ZVD at each cable length
  fig04  Part B residual vs cable length, measured against prediction
  fig05  Shaper sensitivity curves with the three test frequencies marked -
         the design-justification figure for Part B
  fig06  Applied velocity staircases read back from the data, including the
         ZVD amplitude error
  fig07  Tower command vs actual slew velocity - how well each shaper was
         actually executed
  fig08  Tower tangential and radial swing time histories
  fig09  Tower residual vs trolley radius
  fig10  Tower residual split by mode - the two-mode argument
  fig11  Tower swing spectra with both predicted modes marked
  fig12  Tower shaper sensitivity curves with both modes marked
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lab2lib as L

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight",
    "font.size": 9, "axes.grid": True, "grid.alpha": 0.25,
    "axes.titlesize": 9.5, "axes.labelsize": 9, "legend.fontsize": 8,
    "lines.linewidth": 1.3,
})
C = {"Unshaped": "#c1272d", "ZV": "#0072b2", "ZVD": "#009e73",
     "Two-mode ZV": "#7030a0", "ZVD (textbook)": "#999999"}


def save(fig, name):
    os.makedirs(L.FIGURES, exist_ok=True)
    p = os.path.join(L.FIGURES, name)
    fig.savefig(p)
    plt.close(fig)
    print("  wrote", os.path.relpath(p, L.ROOT))
    return p


def pick(tr, **kw):
    """Best trial matching the given conditions: prefers a long residual
    window so the plotted trace actually shows 4-5 oscillations."""
    m = pd.Series(True, index=tr.index)
    for k, v in kw.items():
        m &= tr[k] == v
    sub = tr[m]
    if sub.empty:
        return None
    return sub.sort_values(["cycles_available", "resid_window_s"],
                           ascending=False).iloc[0]


def bridge_trace(row, pad_before=1.0, pad_after=None):
    d = L.load_bridge(os.path.join(L.BRIDGE_DIR, row.file))
    t0, t1 = row.move_t0, row.move_t1
    hi = t1 + (pad_after if pad_after else max(row.resid_window_s, 6.0))
    m = (d["t"] >= t0 - pad_before) & (d["t"] <= hi)
    return d["t"][m] - t0, d["defl"][m], d["vel"][m], d["pos"][m]


# ---------------------------------------------------------------- Part A ----
def fig01(tr):
    tfs = [1100, 1600, 2000]
    fig, axes = plt.subplots(len(tfs), 1, figsize=(7.2, 7.0), sharex=True)
    for ax, tf in zip(axes, tfs):
        for sh in ("Unshaped", "ZV"):
            r = pick(tr, part="A", shaper=sh, tf_ms=float(tf))
            if r is None:
                continue
            t, y, v, _ = bridge_trace(r)
            ax.plot(t, y - np.mean(y[t > r.move_t1 - r.move_t0]),
                    color=C[sh], label=f"{sh}  ({r.amp_pp_mm:.0f} mm p-p)")
            ax.axvspan(0, r.move_t1 - r.move_t0, color="0.85", zorder=0)
        ax.set_ylabel("payload deflection (mm)")
        ax.set_title(f"$t_f$ = {tf} ms   (move distance "
                     f"{tr[(tr.part=='A')&(tr.tf_ms==tf)].move_dist_mm.mean():.0f} mm)")
        ax.legend(loc="upper right")
        ax.axhline(0, color="0.4", lw=0.6)
    axes[-1].set_xlabel("time from start of move (s)  -  grey band = trolley moving")
    fig.suptitle("Part A: ZV input shaping removes the residual payload swing at every "
                 "move distance\n0.8 m cable, 100 % speed, trolley axis", y=1.0)
    fig.tight_layout()
    return save(fig, "fig01_partA_time_histories.png")


def fig02(tr, pred):
    w, T = L.pendulum_freq(0.8)
    v_top = 200.0
    tf = np.linspace(0.6, 2.6, 600)
    # residual p-p of an ideal velocity pulse: (4 v/w) |sin(w tf / 2)|
    pp_pulse = (4 * v_top / w) * np.abs(np.sin(w * tf / 2))
    A, Tt = L.pulse_sequence(1.0, [0.5, 0.5], [0.0, 0.90])
    pp_zv = np.array([(2 * v_top / w) * abs(
        np.sum(np.asarray(a) * np.exp(1j * w * np.asarray(b))))
        for a, b in (L.pulse_sequence(x, [0.5, 0.5], [0.0, 0.90]) for x in tf)])

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(tf * 1000, pp_pulse, color=C["Unshaped"], ls="--", lw=1.1,
            label=r"theory, unshaped pulse: $(4v/\omega)\,|\sin(\omega t_f/2)|$")
    ax.plot(tf * 1000, pp_zv, color=C["ZV"], ls="--", lw=1.1,
            label=r"theory, ZV ($\Delta t$ = 0.90 s)")
    for sh, mk in (("Unshaped", "o"), ("ZV", "s")):
        g = tr[(tr.part == "A") & (tr.shaper == sh)].groupby("tf_ms").amp_pp_mm
        ax.errorbar(g.mean().index, g.mean().values, yerr=g.std().values,
                    fmt=mk, color=C[sh], capsize=3, ms=6, ls="none",
                    label=f"measured, {sh}")
    ax.set_xlim(tf[0] * 1000, tf[-1] * 1000)
    for k in (1, 2):
        if tf[0] < k * T < tf[-1]:
            ax.axvline(k * T * 1000, color="0.6", lw=0.7, ls=":")
            ax.text(k * T * 1000, ax.get_ylim()[1] * 0.96,
                    f" $t_f$ = {k}T = {k*T*1000:.0f} ms",
                    fontsize=7.5, color="0.35", va="top")
    ax.set_xlabel("pulse length $t_f$ (ms)  -  sets the move distance")
    ax.set_ylabel("residual payload swing, peak-to-peak (mm)")
    ax.set_title("Part A: the unshaped residual depends strongly on move distance because\n"
                 "a velocity pulse has nulls at $t_f = nT$; the ZV response is flat and small")
    ax.legend()
    fig.tight_layout()
    return save(fig, "fig02_partA_residual_vs_tf.png")


# ---------------------------------------------------------------- Part B ----
def fig03(tr):
    cables = [600, 900, 1200]
    fig, axes = plt.subplots(len(cables), 1, figsize=(7.2, 7.4), sharex=True)
    for ax, cb in zip(axes, cables):
        for sh in ("Unshaped", "ZV", "ZVD"):
            r = pick(tr, part="B", shaper=sh, cable_nom_mm=float(cb))
            if r is None:
                continue
            t, y, v, _ = bridge_trace(r)
            ax.plot(t, y - np.mean(y[t > r.move_t1 - r.move_t0]), color=C[sh],
                    label=f"{sh}  ({r.amp_pp_mm:.0f} mm p-p)")
        w, Tn = L.pendulum_freq(cb / 1000.0)
        ax.set_ylabel("payload deflection (mm)")
        ax.set_title(f"cable = {cb} mm   (T = {Tn:.2f} s, "
                     f"$\\omega$ = {w:.2f} rad/s)")
        ax.legend(loc="upper right", ncol=3)
        ax.axhline(0, color="0.4", lw=0.6)
    axes[-1].set_xlabel("time from start of move (s)")
    fig.suptitle("Part B: one ZV and one robust shaper across a 2:1 cable-length range\n"
                 "$t_f$ = 1500 ms, 100 % speed", y=1.0)
    fig.tight_layout()
    return save(fig, "fig03_partB_time_histories.png")


def fig04(tr, pred):
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.9))
    ax = axes[0]
    for sh, mk in (("Unshaped", "o"), ("ZV", "s"), ("ZVD", "^")):
        g = tr[(tr.part == "B") & (tr.shaper == sh)].groupby("cable_nom_mm").amp_pp_mm
        ax.errorbar(g.mean().index, g.mean().values, yerr=g.std().values,
                    fmt=mk + "-", color=C[sh], capsize=3, ms=6, label=sh)
    ax.set_xlabel("cable length (mm)")
    ax.set_ylabel("residual swing, peak-to-peak (mm)")
    ax.set_title("measured residual vs cable length")
    ax.set_xticks([600, 900, 1200])
    ax.legend()

    ax = axes[1]
    pb = pred[(pred.rig == "Bridge") & (pred.part == "B")]
    for sh, mk, c in (("ZV", "s", C["ZV"]),
                      ("ZVD (as run)", "^", C["ZVD"]),
                      ("ZVD (textbook)", "v", C["ZVD (textbook)"])):
        g = pb[pb.shaper == sh]
        ax.plot(g.cable_m * 1000, 100 * g.V_rel_vs_unshaped, mk + "--", color=c,
                ms=6, label=f"predicted, {sh}")
    for sh, mk in (("ZV", "s"), ("ZVD", "^")):
        g = tr[(tr.part == "B") & (tr.shaper == sh)]
        u = tr[(tr.part == "B") & (tr.shaper == "Unshaped")].groupby("cable_nom_mm").amp_pp_mm.mean()
        gm = g.groupby("cable_nom_mm").amp_pp_mm.mean()
        ax.plot(gm.index, 100 * gm / u.reindex(gm.index), mk + "-", color=C[sh],
                ms=7, label=f"measured, {sh}")
    ax.set_xlabel("cable length (mm)")
    ax.set_ylabel("residual as % of unshaped")
    ax.set_title("measured vs predicted reduction\n(as-run ZVD amplitudes explain the ZVD result)")
    ax.set_xticks([600, 900, 1200])
    ax.legend(fontsize=7)
    fig.suptitle("Part B: robustness across cable length", y=1.0)
    fig.tight_layout()
    return save(fig, "fig04_partB_residual_vs_cable.png")


def fig05(sc, tr):
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    curves = [("bridge_B_ZV_0.91", "ZV, $\\Delta t$ = 0.91 s (as run)", C["ZV"], "-"),
              ("bridge_B_ZVD_asrun_0.91", "ZVD as run [0.25, 0.50, 0.50]", C["ZVD"], "-"),
              ("bridge_B_ZVD_textbook_0.91", "ZVD textbook [0.25, 0.50, 0.25]",
               C["ZVD (textbook)"], "--"),
              ("bridge_B_EI5_0.91", "EI (5 %), same duration", "#e69f00", ":")]
    for col, lab, c, ls in curves:
        if col in sc:
            ax.plot(sc.w_rad_s, 100 * sc[col], color=c, ls=ls, label=lab)
    u = tr[(tr.part == "B") & (tr.shaper == "Unshaped")].groupby("cable_nom_mm").amp_pp_mm.mean()
    for cb, name in ((600, "0.6 m"), (900, "0.9 m"), (1200, "1.2 m")):
        w, _ = L.pendulum_freq(cb / 1000.0)
        ax.axvline(w, color="0.55", lw=0.8, ls=":")
        ax.text(w, 92, f" {name}\n $\\omega$={w:.2f}", fontsize=7.5, color="0.3", va="top")
        for sh, mk in (("ZV", "s"), ("ZVD", "^")):
            g = tr[(tr.part == "B") & (tr.shaper == sh) & (tr.cable_nom_mm == cb)]
            if len(g) and cb in u.index:
                ax.plot([w], [100 * g.amp_pp_mm.mean() / u.loc[cb]], mk,
                        color=C[sh], ms=8, mec="k", mew=0.6, zorder=5)
    w_mid = 0.5 * (L.pendulum_freq(1.2)[0] + L.pendulum_freq(0.6)[0])
    ax.axvline(w_mid, color="k", lw=1.0)
    ax.text(w_mid, 100, " design $\\omega$ = mean of the\n extremes = %.2f rad/s" % w_mid,
            fontsize=7.5, va="top")
    ax.set_xlim(2.2, 5.0)
    ax.set_ylim(0, 105)
    ax.set_xlabel("actual system frequency $\\omega$ (rad/s)")
    ax.set_ylabel("residual vibration, % of unshaped")
    ax.set_title("Part B design justification: sensitivity curves of the applied commands\n"
                 "(shaper convolved with the $t_f$ = 1.5 s pulse).  Filled markers = measured.")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, fontsize=7.5)
    fig.tight_layout()
    return save(fig, "fig05_sensitivity_curves.png")


def fig06(tr):
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 6.4), sharex=True)
    specs = [("Unshaped", 900.0), ("ZV", 900.0), ("ZVD", 900.0)]
    for ax, (sh, cb) in zip(axes, specs):
        r = pick(tr, part="B", shaper=sh, cable_nom_mm=cb)
        if r is None:
            continue
        t, y, v, p = bridge_trace(r, pad_before=0.8, pad_after=2.0)
        ax.plot(t, np.abs(v), color=C[sh])
        for frac in (0.25, 0.5, 0.75, 1.0, 1.25):
            ax.axhline(frac * 200, color="0.8", lw=0.6, ls=":")
        ax.set_ylabel("|trolley velocity|\n(mm/s)")
        ax.set_title(f"{sh}:  applied impulses {r.applied_amps} at {r.applied_delays_s} s, "
                     f"sum = {r.applied_amp_sum:.2f},  move = {r.move_dist_mm:.0f} mm")
        ax.set_ylim(-10, 230)
    axes[-1].set_xlabel("time from start of move (s)")
    fig.suptitle("The command the bridge actually applied, read back from the measured velocity\n"
                 "(the GUI never logs its command).  Dotted lines mark 25 % steps of top speed.",
                 y=1.0)
    fig.tight_layout()
    return save(fig, "fig06_bridge_velocity_profiles.png")


# ----------------------------------------------------------------- Tower ----
def fig07(tt):
    fig, axes = plt.subplots(2, 2, figsize=(8.8, 5.4), sharex=True, sharey=True)
    for ax, sh in zip(axes.ravel(), ["Unshaped", "ZV", "ZVD", "Two-mode ZV"]):
        r = tt[(tt.shaper == sh) & (~tt.cmd_truncated)].iloc[0]
        d = L.load_tower(os.path.join(L.TOWER_DIR, r.file))
        mo = d["motion"]
        t = mo["Time [s]"].to_numpy(float)
        c = mo["Command Velo Slew [deg/s]"].to_numpy(float)
        if np.nanmax(np.abs(c)) > 200:
            c = c / 100.0
        v = mo["Actual Velo Slew [deg/s]"].to_numpy(float)
        t = t - r.move_t0
        ax.plot(t, c, color="k", lw=1.6, label="command")
        ax.plot(t, v, color=C[sh], lw=1.0, label="actual")
        ax.axhline(32.4, color="0.7", lw=0.7, ls=":")
        ax.set_xlim(-1, 10)
        ax.set_title(f"{sh}   amps {r.impulse_amps}\ntravel {r.actual_travel_deg:.0f}"
                     f"$\\degree$ vs commanded {r.cmd_area_deg:.0f}$\\degree$")
        ax.legend(loc="upper right")
    for ax in axes[:, 0]:
        ax.set_ylabel("slew velocity (deg/s)")
    for ax in axes[-1, :]:
        ax.set_xlabel("time from start of move (s)")
    fig.suptitle("Tower: how faithfully each shaped command was executed.  The ZV trials "
                 "overshoot the\nfirst 16 deg/s step at breakaway; the as-run ZVD asks for "
                 "40.5 deg/s and saturates.", y=1.02)
    fig.tight_layout()
    return save(fig, "fig07_tower_command_tracking.png")


def fig08(tt, radius=700):
    fig, axes = plt.subplots(2, 1, figsize=(7.4, 6.0), sharex=True)
    for sh in ["Unshaped", "ZV", "ZVD", "Two-mode ZV"]:
        sub = tt[(tt.shaper == sh) & (tt.trolley_nom_mm == radius)]
        if sub.empty:
            continue
        r = sub.iloc[0]
        d = L.load_tower(os.path.join(L.TOWER_DIR, r.file))
        vi = d["vision"]
        t = vi["Time [s]"].to_numpy(float) - r.vision_t0
        for ax, chan, tag in ((axes[0], "Tangential Swing [rad]", "tan"),
                              (axes[1], "Radial Swing [rad]", "rad")):
            y = vi[chan].to_numpy(float)
            ok = np.isfinite(y)
            yy = y[ok] - np.polyval(np.polyfit(t[ok], y[ok], 1), t[ok])
            ax.plot(t[ok], yy, color=C[sh],
                    label=f"{sh} ({getattr(r, tag + '_amp_pp_rad'):.3f} rad p-p)")
    axes[0].set_ylabel("tangential swing (rad)")
    axes[1].set_ylabel("radial swing (rad)")
    axes[1].set_xlabel("time from start of the vision record (s)")
    for ax in axes:
        ax.legend(ncol=2, fontsize=7.5)
        ax.axhline(0, color="0.4", lw=0.6)
    axes[0].set_title("along the direction of travel - what the shaper targets")
    axes[1].set_title("along the jib - excited by the rotation, not by the shaper's target mode")
    fig.suptitle(f"Tower residual hook swing, trolley at {radius} mm\n"
                 "the vision record starts ~15 s after the move ends, so these are "
                 "post-decay amplitudes", y=1.0)
    fig.tight_layout()
    return save(fig, "fig08_tower_swing_time_histories.png")


def fig09(tt):
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.9), sharex=True)
    for ax, tag, name in ((axes[0], "tan", "tangential"), (axes[1], "rad", "radial")):
        for sh, mk in (("Unshaped", "o"), ("ZV", "s"), ("ZVD", "^"), ("Two-mode ZV", "D")):
            g = tt[tt.shaper == sh].sort_values("trolley_nom_mm")
            bad = g.cmd_truncated
            ax.plot(g.trolley_nom_mm, g[f"{tag}_amp_pp_rad"], mk + "-",
                    color=C[sh], ms=6, label=sh)
            if bad.any():
                ax.plot(g[bad].trolley_nom_mm, g[bad][f"{tag}_amp_pp_rad"], "x",
                        color="k", ms=9, mew=1.6,
                        label="truncated recording" if tag == "tan" else None)
        ax.set_xlabel("trolley radius on the jib (mm)")
        ax.set_ylabel(f"{name} swing, peak-to-peak (rad)")
        ax.set_title(f"{name} residual swing")
        ax.set_xticks([500, 700, 900])
    axes[0].legend(fontsize=7.5)
    fig.suptitle("Tower: residual swing grows with trolley radius for every shaper;\n"
                 "only the robust and two-mode shapers reduce the radial component", y=1.0)
    fig.tight_layout()
    return save(fig, "fig09_tower_residual_vs_radius.png")


def fig10(tt):
    shapers = ["Unshaped", "ZV", "ZVD", "Two-mode ZV"]
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.9))
    x = np.arange(len(shapers))
    wd = 0.38
    for ax, tag, name in ((axes[0], "tan", "tangential"), (axes[1], "rad", "radial")):
        for k, (mode, hatch) in enumerate(((1, ""), (2, "///"))):
            vals = [tt[tt.shaper == s][f"{tag}_m{mode}_amp_pp_rad"].mean() for s in shapers]
            errs = [tt[tt.shaper == s][f"{tag}_m{mode}_amp_pp_rad"].std() for s in shapers]
            ax.bar(x + (k - 0.5) * wd, vals, wd, yerr=errs, capsize=3,
                   hatch=hatch, edgecolor="k", linewidth=0.6,
                   color=["#b0c4de", "#8fbc8f"][k],
                   label=f"mode {mode}")
        ax.set_xticks(x)
        ax.set_xticklabels(shapers, rotation=18, ha="right")
        ax.set_ylabel(f"{name} swing, p-p (rad)")
        ax.set_title(f"{name}")
    f1 = tt.f1_theory_hz.mean()
    f2 = tt.f2_theory_hz.mean()
    axes[0].legend(title=f"mode 1 ~ {f1:.2f} Hz\nmode 2 ~ {f2:.2f} Hz", fontsize=7.5,
                   title_fontsize=7.5)
    fig.suptitle("Tower, residual split by mode: the single-mode ZV leaves mode 2 untouched "
                 "(slightly worse),\nwhile the two-mode ZV is the only shaper that suppresses "
                 "mode 2", y=1.0)
    fig.tight_layout()
    return save(fig, "fig10_tower_mode_split.png")


def fig11(tt, radius=900):
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    for sh in ["Unshaped", "ZV", "ZVD", "Two-mode ZV"]:
        sub = tt[(tt.shaper == sh) & (tt.trolley_nom_mm == radius)]
        if sub.empty:
            continue
        r = sub.iloc[0]
        d = L.load_tower(os.path.join(L.TOWER_DIR, r.file))
        vi = d["vision"]
        t = vi["Time [s]"].to_numpy(float)
        y = vi["Tangential Swing [rad]"].to_numpy(float)
        ok = np.isfinite(y)
        tt_, yy = t[ok], y[ok]
        yy = yy - np.polyval(np.polyfit(tt_, yy, 1), tt_)
        dt = float(np.median(np.diff(tt_)))
        tu = np.arange(tt_[0], tt_[-1], dt)
        yu = np.interp(tu, tt_, yy) * np.hanning(tu.size)
        n = 1 << int(np.ceil(np.log2(tu.size * 8)))
        spec = np.abs(np.fft.rfft(yu, n)) * 2 / tu.size
        f = np.fft.rfftfreq(n, dt)
        m = (f > 0.15) & (f < 3.0)
        ax.semilogy(f[m], spec[m], color=C[sh], label=sh)
    for fc, lab in ((tt.f1_theory_hz.mean(), "mode 1 (model)"),
                    (tt.f2_theory_hz.mean(), "mode 2 (model)")):
        ax.axvline(fc, color="0.5", lw=0.8, ls="--")
        ax.text(fc, ax.get_ylim()[1], f" {lab}", fontsize=7.5, va="top", color="0.3")
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("tangential swing amplitude (rad)")
    ax.set_title(f"Tower swing spectra, trolley at {radius} mm: both double-pendulum modes are "
                 "present,\nand the measured peaks sit ~10 % below the linearised model")
    ax.legend()
    fig.tight_layout()
    return save(fig, "fig11_tower_spectra.png")


def fig12(sc, tt):
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    for col, lab, c, ls in (
            ("tower_ZV_0.84", "ZV, $\\Delta t$ = 0.84 s (as run)", C["ZV"], "-"),
            ("tower_ZVD_asrun", "ZVD as run [0.25, 0.50, 0.50]", C["ZVD"], "-"),
            ("tower_ZVD_textbook", "ZVD textbook [0.25, 0.50, 0.25]", C["ZVD (textbook)"], "--"),
            ("tower_2modeZV", "two-mode ZV [0.25]x4", C["Two-mode ZV"], "-")):
        if col in sc:
            ax.plot(sc.w_rad_s, 100 * sc[col], color=c, ls=ls, label=lab)
    for w, lab in ((tt.w1_theory.mean(), "$\\omega_1$ model"),
                   (tt.w2_theory.mean(), "$\\omega_2$ model")):
        ax.axvline(w, color="0.5", lw=0.9, ls="--")
        ax.text(w, 103, f" {lab}", fontsize=7.5, va="top", color="0.3")
    for w, lab in ((2 * np.pi * tt.tan_m1_f_meas_hz.mean(), "$\\omega_1$ meas"),
                   (2 * np.pi * tt.tan_m2_f_meas_hz.mean(), "$\\omega_2$ meas")):
        ax.axvline(w, color="#c1272d", lw=0.9, ls=":")
        ax.text(w, 78, f" {lab}", fontsize=7.5, va="top", color="#c1272d")
    ax.set_xlim(1.5, 14)
    ax.set_ylim(0, 108)
    ax.set_xlabel("frequency $\\omega$ (rad/s)")
    ax.set_ylabel("residual vibration, % of unshaped")
    ax.set_title("Tower design justification: only the two-mode ZV has a notch at BOTH modes.\n"
                 "The measured frequencies (dotted red) sit below the model, which is why "
                 "cancellation is partial.")
    ax.legend(loc="center right", fontsize=7.5)
    fig.tight_layout()
    return save(fig, "fig12_tower_sensitivity.png")


def main():
    tr = pd.read_csv(os.path.join(L.TABLES, "bridge_trials.csv"))
    tt = pd.read_csv(os.path.join(L.TABLES, "tower_trials.csv"))
    sc = pd.read_csv(os.path.join(L.TABLES, "sensitivity_curves.csv"))
    pred = pd.read_csv(os.path.join(L.TABLES, "design_predictions.csv"))
    print("figures:")
    fig01(tr); fig02(tr, pred); fig03(tr); fig04(tr, pred); fig05(sc, tr); fig06(tr)
    fig07(tt); fig08(tt); fig09(tt); fig10(tt); fig11(tt); fig12(sc, tt)


if __name__ == "__main__":
    main()

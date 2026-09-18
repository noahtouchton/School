"""
Shared library for ME6404 Lab 2 (Input Shaping) data reduction.

Two rigs, two very different raw formats:

BRIDGE CRANE  (data/Bridge Crane Data/*.csv)
    12 columns, nominally X-dir and Y-dir blocks.  In this data set:
      * ONLY the Y-block carries live signals.  "Time Y-dir (sec)" is the
        clock, "Y Payload Deflection (mm)" is the payload swing, and
        "Y Crane Position (mm)" / "Y Actual Velocity (mm/sec)" describe the
        driven (trolley) axis.
      * The X-block is junk from the GUI export EXCEPT that
        "Time X-dir (sec)" and "X Command Velocity (mm/sec)" are both stuck
        at a constant equal to the CABLE LENGTH IN MM (799 for the 0.8 m
        Part-A trials, 603/901/1201 for the Part-B trials).  We exploit that
        as ground truth to validate the file names.
      * "Y Command Velocity" is identically zero (not logged).
      * Files are padded to 2000 rows with all-zero rows.

TOWER CRANE   (data/Tower Crane Data/*.xlsx)
    One "Data" sheet.  In 7 of the 12 workbooks it holds TWO acquisition
    streams stacked vertically and separated by a jump in "Time [s]"; in the
    other 5 a single block carries everything.  Where there are two:
      * block 1 (dt = 30 ms): motion-control stream.  Slew/trolley/hoist
        positions and the shaped command velocity are live; the vision
        columns (Cable Length, Tangential/Radial Swing) are blank.
      * block 2 (dt = 20 ms): vision stream.  Tangential/Radial swing and
        cable length are live; the motion columns are frozen at their final
        value.

    The two blocks are PARALLEL RECORDINGS OF THE SAME WINDOW on different
    clocks, not consecutive segments.  Block 2's clock is offset by about
    21.2 s, but the two spans have matching durations and the swing onset in
    block 2 lines up with the move start in block 1 to within the detection
    threshold.  So the vision stream DOES capture the move, and the blocks
    must be aligned by subtracting each one's own start time before the
    residual window can be placed after the end of the move.  `load_tower`
    returns `t_vision_aligned` for exactly that.
"""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd

G = 9.80665  # m/s^2

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BRIDGE_DIR = os.path.join(ROOT, "data", "Bridge Crane Data")
TOWER_DIR = os.path.join(ROOT, "data", "Tower Crane Data")
TABLES = os.path.join(ROOT, "results", "tables")
FIGURES = os.path.join(ROOT, "results", "figures")

# Hook / payload parameters, Table 1 of the handout.
M_HOOK = 0.407      # kg
M_PAYLOAD = 0.569   # kg
L2_RIG = 0.200      # m, eye of hook to payload centre


# ----------------------------------------------------------------------------
# generic signal helpers
# ----------------------------------------------------------------------------
def md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def enforce_monotonic(t: np.ndarray, *signals: np.ndarray):
    """Drop samples whose timestamp does not advance (the bridge GUI repeats
    timestamps).  Returns (t, *signals) filtered consistently."""
    keep = np.ones(t.size, dtype=bool)
    last = -np.inf
    for i, ti in enumerate(t):
        if ti > last:
            last = ti
        else:
            keep[i] = False
    return (t[keep],) + tuple(s[keep] for s in signals)


def split_blocks(t: np.ndarray, gap: float = 1.0):
    """Split a record wherever the clock jumps forward by more than `gap`
    or jumps backwards.  Returns a list of (start, stop_exclusive) indices."""
    if t.size == 0:
        return []
    dt = np.diff(t)
    brk = np.nonzero((dt > gap) | (dt < 0))[0] + 1
    edges = [0, *brk.tolist(), t.size]
    return [(a, b) for a, b in zip(edges[:-1], edges[1:]) if b - a > 1]


def find_motion_segments(t, v, thresh, merge_gap=0.8, min_dur=0.3):
    """Locate intervals where |v| exceeds `thresh`, merging intervals that are
    closer together than `merge_gap` (so the accel / cruise / decel phases and
    the individual shaper steps become one move)."""
    active = np.abs(v) > thresh
    segs, on = [], None
    for i, a in enumerate(active):
        if a and on is None:
            on = i
        elif not a and on is not None:
            segs.append([on, i - 1])
            on = None
    if on is not None:
        segs.append([on, active.size - 1])

    merged = []
    for s in segs:
        if merged and t[s[0]] - t[merged[-1][1]] < merge_gap:
            merged[-1][1] = s[1]
        else:
            merged.append(s)
    return [tuple(s) for s in merged if t[s[1]] - t[s[0]] >= min_dur]


def local_extrema(y):
    """Indices of interior local maxima and minima (strict).

    Kept for inspection only.  Amplitude scoring uses
    `alternating_extrema`, which is robust to quantisation chatter.
    """
    y = np.asarray(y, float)
    idx = [i for i in range(1, y.size - 1)
           if (y[i] >= y[i - 1] and y[i] > y[i + 1])
           or (y[i] <= y[i - 1] and y[i] < y[i + 1])]
    return np.asarray(idx, dtype=int)


def alternating_extrema(y, thresh=0.0):
    """Peak/valley indices that strictly alternate, with a hysteresis band.

    Walks the signal tracking a running extreme and commits it only once the
    signal has reversed by more than `thresh`.  That rejects sensor
    quantisation chatter (the bridge logs deflection to 1 mm) and, unlike a
    neighbour-comparison peak search, it cannot return two maxima in a row -
    so consecutive differences of the returned values are always true
    peak-to-peak swings.  On a two-mode signal it follows whichever mode
    dominates locally rather than collapsing, which is why it replaced the
    earlier prune-by-alternation approach.
    """
    y = np.asarray(y, float)
    if y.size < 3:
        return np.array([], dtype=int)
    out = []
    i_max = i_min = 0
    looking = 0              # +1 = a max is next to commit, -1 = a min
    for i in range(1, y.size):
        if y[i] > y[i_max]:
            i_max = i
        if y[i] < y[i_min]:
            i_min = i
        if looking >= 0 and y[i] < y[i_max] - thresh:
            out.append(i_max)
            looking = -1
            i_min = i
        elif looking <= 0 and y[i] > y[i_min] + thresh:
            out.append(i_min)
            looking = 1
            i_max = i
    return np.asarray(out, dtype=int)


def zero_cross_period(t, y):
    """Mean period from upward zero crossings of a zero-mean signal."""
    s = np.signbit(y)
    up = np.nonzero((~s[1:]) & (s[:-1]))[0]
    if up.size < 2:
        return np.nan, 0
    # linear interpolation for sub-sample crossing times
    tc = []
    for i in up:
        y0, y1 = y[i], y[i + 1]
        frac = 0.0 if y1 == y0 else -y0 / (y1 - y0)
        tc.append(t[i] + frac * (t[i + 1] - t[i]))
    tc = np.asarray(tc)
    return float(np.mean(np.diff(tc))), tc.size - 1


def fft_peaks(t, y, n_peaks=2, fmin=0.05, fmax=6.0):
    """Dominant spectral peaks (Hz) of a non-uniformly sampled signal, after
    resampling onto a uniform grid."""
    if t.size < 16:
        return []
    dt = float(np.median(np.diff(t)))
    if not np.isfinite(dt) or dt <= 0:
        return []
    tu = np.arange(t[0], t[-1], dt)
    yu = np.interp(tu, t, y)
    yu = yu - yu.mean()
    yu *= np.hanning(yu.size)
    n = 1 << int(np.ceil(np.log2(max(yu.size * 4, 64))))
    spec = np.abs(np.fft.rfft(yu, n))
    freq = np.fft.rfftfreq(n, dt)
    band = (freq >= fmin) & (freq <= fmax)
    spec, freq = spec[band], freq[band]
    if spec.size == 0:
        return []
    order = np.argsort(spec)[::-1]
    peaks, taken = [], []
    for k in order:
        if any(abs(freq[k] - f) < 0.15 for f in taken):
            continue
        taken.append(freq[k])
        peaks.append((float(freq[k]), float(spec[k] / spec.max())))
        if len(peaks) >= n_peaks:
            break
    return peaks


@dataclass
class Residual:
    """Residual-vibration metrics over one analysis window."""
    t0: float = np.nan
    t1: float = np.nan
    window_s: float = np.nan
    n_extrema: int = 0
    n_cycles: float = np.nan
    amp_pp_mean: float = np.nan     # mean peak-to-peak over the window
    amp_pp_first: float = np.nan    # first peak-to-peak (worst case)
    amp_pp_max: float = np.nan
    amp_single_mean: float = np.nan  # half of amp_pp_mean
    rms: float = np.nan
    period_zc: float = np.nan
    freq_zc_hz: float = np.nan
    freq_fft_hz: float = np.nan
    freq_fft2_hz: float = np.nan
    zeta: float = np.nan            # from log-decrement of the peak envelope
    offset: float = np.nan          # window mean that was removed


def residual_metrics(t, y, t0, t1, min_swing=0.0, n_cycles_cap=None, detrend="mean"):
    """Reduce y(t) on [t0, t1] to a Residual record.

    `min_swing` suppresses quantisation chatter (bridge deflection is logged to
    1 mm).  If `n_cycles_cap` is given the window is truncated to that many
    measured periods so every trial is scored over a comparable span.
    """
    r = Residual(t0=float(t0), t1=float(t1))
    m = (t >= t0) & (t <= t1)
    if m.sum() < 8:
        return r
    tw, yw = t[m], y[m].astype(float)

    if detrend == "linear" and tw.size > 4:
        c = np.polyfit(tw, yw, 1)
        base = np.polyval(c, tw)
    else:
        base = np.full_like(yw, yw.mean())
    r.offset = float(base.mean())
    yc = yw - base

    per, ncyc = zero_cross_period(tw, yc)
    if n_cycles_cap and np.isfinite(per) and per > 0:
        t1b = min(t1, t0 + n_cycles_cap * per)
        m = (t >= t0) & (t <= t1b)
        tw, yw = t[m], y[m].astype(float)
        if tw.size < 8:
            return r
        base = np.polyval(np.polyfit(tw, yw, 1), tw) if detrend == "linear" else np.full_like(yw, yw.mean())
        yc = yw - base
        r.t1 = float(tw[-1])
        per, ncyc = zero_cross_period(tw, yc)

    r.window_s = float(tw[-1] - tw[0])
    r.period_zc = per
    r.freq_zc_hz = 1.0 / per if (np.isfinite(per) and per > 0) else np.nan
    r.rms = float(np.sqrt(np.mean(yc ** 2)))

    pk = fft_peaks(tw, yc, n_peaks=2)
    if pk:
        r.freq_fft_hz = pk[0][0]
    if len(pk) > 1:
        r.freq_fft2_hz = pk[1][0]

    idx = alternating_extrema(yc, min_swing)
    r.n_extrema = int(idx.size)
    if idx.size >= 2:
        swings = np.abs(np.diff(yc[idx]))
        r.amp_pp_mean = float(np.mean(swings))
        r.amp_pp_first = float(swings[0])
        r.amp_pp_max = float(np.max(swings))
        r.amp_single_mean = r.amp_pp_mean / 2.0
        r.n_cycles = idx.size / 2.0
        # log-decrement damping from the |extrema| envelope
        env = np.abs(yc[idx])
        good = env > 0
        if good.sum() >= 4 and np.isfinite(per) and per > 0:
            tt = tw[idx][good]
            slope = np.polyfit(tt, np.log(env[good]), 1)[0]
            wn = 2 * np.pi / per
            r.zeta = float(min(max(-slope / wn, 0.0), 1.0))
    return r


# ----------------------------------------------------------------------------
# input-shaper mathematics
# ----------------------------------------------------------------------------
def double_pendulum_freqs(L1, L2, m_h=M_HOOK, m_p=M_PAYLOAD, g=G):
    """Linearised natural frequencies (rad/s) of the hook/payload double
    pendulum, per the handout."""
    R = m_p / m_h
    s = 1.0 / L1 + 1.0 / L2
    beta = np.sqrt((1 + R) ** 2 * s ** 2 - 4 * (1 + R) / (L1 * L2))
    w1 = np.sqrt(0.5 * g * ((1 + R) * s - beta))
    w2 = np.sqrt(0.5 * g * ((1 + R) * s + beta))
    return float(w1), float(w2), float(R), float(beta)


def pendulum_freq(L, g=G):
    """Single-pendulum natural frequency (rad/s) and period (s)."""
    w = np.sqrt(g / L)
    return float(w), float(2 * np.pi / w)


def zv_shaper(w, zeta=0.0):
    """Zero-vibration shaper (2 impulses)."""
    K = np.exp(-zeta * np.pi / np.sqrt(1 - zeta ** 2)) if zeta else 1.0
    A = np.array([1.0, K]) / (1.0 + K)
    wd = w * np.sqrt(1 - zeta ** 2) if zeta else w
    return A, np.array([0.0, np.pi / wd])


def zvd_shaper(w, zeta=0.0):
    """Zero-vibration-and-derivative shaper (3 impulses)."""
    K = np.exp(-zeta * np.pi / np.sqrt(1 - zeta ** 2)) if zeta else 1.0
    A = np.array([1.0, 2 * K, K ** 2]) / (1 + K) ** 2
    wd = w * np.sqrt(1 - zeta ** 2) if zeta else w
    return A, np.array([0.0, np.pi / wd, 2 * np.pi / wd])


def ei_shaper(w, V=0.05):
    """Extra-insensitive shaper (undamped, tolerance V of residual vibration)."""
    A = np.array([(1 + V) / 4.0, (1 - V) / 2.0, (1 + V) / 4.0])
    T = 2 * np.pi / w
    return A / A.sum(), np.array([0.0, T / 2.0, T])


def convolve_shapers(A1, t1, A2, t2):
    """Convolution of two impulse sequences (multi-mode shaper design)."""
    A, T = [], []
    for a1, tt1 in zip(A1, t1):
        for a2, tt2 in zip(A2, t2):
            A.append(a1 * a2)
            T.append(tt1 + tt2)
    order = np.argsort(T)
    A = np.asarray(A)[order]
    T = np.asarray(T)[order]
    # merge coincident impulses
    Am, Tm = [A[0]], [T[0]]
    for a, tt in zip(A[1:], T[1:]):
        if abs(tt - Tm[-1]) < 1e-9:
            Am[-1] += a
        else:
            Am.append(a)
            Tm.append(tt)
    return np.asarray(Am), np.asarray(Tm)


def residual_vibration(A, T, w, zeta=0.0):
    """Percent residual vibration of an impulse sequence at frequency w.

    V = |sum A_i e^{zeta w t_i} e^{j wd t_i}| / sum A_i   (undamped: zeta=0)
    Accepts scalar or array w.
    """
    w = np.atleast_1d(np.asarray(w, dtype=float))
    A = np.asarray(A, dtype=float)
    T = np.asarray(T, dtype=float)
    wd = w * np.sqrt(1 - zeta ** 2) if zeta else w
    gain = np.exp(zeta * w[:, None] * T[None, :]) if zeta else 1.0
    c = np.sum(A[None, :] * gain * np.cos(wd[:, None] * T[None, :]), axis=1)
    s = np.sum(A[None, :] * gain * np.sin(wd[:, None] * T[None, :]), axis=1)
    amp = np.sqrt(c ** 2 + s ** 2) / A.sum()
    if zeta:
        amp *= np.exp(-zeta * w * T[-1])
    return amp if amp.size > 1 else float(amp[0])


def pulse_sequence(tf, A=None, T=None):
    """The 'pulse'/'return' command the cranes actually apply: a +1 impulse at
    t=0 and a -1 impulse at t=tf, convolved with the shaper (A, T).

    This is what determines the real residual vibration on these rigs, because
    the impulses sum to zero rather than one (see Fig. 1-2 of the handout).
    """
    Ap = np.array([1.0, -1.0])
    Tp = np.array([0.0, float(tf)])
    if A is None:
        return Ap, Tp
    Ac, Tc = [], []
    for a1, tt1 in zip(Ap, Tp):
        for a2, tt2 in zip(A, T):
            Ac.append(a1 * a2)
            Tc.append(tt1 + tt2)
    order = np.argsort(Tc)
    return np.asarray(Ac)[order], np.asarray(Tc)[order]


# ----------------------------------------------------------------------------
# bridge crane I/O
# ----------------------------------------------------------------------------
BRIDGE_COLS = dict(t=6, defl=7, pos=8, vel=9, cable=4)


def load_bridge(path):
    """Return a dict with the usable bridge channels plus the in-file cable
    length tag, with padding rows and repeated timestamps removed."""
    df = pd.read_csv(path)
    cols = list(df.columns)
    df = df[~(df == 0).all(axis=1)].reset_index(drop=True)
    cable = float(np.median(df[cols[BRIDGE_COLS["cable"]]].values))
    t = df[cols[BRIDGE_COLS["t"]]].to_numpy(float)
    defl = df[cols[BRIDGE_COLS["defl"]]].to_numpy(float)
    pos = df[cols[BRIDGE_COLS["pos"]]].to_numpy(float)
    vel = df[cols[BRIDGE_COLS["vel"]]].to_numpy(float)
    t, defl, pos, vel = enforce_monotonic(t, defl, pos, vel)
    return dict(path=path, name=os.path.basename(path), cable_mm=cable,
                t=t, defl=defl, pos=pos, vel=vel, n=t.size)


_RE_UNSHAPED_A = re.compile(r"^Bridge_100_-100_(\d+)_v(\d+)(_del)?$", re.I)
_RE_ZV_A = re.compile(r"^Bridge_ZV_\.5_\.5_\.9_(\d+)_v(\d+)(_del)?$", re.I)
_RE_UNSHAPED_B = re.compile(r"^Bridge_b_(\d+)_100_1500_v(\d+)(_del)?$", re.I)
_RE_ZV_B = re.compile(r"^Bride_ZV_\.5_\.5_\.91_(\d+)_v(\d+)(_del)?$", re.I)
_RE_ZVD_B = re.compile(r"^Bride_ZVD_\.5_\.5_\.91_(\d+)_v(\d+)(_del)?$", re.I)


def parse_bridge_name(name):
    """Decode the lab condition from a bridge file name.

    Naming conventions used during the session:
      Bridge_100_-100_<tf>_v<n>        Part A, unshaped, 0.8 m cable
      Bridge_ZV_.5_.5_.9_<tf>_v<n>     Part A, ZV (dt = 0.90 s), 0.8 m cable
      Bridge_b_<cable>_100_1500_v<n>   Part B, unshaped, tf = 1500 ms
      Bride_ZV_.5_.5_.91_<cable>_v<n>  Part B, ZV  (dt = 0.91 s), tf = 1500 ms
      Bride_ZVD_.5_.5_.91_<cable>_v<n> Part B, ZVD (dt = 0.91 s), tf = 1500 ms
    A trailing '_del' marks a trial the group flagged in the lab.
    """
    stem = os.path.splitext(name)[0]
    out = dict(part=None, shaper=None, tf_ms=np.nan, cable_nom_mm=np.nan,
               trial=np.nan, flagged=False, shaper_dt_s=np.nan, n_impulses=np.nan)
    for rx, part, shaper, kind, dt, nimp in (
        (_RE_UNSHAPED_A, "A", "Unshaped", "tf", np.nan, 1),
        (_RE_ZV_A, "A", "ZV", "tf", 0.90, 2),
        (_RE_UNSHAPED_B, "B", "Unshaped", "cable", np.nan, 1),
        (_RE_ZV_B, "B", "ZV", "cable", 0.91, 2),
        (_RE_ZVD_B, "B", "ZVD", "cable", 0.91, 3),
    ):
        m = rx.match(stem)
        if not m:
            continue
        val = int(m.group(1))
        out.update(part=part, shaper=shaper, trial=int(m.group(2)),
                   flagged=bool(m.group(3)), shaper_dt_s=dt, n_impulses=nimp)
        if kind == "tf":
            out["tf_ms"] = val
            out["cable_nom_mm"] = 800.0
        else:
            out["cable_nom_mm"] = val
            out["tf_ms"] = 1500
        return out
    return out


# ----------------------------------------------------------------------------
# tower crane I/O
# ----------------------------------------------------------------------------
TOWER_MOTION = ["Time [s]", "Position Slew [deg]", "Command Velo Slew [deg/s]",
                "Actual Velo Slew [deg/s]", "Position Trolley [mm]",
                "Position Hoist [mm]"]
TOWER_VISION = ["Time [s]", "Tangential Swing [rad]", "Radial Swing [rad]",
                "Cable Length [mm]"]


def load_tower(path):
    """Split a tower workbook into its motion stream and its vision stream,
    and put the vision stream on the motion stream's clock.

    The two streams are stacked vertically in one sheet and are told apart by
    whether the vision columns are populated - NOT by looking for a jump in
    "Time [s]".  A time-gap split silently fails on 5 of the 12 workbooks,
    where the vision stream's clock carries straight on from the motion
    stream's with no gap, so the whole sheet looks like a single block and the
    swing data then appears to sit at times when the crane was still moving.

    The streams are simultaneous recordings of one event on different clocks:
    the motion stream always starts near t = 0 at 30 ms, the vision stream
    always starts at t ~ 21.2 s at 20 ms.  Aligning their start times aligns
    the event - verified by the swing onset landing on the move start to within
    the detection threshold in all 12 trials.  `t_vision_aligned` is the vision
    time base expressed on the motion clock.
    """
    df = pd.read_excel(path, sheet_name="Data")
    vok = df["Tangential Swing [rad]"].notna().to_numpy()

    motion = df[~vok].reset_index(drop=True) if (~vok).any() else None
    vision = df[vok].reset_index(drop=True) if vok.any() else None
    if motion is not None and len(motion) < 20:
        motion = None
    if vision is not None and len(vision) < 20:
        vision = None
    if motion is None:                      # degenerate: use the whole sheet
        motion = df.reset_index(drop=True)

    offset = 0.0
    t_vision_aligned = None
    if vision is not None:
        tv = vision["Time [s]"].to_numpy(float)
        tm = motion["Time [s]"].to_numpy(float)
        offset = float(tv.min() - tm.min())
        t_vision_aligned = tv - offset
    return dict(path=path, name=os.path.basename(path), raw=df,
                motion=motion, vision=vision,
                n_blocks=len(split_blocks(df["Time [s]"].to_numpy(float))),
                vision_clock_offset_s=offset,
                t_vision_aligned=t_vision_aligned,
                motion_dur_s=float(np.ptp(motion["Time [s]"].to_numpy(float))),
                vision_dur_s=float(np.ptp(vision["Time [s]"].to_numpy(float)))
                if vision is not None else np.nan)


def command_profile(motion):
    """Extract the staircase of commanded slew velocity as (times, levels).

    The tower GUI logs the shaped command with an occasional x100 scale glitch
    and sporadic one-sample intermediate values; both are cleaned here.
    """
    t = motion["Time [s]"].to_numpy(float)
    c = motion["Command Velo Slew [deg/s]"].to_numpy(float)
    if np.nanmax(np.abs(c)) > 200:          # x100 logging glitch
        c = c / 100.0
    # drop single-sample transition artefacts
    c = pd.Series(c).rolling(3, center=True, min_periods=1).median().to_numpy()
    lvl = np.round(c / 0.81) * 0.81         # GUI quantises to 2.5 % of 32.4
    ch = np.nonzero(np.diff(lvl))[0] + 1
    times = [t[0]] + t[ch].tolist()
    levels = [lvl[0]] + lvl[ch].tolist()
    # collapse repeats
    ts, ls = [times[0]], [levels[0]]
    for tt, ll in zip(times[1:], levels[1:]):
        if abs(ll - ls[-1]) < 1e-6:
            continue
        if tt - ts[-1] < 0.06 and len(ls) > 1:   # merge sampling jitter
            ls[-1] = ll
            continue
        ts.append(tt)
        ls.append(ll)
    return np.asarray(ts), np.asarray(ls)


def _staircase_steps(ts, ls, full, sign):
    """Pull one monotonic run of steps out of the command staircase.

    sign=+1 takes the rising run (the shaper convolved with the step up),
    sign=-1 takes the falling run (the mirrored negative impulses that stop the
    crane).  Either one encodes the shaper: the step sizes are the impulse
    amplitudes and the step times are the impulse delays.
    """
    d = np.diff(np.concatenate(([0.0], ls))) if sign > 0 else -np.diff(np.concatenate((ls, [0.0])))
    tt = ts if sign > 0 else ts
    best = None
    i = 0
    n = d.size
    while i < n:
        if d[i] <= 1e-6:
            i += 1
            continue
        j = i
        while j + 1 < n and d[j + 1] > 1e-6:
            j += 1
        run = (i, j)
        if best is None or (j - i) > (best[1] - best[0]):
            best = run
        i = j + 1
    if best is None:
        return np.array([]), np.array([])
    a, b = best
    amps = d[a:b + 1] / full
    times = tt[a:b + 1] if sign > 0 else tt[a:b + 1]
    return amps, times - times[0]


def classify_tower_shaper(motion, full=32.4):
    """Identify which shaper produced a tower trial from its command profile.

    The commanded slew velocity is the shaper convolved with the velocity
    pulse, so its rising staircase steps ARE the shaper impulse amplitudes and
    its step times ARE the impulse delays.  The command falls again in a
    mirror image at the end of the pulse, so when a recording starts mid-move
    and clips the rise, the fall is used instead.
    """
    ts, ls = command_profile(motion)
    if ls.size == 0 or np.max(np.abs(ls)) <= 0:
        return dict(shaper="Unknown", n_impulses=0, amps=[], delays=[],
                    rise_s=np.nan, amp_sum=np.nan, source="none", truncated=True)

    up_a, up_t = _staircase_steps(ts, ls, full, +1)
    dn_a, dn_t = _staircase_steps(ts, ls, full, -1)

    # A recording that starts mid-move already has a non-zero command on the
    # first sample, so its rising staircase is missing its first step(s).
    clipped_rise = ls[0] > 1e-6
    if clipped_rise and dn_a.size >= up_a.size:
        amps, delays, source = dn_a, dn_t, "fall"
    elif up_a.size >= dn_a.size:
        amps, delays, source = up_a, up_t, "rise"
    else:
        amps, delays, source = dn_a, dn_t, "fall"

    n = amps.size
    if n == 0:
        shaper = "Unknown"
    elif n == 1:
        shaper = "Unshaped"
    elif n == 2:
        shaper = "ZV"
    elif n == 3:
        shaper = "ZVD"
    elif n == 4:
        # four near-equal impulses is the convolution of two ZV shapers
        shaper = "Two-mode ZV" if np.ptp(amps) < 0.12 else "4-impulse"
    else:
        shaper = f"{n}-impulse"
    return dict(shaper=shaper, n_impulses=int(n),
                amps=[round(float(a), 3) for a in amps],
                delays=[round(float(d), 3) for d in delays],
                rise_s=float(delays[-1]) if n else np.nan,
                amp_sum=float(np.sum(amps)) if n else np.nan,
                source=source, truncated=bool(clipped_rise))


def tower_move_window(motion, thresh=1.0):
    """(t_start, t_end) of the slew move from the actual slew velocity."""
    t = motion["Time [s]"].to_numpy(float)
    v = motion["Actual Velo Slew [deg/s]"].to_numpy(float)
    segs = find_motion_segments(t, v, thresh=thresh, merge_gap=1.0, min_dur=0.5)
    if not segs:
        return np.nan, np.nan
    a, b = max(segs, key=lambda s: t[s[1]] - t[s[0]])
    return float(t[a]), float(t[b])


def write_table(df, name, index=False):
    os.makedirs(TABLES, exist_ok=True)
    path = os.path.join(TABLES, name)
    df.to_csv(path, index=index)
    return path


# ----------------------------------------------------------------------------
# predicted residual vibration from a MEASURED velocity profile
# ----------------------------------------------------------------------------
def predicted_residual_from_velocity(t, v, w, t0=None, t1=None):
    """Residual payload-swing amplitude predicted by the measured trolley
    velocity profile, for an undamped pendulum of frequency `w`.

    A pendulum whose support moves obeys  y'' + w^2 y = -a(t), where y is the
    payload deflection relative to the support.  Integrating the forced
    response and using v = 0 at both ends of the move gives the residual
    amplitude exactly as the Fourier transform of the velocity profile
    evaluated at the natural frequency:

        Y_residual = | integral v(t) e^{j w t} dt |

    so a single number computed straight from the logged velocity predicts the
    swing amplitude, with no assumption that the profile was an ideal pulse.
    Units follow v (mm/s in -> mm out).  Returns the SINGLE amplitude; double
    it to compare against a peak-to-peak measurement.
    """
    m = np.ones(t.size, dtype=bool)
    if t0 is not None:
        m &= t >= t0
    if t1 is not None:
        m &= t <= t1
    tt, vv = t[m], v[m].astype(float)
    if tt.size < 4:
        return np.nan
    w = float(w)
    c = np.trapezoid(vv * np.cos(w * tt), tt)
    s = np.trapezoid(vv * np.sin(w * tt), tt)
    return float(np.hypot(c, s))


def predicted_residual_vs_freq(t, v, w_grid, t0=None, t1=None):
    """`predicted_residual_from_velocity` swept over an array of frequencies -
    i.e. the measured command's own sensitivity curve."""
    return np.array([predicted_residual_from_velocity(t, v, w, t0, t1) for w in np.atleast_1d(w_grid)])


def pulse_residual_ratio(A, T, tf, w):
    """Residual vibration of (shaper * velocity pulse) RELATIVE to the same
    pulse with no shaper, at frequency w.

    On both of these rigs the applied command is the shaper convolved with a
    pulse of length `tf`, whose impulses are +1 at t=0 and -1 at t=tf.  Those
    sum to zero, so the usual "divide by sum(A)" normalisation blows up; the
    meaningful figure of merit is the ratio to the unshaped pulse:

        V_rel(w) = |sum A_i e^{j w t_i}| (shaped)  /  |1 - e^{j w tf}|

    V_rel = 1 means the shaper did nothing, V_rel = 0 means perfect
    cancellation.  Accepts scalar or array w.
    """
    w = np.atleast_1d(np.asarray(w, dtype=float))

    def mag(Aa, Tt):
        c = np.sum(np.asarray(Aa)[None, :] * np.cos(w[:, None] * np.asarray(Tt)[None, :]), axis=1)
        s = np.sum(np.asarray(Aa)[None, :] * np.sin(w[:, None] * np.asarray(Tt)[None, :]), axis=1)
        return np.hypot(c, s)

    As, Ts = pulse_sequence(tf, np.asarray(A, float), np.asarray(T, float))
    Au, Tu = pulse_sequence(tf)
    den = mag(Au, Tu)
    out = np.where(den > 1e-12, mag(As, Ts) / np.maximum(den, 1e-12), np.nan)
    return out if out.size > 1 else float(out[0])


def bandpass_fft(t, y, f_lo, f_hi):
    """Zero-phase band-pass by masking rFFT bins.  Returns the filtered signal
    resampled back onto the original (possibly non-uniform) time base."""
    finite = np.isfinite(y)
    if finite.sum() < 16:
        return np.full_like(y, np.nan, dtype=float)
    tt, yy = t[finite], y[finite].astype(float)
    dt = float(np.median(np.diff(tt)))
    if not np.isfinite(dt) or dt <= 0:
        return np.full_like(y, np.nan, dtype=float)
    # remove the linear trend first, or slow sensor drift leaks into the
    # lowest band and masquerades as mode-1 content
    yy = yy - np.polyval(np.polyfit(tt, yy, 1), tt)
    tu = np.arange(tt[0], tt[-1], dt)
    yu = np.interp(tu, tt, yy)
    yu = yu - yu.mean()
    Y = np.fft.rfft(yu)
    f = np.fft.rfftfreq(yu.size, dt)
    Y[(f < f_lo) | (f > f_hi)] = 0.0
    yf = np.fft.irfft(Y, yu.size)
    return np.interp(t, tu, yf, left=np.nan, right=np.nan)


def mode_amplitudes(t, y, mode_freqs_hz, bw_frac=0.30, min_swing=0.0,
                    t_score_hi=None):
    """Split a multi-mode signal into one band per mode and score each band.

    A two-mode payload cannot be characterised by zero crossings - they count
    both modes and return a meaningless period.  Band-splitting around each
    predicted mode frequency gives the amplitude actually sitting in each mode,
    which is exactly what separates a single-mode shaper (cancels mode 1,
    leaves mode 2) from a two-mode shaper (cancels both).

    Filtering always uses the WHOLE record, because the frequency resolution of
    a short window is too coarse to locate a mode.  Amplitudes are then scored
    only up to `t_score_hi`, so every trial is scored over the same span even
    though the records have different lengths.  Frequencies therefore come from
    the full record and amplitudes from a fixed window.

    `f_at_band_edge` is set when the in-band spectral peak lands on a band
    limit, which means there is no real mode there - only leakage from outside.
    """
    out = []
    for i, f0 in enumerate(np.atleast_1d(mode_freqs_hz), start=1):
        lo, hi = f0 * (1 - bw_frac), f0 * (1 + bw_frac)
        yf = bandpass_fft(t, y, lo, hi)
        ok = np.isfinite(yf)
        d = dict(mode=i, f_center_hz=float(f0), f_lo_hz=float(lo), f_hi_hz=float(hi))
        if ok.sum() < 16:
            d.update(amp_pp=np.nan, amp_pp_mean=np.nan, rms=np.nan,
                     f_peak_hz=np.nan, f_at_band_edge=True, score_window_s=np.nan)
            out.append(d)
            continue

        tt, yy = t[ok], yf[ok]
        # frequency from the full band-passed record
        pk = fft_peaks(tt, yy, n_peaks=1, fmin=lo, fmax=hi)
        fpk = pk[0][0] if pk else np.nan
        edge = bool(np.isfinite(fpk) and (fpk < lo * 1.03 or fpk > hi * 0.97))

        # amplitude over the common scoring window only
        if t_score_hi is not None:
            sel = tt <= t_score_hi
            if sel.sum() >= 16:
                tt, yy = tt[sel], yy[sel]
        idx = alternating_extrema(yy, min_swing)
        swings = np.abs(np.diff(yy[idx])) if idx.size >= 2 else np.array([np.nan])
        d.update(amp_pp=float(np.nanmax(swings)) if swings.size else np.nan,
                 amp_pp_mean=float(np.nanmean(swings)) if swings.size else np.nan,
                 rms=float(np.sqrt(np.mean(yy ** 2))),
                 f_peak_hz=fpk, f_at_band_edge=edge,
                 score_window_s=float(tt[-1] - tt[0]))
        out.append(d)
    return out


# ----------------------------------------------------------------------------
# reconstructing the bridge command (it is never logged)
# ----------------------------------------------------------------------------
def velocity_plateaus(t, v, v_max=None, tol=0.06, min_hold=0.15):
    """Find the flat levels of a staircase velocity profile.

    The bridge GUI does not log its command (`Y Command Velocity` is all
    zeros), so the shaper that was actually applied has to be read back out of
    the measured trolley velocity.  Each shaper impulse shows up as a plateau
    whose height is the running sum of the impulse amplitudes times the top
    speed.  Returns a list of (t_start, t_end, level) for each plateau that is
    held at least `min_hold` seconds.
    """
    v = np.asarray(v, float)
    if v_max is None:
        v_max = np.max(np.abs(v))
    if v_max <= 0:
        return []
    band = tol * v_max
    plateaus = []
    i = 0
    n = v.size
    while i < n:
        j = i
        while j + 1 < n and abs(v[j + 1] - v[i]) <= band:
            j += 1
        if t[j] - t[i] >= min_hold:
            plateaus.append((float(t[i]), float(t[j]),
                             float(np.median(v[i:j + 1]))))
        i = j + 1
    # merge adjacent plateaus at the same level
    merged = []
    for p in plateaus:
        if merged and abs(p[2] - merged[-1][2]) <= band:
            merged[-1] = (merged[-1][0], p[1], 0.5 * (merged[-1][2] + p[2]))
        else:
            merged.append(list(p) if False else (p[0], p[1], p[2]))
            merged[-1] = tuple(merged[-1])
    out, buf = [], None
    for p in plateaus:
        if buf is not None and abs(p[2] - buf[2]) <= band:
            buf = (buf[0], p[1], 0.5 * (buf[2] + p[2]))
        else:
            if buf is not None:
                out.append(buf)
            buf = p
    if buf is not None:
        out.append(buf)
    return out


def reconstruct_staircase(t, v, t0, t1, v_max=None, min_level_frac=0.06,
                          min_hold=0.10):
    """Read the applied impulse sequence off a measured staircase velocity.

    The bridge GUI logs no command, so this is the only way to know what was
    actually run.  Differencing the plateau staircase recovers the convolved
    impulse train; POSITIVE steps are the shaper itself and NEGATIVE steps are
    the mirrored sequence that stops the crane (the handout's "last n impulses
    mirror the first n with negative amplitude").

    The two halves interleave whenever the button-hold time tf is shorter than
    the shaper duration, which puts a dip in the middle of the profile - so the
    halves must be separated by the SIGN of each step, not by finding the peak.

    Returns normalised shaper amplitudes and delays, the mirrored amplitudes
    and their delays relative to tf, and the inferred tf.
    """
    m = (t >= t0) & (t <= t1)
    tt, vv = t[m], np.abs(np.asarray(v, float)[m])
    if tt.size < 8:
        return {}
    if v_max is None:
        v_max = float(np.max(vv))
    if v_max <= 0:
        return {}

    pl_all = velocity_plateaus(tt, vv, v_max=v_max, min_hold=min_hold)
    # a leading or trailing zero plateau is the rest state, not a command level
    pl_used = [p for p in pl_all if abs(p[2]) >= 0.02 * v_max]
    while pl_used and abs(pl_used[0][2]) < min_level_frac * v_max:
        pl_used = pl_used[1:]
    while pl_used and abs(pl_used[-1][2]) < min_level_frac * v_max:
        pl_used = pl_used[:-1]
    if not pl_used:
        return {}
    levels = np.array([p[2] for p in pl_used]) / v_max
    times = np.array([p[0] for p in pl_used])

    # Time each step at the MIDPOINT of the acceleration ramp between the two
    # plateaus it joins.  Using plateau start times instead biases every delay
    # (and tf) late by most of a ramp, which is ~0.4 s on this rig.
    ends = np.array([p[1] for p in pl_used])
    starts = np.array([p[0] for p in pl_used])
    steps = np.diff(np.concatenate(([0.0], levels, [0.0])))
    t_first = starts[0]
    below = np.nonzero(vv < 0.05 * v_max)[0]
    pre = below[tt[below] < starts[0]]
    if pre.size:
        t_first = 0.5 * (tt[pre[-1]] + starts[0])
    t_last = ends[-1]
    after = below[tt[below] > ends[-1]]
    if after.size:
        t_last = 0.5 * (ends[-1] + tt[after[0]])
    step_t = np.concatenate((
        [t_first],
        0.5 * (ends[:-1] + starts[1:]),
        [t_last]))

    pos = [(float(st), float(a)) for st, a in zip(step_t, steps) if a > 0.05]
    neg = [(float(st), float(a)) for st, a in zip(step_t, steps) if a < -0.05]

    out = dict(v_top=round(v_max, 1), n_plateaus=int(levels.size),
               plateau_fracs=[round(float(x), 3) for x in levels],
               plateau_times=[round(float(x), 3) for x in times])
    if pos:
        t_ref = pos[0][0]
        out["shaper_amps"] = [round(a, 3) for _, a in pos]
        out["shaper_delays"] = [round(st - t_ref, 3) for st, _ in pos]
        out["shaper_amp_sum"] = round(sum(a for _, a in pos), 3)
    if neg:
        t_tf = neg[0][0]
        out["mirror_amps"] = [round(a, 3) for _, a in neg]
        out["mirror_delays"] = [round(st - t_tf, 3) for st, _ in neg]
        out["mirror_amp_sum"] = round(sum(a for _, a in neg), 3)
        if pos:
            out["tf_inferred_s"] = round(t_tf - pos[0][0], 3)
    if pos and neg:
        a_up = np.array([a for _, a in pos])
        a_dn = -np.array([a for _, a in neg])
        out["mirror_matches_shaper"] = bool(
            a_up.size == a_dn.size and np.allclose(a_up, a_dn, atol=0.08))
        out["reconstruction_balanced"] = bool(
            abs(a_up.sum() - a_dn.sum()) < 0.12)
    return out


def name_impulse_pattern(amps, tol=0.09):
    """Label a normalised impulse amplitude pattern (ZV / ZVD / ...)."""
    a = np.asarray(amps, float)
    if a.size == 0:
        return "none"
    if a.size == 1:
        return "unshaped"
    if a.size == 2 and np.allclose(a, [0.5, 0.5], atol=tol):
        return "ZV"
    if a.size == 3 and np.allclose(a, [0.25, 0.5, 0.25], atol=tol):
        return "ZVD"
    if a.size == 3 and np.allclose(a, [0.25, 0.5, 0.5], atol=tol):
        return "ZVD-like (3rd impulse 0.5, sums to 1.25)"
    if a.size == 4 and np.ptp(a) <= 2 * tol:
        return "4 equal impulses (two-mode ZV)"
    return f"{a.size} impulses {np.round(a,2).tolist()}"

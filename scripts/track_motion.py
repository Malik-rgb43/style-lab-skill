#!/usr/bin/env python3
"""Motion curves of one video, read the way the After Effects graph editor shows them.

For every camera move and every animated element: the value graph and the speed graph per
property, peak speed and where it sits in the move, overshoot and settle time, hard stop or soft
landing, and the fitted easing as a cubic bezier plus a GSAP CustomEase string, with the nearest
named ease. Across elements: stagger intervals and entrance vs exit.

Reads the video (never modifies it) and writes motion.json (schema "motion/1").

Usage:
  track_motion.py <video> --measure <measure.json> --out <dir>/motion.json [--analysis-width 320]

How it is measured (all local, free):
  - frames come from measure_video.decode_gray; shots are the spans between the cuts of
    measure.json (frames inside a gradual transition are skipped). Without --measure the cuts are
    detected here with measure_video.detect_cuts.
  - camera: LK optical flow on good features from the shot start, a RANSAC partial affine
    (scale, rotation, translation) from the shot-start frame to each frame; reset at every cut.
    A shot with too few trackable features, or features on one small object only (their box
    spans < 40 % of the frame), keeps the identity (no camera move is invented).
    tx / ty are the motion of the frame centre, in fractions of the frame; rotation in degrees.
  - elements: each frame is warped back into the shot-start frame (camera-stabilised), compared
    with the shot's temporal-median background, and the changed regions become connected
    components, tracked frame to frame by IoU >= 0.3 (a fast move that breaks the overlap is
    still matched by centre distance and a similar area). x / y are the weighted centre in frame
    fractions, scale the weighted spread relative to the track's median, opacity the mean
    contrast relative to the track's 90th percentile. Limitation: an element that stays still for
    most of a shot becomes part of the median background.
  - unique frames: a frame that repeats the one before (on-2s / on-3s animation, a variable
    frame rate made constant) is not a real sample. Only the first frame of a short repeat run
    (<= MAX_STEP_HOLD frames) is a sample; a longer run is a real hold (its first and last frames
    are samples). Moves are found and fitted on these samples, so hold frames never read as easing.
    Where a stepped move really starts and ends is uncertain by up to one picture; those two
    times are searched at quarter-picture offsets for the smoothest fit.
  - a move is a run of at least min_frames frames whose speed is above 5 % of the peak speed,
    widened to where the motion really starts and stops, with an overshoot's way back included.
  - the easing fit works on progress 0..1 over time 0..1: a coarse grid over x1, x2 in [0, 1]
    and y1, y2 in [-0.5, 1.5], then Nelder-Mead (numpy only) from the best cells over (x1, x2),
    with y1, y2 solved by least squares at each step (the curve is linear in them for a fixed
    x1, x2). The bezier is evaluated by solving x(t) = x with Newton's method and a bisection
    fallback. Named eases are compared by curve distance with their standard cubic-bezier
    equivalents (NAMED).

Exit: 0 done, 1 error, 2 bad input, 3 ffmpeg missing (one line, with an install hint).
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import store  # noqa: E402
from cut import MissingTool, require_ffmpeg  # noqa: E402

SCHEMA = "motion/1"
MOVE_SPEED_FRAC = 0.05   # a move: speed above this fraction of the peak speed ...
MIN_FRAMES = 3           # ... for at least this many frames
EXTEND_FRAC = 0.002      # the move is widened while the speed stays above this fraction of its peak
MAX_STEP_HOLD = 3        # a repeat run up to this long is stepped animation, longer is a real hold
HARD_END_FRAC = 0.3      # end speed / peak speed at or above this = hard stop
SETTLE_TOL = 0.01        # settled = within 1 % of the move's delta
MIN_FEATURES = 20        # fewer trackable features = the camera is not measured (identity)
CAM_SPREAD = 0.4         # ... or features that span less than this share of the frame (one object)
CHANGE_PX = 8            # gray levels, as in measure_video
REPEAT_FRAC = 0.0005     # fewer changed pixels than this = a repeated frame (measure_video)
STAGGER_GAP_S = 0.6      # entrances closer than this belong to one stagger group
STAGGER_MIN = 3          # ... of at least this many elements
# smallest |delta| that counts as a move, per property
MIN_DELTA = {"scale": 0.004, "x": 0.003, "y": 0.003, "rotation": 0.25}
EL_MIN_DELTA = {"x": 0.004, "y": 0.004, "scale": 0.03, "opacity": 0.08}
CAM_PROPS = ("scale", "x", "y", "rotation")
EL_PROPS = ("x", "y", "scale", "opacity")

# Named eases as their standard cubic-bezier equivalents (the CSS / easings.net tables that GSAP's
# power eases are usually quoted with: power1 = quad, power2 = cubic, power3 = quart,
# power4 = quint; expo.out = easeOutExpo, back.out = easeOutBack, sine.inOut = easeInOutSine).
NAMED = {
    "linear": (0.0, 0.0, 1.0, 1.0),
    "power1.in": (0.55, 0.085, 0.68, 0.53),
    "power1.out": (0.25, 0.46, 0.45, 0.94),
    "power1.inOut": (0.455, 0.03, 0.515, 0.955),
    "power2.in": (0.55, 0.055, 0.675, 0.19),
    "power2.out": (0.215, 0.61, 0.355, 1.0),
    "power2.inOut": (0.645, 0.045, 0.355, 1.0),
    "power3.in": (0.895, 0.03, 0.685, 0.22),
    "power3.out": (0.165, 0.84, 0.44, 1.0),
    "power3.inOut": (0.77, 0.0, 0.175, 1.0),
    "power4.in": (0.755, 0.05, 0.855, 0.06),
    "power4.out": (0.23, 1.0, 0.32, 1.0),
    "power4.inOut": (0.86, 0.0, 0.07, 1.0),
    "expo.out": (0.19, 1.0, 0.22, 1.0),
    "back.out": (0.175, 0.885, 0.32, 1.275),
    "sine.inOut": (0.445, 0.05, 0.55, 0.95),
}


# ---------------------------------------------------------------- optional deps

def _cv2():
    try:
        import cv2
        return cv2
    except ImportError:
        return None


def _ndimage():
    try:
        from scipy import ndimage
        return ndimage
    except ImportError:
        return None


# ---------------------------------------------------------------- cubic bezier

def _bx(t, x1, x2):
    u = 1.0 - t
    return 3 * u * u * t * x1 + 3 * u * t * t * x2 + t ** 3


def _dbx(t, x1, x2):
    u = 1.0 - t
    return 3 * u * u * x1 + 6 * u * t * (x2 - x1) + 3 * t * t * (1 - x2)


def _solve_t(x, x1, x2):
    """t in [0, 1] with bx(t) = x, broadcast over x, x1, x2 (x1, x2 in [0, 1]: bx is monotonic)."""
    x, x1, x2 = np.broadcast_arrays(np.asarray(x, float), np.asarray(x1, float), np.asarray(x2, float))
    t = x.copy()
    for _ in range(8):
        f = _bx(t, x1, x2) - x
        if np.abs(f).max() < 1e-9:
            break
        d = _dbx(t, x1, x2)
        ok = np.abs(d) > 1e-6
        t = np.clip(t - np.where(ok, f / np.where(ok, d, 1.0), 0.0), 0.0, 1.0)
    bad = np.abs(_bx(t, x1, x2) - x) > 1e-7
    if bad.any():  # bisection where Newton did not converge (flat derivative)
        xb, a, b = x[bad], x1[bad], x2[bad]
        lo, hi = np.zeros_like(xb), np.ones_like(xb)
        for _ in range(48):
            mid = (lo + hi) / 2
            below = _bx(mid, a, b) < xb
            lo, hi = np.where(below, mid, lo), np.where(below, hi, mid)
        t[bad] = (lo + hi) / 2
    return t


def bezier_y(x, bezier):
    """y of the cubic-bezier easing [x1, y1, x2, y2] at time(s) x in [0, 1]."""
    x1, y1, x2, y2 = (float(v) for v in bezier)
    x1, x2 = min(1.0, max(0.0, x1)), min(1.0, max(0.0, x2))
    t = _solve_t(np.clip(np.asarray(x, float), 0, 1), x1, x2)
    u = 1.0 - t
    return 3 * u * u * t * y1 + 3 * u * t * t * y2 + t ** 3


def _nelder_mead(f, x0, step, iters=400, ftol=1e-7, xtol=2e-4):
    """Minimise f from x0 (numpy only): reflection 1, expansion 2, contraction 0.5, shrink 0.5."""
    dim = len(x0)
    pts = [np.asarray(x0, float)]
    for i in range(dim):
        p = pts[0].copy()
        p[i] += step[i]
        pts.append(p)
    pts = np.array(pts)
    vals = np.array([f(p) for p in pts])
    for _ in range(iters):
        order = np.argsort(vals)
        pts, vals = pts[order], vals[order]
        if vals[-1] - vals[0] < ftol and np.abs(pts[1:] - pts[0]).max() < xtol:
            break
        c = pts[:-1].mean(axis=0)
        xr = c + (c - pts[-1])
        fr = f(xr)
        if fr < vals[0]:
            xe = c + 2 * (c - pts[-1])
            fe = f(xe)
            pts[-1], vals[-1] = (xe, fe) if fe < fr else (xr, fr)
        elif fr < vals[-2]:
            pts[-1], vals[-1] = xr, fr
        else:
            xc = c + 0.5 * (pts[-1] - c) if fr >= vals[-1] else c + 0.5 * (xr - c)
            fc = f(xc)
            if fc < min(fr, vals[-1]):
                pts[-1], vals[-1] = xc, fc
            else:
                pts[1:] = pts[0] + 0.5 * (pts[1:] - pts[0])
                vals[1:] = [f(p) for p in pts[1:]]
    i = int(np.argmin(vals))
    return pts[i], float(vals[i])


def _round_bezier(b):
    out = []
    for v in b:
        r = round(float(v), 3)
        out.append(0.0 if r == 0 else r)
    return out


def fit_bezier(progress, times=None):
    """(bezier [x1, y1, x2, y2] rounded to 3 decimals, rmse) of a progress curve.

    progress: values 0..1 (overshoot allowed) at times 0..1 (uniform when times is None)."""
    p = np.asarray(progress, float)
    n = len(p)
    x = np.linspace(0.0, 1.0, n) if times is None else np.asarray(times, float)
    if n < 3:
        b = [0.0, 0.0, 1.0, 1.0]
        return b, float(np.sqrt(np.mean((bezier_y(x, b) - p) ** 2))) if n else 0.0
    xs, ps = x, p
    if n > 60:  # the grid stage uses a subsample, Nelder-Mead all points
        idx = np.unique(np.linspace(0, n - 1, 60).round().astype(int))
        xs, ps = x[idx], p[idx]
    # Coarse grid. For a given (x1, x2) the curve is linear in (y1, y2):
    # y = A*y1 + B*y2 + C with A = 3u^2 t, B = 3u t^2, C = t^3, so t is solved once per (x1, x2).
    gx = np.linspace(0.0, 1.0, 11)
    gy = np.linspace(-0.5, 1.5, 11)
    X1, X2 = (g.ravel() for g in np.meshgrid(gx, gx, indexing="ij"))
    t = _solve_t(xs[None, :], X1[:, None], X2[:, None])
    u = 1.0 - t
    A, B, C = 3 * u * u * t, 3 * u * t * t, t ** 3
    Y1, Y2 = (g.ravel() for g in np.meshgrid(gy, gy, indexing="ij"))
    sse = (((A[:, None, :] * Y1[None, :, None] + B[:, None, :] * Y2[None, :, None] + C[:, None, :])
            - ps[None, None, :]) ** 2).sum(axis=2)  # (x cells, y cells)
    order = np.argsort(sse.ravel())

    def solve_y(x1, x2):
        """(y1, y2, rmse): least squares for y1, y2 at this (x1, x2), y kept in [-1, 2]."""
        tt = _solve_t(x, x1, x2)
        uu = 1.0 - tt
        a, bb, r = 3 * uu * uu * tt, 3 * uu * tt * tt, p - tt ** 3
        aa, ab, b2 = float(a @ a), float(a @ bb), float(bb @ bb)
        det = aa * b2 - ab * ab
        if abs(det) < 1e-14:
            y1 = y2 = float(r @ (a + bb)) / max(1e-14, float((a + bb) @ (a + bb)))
        else:
            ar, br = float(a @ r), float(bb @ r)
            y1, y2 = (b2 * ar - ab * br) / det, (aa * br - ab * ar) / det
        y1, y2 = min(2.0, max(-1.0, y1)), min(2.0, max(-1.0, y2))
        return y1, y2, float(np.sqrt(np.mean((a * y1 + bb * y2 - r) ** 2)))

    def obj(q):
        x1, x2 = min(1.0, max(0.0, q[0])), min(1.0, max(0.0, q[1]))
        return solve_y(x1, x2)[2] + abs(x1 - q[0]) + abs(x2 - q[1])

    best_q, best_f, seen = None, math.inf, set()
    for k in order[:40]:  # Nelder-Mead from the best cells (distinct x cells, at most 3)
        xi = int(k // sse.shape[1])
        if xi in seen:
            continue
        seen.add(xi)
        q, fq = _nelder_mead(obj, [X1[xi], X2[xi]], step=[0.07, 0.07])
        if fq < best_f:
            best_q, best_f = q, fq
        if best_f < 0.002 or len(seen) >= 3:
            break
    x1, x2 = min(1.0, max(0.0, best_q[0])), min(1.0, max(0.0, best_q[1]))
    y1, y2, _ = solve_y(x1, x2)
    b = _round_bezier([x1, y1, x2, y2])
    rmse = float(np.sqrt(np.mean((bezier_y(x, b) - p) ** 2)))
    return b, rmse


_REF_X = np.linspace(0.0, 1.0, 101)
_REF_Y = {}


def curve_distance(a, b):
    """RMS difference of two bezier easings over time 0..1."""
    return float(np.sqrt(np.mean((bezier_y(_REF_X, a) - bezier_y(_REF_X, b)) ** 2)))


def nearest_named(bezier):
    """The named ease (NAMED) whose curve is nearest to this bezier."""
    y = bezier_y(_REF_X, bezier)
    best, bd = "linear", math.inf
    for name, ref in NAMED.items():
        if name not in _REF_Y:
            _REF_Y[name] = bezier_y(_REF_X, ref)
        d = float(np.sqrt(np.mean((y - _REF_Y[name]) ** 2)))
        if d < bd:
            best, bd = name, d
    return best


def _fmt(v):
    r = round(float(v), 3)
    if r == 0:
        r = 0.0
    return f"{r:.3f}".rstrip("0").rstrip(".")


def custom_ease(bezier):
    """GSAP CustomEase path "M0,0 C{x1},{y1} {x2},{y2} 1,1", numbers rounded to 3 decimals."""
    x1, y1, x2, y2 = bezier
    return f"M0,0 C{_fmt(x1)},{_fmt(y1)} {_fmt(x2)},{_fmt(y2)} 1,1"


def mirror_bezier(b):
    """The time-reversed easing (an ease-out becomes the matching ease-in)."""
    x1, y1, x2, y2 = b
    return [round(1 - x2, 3), round(1 - y2, 3), round(1 - x1, 3), round(1 - y1, 3)]


# ---------------------------------------------------------------- samples, moves

def _repeat_mask(v, repeat=None):
    """Frame i repeats frame i-1: the value did not change (or the caller says so)."""
    v = np.asarray(v, float)
    n = len(v)
    rng = float(v.max() - v.min()) if n else 0.0
    same = np.zeros(n, bool)
    if n > 1:
        same[1:] = np.abs(np.diff(v)) <= 1e-4 * rng + 1e-12
    if repeat is not None:
        same = same | np.asarray(repeat, bool)[:n]
    same[0] = False
    return same


def _runs(same):
    """[(i, j)] inclusive runs of frames that repeat the run's first frame."""
    n, out, i = len(same), [], 0
    while i < n:
        j = i
        while j + 1 < n and same[j + 1]:
            j += 1
        out.append((i, j))
        i = j + 1
    return out


def animation_step(values, repeat=None, max_hold=MAX_STEP_HOLD):
    """Frames per drawn picture while the series moves: 1 = every frame, 2 = on twos, ..."""
    runs = _runs(_repeat_mask(values, repeat))
    lens = [j - i + 1 for i, j in runs[1:-1] if j - i + 1 <= max_hold]
    if not lens:
        return 1
    vals, counts = np.unique(lens, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def sample_points(values, repeat=None, max_hold=MAX_STEP_HOLD, _windows=False):
    """(times, indices) of the real samples (see the module docstring, "unique frames").

    The first frame of every run is a sample. A run longer than max_hold is a real hold, so its
    end is a sample too. With stepped animation (k frames per picture) the motion after a hold
    starts somewhere inside the hold's last picture and the next picture's start, so that
    sample's time is the middle of that window: run end - (k - 1) + k / 2."""
    v = np.asarray(values, float)
    n = len(v)
    if n == 0:
        return np.zeros(0), np.zeros(0, int)
    runs = _runs(_repeat_mask(v, repeat))
    k = animation_step(v, repeat, max_hold)
    pts, win = {}, {}
    for r, (i, j) in enumerate(runs):
        pts[i] = float(i)
        if j - i + 1 > max_hold:
            if r == len(runs) - 1 or k == 1:
                pts[j] = float(j)
            else:
                f = max(i + 1, j - (k - 1))
                pts[f] = min(float(j), f + k / 2.0)
                win[f] = (float(f), float(min(j + 1, f + k)))
    idx = np.array(sorted(pts), int)
    times = np.array([pts[i] for i in idx])
    if _windows:  # (times, idx, step, {idx: (lo, hi) window of a hold-end sample})
        return times, idx, k, win
    return times, idx


def true_samples(values, repeat=None, max_hold=MAX_STEP_HOLD):
    """Frame indices of the real samples (sample_points without the times)."""
    return sample_points(values, repeat, max_hold)[1]


def resample_unique(values, repeat=None):
    """The series with held frames replaced by interpolation between the real samples."""
    v = np.asarray(values, float)
    if len(v) < 2:
        return v.copy()
    t, u = sample_points(v, repeat)
    return np.interp(np.arange(len(v)), t, v[u])


def speed_series(values, fps, repeat=None):
    """|dv/dt| per frame step (units per second), on unique frames; len n-1."""
    v = resample_unique(values, repeat)
    return np.abs(np.diff(v)) * float(fps) if len(v) > 1 else np.zeros(0)


def segment_moves(values, fps, min_frames=MIN_FRAMES, repeat=None, min_delta=0.0):
    """[(i0, i1)] frame ranges (inclusive) of the moves in a value series."""
    v = resample_unique(values, repeat)
    n = len(v)
    if n < 2:
        return []
    s = np.abs(np.diff(v))
    peak = float(s.max())
    if peak <= 0:
        return []
    above = s > MOVE_SPEED_FRAC * peak
    runs, k = [], 0
    while k < len(s):
        if above[k]:
            j = k
            while j + 1 < len(s) and above[j + 1]:
                j += 1
            runs.append([k, j])
            k = j + 1
        else:
            k += 1
    moves = []
    for a, b in runs:
        if b - a + 1 < min_frames:
            continue
        rp = float(s[a:b + 1].max())
        while a > 0 and s[a - 1] > EXTEND_FRAC * rp and s[a - 1] <= s[a] * 1.1 + 1e-15:
            a -= 1
        while b + 1 < len(s) and s[b + 1] > EXTEND_FRAC * rp and s[b + 1] <= s[b] * 1.1 + 1e-15:
            b += 1
        i0, i1 = a, b + 1
        if moves and i0 <= moves[-1][1]:
            moves[-1][1] = max(moves[-1][1], i1)
        else:
            moves.append([i0, i1])
    # an overshoot's way back belongs to the move it settles
    merged = []
    for m in moves:
        if merged:
            p = merged[-1]
            d_prev, d_cur = v[p[1]] - v[p[0]], v[m[1]] - v[m[0]]
            if m[0] - p[1] <= 2 and d_prev * d_cur < 0 and abs(d_cur) < 0.5 * abs(d_prev):
                p[1] = m[1]
                continue
        merged.append(m)
    out = []
    for k, (i0, i1) in enumerate(merged):
        d = v[i1] - v[i0]
        nxt = merged[k + 1][0] if k + 1 < len(merged) else n - 1
        j = i1
        while j < nxt and (v[j + 1] - v[j]) * d < 0 and abs(v[j + 1] - v[j]) > 1e-4 * abs(d):
            j += 1
        back = abs(v[j] - v[i1])
        if j - i1 >= 2 and 0.01 * abs(d) <= back < 0.5 * abs(d):
            i1 = j
        if abs(v[i1] - v[i0]) >= min_delta and i1 - i0 >= min_frames:
            out.append((int(i0), int(i1)))
    return out


def _empty_move():
    b = [0.0, 0.0, 1.0, 1.0]
    return {"dur_f": 0, "dur_s": 0.0, "delta": 0.0, "peak_speed": 0.0, "peak_at": 0.0, "overshoot": 0.0,
            "settle_f": 0, "end": "hard", "bezier": b, "custom_ease": custom_ease(b), "rmse": 0.0,
            "ease_name": "linear"}


def _move_from_samples(fu, vu, fps):
    """Move dict from real sample times fu (frames) and values vu, or None when degenerate."""
    span = float(fu[-1] - fu[0])
    if span <= 0:
        return None
    v0, v1 = vu[0], vu[-1]
    delta = v1 - v0
    if abs(delta) < 1e-12:  # out and back: measure against the largest excursion
        delta = vu[int(np.argmax(np.abs(vu - v0)))] - v0
        if abs(delta) < 1e-12:
            return None
    prog = (vu - v0) / delta
    tn = (fu - fu[0]) / span
    sp = np.abs(np.diff(vu)) / np.diff(fu) * float(fps)
    mids = (tn[:-1] + tn[1:]) / 2
    peak = float(sp.max())
    peak_at = float(mids[sp >= 0.95 * peak].mean()) if peak > 0 else 0.0
    tail = sp[mids >= 0.9]
    end_speed = float(tail.mean()) if len(tail) else float(sp[-1])
    end = "hard" if peak > 0 and end_speed >= HARD_END_FRAC * peak else "soft"
    pk = int(np.argmax(prog))
    overshoot = max(0.0, float(prog[pk]) - 1.0)
    settle_f = 0
    if overshoot > SETTLE_TOL:
        off = np.nonzero(np.abs(prog - 1.0) > SETTLE_TOL)[0]
        last = int(off[-1]) if len(off) else pk
        settle_end = fu[min(last + 1, len(fu) - 1)]
        settle_f = max(1, int(round(settle_end - fu[pk])))
    bez, rmse = fit_bezier(prog, tn)
    return {"dur_f": int(round(span)), "dur_s": round(span / float(fps), 4), "delta": round(float(delta), 5),
            "peak_speed": round(peak, 5), "peak_at": round(peak_at, 4), "overshoot": round(overshoot, 4),
            "settle_f": int(settle_f), "end": end, "bezier": bez, "custom_ease": custom_ease(bez),
            "rmse": round(rmse, 5), "ease_name": nearest_named(bez)}


def move_slice(values, i0, i1, pad=MAX_STEP_HOLD + 2):
    """(a, b): the move's frame range widened by up to `pad` still frames on each side, so
    analyse_move sees the holds around a stepped move (where its true start and end can lie)."""
    v = np.asarray(values, float)
    a, b = int(i0), int(i1)
    while a > 0 and i0 - a < pad and v[a - 1] == v[i0]:
        a -= 1
    while b + 1 < len(v) and b - i1 < pad and v[b + 1] == v[i1]:
        b += 1
    return a, b


def analyse_move(values, fps, repeat=None):
    """Move dict of one move's values (one property, the move's frames; holds at either end and
    repeated frames inside are handled). See the module docstring.

    Stepped animation (k frames per picture) leaves the true start and end uncertain by up to
    one picture: the start lies in the last still picture's window, the end after the picture
    before the final one. Both are tried at quarter-picture offsets and the times that give the
    smoothest bezier fit (lowest rmse) are kept."""
    v = np.asarray(values, float)
    if len(v) < 2:
        return _empty_move()
    fu, u, k, win = sample_points(v, repeat, _windows=True)
    lo_hi = None
    while len(u) > 2 and v[u[1]] == v[u[0]]:
        u, fu = u[1:], fu[1:]
    if int(u[0]) in win:
        lo_hi = win[int(u[0])]
    while len(u) > 2 and v[u[-2]] == v[u[-1]]:
        u, fu = u[:-1], fu[:-1]
    vu = v[u]
    best = _move_from_samples(fu, vu, fps)
    if best is None:
        return _empty_move()
    if k > 1 and len(fu) >= 4:
        starts = [lo_hi[0] + q * (lo_hi[1] - lo_hi[0]) for q in (0, 0.25, 0.5, 0.75)] if lo_hi else [fu[0]]
        ends = [fu[-1] - q * k for q in (0, 0.25, 0.5, 0.75) if fu[-1] - q * k > fu[-2]]
        cur = fu.copy()
        for pos, cands in ((0, starts), (-1, ends)):
            for c in cands:
                if c == cur[pos] or not (pos == 0 and c < cur[1] or pos == -1 and c > cur[-2]):
                    continue
                trial = cur.copy()
                trial[pos] = c
                m = _move_from_samples(trial, vu, fps)
                if m is not None and m["rmse"] < best["rmse"] - 1e-6:
                    best, cur = m, trial
    return best


def stagger(onsets, frame_s=0.0):
    """{"intervals", "even", "trend"} of a group's onset times (seconds). Even: every interval
    within 15 % of the mean (or one frame); trend: intervals shrinking = "accel"."""
    o = sorted(float(t) for t in onsets)
    iv = np.diff(o) if len(o) > 1 else np.zeros(0)
    out = {"intervals": [round(float(x), 4) for x in iv], "even": True, "trend": "flat"}
    if len(iv) < 2:
        return out
    mean = float(iv.mean())
    tol = max(0.15 * mean, float(frame_s))
    out["even"] = bool(np.abs(iv - mean).max() <= tol)
    slope = float(np.polyfit(np.arange(len(iv)), iv, 1)[0])
    rel = slope / mean if mean > 0 else 0.0
    out["trend"] = "accel" if rel < -0.1 else "decel" if rel > 0.1 else "flat"
    return out


def _series_moves(series, fps, repeat=None, min_delta=None, props=EL_PROPS):
    """[(prop, i0, i1, Move)] for every move of every property, in time order."""
    md = min_delta or EL_MIN_DELTA
    out = []
    for prop in props:
        vals = series.get(prop)
        if vals is None or len(vals) < 2:
            continue
        vals = np.asarray(vals, float)
        for i0, i1 in segment_moves(vals, fps, repeat=repeat, min_delta=md.get(prop, 0.0)):
            a, b = move_slice(vals, i0, i1)
            rep = None if repeat is None else np.asarray(repeat, bool)[a:b + 1]
            out.append((prop, i0, i1, analyse_move(vals[a:b + 1], fps, rep)))
    return sorted(out, key=lambda m: (m[1], m[0]))


def entrance_exit(track, fps=None, moves=None, edge_f=3):
    """{"in": Move|None, "hold_s", "out": Move|None, "mirrored"} of an element track.

    in: the move that starts within edge_f frames of the track's first frame (the longest if
    several props animate together); out: another move that ends within edge_f frames of the
    track's last frame. mirrored: both exist, durations within 25 %, and the exit curve is the
    time-reversed entrance curve (curve distance <= 0.05)."""
    fps = float(fps or track.get("fps") or 25.0)
    series = track.get("series", {})
    n = max((len(v) for v in series.values()), default=0)
    if moves is None:
        moves = _series_moves(series, fps, track.get("repeat"))
    first = [m for m in moves if m[1] <= edge_f]
    last = [m for m in moves if m[2] >= n - 1 - edge_f]
    m_in = max(first, key=lambda m: m[2] - m[1]) if first else None
    last = [m for m in last if m is not m_in and (m_in is None or m[1] >= m_in[2] - 1)]
    m_out = max(last, key=lambda m: m[2] - m[1]) if last else None
    a = m_in[2] if m_in else 0
    b = m_out[1] if m_out else max(0, n - 1)
    hold_s = round(max(0, b - a) / fps, 4)
    mirrored = False
    if m_in and m_out:
        di, do = m_in[3]["dur_f"], m_out[3]["dur_f"]
        ok_dur = di > 0 and do > 0 and 0.75 <= do / di <= 1.0 / 0.75
        mirrored = bool(ok_dur and curve_distance(mirror_bezier(m_in[3]["bezier"]), m_out[3]["bezier"]) <= 0.05)

    def pub(m):
        return None if m is None else dict(m[3], prop=m[0])
    return {"in": pub(m_in), "hold_s": hold_s, "out": pub(m_out), "mirrored": mirrored}


# ---------------------------------------------------------------- camera

def _decompose(M, w, h):
    a, b = float(M[0, 0]), float(M[1, 0])
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    nx = M[0, 0] * cx + M[0, 1] * cy + M[0, 2]
    ny = M[1, 0] * cx + M[1, 1] * cy + M[1, 2]
    return [math.hypot(a, b), (nx - cx) / w, (ny - cy) / h, math.degrees(math.atan2(b, a))]


def camera_matrix(row, w, h):
    """The 2x3 partial affine of a camera_track row (scale, tx, ty, rotation)."""
    s, tx, ty, rot = (float(v) for v in row)
    r = math.radians(rot)
    a, b = s * math.cos(r), s * math.sin(r)
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    M = np.array([[a, -b, 0.0], [b, a, 0.0]])
    M[0, 2] = cx + tx * w - (a * cx - b * cy)
    M[1, 2] = cy + ty * h - (b * cx + a * cy)
    return M


def _spread_ok(pts, w, h):
    """The tracked features cover the frame (not one small object): their box spans at least
    CAM_SPREAD of the width and of the height."""
    if len(pts) < 2:
        return False
    p = pts.reshape(-1, 2)
    return bool(np.ptp(p[:, 0]) >= CAM_SPREAD * w and np.ptp(p[:, 1]) >= CAM_SPREAD * h)


def camera_track(frames, cut_frames=()):
    """(n, 4) scale, tx, ty, rotation of each frame relative to its shot start (reset at every
    cut frame). Needs cv2; a shot without enough trackable features stays at the identity."""
    cv2 = _cv2()
    if cv2 is None:
        raise ImportError("cv2 is not installed")
    n = len(frames)
    out = np.zeros((n, 4))
    out[:, 0] = 1.0
    if n == 0:
        return out
    h, w = frames[0].shape
    starts = {0} | {int(c) for c in cut_frames if 0 <= int(c) < n}
    feat = dict(maxCorners=300, qualityLevel=0.01, minDistance=max(5, w // 48), blockSize=7)
    lk = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    ident = np.array([[1.0, 0, 0], [0, 1.0, 0]])
    M, ref_pts, cur_pts, prev = ident.copy(), None, None, None

    def detect(img):
        p = cv2.goodFeaturesToTrack(img, **feat)
        return np.zeros((0, 1, 2), np.float32) if p is None else p.astype(np.float32)

    for i in range(n):
        f = np.ascontiguousarray(frames[i])
        if i in starts:
            M = ident.copy()
            cur_pts = detect(f)
            ref_pts = cur_pts.copy()
            prev = f
            continue
        if len(cur_pts) >= MIN_FEATURES:
            nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev, f, cur_pts, None, **lk)
            back, st2, _ = cv2.calcOpticalFlowPyrLK(f, prev, nxt, None, **lk)
            fb = np.linalg.norm((back - cur_pts).reshape(-1, 2), axis=1)
            good = (st.ravel() == 1) & (st2.ravel() == 1) & (fb < 1.0)
            ref_pts, cur_pts = ref_pts[good], nxt[good]
            if len(cur_pts) >= MIN_FEATURES:
                est, inl = cv2.estimateAffinePartial2D(ref_pts, cur_pts, method=cv2.RANSAC,
                                                       ransacReprojThreshold=1.0, maxIters=2000, confidence=0.995)
                spread = inl is not None and _spread_ok(cur_pts[inl.ravel() == 1], w, h)
                if est is not None and int(inl.sum()) >= MIN_FEATURES and spread:
                    M = est
                    keep = inl.ravel() == 1
                    ref_pts, cur_pts = ref_pts[keep], cur_pts[keep]
        out[i] = _decompose(M, w, h)
        if len(cur_pts) < 2 * MIN_FEATURES:  # refill: new features, mapped back to shot-start coords
            fresh = detect(f)
            if len(fresh):
                inv = cv2.invertAffineTransform(M)
                ref_new = cv2.transform(fresh, inv).astype(np.float32)
                cur_pts = np.concatenate([cur_pts, fresh]) if len(cur_pts) else fresh
                ref_pts = np.concatenate([ref_pts, ref_new]) if len(ref_pts) else ref_new
        prev = f
    return out


def camera_moves(cam, fps, shot_ranges=None, repeat=None, t_offset=0.0):
    """Camera moves (Move + {"prop", "i0", "i1", "t0"}) of a camera_track, per shot range."""
    cam = np.asarray(cam, float)
    ranges = shot_ranges or [(0, len(cam) - 1)]
    out = []
    for f0, f1 in ranges:
        for ci, prop in enumerate(CAM_PROPS):
            vals = cam[f0:f1 + 1, ci]
            rep = None if repeat is None else np.asarray(repeat, bool)[f0:f1 + 1]
            for i0, i1 in segment_moves(vals, fps, repeat=rep, min_delta=MIN_DELTA[prop]):
                a, b = move_slice(vals, i0, i1)
                r2 = None if rep is None else rep[a:b + 1]
                m = analyse_move(vals[a:b + 1], fps, r2)
                m.update(prop=prop, i0=f0 + i0, i1=f0 + i1, t0=round(t_offset + (f0 + i0) / float(fps), 4))
                out.append(m)
    return sorted(out, key=lambda m: (m["t0"], m["prop"]))


# ---------------------------------------------------------------- elements

def _components(mask, min_area, max_n=12):
    """[(x, y, w, h, area, sub_label_mask)] of connected components (cv2, else scipy)."""
    cv2 = _cv2()
    if cv2 is not None:
        nl, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
        comps = [tuple(int(v) for v in stats[k]) + (k,) for k in range(1, nl)]
    else:
        nd = _ndimage()
        lab, nl = nd.label(mask)
        comps = []
        for k, sl in enumerate(nd.find_objects(lab), start=1):
            if sl is None:
                continue
            y, x = sl[0].start, sl[1].start
            hh, ww = sl[0].stop - y, sl[1].stop - x
            comps.append((x, y, ww, hh, int((lab[sl] == k).sum()), k))
    comps = [c for c in comps if c[4] >= min_area]
    comps.sort(key=lambda c: -c[4])
    return [(x, y, w, h, a, lab[y:y + h, x:x + w] == k) for x, y, w, h, a, k in comps[:max_n]]


def _iou(a, b):
    ax0, ay0, aw, ah = a
    bx0, by0, bw, bh = b
    ix = max(0, min(ax0 + aw, bx0 + bw) - max(ax0, bx0))
    iy = max(0, min(ay0 + ah, by0 + bh) - max(ay0, by0))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _grad(img):
    """|d/dx| + |d/dy| of an image (numpy only)."""
    gy, gx = np.gradient(np.asarray(img, float))
    return np.abs(gx) + np.abs(gy)


def _shift_or(m, r, fn):
    out = m.copy()
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if dx or dy:
                sh = np.zeros_like(m)
                ys, yd = (slice(dy, None), slice(None, -dy or None)) if dy >= 0 else (slice(None, dy), slice(-dy, None))
                xs, xd = (slice(dx, None), slice(None, -dx or None)) if dx >= 0 else (slice(None, dx), slice(-dx, None))
                sh[ys, xs] = m[yd, xd]
                out = fn(out, sh)
    return out


def _dilate(m, r):
    return _shift_or(m, r, np.logical_or)


def _erode(m, r):
    return ~_shift_or(~m, r, np.logical_or)


def element_tracks(frames, fps, cam=None, thr=14, t_offset=0.0, repeat=None):
    """Tracks of moving elements in one shot's frames (see the module docstring).

    Returns [{"id", "t0", "t1", "fps", "f0", "series": {"x", "y", "scale", "opacity"}, "repeat"}],
    f0 = the track's first frame inside the shot. cam: this shot's camera_track rows (or None)."""
    cv2, nd = _cv2(), _ndimage()
    n = len(frames)
    if n < 3 or (cv2 is None and nd is None):
        return []
    h, w = frames[0].shape
    moving_cam = cam is not None and len(cam) == n and (
        np.abs(cam[:, 0] - 1).max() > 1e-3 or np.abs(cam[:, 1:3]).max() > 5e-4 or np.abs(cam[:, 3]).max() > 0.05)

    def stab(i):
        f = frames[i]
        if not moving_cam or cv2 is None:
            return f, None
        inv = cv2.invertAffineTransform(camera_matrix(cam[i], w, h))
        g = cv2.warpAffine(f, inv, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        ok = cv2.warpAffine(np.full((h, w), 255, np.uint8), inv, (w, h), flags=cv2.INTER_NEAREST) > 0
        return g, ok

    idx = np.unique(np.linspace(0, n - 1, min(n, 61)).round().astype(int))
    bg = np.median(np.stack([stab(i)[0] for i in idx]), axis=0).astype(np.int16)
    min_area = max(12, int(0.0008 * h * w))
    k = max(3, w // 64)
    kern = np.ones((k, k), np.uint8)

    def fg(g, ok):
        d = np.abs(g.astype(np.int16) - bg)
        if ok is not None:
            d[~ok] = 0
        m = d > thr
        if cv2 is not None:
            m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, kern) > 0
            m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8)) > 0
        else:
            m = nd.binary_opening(nd.binary_closing(m, structure=kern), structure=np.ones((2, 2)))
        return d, m

    # Ghosts: an element that rests for most of the shot is part of the median, so where it is
    # absent the median shows a "ghost" of it. A ghost has edges in the background but none in the
    # frame along its outline; there the frame shows the real background, which patches bg.
    # Pass 1 (sampled frames) only repairs bg; pass 2 tracks.
    g_bg = _grad(bg)
    for i in np.unique(np.linspace(0, n - 1, min(n, 120)).round().astype(int)):
        g, ok = stab(i)
        _d, m = fg(g, ok)
        g_cur = None
        for x, y, bw, bh, _a, sub in _components(m, min_area, max_n=24):
            x0, y0 = max(0, x - 2), max(0, y - 2)
            x1, y1 = min(w, x + bw + 2), min(h, y + bh + 2)
            reg = np.zeros((y1 - y0, x1 - x0), bool)
            reg[y - y0:y - y0 + bh, x - x0:x - x0 + bw] = sub
            ring = _dilate(reg, 2) & ~_erode(reg, 2)
            if g_cur is None:
                g_cur = _grad(g)
            eb = float(g_bg[y0:y1, x0:x1][ring].mean()) if ring.any() else 0.0
            ec = float(g_cur[y0:y1, x0:x1][ring].mean()) if ring.any() else 0.0
            if eb > 4.0 and ec < 0.5 * eb:
                patch = _dilate(reg, 2)
                if ok is not None:
                    patch &= ok[y0:y1, x0:x1]
                bg[y0:y1, x0:x1][patch] = g[y0:y1, x0:x1][patch]
                g_bg = _grad(bg)

    active, done, next_id = [], [], 0
    for i in range(n):
        g, ok = stab(i)
        d, m = fg(g, ok)
        obs = []
        for x, y, bw, bh, area, sub in _components(m, min_area):
            wts = np.maximum(0.0, d[y:y + bh, x:x + bw].astype(float) - thr / 2.0) * sub
            sw = float(wts.sum())
            if sw <= 0:
                continue
            yy, xx = np.mgrid[y:y + bh, x:x + bw]
            cx, cy = float((wts * xx).sum() / sw), float((wts * yy).sum() / sw)
            var = float((wts * ((xx - cx) ** 2 + (yy - cy) ** 2)).sum() / sw)
            spread = math.sqrt(max(var, 1e-9))
            obs.append({"box": (x, y, bw, bh), "cx": cx, "cy": cy, "spread": spread, "sum": sw, "area": area})
        matched = set()
        still = []
        for tr in active:
            best, bscore = None, 0.0
            for oi, o in enumerate(obs):
                if oi in matched:
                    continue
                iou = _iou(tr["box"], o["box"])
                if iou >= 0.3 and iou > bscore:
                    best, bscore = oi, iou
            if best is None:  # a fast move: nearest centre with a similar area
                for oi, o in enumerate(obs):
                    if oi in matched:
                        continue
                    diag = math.hypot(*tr["box"][2:]) + math.hypot(*o["box"][2:])
                    dist = math.hypot(o["cx"] - tr["cx"][-1], o["cy"] - tr["cy"][-1])
                    ratio = o["area"] / max(1, tr["area"])
                    if dist <= 0.5 * diag and 1 / 3 <= ratio <= 3 and (best is None or dist < bscore):
                        best, bscore = oi, dist
            if best is None:
                done.append(tr)
                continue
            o = obs[best]
            matched.add(best)
            tr["box"], tr["area"] = o["box"], o["area"]
            for kk in ("cx", "cy", "spread", "sum"):
                tr[kk].append(o[kk])
            still.append(tr)
        for oi, o in enumerate(obs):
            if oi not in matched:
                still.append({"id": next_id, "f0": i, "box": o["box"], "area": o["area"], "cx": [o["cx"]],
                              "cy": [o["cy"]], "spread": [o["spread"]], "sum": [o["sum"]]})
                next_id += 1
        active = still
    done.extend(active)
    out = []
    for tr in sorted(done, key=lambda t: (t["f0"], t["id"])):
        L = len(tr["cx"])
        if L < 3:
            continue
        sp = np.array(tr["spread"])
        scale = sp / max(1e-9, float(np.median(sp)))
        op_raw = np.array(tr["sum"]) / np.maximum(sp ** 2, 1e-9)
        opacity = np.clip(op_raw / max(1e-9, float(np.percentile(op_raw, 90))), 0, 1.5)
        rep = None if repeat is None else np.asarray(repeat, bool)[tr["f0"]:tr["f0"] + L]
        out.append({"id": f"el{len(out)}", "t0": round(t_offset + tr["f0"] / float(fps), 4),
                    "t1": round(t_offset + (tr["f0"] + L - 1) / float(fps), 4), "fps": float(fps), "f0": tr["f0"],
                    "series": {"x": np.array(tr["cx"]) / w, "y": np.array(tr["cy"]) / h, "scale": scale,
                               "opacity": opacity}, "repeat": rep})
    return out


# ---------------------------------------------------------------- the whole video

def _frame_repeats(frames):
    n = len(frames)
    rep = np.zeros(n, bool)
    for i in range(1, n):
        ch = (np.abs(frames[i].astype(np.int16) - frames[i - 1].astype(np.int16)) > CHANGE_PX).mean()
        rep[i] = ch < REPEAT_FRAC
    return rep


def shot_ranges(cuts, n, fps):
    """[(f0, f1)] inclusive frame ranges between cuts; frames inside a gradual transition skipped."""
    starts, ends = [0], []
    for c in sorted(cuts, key=lambda c: c["t"]):
        f = int(round(float(c["t"]) * fps))
        ends.append(f - 1)
        starts.append(f + (int(c.get("dur_f") or 1) if c.get("kind") == "gradual" else 0))
    ends.append(n - 1)
    return [(a, b) for a, b in zip(starts, ends) if 0 <= a <= b < n]


def _stats(vals):
    if not len(vals):
        return {"median": None, "p10": None, "p90": None, "n": 0}
    a = np.asarray(vals, float)
    return {"median": round(float(np.median(a)), 4), "p10": round(float(np.percentile(a, 10)), 4),
            "p90": round(float(np.percentile(a, 90)), 4), "n": int(len(a))}


def _graph(vals, nd=5):
    return [round(float(x), nd) for x in vals]


def track(video, measure=None, *, analysis_width=320):
    """Measure the motion of one video. Returns the motion/1 document (not written)."""
    import measure_video as mv
    video = Path(video)
    sha = mv.sha256_of(video)
    frames, fps = mv.decode_gray(video, analysis_width)
    n = len(frames)
    not_run = []
    if measure is not None:
        cuts = measure.get("cuts") or []
        cad = measure.get("cadence")
    else:
        cuts = mv.detect_cuts(frames, fps) if n >= 2 else []
        cad = None
    if cad is None:
        cad = mv.cadence(frames, fps)
    shots = shot_ranges(cuts, n, fps)
    repeat = _frame_repeats(frames)
    cv2 = _cv2()
    if cv2 is None:
        not_run.append("camera")
    if cv2 is None and _ndimage() is None:
        not_run.append("elements")

    cam_out, elements, speed_g, value_g = [], [], {}, {}
    for si, (f0, f1) in enumerate(shots):
        fr = frames[f0:f1 + 1]  # a view, no copy
        rep = repeat[f0:f1 + 1].copy()
        rep[0] = False
        cam = camera_track(fr) if cv2 is not None else np.tile([1.0, 0, 0, 0], (len(fr), 1))
        for m in camera_moves(cam, fps, repeat=rep, t_offset=f0 / float(fps)):
            mid = f"cam{len(cam_out)}"
            ci = CAM_PROPS.index(m["prop"])
            vals = cam[m["i0"]:m["i1"] + 1, ci]
            r2 = rep[m["i0"]:m["i1"] + 1]
            value_g[mid] = _graph(vals)
            speed_g[mid] = _graph(speed_series(vals, fps, r2))
            m = {k: v for k, v in m.items() if k not in ("i0", "i1")}
            cam_out.append(dict(m, id=mid, shot=si))
        if "elements" in not_run:
            continue
        for tr in element_tracks(fr, fps, cam=cam if cv2 is not None else None, t_offset=f0 / float(fps), repeat=rep):
            L = len(tr["series"]["x"])
            moves = _series_moves(tr["series"], fps, tr["repeat"])
            pops = tr["f0"] > 0 or tr["f0"] + L - 1 < len(fr) - 1
            if not moves and not (pops and L >= max(3, int(0.25 * fps))):
                continue
            eid = f"el{len(elements)}"
            mv_out = []
            for k, (prop, i0, i1, m) in enumerate(moves):
                mid = f"{eid}.{prop}{k}"
                vals = tr["series"][prop][i0:i1 + 1]
                r2 = None if tr["repeat"] is None else tr["repeat"][i0:i1 + 1]
                value_g[mid] = _graph(vals)
                speed_g[mid] = _graph(speed_series(vals, fps, r2))
                mv_out.append(dict(m, id=mid, prop=prop, t0=round(tr["t0"] + i0 / float(fps), 4)))
            ee = entrance_exit(tr, fps, moves)
            elements.append({"id": eid, "shot": si, "t0": tr["t0"], "t1": tr["t1"], "moves": mv_out,
                             "entrance_exit": ee, "_onset": tr["t0"], "_appears": tr["f0"] > 0})

    groups = []
    by_shot = {}
    for e in elements:
        if e["_appears"]:
            by_shot.setdefault(e["shot"], []).append(e)
    for _si, els in sorted(by_shot.items()):
        els.sort(key=lambda e: e["_onset"])
        cur = []
        for e in els:
            if cur and e["_onset"] - cur[-1]["_onset"] > STAGGER_GAP_S:
                if len(cur) >= STAGGER_MIN:
                    groups.append(cur)
                cur = []
            cur.append(e)
        if len(cur) >= STAGGER_MIN:
            groups.append(cur)
    stagger_groups = []
    for g in groups:
        on = [e["_onset"] for e in g]
        stagger_groups.append(dict(stagger(on, 1.0 / fps), members=[e["id"] for e in g],
                                   onsets=[round(t, 4) for t in on]))
    for e in elements:
        e.pop("_onset")
        e.pop("_appears")

    all_moves = cam_out + [m for e in elements for m in e["moves"]]
    names = {}
    for m in all_moves:
        names[m["ease_name"]] = names.get(m["ease_name"], 0) + 1
    doc = {
        "schema": SCHEMA,
        "video": {"id": sha[:12], "sha256": sha, "fps": round(float(fps), 4), "frames": int(n),
                  "duration_s": round(n / float(fps), 4) if fps else 0.0},
        "cadence": cad,
        "shots": [{"t0": round(a / float(fps), 4), "t1": round(b / float(fps), 4)} for a, b in shots],
        "camera_moves": cam_out,
        "elements": elements,
        "stagger_groups": stagger_groups,
        "speed_graphs": speed_g,
        "value_graphs": value_g,
        "summary": {"moves": len(all_moves), "dur_s": _stats([m["dur_s"] for m in all_moves]),
                    "peak_at": _stats([m["peak_at"] for m in all_moves]),
                    "overshoot": _stats([m["overshoot"] for m in all_moves]),
                    "hard_end_share": round(sum(m["end"] == "hard" for m in all_moves) / len(all_moves), 4)
                    if all_moves else None,
                    "ease_names": dict(sorted(names.items(), key=lambda kv: -kv[1]))},
        "not_run": not_run,
    }
    if mv.sha256_of(video) != sha:
        raise RuntimeError("the source video changed while it was measured")
    problems = validate(doc)
    if problems:
        raise RuntimeError("internal: motion.json failed validation: " + "; ".join(problems))
    return doc


_MOVE_KEYS = ("dur_f", "dur_s", "delta", "peak_speed", "peak_at", "overshoot", "settle_f", "end", "bezier",
              "custom_ease", "rmse", "ease_name")


def validate(doc):
    """Problems with a motion/1 document (empty list = valid)."""
    if not isinstance(doc, dict) or doc.get("schema") != SCHEMA:
        return ["schema is not motion/1"]
    p = []
    need = {"video": dict, "camera_moves": list, "elements": list, "stagger_groups": list, "speed_graphs": dict,
            "value_graphs": dict, "not_run": list, "summary": dict}
    for k, t in need.items():
        if not isinstance(doc.get(k), t):
            p.append(f"{k} missing or wrong type")
    if p:
        return p
    moves = list(doc["camera_moves"])
    for e in doc["elements"]:
        for k in ("id", "t0", "moves", "entrance_exit"):
            if k not in e:
                p.append(f"element.{k} missing")
        moves += e.get("moves", [])
    for m in moves:
        for k in _MOVE_KEYS + ("id", "t0", "prop"):
            if k not in m:
                p.append(f"move.{k} missing")
        if m.get("end") not in ("hard", "soft"):
            p.append(f"move.end {m.get('end')!r}")
        if m.get("id") not in doc["speed_graphs"] or m.get("id") not in doc["value_graphs"]:
            p.append(f"graphs missing for {m.get('id')}")
    return p


# ---------------------------------------------------------------- CLI

def main(argv=None):
    store.print_utf8()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--measure", help="measure.json of this video (cuts, cadence); detected here when missing")
    ap.add_argument("--out", required=True, help="output file, e.g. <dir>/motion.json")
    ap.add_argument("--analysis-width", type=int, default=320)
    a = ap.parse_args(argv)
    try:
        require_ffmpeg()
    except MissingTool as e:
        print(str(e), file=sys.stderr)
        return 3
    if not Path(a.video).is_file():
        print(f"error: video not found: {a.video}", file=sys.stderr)
        return 2
    measure = None
    if a.measure:
        try:
            measure = json.loads(Path(a.measure).read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            print(f"error: cannot read --measure {a.measure}: {e}", file=sys.stderr)
            return 2
        if not isinstance(measure, dict) or measure.get("schema") != "measure/1" or not isinstance(
                measure.get("cuts"), list):
            print(f"error: --measure is not a measure/1 document with cuts: {a.measure}", file=sys.stderr)
            return 2
    try:
        doc = track(a.video, measure, analysis_width=a.analysis_width)
        store.write_json(Path(a.out), doc)
    except MissingTool as e:
        print(str(e), file=sys.stderr)
        return 3
    except (RuntimeError, ValueError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    s = doc["summary"]
    print(f"{len(doc['camera_moves'])} camera moves, {len(doc['elements'])} elements, {s['moves']} moves, "
          f"{len(doc['stagger_groups'])} stagger groups; eases {s['ease_names'] or 'none'}; "
          f"not run: {doc['not_run'] or 'none'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

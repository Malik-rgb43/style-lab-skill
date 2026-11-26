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
  - camera: LK optical flow on good features, a RANSAC partial affine (scale, rotation,
    translation) from the reference frame to each frame; the reference is the shot start. When
    tracking fails (too few features, or features on one small object only: their box spans
    < 40 % of the frame) the frame is a gap (NaN) and tracking restarts from the identity after
    it; nothing is carried across a gap. Camera moves shorter than 6 frames or with rmse above
    0.08 are jitter: dropped and counted (camera_jitter_dropped).
    tx / ty are the motion of the frame centre, in fractions of the frame; rotation in degrees.
  - elements: each frame is warped back into the reference frame (camera-stabilised), compared
    with the temporal-median background (ghosts of resting elements repaired, threshold raised
    by the local noise), and the changed regions become connected components, tracked frame to
    frame by IoU >= 0.3 (a fast move that breaks the overlap is still matched by centre distance
    and a similar area). An element lives >= 0.4 s, covers >= 0.4 % of the frame and moves
    rigidly (a similarity warp of its previous picture predicts the next one); non-rigid blobs
    (a talking person, cloth, water) are summarised under nonrigid_regions, not tracked.
    Components that move together are merged. x / y are the weighted centre in frame fractions,
    scale the weighted spread relative to the track's median, opacity the mean contrast relative
    to the track's 90th percentile.
  - noise: per series, the position noise sigma is the smaller of a second-difference MAD and
    the 20th percentile of the frame speed; the noise floor of the speed is 3 * sqrt(2) * sigma.
  - stepping, per series: a frame whose speed is under the noise floor between higher speeds
    (within 3 frames on both sides) is a held picture (on 2s, on 3s, a variable frame rate made
    constant), not a sample. The step period is the autocorrelation peak of the speed.
  - a move is a run of speed above the noise floor, split where the speed dips under 5 % of
    that run's own peak; its delta (medians of the holds on both sides) must be at least
    15 sigma and the property's minimum.
  - the fit works on the samples only: progress 0..1 over time 0..1 with the start and end time
    free (the holds around the move pin v0 and v1 as their medians), so a soft tail lost in the
    noise and the true start of a stepped move are found by the fit. Coarse grid over x1, x2 in
    [0, 1] and y1, y2 in [-0.5, 1.5], then Nelder-Mead (numpy only) over (x1, x2, start, end),
    with y1, y2 solved by least squares at each step (the curve is linear in them). The bezier
    is evaluated by solving x(t) = x with Newton's method and a bisection fallback. peak_at,
    end, overshoot and settle come from the fitted curve. Named eases are compared by curve
    distance with their standard cubic-bezier equivalents (NAMED).

Exit: 0 done, 1 error, 2 bad input (missing or corrupt video, bad measure.json), 3 ffmpeg missing.
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
NOISE_K = 3.0            # the speed noise floor is NOISE_K * sqrt(2) * sigma
MOVE_SPEED_FRAC = 0.05   # a run is split where its speed dips under this fraction of its own peak
MIN_FRAMES = 3           # a move lasts at least this many frames
SNR_DELTA = 15.0         # ... and its delta is at least this many sigma
MAX_STEP_HOLD = 3        # a held picture sits between higher speeds within this many frames
HARD_END_FRAC = 0.3      # end speed / peak speed at or above this = hard stop
SETTLE_TOL = 0.01        # settled = within 1 % of the move's delta
FIT_MAX_PTS = 90         # the fit uses at most this many samples (evenly picked)
FIT_ITERS = 160          # Nelder-Mead iterations per start
MIN_FEATURES = 20        # fewer trackable features = a camera gap
CAM_SPREAD = 0.4         # ... or features that span less than this share of the frame (one object)
CAM_MIN_F = 6            # camera moves shorter than this ...
CAM_MAX_RMSE = 0.08      # ... or with a worse fit are jitter
EL_MIN_LIFE_S = 0.4      # an element lives at least this long ...
EL_MIN_AREA = 0.004      # ... covers at least this share of the frame ...
RIGID_MAX_RES = 0.35     # ... and its similarity-warp residual (median) stays under this
CHANGE_PX = 8            # gray levels, as in measure_video
REPEAT_FRAC = 0.0005     # fewer changed pixels than this = a repeated frame (measure_video)
STAGGER_GAP_S = 0.6      # entrances closer than this belong to one stagger group
STAGGER_SPAN_S = 2.0     # a group spans at most this long
STAGGER_MAX = 12         # ... has at most this many members
STAGGER_MIN = 3          # ... and at least this many
STAGGER_AREA_RATIO = 3.0  # members of similar size
STAGGER_ANGLE = 45.0     # ... and similar entrance direction (degrees)
# smallest |delta| that counts as a move, per property
MIN_DELTA = {"scale": 0.015, "x": 0.008, "y": 0.008, "rotation": 0.5}  # visible camera moves only (1.5 % zoom, 0.8 % of the frame)
EL_MIN_DELTA = {"x": 0.004, "y": 0.004, "scale": 0.03, "opacity": 0.08}
CAM_PROPS = ("scale", "x", "y", "rotation")
EL_PROPS = ("x", "y", "scale", "opacity")


class BadInput(ValueError):
    """A corrupt video or a malformed measure.json: the CLI prints one line and exits 2."""


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


# ---------------------------------------------------------------- noise, stepping, samples

def noise_sigma(values):
    """Position noise (one frame) of a series: the smaller of the second-difference MAD and the
    20th percentile of the frame speed (both scaled to a gaussian sigma). Holds, stepped pictures
    and smooth motion each keep one of the two honest."""
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if len(v) < 4:
        return 0.0
    s = np.abs(np.diff(v))
    sb = float(np.percentile(s, 20)) / (0.2533 * math.sqrt(2))
    d2 = v[2:] - 2 * v[1:-1] + v[:-2]
    sa = 1.4826 * float(np.median(np.abs(d2 - np.median(d2)))) / math.sqrt(6)
    return min(sa, sb)


def _floor(sigma, values):
    v = np.asarray(values, float)
    rng = float(np.nanmax(v) - np.nanmin(v)) if len(v) else 0.0
    return max(NOISE_K * math.sqrt(2) * sigma, 1e-9 * rng + 1e-12)


def _runs_of(mask):
    """[(a, b)] inclusive index runs where mask is True."""
    m = np.asarray(mask, bool)
    if not len(m):
        return []
    d = np.diff(np.r_[0, m.astype(np.int8), 0])
    return list(zip(np.nonzero(d == 1)[0].tolist(), (np.nonzero(d == -1)[0] - 1).tolist()))


def step_holds(values, repeat=None, sigma=None):
    """bool per frame: the frame repeats the picture before it (a held picture inside motion).
    repeat: frames known to repeat (a whole-frame repeat) are holds too."""
    v = np.asarray(values, float)
    n = len(v)
    hold = np.zeros(n, bool)
    if n < 3:
        return hold
    sigma = noise_sigma(v) if sigma is None else sigma
    low = np.abs(np.diff(v)) <= _floor(sigma, v)
    hi = (~low).astype(np.int8)
    k = MAX_STEP_HOLD
    pad = np.r_[np.zeros(k, np.int8), hi, np.zeros(k, np.int8)]
    c = np.cumsum(np.r_[0, pad])
    i = np.arange(len(low))
    left = c[i + k] - c[i] > 0          # a higher speed within k steps before
    right = c[i + 2 * k + 1] - c[i + k + 1] > 0  # ... and within k steps after
    hold[1:] = low & left & right
    if repeat is not None:
        # a repeated whole frame is a held picture only inside motion; the repeats of a static
        # stretch before or after a move are the rest, and must stay samples of it
        rep = np.zeros(n, bool)
        r = np.asarray(repeat, bool)[:n]
        rep[:len(r)] = r
        hold[1:] |= rep[1:] & left & right
    hold[0] = False
    return hold


def animation_step(values, repeat=None):
    """Frames per drawn picture while the series moves (1 = every frame, 2 = on 2s, 3 = on 3s):
    the autocorrelation peak of the speed over the moving part, when held pictures are common."""
    v = np.asarray(values, float)
    if len(v) < 8:
        return 1
    sigma = noise_sigma(v)
    s = np.abs(np.diff(v))
    act = np.nonzero(s > _floor(sigma, v))[0]
    if len(act) < 4:
        return 1
    seg = s[act[0]:act[-1] + 1]
    hold = step_holds(v, repeat, sigma)[act[0] + 1:act[-1] + 2]
    if len(seg) < 8 or hold.mean() < 0.25:
        return 1
    x = seg - seg.mean()
    den = float((x * x).sum())
    if den <= 0:
        return 1
    ac = {lag: float((x[:-lag] * x[lag:]).sum()) / den for lag in (2, 3, 4) if lag < len(x)}
    if not ac:
        return 1
    top = max(ac.values())
    if top < 0.2:
        return 1
    return min(lag for lag, a in ac.items() if a >= 0.8 * top)


def true_samples(values, repeat=None):
    """Frame indices that are real samples (every frame but the held pictures)."""
    return np.nonzero(~step_holds(values, repeat))[0]


def resample_unique(values, repeat=None):
    """The series with held pictures replaced by interpolation between the real samples."""
    v = np.asarray(values, float)
    if len(v) < 2:
        return v.copy()
    u = true_samples(v, repeat)
    return np.interp(np.arange(len(v)), u, v[u])


def speed_series(values, fps, repeat=None):
    """|dv/dt| per frame step (units per second), on the real samples; len n-1."""
    v = resample_unique(values, repeat)
    return np.abs(np.diff(v)) * float(fps) if len(v) > 1 else np.zeros(0)


def _local_delta(v, i0, i1, lo=0, hi=None):
    """(v0, v1): medians of a few frames on each side of a move (inside [lo, hi])."""
    hi = len(v) - 1 if hi is None else hi
    w = max(2, min(6, (i1 - i0) // 2 + 1))
    a = v[max(lo, i0 - w + 1):i0 + 1]
    b = v[i1:min(hi, i1 + w - 1) + 1]
    return float(np.median(a)), float(np.median(b))


def segment_moves(values, fps, min_frames=MIN_FRAMES, repeat=None, min_delta=0.0):
    """[(i0, i1)] frame ranges (inclusive) of the moves in a value series (module docstring)."""
    v = np.asarray(values, float)
    n = len(v)
    if n < 2 or not np.isfinite(v).all():
        return []
    sigma = noise_sigma(v)
    hold = step_holds(v, repeat, sigma)
    u = np.nonzero(~hold)[0]
    vi = np.interp(np.arange(n), u, v[u])
    s = np.abs(np.diff(vi))
    act = s > _floor(sigma, v)
    if len(act) > 2:  # a single quiet step inside a move does not end it
        act[1:-1] |= act[:-2] & act[2:]
    moves = []
    for a, b in _runs_of(act):
        seg = s[a:b + 1]
        core = _runs_of(seg > MOVE_SPEED_FRAC * float(seg.max()))
        cuts = [a]
        for (_c0, c1), (d0, _d1) in zip(core, core[1:]):
            cuts.append(a + c1 + 1 + int(np.argmin(seg[c1 + 1:d0])) if d0 > c1 + 1 else a + d0)
        cuts.append(b + 1)
        for k in range(len(cuts) - 1):
            moves.append([cuts[k], cuts[k + 1]])
    merged = []  # an overshoot's way back belongs to the move it settles
    for m in moves:
        if merged:
            p = merged[-1]
            d_prev = vi[p[1]] - vi[p[0]]
            d_cur = vi[m[1]] - vi[m[0]]
            if m[0] - p[1] <= 3 and d_prev * d_cur < 0 and abs(d_cur) < 0.5 * abs(d_prev):
                p[1] = m[1]
                continue
        merged.append(m)
    out = []
    need = max(float(min_delta), SNR_DELTA * sigma)
    for k, (i0, i1) in enumerate(merged):
        lo = merged[k - 1][1] if k else 0
        hi = merged[k + 1][0] if k + 1 < len(merged) else n - 1
        v0, v1 = _local_delta(vi, i0, i1, lo, hi)
        if i1 - i0 >= min_frames and abs(v1 - v0) >= need and abs(v1 - v0) > 0:
            out.append((int(i0), int(i1)))
    return out


def move_slice(values, i0, i1, lo=0, hi=None, pad=None):
    """(a, b): the move's frame range widened into the holds on both sides (at most `pad`
    frames, default max(8, the move's length), and never past lo / hi), so analyse_move sees
    v0 and v1 and has room to find the true start and end."""
    hi = len(values) - 1 if hi is None else hi
    pad = max(8, int(i1) - int(i0)) if pad is None else pad
    return max(int(lo), int(i0) - pad), min(int(hi), int(i1) + pad)


# ---------------------------------------------------------------- one move

def _empty_move():
    b = [0.0, 0.0, 1.0, 1.0]
    return {"dur_f": 0, "dur_s": 0.0, "delta": 0.0, "peak_speed": 0.0, "peak_at": 0.0, "overshoot": 0.0,
            "settle_f": 0, "end": "hard", "bezier": b, "custom_ease": custom_ease(b), "rmse": 0.0,
            "ease_name": "linear", "step_f": 1}


def _abc(t, ts, te, x1, x2):
    u = np.clip((t - ts) / max(te - ts, 1e-6), 0.0, 1.0)
    tau = _solve_t(u, x1, x2)
    w = 1.0 - tau
    return 3 * w * w * tau, 3 * w * tau * tau, tau ** 3


def _outer_ends(t, vs, r0, r1):
    """(v0, v1): medians of the outer half of the holds before r0 and after r1 (the part far
    from the move, so a soft tail lost in the noise does not pull them), else the end samples."""
    pre = t[t <= r0]
    post = t[t >= r1]
    if len(pre) >= 2:
        before = vs[(t <= r0) & (t <= pre[0] + max(1.0, (r0 - pre[0]) / 2))]
    else:
        before = vs[:1]
    if len(post) >= 2:
        after = vs[(t >= r1) & (t >= post[-1] - max(1.0, (post[-1] - r1) / 2))]
    else:
        after = vs[-1:]
    return float(np.median(before)), float(np.median(after))


def _ly(A, B, r):
    """Least-squares (y1, y2) for A*y1 + B*y2 = r, kept in [-1, 2]."""
    aa, ab, b2 = float(A @ A), float(A @ B), float(B @ B)
    ar, br = float(A @ r), float(B @ r)
    det = aa * b2 - ab * ab
    if abs(det) < 1e-14:
        y1 = y2 = (ar + br) / max(1e-14, aa + 2 * ab + b2)
    else:
        y1, y2 = (b2 * ar - ab * br) / det, (aa * br - ab * ar) / det
    return min(2.0, max(-1.0, y1)), min(2.0, max(-1.0, y2))


def _fit_move(t, vs, ts_rng, te_rng):
    """Fit (x1, y1, x2, y2, ts, te, v0, v1) to samples (t frames, vs values); ts_rng / te_rng are
    (low, start, high). Returns a dict or None (no change in value)."""
    if len(t) > FIT_MAX_PTS:
        keep = np.unique(np.linspace(0, len(t) - 1, FIT_MAX_PTS).round().astype(int))
        t, vs = t[keep], vs[keep]
    (ts_lo, ts0, ts_hi), (te_lo, te0, te_hi) = ts_rng, te_rng
    min_len = 1.0
    v0, v1 = _outer_ends(t, vs, ts0, te0)
    d = v1 - v0
    if abs(d) < 1e-12:
        return None

    def clamp(q):
        x1, x2 = min(1.0, max(0.0, q[0])), min(1.0, max(0.0, q[1]))
        ts = min(ts_hi, max(ts_lo, q[2]))
        te = min(te_hi, max(te_lo, q[3]))
        if te - ts < min_len:
            te = ts + min_len
        return x1, x2, ts, te

    def evaluate(x1, x2, ts, te):
        A, B, C = _abc(t, ts, te, x1, x2)
        r = (vs - v0) / d - C
        y1, y2 = _ly(A, B, r)
        res = A * y1 + B * y2 - r
        return float(np.sqrt(np.mean(res * res))), y1, y2, v0, v1

    def obj(q):
        x1, x2, ts, te = clamp(q)
        e = evaluate(x1, x2, ts, te)
        pen = abs(x1 - q[0]) + abs(x2 - q[1]) + 0.01 * (abs(ts - q[2]) + abs(te - q[3]))
        return (e[0] if e else 10.0) + pen

    # coarse grid over x1, x2 (y1, y2 by least squares) at a few start / end pairs
    gx = np.linspace(0.0, 1.0, 11)
    X1, X2 = (g.ravel() for g in np.meshgrid(gx, gx, indexing="ij"))
    starts = []
    for ts in sorted({ts0, max(ts_lo, ts0 - (ts0 - ts_lo) / 2)}):
        for te in sorted({te0, min(te_hi, te0 + (te_hi - te0) / 2)}):
            if te - ts < min_len:
                continue
            u = np.clip((t - ts) / (te - ts), 0, 1)
            tau = _solve_t(u[None, :], X1[:, None], X2[:, None])
            w = 1.0 - tau
            A, B, C = 3 * w * w * tau, 3 * w * tau * tau, tau ** 3
            r = (vs - v0) / d - C
            aa, ab, b2 = (A * A).sum(1), (A * B).sum(1), (B * B).sum(1)
            ar, br = (A * r).sum(1), (B * r).sum(1)
            det = aa * b2 - ab * ab
            ok = np.abs(det) > 1e-14
            sd = np.where(ok, det, 1.0)
            y1 = np.clip(np.where(ok, (b2 * ar - ab * br) / sd, 0.0), -0.5, 1.5)
            y2 = np.clip(np.where(ok, (aa * br - ab * ar) / sd, 1.0), -0.5, 1.5)
            sse = (((A * y1[:, None] + B * y2[:, None]) - r) ** 2).sum(1)
            gi = int(np.argmin(sse))
            starts.append((float(sse[gi]), [X1[gi], X2[gi], ts, te]))
    if not starts:
        return None
    starts.sort(key=lambda z: z[0])
    span = max(1.0, te0 - ts0)
    best_q, best_f = None, math.inf
    for _sse, q0 in starts[:2]:
        q, fq = _nelder_mead(obj, q0, step=[0.07, 0.07, max(0.5, 0.1 * span), max(0.5, 0.1 * span)],
                             iters=FIT_ITERS)
        if fq < best_f:
            best_q, best_f = q, fq
    x1, x2, ts, te = clamp(best_q)
    e = evaluate(x1, x2, ts, te)
    if e is None:
        return None
    _rm, y1, y2, v0, v1 = e
    return {"bezier": _round_bezier([x1, y1, x2, y2]), "ts": ts, "te": te, "v0": v0, "v1": v1, "t": t, "vs": vs}


def _analyse(values, fps, repeat=None):
    """(Move, ts, te) of the main move in values (frames ts..te, fractional) or (empty, None, None)."""
    v = np.asarray(values, float)
    n = len(v)
    if n < 2 or not np.isfinite(v).all():
        return _empty_move(), None, None
    sigma = noise_sigma(v)
    hold = step_holds(v, repeat, sigma)
    step = animation_step(v, repeat)
    idx = np.nonzero(~hold)[0]
    if step > 1:
        # on 2s / 3s the first picture of a move is held like the others, but the frame before it
        # is static, so step_holds cannot see the hold: drop the (step-1) static frames just before
        # the first change (they show the first picture, not the rest before the move)
        moving = np.nonzero(np.abs(np.diff(v)) > _floor(sigma, v))[0]
        if len(moving):
            first = int(moving[0]) + 1
            idx = idx[(idx <= first - step) | (idx >= first)]
    t, vs = idx.astype(float), v[idx]
    rough = segment_moves(v, fps, min_frames=1, repeat=repeat)
    if rough:
        vi = np.interp(np.arange(n), idx, vs)
        r0, r1 = max(rough, key=lambda m: abs(np.subtract(*_local_delta(vi, m[0], m[1]))))
    else:
        moving = np.nonzero(np.abs(np.diff(v)) > 0)[0]
        if not len(moving):
            return _empty_move(), None, None
        r0, r1 = int(moving[0]), int(moving[-1]) + 1
    # a soft tail slows below the per-frame noise floor before it stops: extend the end while the
    # 3-frame average still approaches the end value by more than the noise over 3 frames
    vi_all = np.interp(np.arange(n), idx, vs) if len(idx) else v
    sm = np.convolve(vi_all, np.ones(3) / 3, mode="same")
    sgn = np.sign(vi_all[min(n - 1, r1)] - vi_all[r0]) or 1.0
    thr = max(2.0 * sigma, 1e-6 * float(np.ptp(v)))
    while r1 + 3 < n - 1 and min(sm[r1 + 3] - sm[r1], vi_all[r1 + 3] - vi_all[r1]) * sgn > thr:
        r1 += 1
    while r0 - 3 > 0 and min(sm[r0] - sm[r0 - 3], vi_all[r0] - vi_all[r0 - 3]) * sgn > thr:
        r0 -= 1
    dur0 = max(1.0, float(r1 - r0))
    ext = max(4.0, 1.5 * dur0)
    inner = min(2.0, dur0 / 3)
    ts_rng = (max(t[0], r0 - ext), float(r0), r0 + inner)
    te_rng = (r1 - inner, float(r1), min(t[-1], r1 + ext))
    # no hold before (after) the move: the series starts (ends) with it, so the first (last)
    # sample is the start (end); a free start there would trade duration against the ease
    fl = _floor(sigma, v)
    if r0 - t[0] < 2 or (len(vs) > 1 and abs(vs[1] - vs[0]) > fl):
        ts_rng = (float(t[0]),) * 3
    if t[-1] - r1 < 2 or (len(vs) > 1 and abs(vs[-1] - vs[-2]) > fl):
        te_rng = (float(t[-1]),) * 3
    f = _fit_move(t, vs, ts_rng, te_rng)
    if f is None:
        return _empty_move(), None, None
    bez, ts, te, v0, v1 = f["bezier"], f["ts"], f["te"], f["v0"], f["v1"]
    d = v1 - v0
    dur = te - ts
    inside = (f["t"] >= ts) & (f["t"] <= te)
    if inside.sum() < 2:
        inside = np.ones(len(f["t"]), bool)
    u = np.clip((f["t"][inside] - ts) / dur, 0, 1)
    prog = (f["vs"][inside] - v0) / d
    rmse = float(np.sqrt(np.mean((bezier_y(u, bez) - prog) ** 2)))
    uu = np.linspace(0.0, 1.0, 201)
    yy = bezier_y(uu, bez)
    sp = np.abs(np.gradient(yy, uu))
    peak = float(sp.max())
    peak_at = float(uu[sp >= 0.98 * peak].mean()) if peak > 0 else 0.0
    end = "hard" if peak > 0 and float(sp[uu >= 0.9].mean()) >= HARD_END_FRAC * peak else "soft"
    pk = int(np.argmax(yy))
    overshoot = max(0.0, float(yy[pk]) - 1.0)
    settle_f = 0
    if overshoot > SETTLE_TOL:
        off = np.nonzero(np.abs(yy - 1.0) > SETTLE_TOL)[0]
        last = int(off[-1]) if len(off) else pk
        settle_f = max(1, int(round((uu[min(last + 1, len(uu) - 1)] - uu[pk]) * dur)))
    move = {"dur_f": int(round(dur)), "dur_s": round(dur / float(fps), 4), "delta": round(d, 5),
            "peak_speed": round(abs(d) * peak / (dur / float(fps)), 5), "peak_at": round(peak_at, 4),
            "overshoot": round(overshoot, 4), "settle_f": int(settle_f), "end": end, "bezier": bez,
            "custom_ease": custom_ease(bez), "rmse": round(rmse, 5), "ease_name": nearest_named(bez),
            "step_f": int(step)}
    return move, ts, te


def analyse_move(values, fps, repeat=None):
    """Move dict of the main move in `values` (one property; best with the holds on both sides,
    see move_slice). Noise, held pictures and an uncertain start / end are handled by the fit
    (module docstring). Extra key: step_f, the frames per drawn picture."""
    return _analyse(values, fps, repeat)[0]


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
    """[(prop, i0, i1, Move, ts, te)] for every move of every property, in time order.
    i0 / i1: the segment's frames; ts / te: the fitted start and end (fractional frames)."""
    md = min_delta or EL_MIN_DELTA
    out = []
    for prop in props:
        vals = series.get(prop)
        if vals is None or len(vals) < 2:
            continue
        vals = np.asarray(vals, float)
        segs = segment_moves(vals, fps, repeat=repeat, min_delta=md.get(prop, 0.0))
        for k, (i0, i1) in enumerate(segs):
            lo = (segs[k - 1][1] + i0) // 2 if k else 0
            hi = (i1 + segs[k + 1][0]) // 2 if k + 1 < len(segs) else len(vals) - 1
            a, b = move_slice(vals, i0, i1, lo, hi)
            rep = None if repeat is None else np.asarray(repeat, bool)[a:b + 1]
            m, ts, te = _analyse(vals[a:b + 1], fps, rep)
            if ts is None:
                continue
            out.append((prop, i0, i1, m, a + ts, a + te))
    return sorted(out, key=lambda m: (m[4], m[0]))


def entrance_exit(track, fps=None, moves=None, edge_f=3):
    """{"in": Move|None, "hold_s", "out": Move|None, "mirrored"} of an element track.

    in: the move that starts within edge_f frames of the track's first frame (the longest if
    several props animate together), unless the element was already there at the cut
    (track["present_at_cut"]); out: another move that ends within edge_f frames of the track's
    last frame, unless the element is still there at the next cut (track["present_at_end"]).
    mirrored: both exist, durations within 25 %, and the exit curve is the time-reversed entrance
    curve (curve distance <= 0.05)."""
    fps = float(fps or track.get("fps") or 25.0)
    series = track.get("series", {})
    n = max((len(v) for v in series.values()), default=0)
    if moves is None:
        moves = _series_moves(series, fps, track.get("repeat"))
    first = [] if track.get("present_at_cut") else [m for m in moves if m[4] <= edge_f]
    last = [] if track.get("present_at_end") else [m for m in moves if m[5] >= n - 1 - edge_f]
    m_in = max(first, key=lambda m: m[5] - m[4]) if first else None
    last = [m for m in last if m is not m_in and (m_in is None or m[4] >= m_in[5] - 1)]
    m_out = max(last, key=lambda m: m[5] - m[4]) if last else None
    a = m_in[5] if m_in else 0
    b = m_out[4] if m_out else max(0, n - 1)
    hold_s = round(max(0.0, b - a) / fps, 4)
    mirrored = False
    if m_in and m_out:
        di, do = m_in[3]["dur_f"], m_out[3]["dur_f"]
        ok_dur = di > 0 and do > 0 and 0.75 <= do / di <= 1.0 / 0.75
        mirrored = bool(ok_dur and curve_distance(mirror_bezier(m_in[3]["bezier"]), m_out[3]["bezier"]) <= 0.05)

    def pub(m):
        return None if m is None else dict(m[3], prop=m[0])
    return {"in": pub(m_in), "hold_s": hold_s, "out": pub(m_out), "mirrored": mirrored}


def _similar_dir(a, b):
    if a is None or b is None:
        return a is None and b is None
    ang = abs(math.degrees(math.atan2(a[1], a[0]) - math.atan2(b[1], b[0]))) % 360
    return min(ang, 360 - ang) <= STAGGER_ANGLE


def stagger_groups(entries, fps):
    """Stagger groups from [{"id", "shot", "onset", "area", "dir"}] (dir: (dx, dy) of the entrance
    or None for a fade / pop): onsets less than STAGGER_GAP_S apart, a group spans at most
    STAGGER_SPAN_S and STAGGER_MAX members, of similar size and entrance direction."""
    groups = []
    for shot in sorted({e["shot"] for e in entries}):
        cur = []
        for e in sorted((e for e in entries if e["shot"] == shot), key=lambda e: e["onset"]):
            if cur:
                med_area = float(np.median([c["area"] for c in cur]))
                fits = (e["onset"] - cur[-1]["onset"] <= STAGGER_GAP_S
                        and e["onset"] - cur[0]["onset"] <= STAGGER_SPAN_S and len(cur) < STAGGER_MAX
                        and max(e["area"], med_area) <= STAGGER_AREA_RATIO * max(1e-9, min(e["area"], med_area))
                        and _similar_dir(e["dir"], cur[0]["dir"]))
                if not fits:
                    if len(cur) >= STAGGER_MIN:
                        groups.append(cur)
                    cur = []
            cur.append(e)
        if len(cur) >= STAGGER_MIN:
            groups.append(cur)
    out = []
    for g in groups:
        on = [e["onset"] for e in g]
        out.append(dict(stagger(on, 1.0 / fps), members=[e["id"] for e in g], onsets=[round(t, 4) for t in on]))
    return out


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
    """(n, 4) scale, tx, ty, rotation of each frame relative to its reference frame: the shot
    start (every cut frame), or the first good frame after a gap. Gap frames (tracking failed)
    are NaN; nothing is carried across a gap. Needs cv2."""
    cv2 = _cv2()
    if cv2 is None:
        raise ImportError("cv2 is not installed")
    n = len(frames)
    out = np.full((n, 4), np.nan)
    if n == 0:
        return out
    h, w = frames[0].shape
    starts = {0} | {int(c) for c in cut_frames if 0 <= int(c) < n}
    feat = dict(maxCorners=300, qualityLevel=0.01, minDistance=max(5, w // 48), blockSize=7)
    lk = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    ident = np.array([[1.0, 0, 0], [0, 1.0, 0]])
    tracking, M, ref_pts, cur_pts, prev = False, ident.copy(), None, None, None

    def detect(img):
        p = cv2.goodFeaturesToTrack(img, **feat)
        return np.zeros((0, 1, 2), np.float32) if p is None else p.astype(np.float32)

    def begin(img):
        p = detect(img)
        if len(p) >= MIN_FEATURES and _spread_ok(p, w, h):
            return True, p
        return False, p

    for i in range(n):
        f = np.ascontiguousarray(frames[i])
        if i in starts or not tracking:
            if i in starts or i % 3 == 0 or prev is None:  # a gap retries every 3rd frame
                ok, pts = begin(f)
                tracking = ok
                if ok:
                    M, ref_pts, cur_pts = ident.copy(), pts.copy(), pts
                    out[i] = [1.0, 0.0, 0.0, 0.0]
            prev = f
            continue
        good_m = False
        if len(cur_pts) >= MIN_FEATURES:
            nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev, f, cur_pts, None, **lk)
            back, st2, _ = cv2.calcOpticalFlowPyrLK(f, prev, nxt, None, **lk)
            fb = np.linalg.norm((back - cur_pts).reshape(-1, 2), axis=1)
            good = (st.ravel() == 1) & (st2.ravel() == 1) & (fb < 1.0)
            ref_pts, cur_pts = ref_pts[good], nxt[good]
            if len(cur_pts) >= MIN_FEATURES:
                est, inl = cv2.estimateAffinePartial2D(ref_pts, cur_pts, method=cv2.RANSAC,
                                                       ransacReprojThreshold=1.0, maxIters=2000, confidence=0.995)
                if est is not None and inl is not None and int(inl.sum()) >= MIN_FEATURES:
                    keep = inl.ravel() == 1
                    if _spread_ok(cur_pts[keep], w, h):
                        M, good_m = est, True
                        ref_pts, cur_pts = ref_pts[keep], cur_pts[keep]
        if not good_m:  # a gap: this frame is unknown, restart after it
            tracking = False
            prev = f
            continue
        out[i] = _decompose(M, w, h)
        if len(cur_pts) < 2 * MIN_FEATURES:  # refill: new features, mapped back to reference coords
            fresh = detect(f)
            if len(fresh):
                inv = cv2.invertAffineTransform(M)
                ref_new = cv2.transform(fresh, inv).astype(np.float32)
                cur_pts = np.concatenate([cur_pts, fresh]) if len(cur_pts) else fresh
                ref_pts = np.concatenate([ref_pts, ref_new]) if len(ref_pts) else ref_new
        prev = f
    return out


def camera_segments(cam):
    """[(f0, f1)] inclusive runs of measured (non-NaN) camera rows."""
    ok = np.isfinite(np.asarray(cam, float)).all(axis=1) if len(cam) else np.zeros(0, bool)
    return _runs_of(ok)


def camera_moves(cam, fps, shot_ranges=None, repeat=None, t_offset=0.0, stats=None):
    """Camera moves (Move + {"prop", "i0", "i1", "t0"}) of a camera_track, per measured segment
    of each shot range. Moves shorter than CAM_MIN_F frames or with rmse above CAM_MAX_RMSE are
    jitter: dropped and counted in stats["camera_jitter_dropped"]."""
    cam = np.asarray(cam, float)
    ranges = shot_ranges or [(0, len(cam) - 1)]
    out = []
    for f0, f1 in ranges:
        for s0, s1 in camera_segments(cam[f0:f1 + 1]):
            g0, g1 = f0 + s0, f0 + s1
            rep = None if repeat is None else np.asarray(repeat, bool)[g0:g1 + 1]
            for ci, prop in enumerate(CAM_PROPS):
                series = {prop: cam[g0:g1 + 1, ci]}
                for _p, i0, i1, m, ts, te in _series_moves(series, fps, rep, MIN_DELTA, (prop,)):
                    if m["dur_f"] < CAM_MIN_F or m["rmse"] > CAM_MAX_RMSE:
                        if stats is not None:
                            stats["camera_jitter_dropped"] = stats.get("camera_jitter_dropped", 0) + 1
                        continue
                    m.update(prop=prop, i0=g0 + int(math.floor(ts)), i1=g0 + int(math.ceil(te)),
                             t0=round(t_offset + (g0 + ts) / float(fps), 4))
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


def _rigid_residual(cv2, prev, cur):
    """How badly a similarity warp of the previous picture of an element predicts the current one
    (0 = exactly, 1 = not at all). prev / cur: dicts with the masked difference crop "dpatch", its
    box origin "box", centre "cx", "cy" and "spread"."""
    s = cur["spread"] / max(prev["spread"], 1e-9)
    px0, py0 = prev["box"][:2]
    bx0, by0, bw, bh = cur["box"]
    M = np.array([[s, 0.0, cur["cx"] - s * prev["cx"] + s * px0 - bx0],
                  [0.0, s, cur["cy"] - s * prev["cy"] + s * py0 - by0]], np.float32)
    W = cv2.warpAffine(prev["dpatch"], M, (bw, bh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    D = cur["dpatch"]
    tot = float(D.sum())
    if tot <= 0:
        return 1.0
    alpha = tot / max(float(W.sum()), 1e-9)
    return float(np.abs(D - alpha * W).sum()) / tot


def _merge_tracks(raw, w, h):
    """Merge raw tracks that move together (letters of a word, parts of one card): they share
    >= 80 % of both lifetimes, stay near each other, and their displacement and scale change
    agree within 1.5 px / 5 %."""
    n = len(raw)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    tol = max(1.5, 0.004 * w)
    for i in range(n):
        A = raw[i]
        la = len(A["cx"])
        for j in range(i + 1, n):
            B = raw[j]
            lb = len(B["cx"])
            s0, s1 = max(A["f0"], B["f0"]), min(A["f0"] + la, B["f0"] + lb)
            ov = s1 - s0
            if ov < 3 or ov < 0.8 * la or ov < 0.8 * lb:
                continue
            ia, ib = slice(s0 - A["f0"], s1 - A["f0"]), slice(s0 - B["f0"], s1 - B["f0"])
            ax, ay = np.array(A["cx"][ia]), np.array(A["cy"][ia])
            bx, by = np.array(B["cx"][ib]), np.array(B["cy"][ib])
            near = math.hypot(ax[0] - bx[0], ay[0] - by[0]) <= 2.5 * (A["spread"][s0 - A["f0"]] + B["spread"][s0 - B["f0"]])
            if not near:
                continue
            dmov = max(np.abs((ax - ax[0]) - (bx - bx[0])).mean(), np.abs((ay - ay[0]) - (by - by[0])).mean())
            sa, sb = np.log(np.array(A["spread"][ia]) + 1e-9), np.log(np.array(B["spread"][ib]) + 1e-9)
            dsc = np.abs((sa - sa[0]) - (sb - sb[0])).mean()
            if dmov <= tol and dsc <= 0.05:
                parent[find(j)] = find(i)
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(raw[i])
    out = []
    for g in groups.values():
        if len(g) == 1:
            out.append(g[0])
            continue
        s0 = max(t["f0"] for t in g)
        s1 = min(t["f0"] + len(t["cx"]) for t in g)
        m = {"f0": s0, "cx": [], "cy": [], "spread": [], "sum": [], "area": [],
             "res": [r for t in g for r in t["res"]], "boxsum": np.sum([t["boxsum"] for t in g], axis=0),
             "nbox": sum(t["nbox"] for t in g)}
        for f in range(s0, s1):
            k = [f - t["f0"] for t in g]
            ws = np.array([t["sum"][kk] for t, kk in zip(g, k)])
            cxs = np.array([t["cx"][kk] for t, kk in zip(g, k)])
            cys = np.array([t["cy"][kk] for t, kk in zip(g, k)])
            sps = np.array([t["spread"][kk] for t, kk in zip(g, k)])
            W = max(float(ws.sum()), 1e-9)
            cx, cy = float((ws * cxs).sum() / W), float((ws * cys).sum() / W)
            var = float((ws * (sps ** 2 + (cxs - cx) ** 2 + (cys - cy) ** 2)).sum() / W)
            m["cx"].append(cx)
            m["cy"].append(cy)
            m["spread"].append(math.sqrt(max(var, 1e-9)))
            m["sum"].append(W)
            m["area"].append(sum(t["area"][kk] for t, kk in zip(g, k)))
        out.append(m)
    return out


def _track_range(frames, a0, a1, cam, thr, stabilise):
    """Raw tracks of frames a0..a1 of one shot (one measured camera segment)."""
    cv2, nd = _cv2(), _ndimage()
    h, w = frames[0].shape
    n = a1 - a0 + 1

    def stab(i):
        f = frames[i]
        if not stabilise or cv2 is None:
            return f, None
        inv = cv2.invertAffineTransform(camera_matrix(cam[i], w, h))
        g = cv2.warpAffine(f, inv, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        ok = cv2.warpAffine(np.full((h, w), 255, np.uint8), inv, (w, h), flags=cv2.INTER_NEAREST) > 0
        return g, ok

    idx = np.unique(np.linspace(a0, a1, min(n, 61)).round().astype(int))
    stack = np.stack([stab(i)[0] for i in idx])
    bg = np.median(stack, axis=0).astype(np.int16)
    # the threshold rises with the local noise (grain, water, a moving face)
    noise = np.median(np.abs(stack.astype(np.int16) - bg[None]), axis=0).astype(np.float32)
    del stack
    if cv2 is not None:
        noise = cv2.blur(noise, (5, 5))
    thr_map = np.maximum(float(thr), 3.0 * noise).astype(np.int16)
    min_area = max(12, int(0.0008 * h * w))
    k = max(3, w // 64)
    kern = np.ones((k, k), np.uint8)

    def fg(g, ok):
        d = np.abs(g.astype(np.int16) - bg)
        if ok is not None:
            d[~ok] = 0
        m = d > thr_map
        if cv2 is not None:
            m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, kern) > 0
            m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8)) > 0
        else:
            m = nd.binary_opening(nd.binary_closing(m, structure=kern), structure=np.ones((2, 2)))
        return d, m

    # Ghosts: an element that rests for most of the range is part of the median, so where it is
    # absent the median shows a "ghost" of it. A ghost has edges in the background but none in the
    # frame along its outline; there the frame shows the real background, which patches bg.
    # Pass 1 (sampled frames) only repairs bg; pass 2 tracks.
    g_bg = _grad(bg)
    for i in np.unique(np.linspace(a0, a1, min(n, 60)).round().astype(int)):
        g, ok = stab(i)
        _d, m = fg(g, ok)
        g_cur = None
        for x, y, bw, bh, _a, sub in _components(m, min_area, max_n=24):
            x0, y0 = max(0, x - 2), max(0, y - 2)
            x1, y1 = min(w, x + bw + 2), min(h, y + bh + 2)
            reg = np.zeros((y1 - y0, x1 - x0), bool)
            reg[y - y0:y - y0 + bh, x - x0:x - x0 + bw] = sub
            ring = _dilate(reg, 2) & ~_erode(reg, 2)
            if not ring.any():
                continue
            if g_cur is None:
                g_cur = _grad(g)
            eb = float(g_bg[y0:y1, x0:x1][ring].mean())
            ec = float(g_cur[y0:y1, x0:x1][ring].mean())
            if eb > 4.0 and ec < 0.5 * eb:
                patch = _dilate(reg, 2)
                if ok is not None:
                    patch &= ok[y0:y1, x0:x1]
                bg[y0:y1, x0:x1][patch] = g[y0:y1, x0:x1][patch]
                g_bg[max(0, y0 - 1):y1 + 1, max(0, x0 - 1):x1 + 1] = _grad(
                    bg[max(0, y0 - 1):y1 + 1, max(0, x0 - 1):x1 + 1])

    active, done = [], []
    for i in range(a0, a1 + 1):
        g, ok = stab(i)
        d, m = fg(g, ok)
        obs = []
        for x, y, bw, bh, area, sub in _components(m, min_area):
            wts = np.maximum(0.0, (d[y:y + bh, x:x + bw] - thr_map[y:y + bh, x:x + bw] / 2.0).astype(np.float32)) * sub
            sw = float(wts.sum())
            if sw <= 0:
                continue
            yy, xx = np.mgrid[y:y + bh, x:x + bw]
            cx, cy = float((wts * xx).sum() / sw), float((wts * yy).sum() / sw)
            var = float((wts * ((xx - cx) ** 2 + (yy - cy) ** 2)).sum() / sw)
            obs.append({"box": (x, y, bw, bh), "cx": cx, "cy": cy, "spread": math.sqrt(max(var, 1e-9)),
                        "sum": sw, "area": area, "dpatch": wts})
        matched, still = set(), []
        for tr in active:
            best, bscore = None, 0.0
            for oi, o in enumerate(obs):
                if oi in matched:
                    continue
                iou = _iou(tr["last"]["box"], o["box"])
                if iou >= 0.3 and iou > bscore:
                    best, bscore = oi, iou
            if best is None:  # a fast move: nearest centre with a similar area
                for oi, o in enumerate(obs):
                    if oi in matched:
                        continue
                    diag = math.hypot(*tr["last"]["box"][2:]) + math.hypot(*o["box"][2:])
                    dist = math.hypot(o["cx"] - tr["cx"][-1], o["cy"] - tr["cy"][-1])
                    ratio = o["area"] / max(1, tr["area"][-1])
                    if dist <= 0.5 * diag and 1 / 3 <= ratio <= 3 and (best is None or dist < bscore):
                        best, bscore = oi, dist
            if best is None:
                done.append(tr)
                continue
            o = obs[best]
            matched.add(best)
            if cv2 is not None and len(tr["cx"]) % 2 == 1:
                tr["res"].append(_rigid_residual(cv2, tr["last"], o))
            _append_obs(tr, o)
            still.append(tr)
        for oi, o in enumerate(obs):
            if oi not in matched:
                tr = {"f0": i, "cx": [], "cy": [], "spread": [], "sum": [], "area": [], "res": [],
                      "boxsum": np.zeros(4), "nbox": 0}
                _append_obs(tr, o)
                still.append(tr)
        active = still
    done.extend(active)
    for tr in done:
        tr.pop("last", None)
    return done


def _append_obs(tr, o):
    for kk in ("cx", "cy", "spread", "sum", "area"):
        tr[kk].append(o[kk])
    x, y, bw, bh = o["box"]
    tr["boxsum"] = tr["boxsum"] + np.array([x, y, x + bw, y + bh], float)
    tr["nbox"] += 1
    tr["last"] = o


def element_tracks(frames, fps, cam=None, thr=14, t_offset=0.0, repeat=None, nonrigid=None):
    """Tracks of rigid moving elements in one shot's frames (see the module docstring).

    Returns [{"id", "t0", "t1", "fps", "f0", "present_at_cut", "present_at_end", "area",
    "series": {"x", "y", "scale", "opacity"}, "repeat"}]; f0 = first frame inside the shot,
    area = median share of the frame. cam: this shot's camera_track rows (NaN = gap) or None.
    nonrigid: a list that receives {"t0", "t1", "box", "area"} of blobs that live long enough
    but do not move rigidly."""
    cv2, nd = _cv2(), _ndimage()
    n = len(frames)
    if n < 3 or (cv2 is None and nd is None):
        return []
    h, w = frames[0].shape
    segs = camera_segments(cam) if cam is not None and len(cam) == n else []
    if segs:
        ranges = [(a, b, True) for a, b in segs if b - a + 1 >= 3]
    else:
        ranges = [(0, n - 1, False)]
    min_life = max(3, int(math.ceil(EL_MIN_LIFE_S * fps)))
    min_area = EL_MIN_AREA * h * w
    out = []
    for a0, a1, moving_ok in ranges:
        stabilise = False
        if moving_ok and cam is not None:
            c = cam[a0:a1 + 1]
            stabilise = bool(np.abs(c[:, 0] - 1).max() > 1e-3 or np.abs(c[:, 1:3]).max() > 5e-4
                             or np.abs(c[:, 3]).max() > 0.05)
        raw = [t for t in _track_range(frames, a0, a1, cam, thr, stabilise) if len(t["cx"]) >= 3]
        for tr in _merge_tracks(raw, w, h):
            L = len(tr["cx"])
            area = float(np.median(tr["area"]))
            if L < min_life or area < min_area:
                continue
            rigid = not tr["res"] or float(np.median(tr["res"])) <= RIGID_MAX_RES
            t0 = round(t_offset + tr["f0"] / float(fps), 4)
            t1 = round(t_offset + (tr["f0"] + L - 1) / float(fps), 4)
            if not rigid:
                if nonrigid is not None:
                    bx = tr["boxsum"] / max(1, tr["nbox"])
                    nonrigid.append({"t0": t0, "t1": t1, "area": round(area / (h * w), 4),
                                     "box": [round(bx[0] / w, 4), round(bx[1] / h, 4),
                                             round((bx[2] - bx[0]) / w, 4), round((bx[3] - bx[1]) / h, 4)]})
                continue
            sp = np.array(tr["spread"])
            scale = sp / max(1e-9, float(np.median(sp)))
            op_raw = np.array(tr["sum"]) / np.maximum(sp ** 2, 1e-9)
            opacity = np.clip(op_raw / max(1e-9, float(np.percentile(op_raw, 90))), 0, 1.5)
            rep = None if repeat is None else np.asarray(repeat, bool)[tr["f0"]:tr["f0"] + L]
            out.append({"id": f"el{len(out)}", "t0": t0, "t1": t1, "fps": float(fps), "f0": tr["f0"],
                        "present_at_cut": tr["f0"] == 0, "present_at_end": tr["f0"] + L - 1 == n - 1,
                        "area": round(area / (h * w), 5),
                        "series": {"x": np.array(tr["cx"]) / w, "y": np.array(tr["cy"]) / h, "scale": scale,
                                   "opacity": opacity}, "repeat": rep})
    out.sort(key=lambda e: (e["f0"], e["id"]))
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


def _entry_dir(tr, moves):
    """(dx, dy) of an element's entrance (frame fractions) or None for a fade / pop."""
    ins = [m for m in moves if m[0] in ("x", "y") and m[4] <= 3]
    if not ins:
        return None
    a = int(max(0, math.floor(min(m[4] for m in ins))))
    b = int(min(len(tr["series"]["x"]) - 1, math.ceil(max(m[5] for m in ins))))
    dx = float(tr["series"]["x"][b] - tr["series"]["x"][a])
    dy = float(tr["series"]["y"][b] - tr["series"]["y"][a])
    return (dx, dy) if math.hypot(dx, dy) >= 0.01 else None


def track(video, measure=None, *, analysis_width=320):
    """Measure the motion of one video. Returns the motion/1 document (not written).
    Raises BadInput for a video that cannot be read."""
    import measure_video as mv
    video = Path(video)
    sha = mv.sha256_of(video)
    try:
        frames, fps = mv.decode_gray(video, analysis_width)
    except RuntimeError as e:
        raise BadInput(str(e)) from None
    n = len(frames)
    if n == 0 or not fps:
        raise BadInput("the video has no decodable frames")
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

    cam_out, elements, speed_g, value_g, nonrigid = [], [], {}, {}, []
    stats = {"camera_jitter_dropped": 0}
    entries = []
    for si, (f0, f1) in enumerate(shots):
        fr = frames[f0:f1 + 1]  # a view, no copy
        rep = repeat[f0:f1 + 1].copy()
        rep[0] = False
        cam = camera_track(fr) if cv2 is not None else None
        if cam is not None:
            for m in camera_moves(cam, fps, repeat=rep, t_offset=f0 / float(fps), stats=stats):
                mid = f"cam{len(cam_out)}"
                ci = CAM_PROPS.index(m["prop"])
                a, b = max(0, m["i0"]), min(len(cam) - 1, m["i1"])
                vals = cam[a:b + 1, ci]
                value_g[mid] = _graph(vals)
                speed_g[mid] = _graph(speed_series(vals, fps, rep[a:b + 1]))
                m = {k: v for k, v in m.items() if k not in ("i0", "i1")}
                cam_out.append(dict(m, id=mid, shot=si))
        if "elements" in not_run:
            continue
        nr = []
        for tr in element_tracks(fr, fps, cam=cam, t_offset=f0 / float(fps), repeat=rep, nonrigid=nr):
            L = len(tr["series"]["x"])
            moves = _series_moves(tr["series"], fps, tr["repeat"])
            pops = not tr["present_at_cut"] or not tr["present_at_end"]
            if not moves and not pops:
                continue
            eid = f"el{len(elements)}"
            mv_out = []
            for k, (prop, _i0, _i1, m, ts, te) in enumerate(moves):
                mid = f"{eid}.{prop}{k}"
                a, b = max(0, int(math.floor(ts))), min(L - 1, int(math.ceil(te)))
                vals = tr["series"][prop][a:b + 1]
                r2 = None if tr["repeat"] is None else tr["repeat"][a:b + 1]
                value_g[mid] = _graph(vals)
                speed_g[mid] = _graph(speed_series(vals, fps, r2))
                mv_out.append(dict(m, id=mid, prop=prop, t0=round(tr["t0"] + ts / float(fps), 4)))
            ee = entrance_exit(tr, fps, moves)
            elements.append({"id": eid, "shot": si, "t0": tr["t0"], "t1": tr["t1"], "area": tr["area"],
                             "moves": mv_out, "entrance_exit": ee})
            if not tr["present_at_cut"]:
                in_ts = min((m[4] for m in moves if m[4] <= 3), default=0.0) if ee["in"] else 0.0
                onset = tr["t0"] + in_ts / float(fps)
                entries.append({"id": eid, "shot": si, "onset": round(onset, 4), "area": tr["area"],
                                "dir": _entry_dir(tr, moves)})
        for r in nr:
            nonrigid.append(dict(r, shot=si))

    all_moves = cam_out + [m for e in elements for m in e["moves"]]
    names = {}
    for m in all_moves:
        names[m["ease_name"]] = names.get(m["ease_name"], 0) + 1
    nonrigid.sort(key=lambda r: -(r["t1"] - r["t0"]))
    doc = {
        "schema": SCHEMA,
        "video": {"id": sha[:12], "sha256": sha, "fps": round(float(fps), 4), "frames": int(n),
                  "duration_s": round(n / float(fps), 4)},
        "cadence": cad,
        "shots": [{"t0": round(a / float(fps), 4), "t1": round(b / float(fps), 4)} for a, b in shots],
        "camera_moves": cam_out,
        "camera_jitter_dropped": int(stats["camera_jitter_dropped"]),
        "elements": elements,
        "nonrigid_regions": {"count": len(nonrigid),
                             "time_s": round(sum(r["t1"] - r["t0"] for r in nonrigid), 3),
                             "area": _stats([r["area"] for r in nonrigid]),
                             "longest": nonrigid[:10]},
        "stagger_groups": stagger_groups(entries, fps),
        "speed_graphs": speed_g,
        "value_graphs": value_g,
        "summary": {"moves": len(all_moves), "dur_s": _stats([m["dur_s"] for m in all_moves]),
                    "peak_at": _stats([m["peak_at"] for m in all_moves]),
                    "overshoot": _stats([m["overshoot"] for m in all_moves]),
                    "hard_end_share": round(sum(m["end"] == "hard" for m in all_moves) / len(all_moves), 4)
                    if all_moves else None,
                    "stepped_share": round(sum(m.get("step_f", 1) > 1 for m in all_moves) / len(all_moves), 4)
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
            "value_graphs": dict, "not_run": list, "summary": dict, "nonrigid_regions": dict,
            "camera_jitter_dropped": int}
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

def _check_measure(measure):
    """A reason the measure/1 document cannot be used, or None."""
    if not isinstance(measure, dict) or measure.get("schema") != "measure/1" or not isinstance(
            measure.get("cuts"), list):
        return "not a measure/1 document with cuts"
    for c in measure["cuts"]:
        t = c.get("t") if isinstance(c, dict) else None
        if isinstance(t, bool) or not isinstance(t, (int, float)) or not math.isfinite(t) or t < 0:
            return f"a cut has no valid time t: {c!r}"
        df = c.get("dur_f", 1)
        if df is not None and (isinstance(df, bool) or not isinstance(df, (int, float)) or df < 0):
            return f"a cut has a bad dur_f: {c!r}"
    return None


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
        why = _check_measure(measure)
        if why:
            print(f"error: --measure {a.measure}: {why}", file=sys.stderr)
            return 2
    cv2 = _cv2()
    cv2_error = getattr(cv2, "error", None) if cv2 is not None else None
    try:
        doc = track(a.video, measure, analysis_width=a.analysis_width)
        store.write_json(Path(a.out), doc)
    except MissingTool as e:
        print(str(e), file=sys.stderr)
        return 3
    except BadInput as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except Exception as e:  # noqa: BLE001 - one line, never a traceback
        if cv2_error is not None and isinstance(e, cv2_error):
            msg = str(e).strip().splitlines()
            print(f"error: OpenCV could not process the video: {msg[-1] if msg else 'cv2.error'}", file=sys.stderr)
            return 2
        if isinstance(e, (RuntimeError, ValueError, OSError)):
            print(f"error: {e}", file=sys.stderr)
            return 1
        raise
    s = doc["summary"]
    print(f"{len(doc['camera_moves'])} camera moves ({doc['camera_jitter_dropped']} jitter dropped), "
          f"{len(doc['elements'])} elements, {doc['nonrigid_regions']['count']} non-rigid regions, {s['moves']} moves, "
          f"{len(doc['stagger_groups'])} stagger groups; eases {s['ease_names'] or 'none'}; "
          f"not run: {doc['not_run'] or 'none'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

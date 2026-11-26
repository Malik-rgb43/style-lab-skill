#!/usr/bin/env python3
"""Look and finish of one video, measured: grain, vignette, sharpness and depth of field, tone curve,
split toning, halation, chromatic aberration, letterbox, flicker, and the finish of its text.

Reads the video (never modifies it; sha256 before and after) and writes look.json (schema "look/1").

Usage:
  look.py <video> --measure <measure.json> --out <dir>/look.json [--analysis-width 1280]

How it is measured (all local, free, numpy; cv2 only speeds up blurs and dilations):
  - shots come from measure.json "shots" [{t0, t1}] (without --measure the cuts are detected here with
    measure_video.detect_cuts). Per shot at least 8 frames are sampled, evenly spread, and decoded in
    RGB at the analysis width (default 1280, never above the source width) through ONE ffmpeg pipe
    that only outputs the sampled frames. Every metric is a per-frame pure function; the shot value is
    the median over its sampled frames. The letterbox bars are cut off before the other metrics run.
  - grain: the frame is cut into 24 px blocks; flat blocks are those whose structure (the frame blurred
    by sigma 8, minus each block's best-fit plane) has a variance at or below its p20; blocks near
    black or white are skipped (there the grain is clipped). std = RMS of the plane-detrended frame in
    the flat blocks; size_px = the full width at half maximum of its autocorrelation (1 px = per-pixel
    noise; Gaussian grain of sigma s reads about 3.3 s).
  - vignette: radial luma profile (median per ring; radius 1 = the corners, aspect-normalised),
    fitted as L(r) = c * (1 - falloff * r^2). shape "round" when falloff >= 0.08 and all four
    corners are darker than the centre, else "none".
  - sharpness: variance of the 4-neighbour Laplacian; centre_edge_ratio = Laplacian variance of the
    middle 40 % box over that of the frame outside the middle 70 % box (high = shallow depth of field
    or a tilt-shift look; about 1 = deep focus).
  - tone: luma percentiles: black_pt p0.5, white_pt p99.5, midtone p50, curve [p5, p25, p50, p75,
    p95] (gray levels 0..255); contrast = RMS contrast (luma std / 255).
  - grade (split toning): mean OKLab a, b of the darkest 15 % and the brightest 15 % of the pixels
    (by OKLab L; palette.py converters through a 32^3 lookup table); sat = mean OKLab chroma.
  - halation: mean luma in a ring just outside the highlights (luma >= 230) minus that of the ring
    beyond it, / 255. None when the frame has no highlights.
  - chroma_ab: R vs B channel offset by phase correlation in the 8 outer tiles of a 3x3 grid, the
    radial component; the absolute median, in px. Lateral aberration shows here, content does not.
  - letterbox: dark (mean <= 24), flat (std <= 6) rows from the top and from the bottom, as fractions
    of the frame height. A frame that is mostly dark (a fade to black) has no letterbox.
  - flicker: per shot, the per-frame mean luma (from a small gray decode of every frame), detrended
    by a centred 9-frame moving average; the residual std in gray levels.
  - text_finish: on the measure.json text boxes [{t, box}] (up to 40, spread over time): fill = the
    bright or dark class against the background (the padded crop's border); a stroke is a class that
    encloses the fill (its width = rings around the fill that hold it); the shadow is a fit of a
    shifted, blurred copy of the glyph to the darkening around it (dx, dy, blur_px = Gaussian sigma,
    alpha); a glow is an isotropic, fading bright ring; fill_kind "gradient" when a plane fit across
    the fill spans >= 20 levels.
  - every length in px (grain size, chroma_ab, stroke, shadow) is given at a 1920 px wide frame
    ("px_ref_width": 1920), so videos of different sizes compare.

Output: {schema, video {sha256, width, height, fps, duration_s, frames}, analysis_width, px_ref_width,
per_shot [...], summary {metric: {median, p10, p90, n}}, text_finish {n, stroke_px, shadow_share,
glow_share, gradient_share, shadow {dx, dy, blur_px, alpha}, boxes}, not_run, notes}.

Exit: 0 done, 1 error (or the source changed while it was read), 2 bad input, 3 ffmpeg missing.
"""
import argparse
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import palette  # noqa: E402
import store  # noqa: E402
from cut import MissingTool, require_ffmpeg  # noqa: E402

SCHEMA = "look/1"
ANALYSIS_WIDTH = 1280      # RGB decode width (capped at the source width)
FLICKER_WIDTH = 160        # gray decode width for the per-frame luma series
PX_REF_WIDTH = 1920        # px metrics are reported at this frame width
LOWFREQ_WIDTH = 480        # vignette and halation run on a block-mean copy at most this wide
SUBSAMPLE = 2              # tone and grade read every 2nd pixel of every 2nd row
SAMPLES_PER_SHOT = 8       # at least this many frames per shot (all of them when the shot is shorter)
MAX_SAMPLES = 1200         # ... unless the video has so many shots that the total passes this

GRAIN_BLOCK = 24           # px, block size for the flat-region search
GRAIN_FLAT_PCT = 20        # flat = low-pass block variance at or below this percentile
GRAIN_FLAT_SIGMA = 8.0     # flatness is judged on the frame blurred by this sigma (structure, not grain)
GRAIN_CLIP = (12.0, 243.0)  # blocks whose mean is outside this range hold clipped grain
GRAIN_ACF_MAX = 8          # px, longest autocorrelation lag looked at

VIG_BINS = 12
VIG_MIN = 0.08             # falloff at or above this (and darker corners) = "round"

SHARP_CENTRE = 0.40        # middle box side (fraction of the frame) for the centre
SHARP_EDGE = 0.70          # outside this middle box = the edge
SHARP_EPS = 1.0            # added to both variances so flat frames read 1.0
SHARP_MAX = 100.0

TONE_BLACK_PCT, TONE_WHITE_PCT = 0.5, 99.5
TONE_CURVE_PCTS = (5, 25, 50, 75, 95)
GRADE_SHARE = 0.15         # darkest / brightest share of the pixels for the split toning

HALO_HI = 230.0            # highlight luma
HALO_MIN_SHARE, HALO_MAX_SHARE = 0.0005, 0.4
HALO_NEAR = 0.006          # near ring radius, fraction of the frame width (the far ring is 3x)

CA_MIN_EDGE = 2.0          # a tile needs this much high-pass std to be measured
CA_MIN_PEAK = 0.03         # phase-correlation peak below this = no reliable shift
CA_MIN_SIZE = 48           # px, smaller frames are not measured
CA_PATCH = 128             # px, the phase correlation runs on a square patch at each outer tile centre

LB_DARK, LB_STD = 24.0, 6.0
LB_MIN = 0.01              # bars thinner than this fraction are not a letterbox
LB_MAX_TOTAL = 0.6         # more dark rows than this = a dark frame, not a letterbox

FLICKER_WIN = 9            # frames, detrending window
FLICKER_MIN = 6            # frames needed

TEXT_MAX_BOXES = 40
TEXT_MIN_H = 6             # px, smaller boxes are not measured
TEXT_MAX_H = 64            # px, larger boxes are measured on a block-mean downscaled crop
TEXT_PAD = 0.6             # crop padding, fraction of the box height
TEXT_CONTRAST = 40.0       # levels from the background to count as glyph (fill or stroke)
TEXT_MIN_AREA = 0.02       # a fill covers at least this share of the box
STROKE_ENCLOSE = 0.6       # a class that holds this share of the ring around the fill is a stroke
STROKE_RING = 0.5          # a ring around the fill is stroke when this share of it is stroke class
SHADOW_MAX = 16            # px (crop scale), largest shadow offset searched
SHADOW_SIGMAS = (0.0, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
SHADOW_MIN_ALPHA = 0.10
SHADOW_MIN_R2 = 0.35       # the shadow model explains at least this share of the darkening
SHADOW_MIN_BG = 30.0       # on a darker background a shadow cannot be seen
GLOW_DIFF = 12.0           # levels above the background just outside the glyph
GLOW_ISO = 0.6             # ... in at least this share of that ring
FILL_GRAD = 20.0           # levels spanned by a plane fit across the fill = gradient

# metric -> per-shot field; these are the summary keys (and the fingerprint's layers.look keys)
SHOT_KEYS = ("grain_std", "grain_size_px", "vignette_falloff", "sharp_centre_edge_ratio", "lap_var",
             "tone_black_pt", "tone_white_pt", "tone_midtone", "tone_contrast", "shadow_tint_a",
             "shadow_tint_b", "highlight_tint_a", "highlight_tint_b", "sat", "halation", "chroma_ab_px",
             "letterbox_top", "letterbox_bottom", "flicker")


class InputError(ValueError):
    """Bad input: the CLI exits 2."""


# ---------------------------------------------------------------- small helpers

def _cv2():
    try:
        import cv2
        return cv2
    except ImportError:
        return None


def _gauss_kernel(sigma):
    r = max(1, int(math.ceil(3 * sigma)))
    x = np.arange(-r, r + 1, dtype=np.float32)
    k = np.exp(-x * x / (2 * sigma * sigma))
    return k / k.sum(), r


def _blur(img, sigma):
    """Gaussian blur of a 2-D float32 image (reflected borders)."""
    img = np.asarray(img, dtype=np.float32)
    if sigma <= 0 or img.size == 0:
        return img.copy()
    cv2 = _cv2()
    if cv2 is not None:
        return cv2.GaussianBlur(img, (0, 0), sigmaX=float(sigma), sigmaY=float(sigma),
                                borderType=cv2.BORDER_REFLECT)
    k, r = _gauss_kernel(sigma)
    out = img
    for axis in (0, 1):
        n = out.shape[axis]
        mode = "reflect" if r < n else "edge"
        pad = [(0, 0), (0, 0)]
        pad[axis] = (r, r)
        p = np.pad(out, pad, mode=mode)
        acc = np.zeros_like(out)
        for i, w in enumerate(k):
            acc += w * (p[i:i + out.shape[0]] if axis == 0 else p[:, i:i + out.shape[1]])
        out = acc
    return out


def _dilate(mask, r):
    """Square dilation of a bool mask by r px."""
    mask = np.asarray(mask, dtype=bool)
    if r <= 0 or not mask.any():
        return mask.copy()
    cv2 = _cv2()
    if cv2 is not None:
        k = np.ones((2 * r + 1, 2 * r + 1), np.uint8)
        return cv2.dilate(mask.astype(np.uint8), k).astype(bool)
    out = mask.copy()
    for axis in (0, 1):
        src = out.copy()
        n = src.shape[axis]
        for s in range(1, min(r, n - 1) + 1):
            if axis == 0:
                out[s:] |= src[:-s]
                out[:-s] |= src[s:]
            else:
                out[:, s:] |= src[:, :-s]
                out[:, :-s] |= src[:, s:]
    return out


def _grow(mask):
    """One step of a 4-neighbour (cross) dilation."""
    out = mask.copy()
    out[1:] |= mask[:-1]
    out[:-1] |= mask[1:]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    return out


def _erode(mask):
    return ~_grow(~mask)


def _luma(rgb):
    a = np.asarray(rgb, dtype=np.float32)
    return a[..., 0] * 0.299 + a[..., 1] * 0.587 + a[..., 2] * 0.114


def _gray(frame):
    a = np.asarray(frame)
    return _luma(a) if a.ndim == 3 else a.astype(np.float32)


def _r(v, nd=4):
    return None if v is None or not math.isfinite(float(v)) else round(float(v), nd)


def _stats(vals):
    v = [float(x) for x in vals if x is not None and math.isfinite(float(x))]
    if not v:
        return {"median": None, "p10": None, "p90": None, "n": 0}
    a = np.asarray(v, float)
    return {"median": round(float(np.median(a)), 4), "p10": round(float(np.percentile(a, 10)), 4),
            "p90": round(float(np.percentile(a, 90)), 4), "n": len(v)}


def _median(vals):
    v = [float(x) for x in vals if x is not None and math.isfinite(float(x))]
    return float(np.median(v)) if v else None


# ---------------------------------------------------------------- OKLab (palette.py converters)

_LUT = None


def _oklab_lut():
    """(32, 32, 32, 3) OKLab of the centre of every 5-bit sRGB cell, from palette.py's converters."""
    global _LUT
    if _LUT is None:
        lin = [palette._lin((q * 8 + 4) / 255.0) for q in range(32)]
        lut = np.empty((32, 32, 32, 3), np.float32)
        for i, r in enumerate(lin):
            for j, g in enumerate(lin):
                for k, b in enumerate(lin):
                    lut[i, j, k] = palette._lin_to_oklab(r, g, b)
        _LUT = lut
    return _LUT


def to_oklab(rgb):
    """(..., 3) uint8 sRGB -> (..., 3) float32 OKLab (5-bit lookup)."""
    q = np.asarray(rgb, dtype=np.uint8) >> 3
    return _oklab_lut()[q[..., 0], q[..., 1], q[..., 2]]


# ---------------------------------------------------------------- per-frame metrics

def letterbox(frame_gray):
    """[top, bottom] letterbox bars as fractions of the frame height."""
    g = _gray(frame_gray)
    h = g.shape[0]
    if h < 4 or g.shape[1] < 2:
        return [0.0, 0.0]
    dark = (g.mean(axis=1) <= LB_DARK) & (g.std(axis=1) <= LB_STD)
    top = int(np.argmin(dark)) if not dark.all() else h
    bot = int(np.argmin(dark[::-1])) if not dark.all() else h
    if top + bot >= LB_MAX_TOTAL * h:
        return [0.0, 0.0]
    t, b = top / h, bot / h
    return [round(t, 4) if t >= LB_MIN else 0.0, round(b, 4) if b >= LB_MIN else 0.0]


def _active(frame, lb):
    """The frame without its letterbox bars."""
    h = frame.shape[0]
    t, b = int(round(lb[0] * h)), int(round(lb[1] * h))
    return frame[t:h - b] if h - b - t >= 4 else frame


def grain(frame_gray):
    """{"std", "size_px"} of the noise in flat regions (None when the frame is too small or clean)."""
    g = _gray(frame_gray)
    B = GRAIN_BLOCK
    h, w = g.shape
    if h < 2 * B or w < 2 * B:
        return {"std": None, "size_px": None}
    lo = _blur(g, GRAIN_FLAT_SIGMA)
    ny, nx = h // B, w // B
    y0, x0 = (h - ny * B) // 2, (w - nx * B) // 2

    def blocks(a):
        return a[y0:y0 + ny * B, x0:x0 + nx * B].reshape(ny, B, nx, B).transpose(0, 2, 1, 3).reshape(-1, B, B)
    gb, lb_ = blocks(g), blocks(lo)
    mean = gb.mean(axis=(1, 2))
    # remove each block's best-fit plane: a gentle gradient is not grain
    yy, xx = np.mgrid[0:B, 0:B].astype(np.float32)
    A = np.stack([np.ones(B * B, np.float32), xx.ravel(), yy.ravel()], axis=1)
    Ap = np.linalg.pinv(A).astype(np.float32)

    def detrend(bl):
        flat = bl.reshape(len(bl), -1)
        return flat - (flat @ Ap.T) @ A.T
    rb = detrend(gb).reshape(-1, B, B)
    svar = detrend(lb_).var(axis=1)
    valid = (mean >= GRAIN_CLIP[0]) & (mean <= GRAIN_CLIP[1])
    if not valid.any():
        return {"std": None, "size_px": None}
    thr = np.percentile(svar[valid], GRAIN_FLAT_PCT)
    sel = np.where(valid & (svar <= thr))[0]
    rsel = rb[sel]
    rsel = rsel - rsel.mean(axis=(1, 2), keepdims=True)
    std = float(np.sqrt((rsel ** 2).mean()))
    if std < 1e-3:
        return {"std": round(std, 4), "size_px": None}
    # autocorrelation (zero-padded, normalised by the overlap) summed over the flat blocks
    if len(rsel) > 400:
        rsel = rsel[np.linspace(0, len(rsel) - 1, 400).astype(int)]
    F = np.fft.rfft2(rsel, s=(2 * B, 2 * B))
    acf = np.fft.irfft2((F * np.conj(F)).sum(axis=0), s=(2 * B, 2 * B))
    lags = np.arange(GRAIN_ACF_MAX + 1)
    prof_x = acf[0, lags] / (B * (B - lags))
    prof_y = acf[lags, 0] / (B * (B - lags))
    prof = (prof_x + prof_y) / 2.0
    prof = prof / prof[0]
    hw = float(GRAIN_ACF_MAX)
    for d in range(1, len(prof)):
        if prof[d] < 0.5:
            hw = (d - 1) + (prof[d - 1] - 0.5) / (prof[d - 1] - prof[d])
            break
    return {"std": round(std, 4), "size_px": round(float(2.0 * hw), 4)}


def _radius(h, w):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    return np.sqrt(((xx - cx) / max(cx, 0.5)) ** 2 + ((yy - cy) / max(cy, 0.5)) ** 2) / math.sqrt(2.0), yy, xx


def vignette(frame_gray):
    """{"falloff": luma lost at the corners as a share of the centre, "shape": "round"|"none"}."""
    g = _gray(frame_gray)
    h, w = g.shape
    if h < 8 or w < 8:
        return {"falloff": 0.0, "shape": "none"}
    r, yy, xx = _radius(h, w)
    edges = np.linspace(0, 1.0, VIG_BINS + 1)
    rb, med, cnt = [], [], []
    idx = np.clip(np.digitize(r.ravel(), edges) - 1, 0, VIG_BINS - 1)
    flat = g.ravel()
    for k in range(VIG_BINS):
        m = idx == k
        if m.sum() >= 4:
            rb.append((edges[k] + edges[k + 1]) / 2)
            med.append(float(np.median(flat[m])))
            cnt.append(float(m.sum()))
    if len(rb) < 3:
        return {"falloff": 0.0, "shape": "none"}
    x = np.asarray(rb) ** 2
    yv = np.asarray(med)
    wt = np.sqrt(np.asarray(cnt))
    A = np.stack([np.ones_like(x), x], axis=1) * wt[:, None]
    (a, b), *_ = np.linalg.lstsq(A, yv * wt, rcond=None)
    if a <= 1.0:
        return {"falloff": 0.0, "shape": "none"}
    fall = float(np.clip(-b / a, -1.0, 1.0))
    centre = float(np.median(g[r <= 0.3])) if (r <= 0.3).any() else float(a)
    corner_ok = True
    for qy in (yy < h / 2.0, yy >= h / 2.0):
        for qx in (xx < w / 2.0, xx >= w / 2.0):
            m = qy & qx & (r >= 0.85)
            if not m.any() or float(np.median(g[m])) >= centre * (1 - VIG_MIN / 2):
                corner_ok = False
    shape = "round" if fall >= VIG_MIN and corner_ok else "none"
    return {"falloff": round(fall, 4), "shape": shape}


def sharpness(frame_gray):
    """{"lap_var", "centre_edge_ratio"}: Laplacian variance, and centre over edge (shallow focus > 1)."""
    g = _gray(frame_gray)
    h, w = g.shape
    if h < 8 or w < 8:
        return {"lap_var": None, "centre_edge_ratio": None}
    lap = 4 * g[1:-1, 1:-1] - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:]
    hh, ww = lap.shape

    def box(frac):
        y0, x0 = int(hh * (1 - frac) / 2), int(ww * (1 - frac) / 2)
        return slice(y0, max(y0 + 1, hh - y0)), slice(x0, max(x0 + 1, ww - x0))
    c = lap[box(SHARP_CENTRE)]
    mask = np.ones(lap.shape, bool)
    mask[box(SHARP_EDGE)] = False
    e = lap[mask]
    vc = float(c.var()) if c.size else 0.0
    ve = float(e.var()) if e.size else 0.0
    ratio = min(SHARP_MAX, (vc + SHARP_EPS) / (ve + SHARP_EPS))
    return {"lap_var": round(float(lap.var()), 4), "centre_edge_ratio": round(ratio, 4)}


def tone(frame_rgb):
    """{"black_pt", "white_pt", "midtone", "contrast", "curve": [p5, p25, p50, p75, p95]} (luma 0..255)."""
    y = _gray(frame_rgb).ravel()
    if y.size == 0:
        return {"black_pt": None, "white_pt": None, "midtone": None, "contrast": None, "curve": []}
    p = np.percentile(y, [TONE_BLACK_PCT, TONE_WHITE_PCT] + list(TONE_CURVE_PCTS))
    return {"black_pt": round(float(p[0]), 2), "white_pt": round(float(p[1]), 2),
            "midtone": round(float(p[4]), 2), "contrast": round(float(y.std() / 255.0), 4),
            "curve": [round(float(v), 2) for v in p[2:]]}


def grade(frame_rgb):
    """{"shadow_tint": [a, b], "highlight_tint": [a, b], "sat"}: mean OKLab a, b of the darkest and the
    brightest 15 % of the pixels (split toning) and the mean OKLab chroma."""
    rgb = np.asarray(frame_rgb, dtype=np.uint8)
    lab = to_oklab(rgb.reshape(-1, 3))
    n = len(lab)
    if n == 0:
        return {"shadow_tint": None, "highlight_tint": None, "sat": None}
    k = max(1, int(GRADE_SHARE * n))
    L = lab[:, 0]
    if k >= n:
        lo = hi = np.arange(n)
    else:
        lo = np.argpartition(L, k - 1)[:k]
        hi = np.argpartition(L, n - k)[n - k:]
    sh, hl = lab[lo, 1:].mean(axis=0), lab[hi, 1:].mean(axis=0)
    sat = float(np.hypot(lab[:, 1], lab[:, 2]).mean())
    return {"shadow_tint": [round(float(sh[0]), 4), round(float(sh[1]), 4)],
            "highlight_tint": [round(float(hl[0]), 4), round(float(hl[1]), 4)], "sat": round(sat, 4)}


def halation(frame_rgb):
    """Glow energy around highlights: near-ring luma minus far-ring luma, / 255 (None: no highlights)."""
    y = _gray(frame_rgb)
    h, w = y.shape
    if min(h, w) < 16:
        return None
    hi = y >= HALO_HI
    share = float(hi.mean())
    if share < HALO_MIN_SHARE or share > HALO_MAX_SHARE:
        return None
    r1 = max(2, int(round(w * HALO_NEAR)))
    r0 = max(1, r1 // 4)
    d0, d1, d2 = _dilate(hi, r0), _dilate(hi, r1), _dilate(hi, 3 * r1)
    near, far = d1 & ~d0, d2 & ~d1
    if near.sum() < 4 or far.sum() < 4:
        return None
    return round(max(0.0, float(y[near].mean() - y[far].mean()) / 255.0), 4)


def _phase_shift(a, b):
    """(dx, dy, peak): the shift that moves image a onto image b (b(x) = a(x - d)), sub-pixel."""
    h, w = a.shape
    win = np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)
    A = np.fft.fft2((a - a.mean()) * win)
    Bf = np.fft.fft2((b - b.mean()) * win)
    R = Bf * np.conj(A)
    R /= np.abs(R) + 1e-9
    c = np.fft.ifft2(R).real
    py, px = np.unravel_index(int(np.argmax(c)), c.shape)
    peak = float(c[py, px])

    def sub(m, z, p):
        d = m - 2 * z + p
        return 0.0 if abs(d) < 1e-12 else 0.5 * (m - p) / d
    dy = py + sub(c[(py - 1) % h, px], peak, c[(py + 1) % h, px])
    dx = px + sub(c[py, (px - 1) % w], peak, c[py, (px + 1) % w])
    if dy > h / 2:
        dy -= h
    if dx > w / 2:
        dx -= w
    return float(dx), float(dy), peak


def chroma_ab(frame_rgb):
    """Lateral chromatic aberration: radial R vs B offset (px) in the outer tiles; None when unmeasurable."""
    rgb = np.asarray(frame_rgb, dtype=np.float32)
    h, w = rgb.shape[:2]
    if rgb.ndim != 3 or min(h, w) < CA_MIN_SIZE:
        return None
    R = rgb[..., 0] - _blur(np.ascontiguousarray(rgb[..., 0]), 2.0)
    Bc = rgb[..., 2] - _blur(np.ascontiguousarray(rgb[..., 2]), 2.0)
    th, tw = h // 3, w // 3
    radial = []
    for iy in range(3):
        for ix in range(3):
            if iy == 1 and ix == 1:
                continue
            ps = 1 << int(math.floor(math.log2(max(16, min(th, tw, CA_PATCH)))))
            cy, cx = iy * th + th // 2, ix * tw + tw // 2
            sl = (slice(max(0, cy - ps // 2), max(0, cy - ps // 2) + ps), slice(max(0, cx - ps // 2), max(0, cx - ps // 2) + ps))
            a, b = R[sl], Bc[sl]
            if a.std() < CA_MIN_EDGE or b.std() < CA_MIN_EDGE:
                continue
            dx, dy, peak = _phase_shift(a, b)
            if peak < CA_MIN_PEAK:
                continue
            ux, uy = (ix - 1) * tw, (iy - 1) * th
            nrm = math.hypot(ux, uy)
            radial.append((dx * ux + dy * uy) / nrm)
    if not radial:
        return None
    return round(abs(float(np.median(radial))), 4)


def flicker(series_luma):
    """Frame-to-frame luma std (gray levels) after a centred 9-frame moving-average detrend; None if short."""
    x = np.asarray([v for v in series_luma if v is not None], dtype=float)
    if len(x) < FLICKER_MIN:
        return None
    k = min(FLICKER_WIN, len(x) - (1 - len(x) % 2))
    k = max(3, k if k % 2 else k - 1)
    p = np.pad(x, (k // 2, k // 2), mode="reflect", reflect_type="odd")  # keeps a linear trend at the ends
    trend = np.convolve(p, np.ones(k) / k, mode="valid")
    return round(float((x - trend).std()), 4)


# ---------------------------------------------------------------- text finish

def _otsu_like_bg(y):
    border = np.concatenate([y[:2].ravel(), y[-2:].ravel(), y[:, :2].ravel(), y[:, -2:].ravel()])
    return float(np.median(border))


def _rings(mask, n):
    """[ring_1 .. ring_n] around mask (cross-metric distance 1..n) and the grown masks [d0 .. dn]."""
    grown = [mask]
    for _ in range(n):
        grown.append(_grow(grown[-1]))
    return [grown[d] & ~grown[d - 1] for d in range(1, n + 1)], grown


def _corr(a, b, shape):
    """c[s] = sum_q a(q + s) * b(q) for every shift (circular in `shape`, zero padded)."""
    Fa = np.fft.rfft2(a, s=shape)
    Fb = np.fft.rfft2(b, s=shape)
    return np.fft.irfft2(Fa * np.conj(Fb), s=shape)


def _fit_shadow(y, glyph, bg, smax):
    """The best shifted, blurred copy of the glyph that explains the darkening around it.
    Returns (dx, dy, sigma, alpha, r2) in crop px, or None."""
    if bg < SHADOW_MIN_BG:
        return None
    vis = ~_dilate(glyph, 1)
    D = np.clip(bg - y, 0, None) * vis
    tot = float((D * D).sum())
    if tot <= 1e-6:
        return None
    h, w = y.shape
    pad = smax + 1
    shape = (h + 2 * pad, w + 2 * pad)
    best = None
    Dv = D.astype(np.float64)
    V = vis.astype(np.float64)
    for sig in SHADOW_SIGMAS:
        Fb = _blur(glyph.astype(np.float32), sig).astype(np.float64)
        num = _corr(Dv, Fb, shape)
        den = _corr(V, Fb * Fb, shape)
        # shifts -smax..smax live at indices (s mod shape)
        ys = np.arange(-smax, smax + 1)
        xs = np.arange(-smax, smax + 1)
        nu = num[np.ix_(ys % shape[0], xs % shape[1])]
        de = den[np.ix_(ys % shape[0], xs % shape[1])]
        sc = np.where((nu > 0) & (de > 1e-6), nu * nu / np.maximum(de, 1e-6), 0.0)
        k = int(np.argmax(sc))
        iy, ix = np.unravel_index(k, sc.shape)
        if best is None or sc[iy, ix] > best[0]:
            best = (float(sc[iy, ix]), sig, iy, ix, sc, nu[iy, ix] / max(de[iy, ix], 1e-6))
    score, sig, iy, ix, sc, amp = best
    r2 = score / tot
    alpha = float(np.clip(amp / max(bg, 1.0), 0.0, 1.0))
    if r2 < SHADOW_MIN_R2 or alpha < SHADOW_MIN_ALPHA:
        return None

    def sub(m, z, p):
        d = m - 2 * z + p
        return 0.0 if abs(d) < 1e-12 else float(np.clip(0.5 * (m - p) / d, -0.5, 0.5))
    fy = sub(sc[iy - 1, ix], sc[iy, ix], sc[iy + 1, ix]) if 0 < iy < sc.shape[0] - 1 else 0.0
    fx = sub(sc[iy, ix - 1], sc[iy, ix], sc[iy, ix + 1]) if 0 < ix < sc.shape[1] - 1 else 0.0
    return (ix - smax + fx, iy - smax + fy, sig, alpha, r2)


def _hex(rgb):
    return "#" + "".join(f"{int(round(min(255.0, max(0.0, c)))):02x}" for c in rgb)


def text_finish(frame_rgb, box):
    """{"stroke_px", "shadow": {dx, dy, blur_px, alpha} | None, "glow": bool, "fill_kind": "flat"|"gradient",
    "fill_hex"} of the text in `box` ([x, y, w, h] as fractions of the frame), lengths in frame px.
    None when the box is too small or holds no glyph against its background."""
    rgb = np.asarray(frame_rgb)
    if rgb.ndim != 3:
        return None
    H, W = rgb.shape[:2]
    try:
        bx, by, bw, bh = (float(v) for v in box[:4])
    except (TypeError, ValueError, IndexError):
        return None
    x0, y0 = int(round(bx * W)), int(round(by * H))
    x1, y1 = int(round((bx + bw) * W)), int(round((by + bh) * H))
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
    if x1 - x0 < TEXT_MIN_H or y1 - y0 < TEXT_MIN_H:
        return None
    bh_px = y1 - y0
    pad = max(3, int(round(TEXT_PAD * bh_px)))
    cx0, cy0, cx1, cy1 = max(0, x0 - pad), max(0, y0 - pad), min(W, x1 + pad), min(H, y1 + pad)
    crop = rgb[cy0:cy1, cx0:cx1].astype(np.float32)
    k = max(1, int(math.ceil(bh_px / TEXT_MAX_H)))
    if k > 1:
        hh, ww = (crop.shape[0] // k) * k, (crop.shape[1] // k) * k
        crop = crop[:hh, :ww].reshape(hh // k, k, ww // k, k, 3).mean(axis=(1, 3))
    y = _luma(crop)
    h, w = y.shape
    if h < 6 or w < 6:
        return None
    T = np.zeros((h, w), bool)
    T[max(0, (y0 - cy0) // k):min(h, -(-(y1 - cy0) // k)), max(0, (x0 - cx0) // k):min(w, -(-(x1 - cx0) // k))] = True
    bg = _otsu_like_bg(y)
    cands = {"bright": y > bg + TEXT_CONTRAST, "dark": y < bg - TEXT_CONTRAST}
    tarea = float(T.sum())
    cands = {nm: m for nm, m in cands.items() if (m & T).sum() >= TEXT_MIN_AREA * tarea}
    if not cands:
        return None
    gy, gx = np.gradient(y)
    gm = np.hypot(gx, gy)

    def enclosed(inner, outer):
        ring = _grow(inner) & ~inner
        return float((ring & outer).sum() / ring.sum()) if ring.any() else 0.0

    def edge_strength(m):
        ring = (_grow(m) & ~m) | (m & ~_erode(m))
        return float(gm[ring].mean()) if ring.any() else 0.0
    fill_name, stroke_mask = None, None
    if len(cands) == 2:
        for inner, outer in (("bright", "dark"), ("dark", "bright")):
            if enclosed(cands[inner], cands[outer]) >= STROKE_ENCLOSE:
                fill_name, stroke_mask = inner, cands[outer]
                break
    if fill_name is None:
        fill_name = max(cands, key=lambda nm: (edge_strength(cands[nm] & _dilate(T, 1)), (cands[nm] & T).sum()))
    fill = cands[fill_name] & _dilate(T, 1)
    smax = max(3, min(SHADOW_MAX, int(round(0.5 * bh_px / k))))
    stroke = 0
    glyph = fill
    if stroke_mask is not None:
        rings, grown = _rings(fill, smax)
        for rg in rings:
            if rg.any() and (rg & stroke_mask).sum() / rg.sum() >= STROKE_RING:
                stroke += 1
            else:
                break
        if stroke:
            glyph = fill | (stroke_mask & grown[stroke + 1 if stroke + 1 < len(grown) else stroke])
    # glow: an isotropic, fading bright ring just outside the glyph
    glow = False
    fy = y[fill]
    if fy.size and float(fy.mean()) > bg:
        nr = max(4, int(round(0.2 * bh_px / k)))
        rings, _ = _rings(glyph, nr)
        ex = [float(y[rg].mean() - bg) if rg.any() else 0.0 for rg in rings]
        r2 = rings[1] if len(rings) > 1 else rings[0]
        iso = float((y[r2] > bg + GLOW_DIFF / 2).mean()) if r2.any() else 0.0
        glow = bool(len(ex) > 2 and ex[1] >= GLOW_DIFF and iso >= GLOW_ISO and ex[-1] < 0.7 * ex[1])
    shadow = None
    if not glow:
        sh = _fit_shadow(y, glyph, bg, smax)
        if sh is not None:
            dx, dy, sig, alpha, _r2 = sh
            shadow = {"dx": round(float(dx * k), 3), "dy": round(float(dy * k), 3),
                      "blur_px": round(float(sig * k), 3), "alpha": round(float(alpha), 3)}
    core = _erode(fill)
    core = core if core.sum() >= 10 else fill
    yy, xx = np.nonzero(core)
    cols = crop[core]
    kind = "flat"
    if len(cols) >= 10:
        A = np.stack([np.ones(len(xx)), xx.astype(float), yy.astype(float)], axis=1)
        for c in range(3):
            coef, *_ = np.linalg.lstsq(A, cols[:, c], rcond=None)
            pred = A @ coef
            var = float(cols[:, c].var())
            expl = 1.0 - float(((cols[:, c] - pred) ** 2).mean()) / var if var > 1e-6 else 0.0
            if float(pred.max() - pred.min()) >= FILL_GRAD and expl >= 0.5:
                kind = "gradient"
                break
    return {"stroke_px": float(stroke * k), "shadow": shadow, "glow": glow, "fill_kind": kind,
            "fill_hex": _hex(cols.mean(axis=0)) if len(cols) else None}


# ---------------------------------------------------------------- one frame, one shot

def _shrink(a, width):
    """Block-mean downscale (integer factor) so the width is at most `width`."""
    k = int(math.ceil(a.shape[1] / float(width)))
    if k <= 1:
        return np.asarray(a, dtype=np.float32)
    h, w = (a.shape[0] // k) * k, (a.shape[1] // k) * k
    if h == 0 or w == 0:
        return np.asarray(a, dtype=np.float32)
    shp = (h // k, k, w // k, k) + a.shape[2:]
    return np.asarray(a[:h, :w], dtype=np.float32).reshape(shp).mean(axis=(1, 3))


def frame_look(rgb):
    """Every per-frame metric of one RGB frame (lengths in frame px). Grain, sharpness, chroma_ab and
    the letterbox use the full frame; vignette and halation (low-frequency) a copy at most 480 px
    wide; tone and grade every SUBSAMPLE-th pixel (same levels, fewer pixels)."""
    rgb = np.asarray(rgb, dtype=np.uint8)
    g = _luma(rgb)
    lb = letterbox(g)
    ga, ra = _active(g, lb), _active(rgb, lb)
    sub = ra[::SUBSAMPLE, ::SUBSAMPLE] if min(ra.shape[:2]) >= 8 * SUBSAMPLE else ra
    small = _shrink(ra, LOWFREQ_WIDTH)
    gr, vg, sp, tn, gd = grain(ga), vignette(_luma(small)), sharpness(ga), tone(sub), grade(sub)
    return {"grain_std": gr["std"], "grain_size_px": gr["size_px"], "vignette_falloff": vg["falloff"],
            "vignette_round": vg["shape"] == "round", "lap_var": sp["lap_var"],
            "sharp_centre_edge_ratio": sp["centre_edge_ratio"], "tone_black_pt": tn["black_pt"],
            "tone_white_pt": tn["white_pt"], "tone_midtone": tn["midtone"], "tone_contrast": tn["contrast"],
            "tone_curve": tn["curve"],
            "shadow_tint_a": gd["shadow_tint"][0] if gd["shadow_tint"] else None,
            "shadow_tint_b": gd["shadow_tint"][1] if gd["shadow_tint"] else None,
            "highlight_tint_a": gd["highlight_tint"][0] if gd["highlight_tint"] else None,
            "highlight_tint_b": gd["highlight_tint"][1] if gd["highlight_tint"] else None,
            "sat": gd["sat"], "halation": halation(np.clip(small, 0, 255).astype(np.uint8)),
            "chroma_ab_px": chroma_ab(ra),
            "letterbox_top": lb[0], "letterbox_bottom": lb[1]}


PX_KEYS = ("grain_size_px", "chroma_ab_px")


def shot_look(frames_look, px_scale=1.0):
    """Median over the sampled frames of one shot; px lengths times px_scale."""
    out = {}
    for k in SHOT_KEYS:
        if k == "flicker":
            continue
        v = _median([f.get(k) for f in frames_look])
        if v is not None and k in PX_KEYS:
            v *= px_scale
        out[k] = _r(v)
    curves = [f["tone_curve"] for f in frames_look if f.get("tone_curve")]
    out["tone_curve"] = [round(float(np.median([c[i] for c in curves])), 2) for i in range(len(TONE_CURVE_PCTS))] \
        if curves else []
    rnd = [f["vignette_round"] for f in frames_look]
    out["vignette_shape"] = "round" if rnd and sum(rnd) * 2 > len(rnd) else "none"
    return out


# ---------------------------------------------------------------- decoding

def sample_indices(f0, f1, n=SAMPLES_PER_SHOT):
    """At least n frame indices spread evenly over [f0, f1] (all of them when the shot is shorter)."""
    if f1 < f0:
        return []
    if f1 - f0 + 1 <= n:
        return list(range(f0, f1 + 1))
    return sorted(set(int(round(v)) for v in np.linspace(f0, f1, n)))


def iter_frames(video, info, indices, width):
    """Yield (index, rgb uint8 (h, w, 3)) for the sorted unique indices, one ffmpeg pipe; only the selected
    frames leave ffmpeg. Indices count frames at the probe's constant fps (as measure_video does)."""
    import measure_video as mv
    idx = sorted(set(int(i) for i in indices if i >= 0))
    if not idx:
        return
    w, h = mv._scaled_size(info, width)
    expr = "+".join(f"eq(n\\,{i})" for i in idx)
    vf = f"fps={info['fps']:.6f},select={expr},scale={w}:{h}"
    fsize = w * h * 3
    with tempfile.TemporaryFile() as err:
        proc = subprocess.Popen(["ffmpeg", "-v", "error", "-nostdin", "-i", str(video), "-an", "-vf", vf,
                                 "-fps_mode", "passthrough", "-pix_fmt", "rgb24", "-f", "rawvideo", "-"],
                                stdout=subprocess.PIPE, stderr=err)
        got_any = False
        try:
            for i in idx:
                buf = bytearray(fsize)
                view = memoryview(buf)
                got = 0
                while got < fsize:
                    k = proc.stdout.readinto(view[got:])
                    if not k:
                        break
                    got += k
                if got < fsize:
                    break
                got_any = True
                yield i, np.frombuffer(bytes(buf), np.uint8).reshape(h, w, 3)
        finally:
            proc.stdout.close()
            rc = proc.wait()
        if rc != 0 and not got_any:
            err.seek(0)
            msg = err.read().decode("utf-8", "replace").strip().splitlines()
            raise RuntimeError("ffmpeg could not decode the video: " + (msg[-1] if msg else "unknown error"))


def _num(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def shot_frames(measure, n, fps, gray=None):
    """[(f0, f1)] frame ranges of the shots (measure.json shots, or cuts detected on `gray`)."""
    spans = []
    for s in (measure or {}).get("shots") or []:
        if not isinstance(s, dict):
            continue
        t0, t1 = _num(s.get("t0")), _num(s.get("t1"))
        if t0 is None or t1 is None:
            continue
        f0 = min(n - 1, max(0, int(round(t0 * fps))))
        f1 = min(n - 1, max(f0, int(math.ceil(t1 * fps)) - 1))
        spans.append((f0, f1))
    if spans or measure is not None or gray is None or n < 2:
        return spans or [(0, max(0, n - 1))]
    import measure_video as mv
    bounds = [0]
    for c in mv.detect_cuts(gray, fps):
        if c.get("kind") == "jump":
            continue
        bounds.append(int(round(c["t"] * fps)))
    bounds = sorted(set(b for b in bounds if 0 <= b < n)) + [n]
    return [(a, b - 1) for a, b in zip(bounds, bounds[1:]) if b - 1 >= a] or [(0, n - 1)]


def text_records(measure, max_boxes=TEXT_MAX_BOXES):
    """Up to max_boxes [{t, box}] from measure.json text.boxes, spread evenly over time."""
    txt = (measure or {}).get("text")
    boxes = txt.get("boxes") if isinstance(txt, dict) else None
    out = []
    for b in boxes or []:
        if not isinstance(b, dict) or _num(b.get("t")) is None:
            continue
        bx = b.get("box")
        if isinstance(bx, (list, tuple)) and len(bx) >= 4 and all(_num(v) is not None for v in bx[:4]):
            out.append({"t": float(b["t"]), "box": [float(v) for v in bx[:4]]})
    out.sort(key=lambda r: r["t"])
    if len(out) > max_boxes:
        out = [out[int(i)] for i in np.linspace(0, len(out) - 1, max_boxes).round()]
    return out


def text_summary(results, px_scale=1.0):
    ok = [r for r in results if r.get("finish")]
    n = len(ok)
    sh = [r["finish"]["shadow"] for r in ok if r["finish"]["shadow"]]
    return {"n": n,
            "stroke_px": _stats([r["finish"]["stroke_px"] * px_scale for r in ok]),
            "shadow_share": round(len(sh) / n, 4) if n else None,
            "glow_share": round(sum(1 for r in ok if r["finish"]["glow"]) / n, 4) if n else None,
            "gradient_share": round(sum(1 for r in ok if r["finish"]["fill_kind"] == "gradient") / n, 4)
            if n else None,
            "shadow": {"dx": _stats([s["dx"] * px_scale for s in sh]), "dy": _stats([s["dy"] * px_scale for s in sh]),
                       "blur_px": _stats([s["blur_px"] * px_scale for s in sh]),
                       "alpha": _stats([s["alpha"] for s in sh])},
            "boxes": results}


# ---------------------------------------------------------------- the whole video

def analyse(video, measure=None, *, analysis_width=ANALYSIS_WIDTH):
    """The look/1 document of one video (not written)."""
    import measure_video as mv
    video = Path(video)
    sha = mv.sha256_of(video)
    info = mv.probe(video)
    fps = info["fps"]
    width = max(8, min(int(analysis_width), int(info["width"])))
    aw, _ah = mv._scaled_size(info, width)
    px_scale = PX_REF_WIDTH / float(aw)
    gray, _ = mv.decode_gray(video, min(FLICKER_WIDTH, width))
    n = len(gray)
    not_run, notes = [], []
    if n == 0:
        raise RuntimeError("the video has no decodable frames")
    luma = gray.reshape(n, -1).mean(axis=1)
    if measure is None:
        notes.append("no --measure: shots from cuts detected here")
    spans = shot_frames(measure, n, fps, gray)
    del gray
    per_shot_idx = [sample_indices(f0, f1) for f0, f1 in spans]
    total = sum(len(s) for s in per_shot_idx)
    if total > MAX_SAMPLES:
        per = max(3, MAX_SAMPLES // max(1, len(spans)))
        per_shot_idx = [sample_indices(f0, f1, per) for f0, f1 in spans]
        notes.append(f"samples_capped ({per} frames per shot)")
    texts = text_records(measure)
    if measure is None:
        not_run.append("text_finish (no --measure)")
    elif not texts:
        not_run.append("text_finish (no text boxes in measure.json)")
    text_at = {}
    for r in texts:
        text_at.setdefault(min(n - 1, max(0, int(round(r["t"] * fps)))), []).append(r)
    shot_set = set(i for s in per_shot_idx for i in s)
    looks, text_res = {}, []
    for i, rgb in iter_frames(video, info, shot_set | set(text_at), width):
        if i in shot_set:
            looks[i] = frame_look(rgb)
        for r in text_at.get(i, []):
            fin = text_finish(rgb, r["box"])
            text_res.append({"t": round(r["t"], 4), "box": r["box"], "finish": fin})
    per_shot = []
    for k, ((f0, f1), idx) in enumerate(zip(spans, per_shot_idx)):
        fl = [looks[i] for i in idx if i in looks]
        row = {"shot": k, "t0": round(f0 / fps, 4), "t1": round((f1 + 1) / fps, 4), "n_frames": len(fl)}
        if fl:
            row.update(shot_look(fl, px_scale))
        else:
            row.update({key: None for key in SHOT_KEYS if key != "flicker"})
            row.update({"tone_curve": [], "vignette_shape": "none"})
        ser = luma[f0 + 2:f1 - 1] if f1 - f0 + 1 > FLICKER_MIN + 4 else luma[f0:f1 + 1]
        row["flicker"] = flicker(ser)
        per_shot.append(row)
    if not looks:
        not_run.append("look (no sampled frame could be decoded)")
    summary = {key: _stats([s.get(key) for s in per_shot]) for key in SHOT_KEYS}
    rounds = [s["vignette_shape"] == "round" for s in per_shot if s.get("n_frames")]
    summary["vignette_round_share"] = round(sum(rounds) / len(rounds), 4) if rounds else None
    curves = [s["tone_curve"] for s in per_shot if s.get("tone_curve")]
    summary["tone_curve"] = [round(float(np.median([c[i] for c in curves])), 2)
                             for i in range(len(TONE_CURVE_PCTS))] if curves else []
    for r in text_res:
        if r["finish"] is not None:
            f = r["finish"]
            f["stroke_px"] = round(f["stroke_px"], 3)
    tf = text_summary(text_res, px_scale)
    if mv.sha256_of(video) != sha:
        raise RuntimeError("the source changed while it was read (sha256 before and after differ)")
    return {"schema": SCHEMA,
            "video": {"sha256": sha, "width": info["width"], "height": info["height"], "fps": round(fps, 4),
                      "duration_s": info.get("duration_s"), "frames": n},
            "analysis_width": aw, "px_ref_width": PX_REF_WIDTH,
            "per_shot": per_shot, "summary": summary, "text_finish": tf, "not_run": not_run, "notes": notes}


def main(argv=None):
    store.print_utf8()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--measure", help="measure.json of this video (shots, text boxes)")
    ap.add_argument("--out", required=True, help="output file, e.g. <dir>/look.json")
    ap.add_argument("--analysis-width", type=int, default=ANALYSIS_WIDTH)
    a = ap.parse_args(argv)
    try:
        require_ffmpeg()
    except MissingTool as e:
        print(str(e), file=sys.stderr)
        return 3
    if not Path(a.video).is_file():
        print(f"error: video not found: {a.video}", file=sys.stderr)
        return 2
    if a.analysis_width < 8:
        print("error: --analysis-width must be at least 8", file=sys.stderr)
        return 2
    measure = None
    if a.measure:
        try:
            measure = json.loads(Path(a.measure).read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            print(f"error: cannot read --measure {a.measure}: {e}", file=sys.stderr)
            return 2
        if not isinstance(measure, dict) or measure.get("schema") != "measure/1":
            print(f"error: --measure is not a measure/1 document: {a.measure}", file=sys.stderr)
            return 2
    import measure_video as mv
    try:
        doc = analyse(a.video, measure, analysis_width=a.analysis_width)
        store.write_json(Path(a.out), doc)
    except mv.InputError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except MissingTool as e:
        print(str(e), file=sys.stderr)
        return 3
    except (RuntimeError, ValueError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    s = doc["summary"]
    print(f"{len(doc['per_shot'])} shots; grain {s['grain_std']['median']}, vignette {s['vignette_falloff']['median']}, "
          f"black/white {s['tone_black_pt']['median']}/{s['tone_white_pt']['median']}, "
          f"{doc['text_finish']['n']} text boxes; not run: {'; '.join(doc['not_run']) or 'none'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Image metrics for generated assets: decoding through ffmpeg, look stats in OKLab, palette distance, SSIM.

Python only (imported by lookcheck.py and buildcheck.py):
    decode_image(path, width=256, ss=None) -> float32 (h, w, 3) in 0..1 (works on a video at `ss` seconds)
    frames_of(video, n=5, width=256) -> [image]
    look_stats(img) -> {mean_L, contrast, chroma, grain, vignette, sharpness, palette: [{hex, share}]}
    palette_distance(palette, ref_hexes) -> share-weighted mean of the minimum ΔE2000
    ssim(a, b) -> 8x8 block SSIM on grey (resized to the smaller size), over the blocks with content
"""
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

KMEANS_K = 5
KMEANS_SEED = 7
KMEANS_ITERS = 12
FLAT_VAR = 1e-4   # an 8x8 block with less L variance than this is flat


class MissingTool(Exception):
    pass


def _ffmpeg():
    ff = shutil.which("ffmpeg")
    if not ff:
        raise MissingTool("ffmpeg is missing: install it (winget install Gyan.FFmpeg, or brew install ffmpeg)")
    return ff


def _probe_size(path, ss=None):
    ffp = shutil.which("ffprobe")
    if not ffp:
        raise MissingTool("ffprobe is missing: install ffmpeg")
    r = subprocess.run([ffp, "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
                        "-of", "csv=p=0", str(path)], capture_output=True, text=True, encoding="utf-8",
                       errors="replace")
    try:
        w, h = (int(x) for x in r.stdout.strip().splitlines()[0].split(",")[:2])
    except (ValueError, IndexError):
        raise ValueError(f"cannot read the image size of {Path(path).name}")
    return w, h


def decode_image(path, width=256, ss=None):
    ff = _ffmpeg()
    w0, h0 = _probe_size(path)
    w = max(2, min(int(width), w0))
    h = max(2, int(round(h0 * w / w0)))
    cmd = [ff, "-v", "error", "-nostdin"]
    if ss is not None:
        cmd += ["-ss", f"{float(ss):.3f}"]
    cmd += ["-i", str(path), "-frames:v", "1", "-vf", f"scale={w}:{h}", "-pix_fmt", "rgb24", "-f", "rawvideo", "-"]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0 or len(r.stdout) < w * h * 3:
        raise ValueError(f"ffmpeg could not decode {Path(path).name}: {r.stderr.decode('utf-8', 'replace')[-200:]}")
    return np.frombuffer(r.stdout[: w * h * 3], np.uint8).reshape(h, w, 3).astype(np.float32) / 255.0


def _duration(video):
    ffp = shutil.which("ffprobe")
    r = subprocess.run([ffp, "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(video)],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    try:
        return float(r.stdout.strip())
    except ValueError:
        return 0.0


def frames_of(video, n=5, width=256):
    d = _duration(video)
    ts = [d * (k + 0.5) / n for k in range(n)] if d > 0 else [0.0]
    return [decode_image(video, width, ss=t) for t in ts]


# ---------------------------------------------------------------- colour

def _lin(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def to_oklab(img):
    """(..., 3) sRGB 0..1 -> (..., 3) OKLab."""
    x = _lin(np.clip(img, 0, 1).astype(np.float64))
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    l_ = np.cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b)
    m_ = np.cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b)
    s_ = np.cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b)
    return np.stack([0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_,
                     1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_,
                     0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_], axis=-1)


def _oklab_to_hex(lab):
    L, a, b = lab
    l_ = L + 0.3963377774 * a + 0.2158037573 * b
    m_ = L - 0.1055613458 * a - 0.0638541728 * b
    s_ = L - 0.0894841775 * a - 1.2914855480 * b
    l, m, s = l_ ** 3, m_ ** 3, s_ ** 3
    rgb = np.array([4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
                    -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
                    -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s])
    rgb = np.clip(rgb, 0, 1)
    srgb = np.where(rgb <= 0.0031308, 12.92 * rgb, 1.055 * rgb ** (1 / 2.4) - 0.055)
    return "#" + "".join(f"{int(round(v * 255)):02x}" for v in np.clip(srgb, 0, 1))


def kmeans_palette(lab, k=KMEANS_K, seed=KMEANS_SEED):
    px = lab.reshape(-1, 3)
    if len(px) > 20000:
        px = px[np.random.default_rng(seed).choice(len(px), 20000, replace=False)]
    rng = np.random.default_rng(seed)
    k = min(k, len(np.unique(np.round(px, 3), axis=0)))
    cent = px[rng.choice(len(px), k, replace=False)]
    for _ in range(KMEANS_ITERS):
        lab_ = np.argmin(((px[:, None, :] - cent[None]) ** 2).sum(-1), axis=1)
        for j in range(k):
            if (lab_ == j).any():
                cent[j] = px[lab_ == j].mean(axis=0)
    lab_ = np.argmin(((px[:, None, :] - cent[None]) ** 2).sum(-1), axis=1)
    out = [{"hex": _oklab_to_hex(cent[j]), "share": round(float((lab_ == j).mean()), 4)} for j in range(k)]
    return sorted([o for o in out if o["share"] > 0], key=lambda o: -o["share"])


# ---------------------------------------------------------------- stats

def _median3(x):
    p = np.pad(x, 1, mode="edge")
    stack = [p[i:i + x.shape[0], j:j + x.shape[1]] for i in range(3) for j in range(3)]
    return np.median(np.stack(stack), axis=0)


def look_stats(img):
    lab = to_oklab(img)
    L = lab[..., 0]
    C = np.hypot(lab[..., 1], lab[..., 2])
    h, w = L.shape
    grain = float((L - _median3(L)).std()) if h >= 3 and w >= 3 else 0.0
    cy0, cy1, cx0, cx1 = int(h * 0.3), int(h * 0.7), int(w * 0.3), int(w * 0.7)
    centre = L[cy0:cy1, cx0:cx1].mean() if cy1 > cy0 and cx1 > cx0 else L.mean()
    ey, ex = max(1, int(h * 0.15)), max(1, int(w * 0.15))
    corners = np.concatenate([L[:ey, :ex].ravel(), L[:ey, -ex:].ravel(), L[-ey:, :ex].ravel(), L[-ey:, -ex:].ravel()])
    vignette = float(corners.mean() / centre) if centre > 1e-6 else 1.0
    if h >= 3 and w >= 3:
        lap = L[:-2, 1:-1] + L[2:, 1:-1] + L[1:-1, :-2] + L[1:-1, 2:] - 4 * L[1:-1, 1:-1]
        sharp = float(lap.var())
    else:
        sharp = 0.0
    return {"mean_L": round(float(L.mean()), 5), "contrast": round(float(L.std()), 5),
            "chroma": round(float(C.mean()), 5), "grain": round(grain, 6), "vignette": round(vignette, 5),
            "sharpness": round(sharp, 7), "palette": kmeans_palette(lab)}


def palette_distance(pal, ref_hexes):
    import palette
    refs = [h for h in ref_hexes if h]
    if not pal or not refs:
        return None
    tot = sum(p["share"] for p in pal) or 1.0
    return round(sum(p["share"] * min(palette.delta_e2000(p["hex"], r) for r in refs) for p in pal) / tot, 3)


def _resize(g, h, w):
    ys = (np.arange(h) * g.shape[0] / h).astype(int)
    xs = (np.arange(w) * g.shape[1] / w).astype(int)
    return g[ys][:, xs]


def ssim(a, b):
    ga = to_oklab(a)[..., 0]
    gb = to_oklab(b)[..., 0]
    h, w = min(ga.shape[0], gb.shape[0]), min(ga.shape[1], gb.shape[1])
    ga, gb = _resize(ga, h, w), _resize(gb, h, w)
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    vals = []
    for y in range(0, h - 7, 8):
        for x in range(0, w - 7, 8):
            p, q = ga[y:y + 8, x:x + 8], gb[y:y + 8, x:x + 8]
            mp, mq = p.mean(), q.mean()
            vp, vq = p.var(), q.var()
            cov = ((p - mp) * (q - mq)).mean()
            v = ((2 * mp * mq + c1) * (2 * cov + c2)) / ((mp ** 2 + mq ** 2 + c1) * (vp + vq + c2))
            vals.append((v, max(vp, vq) > FLAT_VAR or abs(mp - mq) > 0.05))
    # flat blocks that match (an empty ground in both) would read as perfect agreement and hide a different
    # design: only blocks with content in either image, or a different flat level, are scored
    live = [v for v, keep in vals if keep]
    return round(float(np.mean(live)), 4) if live else 1.0

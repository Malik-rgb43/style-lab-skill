#!/usr/bin/env python3
"""Cover-frame candidates for a trained style: frames that look typical of it.

    cover.py candidates <pack> [--n 3] [--per-video 40]

Samples frames from each reference video (skipping the first and last 0.5 s and any frame within 3 frames
of a cut), scores each by how typical it is of the style, and writes the best frame of up to N different
videos to <pack>/cover-candidates/ plus <pack>/cover-candidates.json:
    {"candidates": [{file, video, t, score, why: {...}}], "not_run": [...]}
The script NEVER sets the cover. The owner picks one; `styles.py set-cover <slug> <image>` stores it.

Score parts (each 0..1, weights renormalised over the parts that could be measured):
    palette   frame colour close to the fingerprint colour roles (layers.colour)
    look      grain, tone and vignette close to the look summary (layers.look or look.json), when present
    text      text present, when the style's text share is high
    face      a face present, when the style is talking-head
    sharpness a motion-blurred frame is a bad cover
    colourful only without a fingerprint (with sharpness)

Exit: 0 ok, 2 bad input, 3 ffmpeg missing. Inputs are read-only; no network.
"""
import argparse
import json
import math
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import measure_video as mv  # noqa: E402
import palette  # noqa: E402
import store  # noqa: E402
from cut import MissingTool, require_ffmpeg  # noqa: E402

EDGE_S = 0.5
CUT_GUARD_FRAMES = 3
SCORE_WIDTH = 320
HIGH_TEXT_SHARE = 0.5
TALKING_FACE_SHARE = 0.5
PALETTE_SCALE = 0.12  # OKLab distance at which palette closeness falls to 1/e


class Bad(Exception):
    pass


def _num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _med(m):
    """A number from a plain number or a {median: ...} stat."""
    if isinstance(m, dict):
        m = m.get("median")
    return float(m) if _num(m) else None


def sharpness(gray):
    """0..1 from the variance of the Laplacian (a blurred frame scores low)."""
    g = np.asarray(gray, dtype=np.float32)
    if g.shape[0] < 3 or g.shape[1] < 3:
        return 0.0
    lap = g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:] - 4 * g[1:-1, 1:-1]
    v = float(lap.var())
    return v / (v + 60.0)


def colourfulness(rgb):
    """0..1, Hasler and Suesstrunk's colourfulness squashed."""
    x = np.asarray(rgb, dtype=np.float32)
    rg = x[..., 0] - x[..., 1]
    yb = 0.5 * (x[..., 0] + x[..., 1]) - x[..., 2]
    m = math.hypot(float(rg.std()), float(yb.std())) + 0.3 * math.hypot(float(rg.mean()), float(yb.mean()))
    return m / (m + 40.0)


def _hex(rgb_mean):
    r, g, b = (int(round(min(255.0, max(0.0, v)))) for v in rgb_mean)
    return f"#{r:02x}{g:02x}{b:02x}"


def _lab(hexc):
    L, C, H = palette.hex_to_oklch(hexc)
    h = math.radians(H)
    return L, C * math.cos(h), C * math.sin(h)


def _dist(p, q):
    return math.dist(p, q)


def _role_hexes(fp):
    col = ((fp or {}).get("layers") or {}).get("colour") or {}
    out = []
    for key in ("palette", "roles"):
        v = col.get(key)
        vals = list(v.values()) if isinstance(v, dict) else v if isinstance(v, list) else []
        for x in vals:
            if isinstance(x, dict):
                x = x.get("hex")
            if isinstance(x, str) and re.fullmatch(r"#[0-9a-fA-F]{6}", x):
                out.append(x)
    return out


def _palette_closeness(mean_rgb, fp):
    """0..1 or None when the fingerprint has no colour information."""
    lab = _lab(_hex(mean_rgb))
    roles = _role_hexes(fp)
    if roles:
        d = min(_dist(lab, _lab(h)) for h in roles)
        return math.exp(-d / PALETTE_SCALE)
    metrics = (((fp or {}).get("layers") or {}).get("colour") or {}).get("metrics") or {}
    L, C = _med(metrics.get("dominant_L")), _med(metrics.get("dominant_chroma"))
    if L is None and C is None:
        return None
    fl, fa, fb = lab
    fc = math.hypot(fa, fb)
    d = math.hypot((fl - L) if L is not None else 0.0, (fc - C) if C is not None else 0.0)
    return math.exp(-d / PALETTE_SCALE)


def look_measures(gray):
    """{grain, tone, vignette} of one frame, each roughly 0..1."""
    g = np.asarray(gray, dtype=np.float32) / 255.0
    if g.shape[0] < 6 or g.shape[1] < 6:
        return {"grain": 0.0, "tone": float(g.mean()), "vignette": 0.0}
    box = sum(g[i:g.shape[0] - 2 + i, j:g.shape[1] - 2 + j] for i in range(3) for j in range(3)) / 9.0
    grain = float((g[1:-1, 1:-1] - box).std())
    h, w = g.shape
    centre = float(g[h // 4:3 * h // 4, w // 4:3 * w // 4].mean())
    edge_mask = np.ones_like(g, dtype=bool)
    edge_mask[h // 6:5 * h // 6, w // 6:5 * w // 6] = False
    edge = float(g[edge_mask].mean())
    return {"grain": grain, "tone": float(g.mean()), "vignette": float(min(1.0, max(0.0, 1 - edge / centre)))
            if centre > 1e-6 else 0.0}


def _look_closeness(gray, look):
    if not isinstance(look, dict):
        return None
    meas = look_measures(gray)
    ds = []
    for k, v in meas.items():
        ref = _med(look.get(k))
        if ref is not None:
            ds.append(math.exp(-abs(v - ref) / 0.1))
    return sum(ds) / len(ds) if ds else None


def _metric(fp, layer, names):
    m = (((fp or {}).get("layers") or {}).get(layer) or {}).get("metrics") or {}
    for n in names:
        v = _med(m.get(n))
        if v is not None:
            return v
    return None


def score_frame(rgb, gray, fingerprint=None, look_summary=None, faces=None, text=None):
    """(score 0..1, why) for one frame. Pure. rgb (h, w, 3) uint8, gray (h, w) uint8. faces/text are the
    detector results (True/False, or None when the detector did not run)."""
    parts = [("sharpness", sharpness(gray), 0.25 if fingerprint else 0.6)]
    why = {}
    if not fingerprint:
        parts.append(("colourful", colourfulness(rgb), 0.4))
    else:
        pal = _palette_closeness(np.asarray(rgb, dtype=np.float32).reshape(-1, 3).mean(axis=0), fingerprint)
        if pal is None:
            parts.append(("colourful", colourfulness(rgb), 0.15))
        else:
            parts.append(("palette", pal, 0.35))
        look = _look_closeness(gray, look_summary)
        if look is not None:
            parts.append(("look", look, 0.15))
        share = _metric(fingerprint, "text", ("text_share", "text_frame_share", "coverage"))
        if share is not None and share >= HIGH_TEXT_SHARE and text is not None:
            parts.append(("text", 1.0 if text else 0.3, 0.1))
        face = _metric(fingerprint, "broll", ("face_share",))
        if face is not None and face >= TALKING_FACE_SHARE and faces is not None:
            parts.append(("face", 1.0 if faces else 0.2, 0.15))
    total = sum(w for _, _, w in parts)
    score = sum(v * w for _, v, w in parts) / total
    for name, v, w in parts:
        why[name] = round(float(v), 4)
    return round(float(score), 4), why


def pick(cands, n):
    """The best frame of each video, then the top n of those."""
    best = {}
    for c in cands:
        v = c.get("video")
        if v not in best or c["score"] > best[v]["score"]:
            best[v] = c
    return sorted(best.values(), key=lambda c: -c["score"])[:max(0, n)]


# ---------------------------------------------------------------- video IO

def _grab_rgb(video, t, w, h):
    p = subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-ss", f"{t:.3f}", "-i", str(video), "-frames:v", "1",
                        "-vf", f"scale={w}:{h}", "-pix_fmt", "rgb24", "-f", "rawvideo", "-"],
                       capture_output=True)
    if len(p.stdout) < w * h * 3:
        return None
    return np.frombuffer(p.stdout[:w * h * 3], dtype=np.uint8).reshape(h, w, 3)


def _save_jpg(video, t, out):
    subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-y", "-ss", f"{t:.3f}", "-i", str(video),
                    "-frames:v", "1", "-q:v", "2", str(out)], check=True, capture_output=True)


def sample_times(duration, fps, cuts, k):
    """k times spread over [EDGE_S, duration - EDGE_S], none within CUT_GUARD_FRAMES frames of a cut."""
    lo, hi = EDGE_S, duration - EDGE_S
    if hi <= lo or k <= 0:
        return []
    guard = CUT_GUARD_FRAMES / max(fps, 1.0)
    out = []
    for i in range(k):
        t = lo + (hi - lo) * (i + 0.5) / k
        if all(abs(t - c) > guard for c in cuts):
            out.append(round(t, 3))
    return out


def _read(p):
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _cut_times(measure):
    out = []
    for c in (measure or {}).get("cuts") or []:
        t = c.get("t") if isinstance(c, dict) else c
        if _num(t):
            out.append(float(t))
    return out


def _source(adir, measure):
    sj = _read(adir / "source.json")
    for v in ((sj or {}).get("source"), (measure or {}).get("source")):
        if isinstance(v, str) and v and Path(v).exists():
            return Path(v)
    return None


def cmd_candidates(a):
    pack = Path(a.pack)
    if not pack.is_dir():
        raise Bad(f"pack folder not found: {pack}")
    adir = pack / "analysis"
    fp = _read(pack / "fingerprint.json")
    fp = fp if isinstance(fp, dict) and fp.get("layers") else None
    not_run = [] if fp else ["fingerprint"]
    ids = (fp or {}).get("trained_on") or (sorted(d.name for d in adir.iterdir() if d.is_dir())
                                           if adir.is_dir() else [])
    if not ids:
        raise Bad("no analysed videos in the pack (run measure_video.py first)")
    look = ((fp or {}).get("layers") or {}).get("look") or {}
    look = look.get("metrics") or look
    lj = _read(pack / "look.json")
    if isinstance(lj, dict):
        look = lj.get("summary") or lj.get("metrics") or look
    require_ffmpeg()
    det = mv._face_detector()
    cands = []
    for vid in ids:
        measure = _read(adir / vid / "measure.json")
        src = _source(adir / vid, measure)
        if src is None:
            not_run.append(f"video:{vid}:no_source")
            continue
        info = mv.probe(src)
        w, h = mv._scaled_size(info, SCORE_WIDTH)
        for t in sample_times(info.get("duration_s") or 0, info["fps"], _cut_times(measure), a.per_video):
            rgb = _grab_rgb(src, t, w, h)
            if rgb is None:
                continue
            gray = (rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)).astype(np.uint8)
            faces = bool(det(gray)) if det else None
            try:
                text = bool(mv.text_boxes(gray))
            except ImportError:
                text = None
            score, why = score_frame(rgb, gray, fp, look, faces=faces, text=text)
            cands.append({"video": vid, "t": t, "score": score, "why": why, "_src": str(src)})
    if det is None and fp:
        not_run.append("faces")
    top = pick(cands, a.n)
    out_dir = pack / "cover-candidates"
    out_dir.mkdir(exist_ok=True)
    for old in out_dir.glob("*.jpg"):
        old.unlink()
    rows = []
    for i, c in enumerate(top, 1):
        name = f"{i}-{re.sub(r'[^0-9A-Za-z_-]+', '_', str(c['video']))}-t{int(c['t'] * 1000)}.jpg"
        _save_jpg(c.pop("_src"), c["t"], out_dir / name)
        rows.append({"file": f"cover-candidates/{name}", **c})
    store.write_json(pack / "cover-candidates.json", {"candidates": rows, "not_run": not_run})
    for r in rows:
        print(pack / r["file"])
    if not rows:
        print("no candidate frames (see not_run in cover-candidates.json)", file=sys.stderr)
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("candidates")
    a.add_argument("pack"); a.add_argument("--n", type=int, default=3)
    a.add_argument("--per-video", type=int, default=40)
    args = p.parse_args(argv)
    store.print_utf8()
    try:
        return cmd_candidates(args)
    except Bad as e:
        print(str(e), file=sys.stderr)
        return 2
    except MissingTool as e:
        print(str(e), file=sys.stderr)
        return 3
    except (RuntimeError, subprocess.CalledProcessError) as e:
        print(f"could not read a video: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())

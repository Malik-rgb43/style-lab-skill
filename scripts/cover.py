#!/usr/bin/env python3
"""Cover-frame candidates for a trained style: frames that look typical of it.

    cover.py candidates <pack> [--n 3] [--per-video 40]

Samples frames from each reference video and writes the best frame of up to N different videos (a single
video offers its 2nd and 3rd best at least 2 s apart) to <pack>/cover-candidates/ plus
<pack>/cover-candidates.json:
    {"candidates": [{file, video, t, score, why: {...}}], "not_run": [...]}
The script NEVER sets the cover. The owner picks one; `styles.py set-cover <slug> <image>` stores it.

Hard rejects (a frame is never offered; a video whose every frame is rejected is retried once with
relaxed luma and contrast bars, black frames and transitions still rejected, else it is `no_candidates`): mean luma below 0.06 or above 0.97, luma std below 0.04, within 3
frames of any cut or jump of measure.json, inside a gradual transition (its dur_f), and directional motion
blur (a one-sided gradient ratio above 3 AND a sharpness percentile under 0.3 AND raw sharpness under half the
video's median, so sharp text lines and UI rows are kept). An empty result keeps the old cover-candidates/
folder and exits 1.

Score parts (each 0..1, weights renormalised over the parts that could be measured), from the fingerprint:
    palette   the frame's DOMINANT colour (measure_video.shot_palette, as fingerprint.py reads it) against
              layers.colour.metrics.dominant_L / dominant_chroma
    look      optional; keys follow layers.look.metrics of Task 10 (look.py): grain_std, vignette_falloff, tone_contrast, sat (when present), measured on
              0..1 luma: grain = std of the 3x3 high-pass residual, vignette = 1 - edge/centre mean,
              tone = p95 - p5, sat = mean (max - min)/max; each within 25 % of the style median
    text      layers.type.metrics text_height_pct / big_text_per_min say the style has on-screen text: the
              frame should show text boxes of about that height
    face      layers.broll.metrics.face_share median >= 0.5 (talking head): the frame should show a face
    sharpness the frame's percentile among its own video's sampled frames (a flat graphic is not punished)
    colourful only without a fingerprint (with sharpness)

Exit: 0 ok, 2 bad input, 3 ffmpeg missing. Inputs are read-only; no network.
"""
import argparse
import json
import math
import os
import re
import shutil
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
PALETTE_WIDTH = 64
TALKING_FACE_SHARE = 0.5
FACES_RELEVANT_SHARE = 0.2
PALETTE_SCALE = 0.12  # OKLab distance at which palette closeness falls to 1/e
LUMA_MIN, LUMA_MAX, LUMA_STD_MIN = 0.06, 0.97, 0.04
RELAXED = (0.02, 0.99, 0.015)  # luma min, luma max, std min for a video whose every frame was rejected
BLUR_RATIO, BLUR_PCT, BLUR_VS_MEDIAN = 3.0, 0.3, 0.5
MIN_GAP_S = 2.0
LOOK_KEYS = ("grain_std", "vignette_falloff", "tone_contrast", "sat")


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
    """Raw sharpness: the variance of the Laplacian."""
    g = np.asarray(gray, dtype=np.float32)
    if g.shape[0] < 3 or g.shape[1] < 3:
        return 0.0
    lap = g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:] - 4 * g[1:-1, 1:-1]
    return float(lap.var())


def _squash(v):
    return v / (v + 60.0)


def sharpness_percentiles(values):
    """Percentile (0..1) of each value among the list; ties share the middle, so a flat video gives 0.5.
    None for every value when fewer than 3 frames."""
    n = len(values)
    if n < 3:
        return [None] * n
    a = np.asarray(values, dtype=float)
    return [float(((a < v).sum() + 0.5 * ((a == v).sum() - 1)) / (n - 1)) for v in a]


def colourfulness(rgb):
    """0..1, Hasler and Suesstrunk's colourfulness squashed."""
    x = np.asarray(rgb, dtype=np.float32)
    rg = x[..., 0] - x[..., 1]
    yb = 0.5 * (x[..., 0] + x[..., 1]) - x[..., 2]
    m = math.hypot(float(rg.std()), float(yb.std())) + 0.3 * math.hypot(float(rg.mean()), float(yb.mean()))
    return m / (m + 40.0)


def directional_ratio(gray):
    """max(gx/gy, gy/gx): how one-sided the gradient energy is (directional motion blur makes it large)."""
    g = np.asarray(gray, dtype=np.float32)
    if g.shape[0] < 3 or g.shape[1] < 3:
        return 1.0
    gx = float((np.diff(g, axis=1) ** 2).mean())
    gy = float((np.diff(g, axis=0) ** 2).mean())
    lo, hi = sorted((gx, gy))
    if lo > 1e-9:
        return hi / lo
    return 1.0 if hi <= 1e-9 else float("inf")


def reject_reason(gray, sharp_pct=None, raw_sharp=None, video_median=None, relax=False):
    """Why a frame can never be a cover, or None. Directional blur needs ALL of: sharpness percentile under
    0.3, raw Laplacian variance under half the video's median, and a one-sided gradient ratio above 3 (a
    sharp frame of text lines or UI rows is one-sided too, and is kept). relax=True lowers the luma and
    contrast bars (RELAXED) for a dark or flat style; a black frame is still rejected."""
    lmin, lmax, smin = RELAXED if relax else (LUMA_MIN, LUMA_MAX, LUMA_STD_MIN)
    g = np.asarray(gray, dtype=np.float32) / 255.0
    if g.mean() < lmin:
        return "too dark"
    if g.mean() > lmax:
        return "too bright"
    if g.std() < smin:
        return "flat (no contrast)"
    if (sharp_pct is not None and raw_sharp is not None and video_median is not None
            and sharp_pct < BLUR_PCT and raw_sharp < BLUR_VS_MEDIAN * video_median
            and directional_ratio(gray) > BLUR_RATIO):
        return "directional motion blur"
    return None


def _frame_dominant_lc(rgb):
    """(L, C) of the frame's dominant colour, the way fingerprint._dominant_lch reads a shot's palette."""
    x = np.asarray(rgb, dtype=np.uint8)
    step = max(1, x.shape[1] // PALETTE_WIDTH)
    pal = mv.shot_palette([x[::step, ::step]], k=5)
    if not pal:
        return None
    L, C, _ = palette.hex_to_oklch(pal[0]["hex"])
    return L, C


def _palette_closeness(rgb, fp):
    """0..1, or None when the fingerprint has no colour layer metrics."""
    m = (((fp or {}).get("layers") or {}).get("colour") or {}).get("metrics") or {}
    L, C = _med(m.get("dominant_L")), _med(m.get("dominant_chroma"))
    if L is None and C is None:
        return None
    lc = _frame_dominant_lc(rgb)
    if lc is None:
        return None
    d = math.hypot((lc[0] - L) if L is not None else 0.0, (lc[1] - C) if C is not None else 0.0)
    return math.exp(-d / PALETTE_SCALE)


def look_measures(rgb, gray):
    """{grain_std, vignette_falloff, tone_contrast, sat} of one frame (0..1 units)."""
    g = np.asarray(gray, dtype=np.float32) / 255.0
    h, w = g.shape
    out = {"grain_std": 0.0, "vignette_falloff": 0.0}
    if h >= 6 and w >= 6:
        box = sum(g[i:h - 2 + i, j:w - 2 + j] for i in range(3) for j in range(3)) / 9.0
        out["grain_std"] = float((g[1:-1, 1:-1] - box).std())
        centre = float(g[h // 4:3 * h // 4, w // 4:3 * w // 4].mean())
        mask = np.ones_like(g, dtype=bool)
        mask[h // 6:5 * h // 6, w // 6:5 * w // 6] = False
        edge = float(g[mask].mean())
        out["vignette_falloff"] = float(min(1.0, max(0.0, 1 - edge / centre))) if centre > 1e-6 else 0.0
    out["tone_contrast"] = float(np.percentile(g, 95) - np.percentile(g, 5))
    x = np.asarray(rgb, dtype=np.float32)
    mx, mn = x.max(axis=-1), x.min(axis=-1)
    out["sat"] = float(np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0.0).mean())
    return out


def _look_closeness(rgb, gray, look):
    if not isinstance(look, dict):
        return None
    meas = look_measures(rgb, gray)
    ds = []
    for k in LOOK_KEYS:
        ref = _med(look.get(k))
        if ref is not None:
            ds.append(math.exp(-abs(meas[k] - ref) / max(0.25 * abs(ref), 0.01)))
    return sum(ds) / len(ds) if ds else None


def _metric(fp, layer, name):
    m = (((fp or {}).get("layers") or {}).get(layer) or {}).get("metrics") or {}
    return _med(m.get(name))


def _text_score(text, fp):
    """None when the style has no text or no detector ran; else 0..1."""
    h_ref = _metric(fp, "type", "text_height_pct")
    big = _metric(fp, "type", "big_text_per_min")
    if not ((h_ref is not None and h_ref > 0) or (big is not None and big > 0)) or text is None:
        return None
    boxes = text if isinstance(text, list) else ([[0, 0, 0, 0]] if text else [])
    if not boxes:
        return 0.3
    hs = [b[3] * 100 for b in boxes if isinstance(b, (list, tuple)) and len(b) == 4 and _num(b[3]) and b[3] > 0]
    if h_ref and hs:
        return math.exp(-abs(float(np.median(hs)) - h_ref) / max(h_ref, 1.0))
    return 1.0


def score_frame(rgb, gray, fingerprint=None, look_summary=None, faces=None, text=None, sharp_pct=None,
                raw_sharp=None, video_median=None, relax=False):
    """(score 0..1, why) for one frame. Pure. rgb (h, w, 3) uint8, gray (h, w) uint8. faces: True/False/None
    (None = no detector); text: a list of [x, y, w, h] fraction boxes, a bool or None; sharp_pct: the frame's
    sharpness percentile within its video (the raw sharpness is used when it is None). A hard-rejected frame
    scores 0 with why {"rejected": reason}. raw_sharp / video_median (raw Laplacian variance of the frame and
    the median of its video) enable the directional-blur reject; relax: see reject_reason."""
    why_r = reject_reason(gray, sharp_pct, raw_sharp, video_median, relax)
    if why_r:
        return 0.0, {"rejected": why_r}
    sharp = sharp_pct if sharp_pct is not None else _squash(sharpness(gray))
    parts = [("sharpness", sharp, 0.25 if fingerprint else 0.6)]
    if not fingerprint:
        parts.append(("colourful", colourfulness(rgb), 0.4))
    else:
        pal = _palette_closeness(rgb, fingerprint)
        parts.append(("palette", pal, 0.35) if pal is not None else ("colourful", colourfulness(rgb), 0.15))
        look = _look_closeness(rgb, gray, ((fingerprint.get("layers") or {}).get("look") or {}).get("metrics")
                               or look_summary)
        if look is not None:
            parts.append(("look", look, 0.15))
        ts = _text_score(text, fingerprint)
        if ts is not None:
            parts.append(("text", ts, 0.1))
        face = _metric(fingerprint, "broll", "face_share")
        if face is not None and face >= TALKING_FACE_SHARE and faces is not None:
            parts.append(("face", 1.0 if faces else 0.2, 0.15))
    total = sum(w for _, _, w in parts)
    score = sum(v * w for _, v, w in parts) / total
    return round(float(score), 4), {name: round(float(v), 4) for name, v, _ in parts}


def pick(cands, n, min_gap_s=MIN_GAP_S):
    """The best frame of each video, then the top n of those. When there are fewer videos than n, the next
    best frames fill the rest, never within min_gap_s seconds of a frame already chosen from the same video."""
    best = {}
    for c in cands:
        v = c.get("video")
        if v not in best or c["score"] > best[v]["score"]:
            best[v] = c
    out = sorted(best.values(), key=lambda c: -c["score"])[:max(0, n)]
    if len(out) < n:
        for c in sorted(cands, key=lambda c: -c["score"]):
            if len(out) >= n:
                break
            if c in out:
                continue
            if all(o.get("video") != c.get("video") or abs(o["t"] - c["t"]) >= min_gap_s for o in out):
                out.append(c)
    return out


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


def cut_intervals(cuts, fps):
    """Seconds spans a frame must stay out of: +-3 frames around every cut or jump, and a gradual
    transition's whole dur_f (plus the guard). cuts: measure.json dicts {t, kind, dur_f} or plain times."""
    guard = CUT_GUARD_FRAMES / max(fps, 1.0)
    out = []
    for c in cuts or []:
        t = c.get("t") if isinstance(c, dict) else c
        if not _num(t):
            continue
        end = float(t)
        if isinstance(c, dict) and c.get("kind") == "gradual" and _num(c.get("dur_f")):
            end += c["dur_f"] / max(fps, 1.0)
        out.append((float(t) - guard, end + guard))
    return out


def sample_times(duration, fps, cuts, k):
    """k times spread over [EDGE_S, duration - EDGE_S], none inside a cut interval (see cut_intervals)."""
    lo, hi = EDGE_S, duration - EDGE_S
    if hi <= lo or k <= 0:
        return []
    spans = cut_intervals(cuts, fps)
    out = []
    for i in range(k):
        t = lo + (hi - lo) * (i + 0.5) / k
        if not any(a <= t <= b for a, b in spans):
            out.append(round(t, 3))
    return out


def _read(p):
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _source(adir, measure):
    sj = _read(adir / "source.json")
    for v in ((sj or {}).get("source") if isinstance(sj, dict) else None, (measure or {}).get("source")):
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
    look = _read(pack / "look.json")
    look = (look.get("summary") or look.get("metrics")) if isinstance(look, dict) else None
    require_ffmpeg()
    det = mv._face_detector()
    cands, rejects = [], {}
    for vid in ids:
        measure = _read(adir / vid / "measure.json")
        src = _source(adir / vid, measure)
        if src is None:
            not_run.append(f"video:{vid}:no_source")
            continue
        info = mv.probe(src)
        w, h = mv._scaled_size(info, SCORE_WIDTH)
        frames = []
        for t in sample_times(info.get("duration_s") or 0, info["fps"], (measure or {}).get("cuts"), a.per_video):
            rgb = _grab_rgb(src, t, w, h)
            if rgb is not None:
                frames.append((t, rgb, (rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)).astype(np.uint8)))
        raws = [sharpness(g) for _, _, g in frames]
        pcts = sharpness_percentiles(raws)
        med = float(np.median(raws)) if raws else None
        if not frames:
            not_run.append(f"video:{vid}:no_frames")
            continue
        found = None
        for relax in (False, True):
            counts, kept = {}, []
            for (t, rgb, gray), raw, pct in zip(frames, raws, pcts):
                faces = bool(det(gray)) if det else None
                try:
                    text = mv.text_boxes(gray)
                except ImportError:
                    text = None
                score, why = score_frame(rgb, gray, fp, look, faces=faces, text=text, sharp_pct=pct,
                                         raw_sharp=raw, video_median=med, relax=relax)
                if "rejected" in why:
                    counts[why["rejected"]] = counts.get(why["rejected"], 0) + 1
                else:
                    kept.append({"video": vid, "t": t, "score": score, "why": why, "_src": str(src)})
            if kept:
                found = kept
                break
            rejects[vid] = counts
            if not relax:
                not_run.append(f"video:{vid}:all_rejected")  # then retried with the relaxed bars
        if found is None:
            not_run.append(f"video:{vid}:no_candidates")
        else:
            cands.extend(found)
    face_share = _metric(fp, "broll", "face_share")
    if det is None and face_share is not None and face_share > FACES_RELEVANT_SHARE:
        not_run.append("faces")
    top = pick(cands, a.n)
    if not top:
        print("no usable cover frame in any video (" + "; ".join(not_run[:4] or ["no frames"])
              + "); the old cover-candidates folder was left as it is", file=sys.stderr)
        return 1
    tmp_dir = pack / "cover-candidates.tmp"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir()
    rows = []
    try:
        for i, c in enumerate(top, 1):
            name = f"{i}-{re.sub(r'[^0-9A-Za-z_-]+', '_', str(c['video']))}-t{int(c['t'] * 1000)}.jpg"
            _save_jpg(c.pop("_src"), c["t"], tmp_dir / name)
            rows.append({"file": f"cover-candidates/{name}", **c})
        out_dir = pack / "cover-candidates"
        if out_dir.exists():
            shutil.rmtree(out_dir)
        os.replace(tmp_dir, out_dir)  # swap in only after every frame was written
    finally:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
    doc = {"candidates": rows, "not_run": not_run}
    if rejects:
        doc["rejects"] = rejects  # reject counts by reason, per video that had every frame rejected
    store.write_json(pack / "cover-candidates.json", doc)
    for r in rows:
        print(pack / r["file"])
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

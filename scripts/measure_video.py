#!/usr/bin/env python3
"""Base measurement of one video: cuts, cadence, shots, palette, faces, text, loudness, pace.

Reads the video (never modifies it), decodes small gray frames through one ffmpeg pipe, and writes
<out>/measure.json (schema "measure/1") and <out>/frames.csv (one row per frame).

Usage:
  measure_video.py <video> --out <dir> [--words words.json] [--from-analysis <video-analysis dir>]
                   [--faces auto|off] [--text auto|ocr|off] [--analysis-width 320]

What is measured (all local, free):
  - cuts: a frame-to-frame distance d = max(histogram distance, mean pixel change), both 0..1
    (the histogram distance is a 1-D earth mover's distance, so a slow dissolve stays small per
    frame). Hard cut: d above median + 6 MAD of the whole clip and above 0.25. Gradual cut: a run
    of at least 6 frames of sustained moderate change whose summed distance is above 0.5 and whose
    end frames differ by more than 0.35. Jump cut (kind "jump"): a single-frame spike of the pixel
    change inside one shot (a talking-head jump), >= 5x the median change of the MOVING frames
    around it and >= 0.03; stepped or frame-rate-converted hold frames do not count. A spike whose
    changed pixels fit in < 15 % of the frame is a screen repaint: it goes to "ui_changes", never to
    "cuts". NOTE: "cuts" entries of kind "jump" are EDIT POINTS, not shot boundaries (they do not
    start a new shot); only hard and gradual cuts split shots.
  - cadence: how many frames are real new pictures (on-twos animation, 24 fps in a 30 fps file).
    Measured only where something moves, so a static clip does not read as "all repeats".
  - shots: the spans between cuts, each with an OKLab palette (palette.py converters), the share of
    sampled frames with a face and the face headroom (fraction of the frame above the face box).
  - faces: OpenCV Haar frontal cascade (needs cv2). Text: morphological-gradient text boxes (needs
    cv2); OCR only when rapidocr_onnxruntime imports, else tesseract on PATH, else "not_run".
    A measurement that cannot run is listed in "not_run", never a crash.
  - loudness: ffmpeg ebur128 (integrated LUFS, true peak dBTP).
  - words_per_s: from --words (a list, or {"words": [...]}, seconds or start_ms/end_ms).
  - --from-analysis: reads a video-analysis folder (measurements.json edit_points + pacing,
    audio.json loudness, transcript.json words) and imports only the fields that exist; cuts then
    come from there. What was imported is recorded under "imported".

Exit: 0 done, 1 error, 2 bad input, 3 ffmpeg missing (one line, with an install hint).
"""
import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import palette  # noqa: E402
import store  # noqa: E402
from cut import MissingTool, require_ffmpeg  # noqa: E402

SCHEMA = "measure/1"
HARD_MIN = 0.25         # a hard cut needs at least this much change
HARD_MAD_K = 6.0        # ... and to stand out of the clip's own noise by this many MAD
GRAD_MIN_RUN = 6        # a gradual transition lasts at least this many frames
GRAD_SUM_MIN = 0.5      # ... changes this much in total
GRAD_ENDS_MIN = 0.35    # ... and its two ends differ by more than this
CHANGE_PX = 8           # a pixel counts as changed above this many gray levels
REPEAT_FRAC = 0.0005    # fewer changed pixels than this fraction = a repeated frame
HEATMAP = 8


# ---------------------------------------------------------------- optional deps

def _cv2():
    """cv2, or None when it is not installed (a None entry in sys.modules also reads as missing)."""
    try:
        import cv2
        return cv2
    except ImportError:
        return None


def _ocr_available():
    """True when rapidocr_onnxruntime is importable or tesseract is on PATH (cheap: builds nothing)."""
    try:
        import importlib.util
        if importlib.util.find_spec("rapidocr_onnxruntime") is not None:
            return True
    except (ImportError, ValueError):
        pass
    return shutil.which("tesseract") is not None


def _ocr_engine():
    """("rapidocr", engine) | ("tesseract", path) | None. Never downloads anything."""
    if not _ocr_available():
        return None
    try:
        from rapidocr_onnxruntime import RapidOCR
        return "rapidocr", RapidOCR()
    except Exception:
        pass
    t = shutil.which("tesseract")
    return ("tesseract", t) if t else None


# ---------------------------------------------------------------- ffmpeg / ffprobe

class InputError(RuntimeError):
    """The input is not a readable video (the CLI exits 2)."""


def probe(video):
    """{width, height, fps, vfr, notes, duration_s, has_audio} from ffprobe. fps is avg_frame_rate
    (r_frame_rate only when avg is missing); above 120 it is clamped to 30 with note "vfr_clamped";
    vfr is True when avg and r differ."""
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                        "stream=codec_type,width,height,r_frame_rate,avg_frame_rate,duration:format=duration",
                        "-of", "json", str(video)], capture_output=True, text=True, encoding="utf-8")
    if r.returncode != 0:
        raise InputError("not a readable video: " + (r.stderr.strip().splitlines() or ["?"])[-1])
    try:
        doc = json.loads(r.stdout or "{}")
    except ValueError:
        raise InputError("not a readable video: ffprobe gave no data")
    vs = [s for s in doc.get("streams", []) if s.get("codec_type") == "video"]
    if not vs:
        raise InputError("not a readable video: the file has no video stream")
    v = vs[0]

    def frac(s):
        try:
            a, b = str(s).split("/")
            return float(a) / float(b) if float(b) else 0.0
        except (ValueError, ZeroDivisionError):
            return 0.0
    avg, rfr = frac(v.get("avg_frame_rate")), frac(v.get("r_frame_rate"))
    fps = avg or rfr or 25.0
    notes = []
    vfr = bool(avg and rfr and abs(avg - rfr) > 0.01 * max(avg, rfr))
    if fps > 120:
        fps = 30.0
        notes.append("vfr_clamped")
    dur = None
    for raw in (v.get("duration"), (doc.get("format") or {}).get("duration")):
        try:
            dur = float(raw)
            break
        except (TypeError, ValueError):
            pass
    return {"width": int(v["width"]), "height": int(v["height"]), "fps": fps, "vfr": vfr, "notes": notes,
            "duration_s": dur,
            "has_audio": any(s.get("codec_type") == "audio" for s in doc.get("streams", []))}


def _scaled_size(info, width):
    w = max(8, int(width))
    h = max(2, int(round(w * info["height"] / info["width"] / 2.0)) * 2)
    return w, h


def _decode(video, info, width, pix_fmt, channels):
    """Frames through an ffmpeg pipe read straight into one preallocated array (peak memory is the
    array alone; it grows only if the stream holds more frames than the probe promised)."""
    w, h = _scaled_size(info, width)
    shape = (h, w) if channels == 1 else (h, w, channels)
    fsize = w * h * channels
    guess = int(math.ceil((info.get("duration_s") or 0) * info["fps"])) + 2
    buf = np.empty((max(guess, 16),) + shape, dtype=np.uint8)
    n = 0
    with tempfile.TemporaryFile() as err:
        proc = subprocess.Popen(["ffmpeg", "-v", "error", "-nostdin", "-i", str(video), "-an", "-vf",
                                 f"scale={w}:{h}", "-r", f"{info['fps']:.6f}", "-pix_fmt", pix_fmt,
                                 "-f", "rawvideo", "-"], stdout=subprocess.PIPE, stderr=err)
        try:
            while True:
                if n == len(buf):
                    nb = np.empty((len(buf) + max(64, len(buf) // 8),) + shape, dtype=np.uint8)
                    nb[:n] = buf
                    buf = nb
                view = memoryview(buf[n].reshape(-1))
                got = 0
                while got < fsize:
                    k = proc.stdout.readinto(view[got:])
                    if not k:
                        break
                    got += k
                if got < fsize:
                    break
                n += 1
        finally:
            proc.stdout.close()
            rc = proc.wait()
        if rc != 0 and n == 0:
            err.seek(0)
            msg = err.read().decode("utf-8", "replace").strip().splitlines()
            raise RuntimeError("ffmpeg could not decode the video: " + (msg[-1] if msg else "unknown error"))
    return buf[:n]


def decode_gray(video, width=320):
    """(frames (n, h, w) uint8 gray, fps) through one ffmpeg pipe. A variable-frame-rate source is
    made constant by ffmpeg (repeated frames), which cadence() then reads as repeats."""
    info = probe(video)
    return _decode(video, info, width, "gray", 1), info["fps"]


def decode_rgb(video, width=64):
    info = probe(video)
    return _decode(video, info, width, "rgb24", 3)


def loudness(video):
    """{"lufs", "tp"} from ffmpeg ebur128, or None when there is no audio or it is silent."""
    r = subprocess.run(["ffmpeg", "-nostdin", "-hide_banner", "-i", str(video), "-vn", "-af",
                        "ebur128=peak=true", "-f", "null", "-"], capture_output=True, text=True, encoding="utf-8",
                       errors="replace")
    if r.returncode != 0:
        return None
    tail = r.stderr.split("Summary:")[-1]
    m_i = re.search(r"\bI:\s*(-?[\d.]+|-?inf)\s*LUFS", tail)
    m_p = re.search(r"\bPeak:\s*(-?[\d.]+|-?inf)\s*dBFS", tail)

    def num(m):
        try:
            v = float(m.group(1)) if m else None
        except ValueError:
            return None
        return v if v is not None and math.isfinite(v) else None
    lufs, tp = num(m_i), num(m_p)
    return None if lufs is None else {"lufs": round(lufs, 1), "tp": None if tp is None else round(tp, 1)}


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------- frame-to-frame signals

def frame_distances(frames):
    """Per frame i (i >= 1, index 0 is 0): hist (1-D earth mover's distance of the 64-bin gray
    histograms, 0..1), pix (mean absolute pixel change, 0..1), changed (fraction of pixels that
    moved by more than CHANGE_PX gray levels)."""
    n = len(frames)
    hist = np.zeros(n)
    pix = np.zeros(n)
    changed = np.zeros(n)
    if n < 2:
        return hist, pix, changed
    cdf = np.zeros((n, 64))
    for i in range(n):
        cdf[i] = np.cumsum(np.bincount((frames[i] >> 2).ravel(), minlength=64)) / frames[i].size
    hist[1:] = np.abs(cdf[1:] - cdf[:-1]).mean(axis=1)
    for i in range(1, n):
        d = np.abs(frames[i].astype(np.int16) - frames[i - 1].astype(np.int16))
        pix[i] = d.mean() / 255.0
        changed[i] = (d > CHANGE_PX).mean()
    return hist, pix, changed


def _med_mad(x):
    med = float(np.median(x)) if len(x) else 0.0
    mad = float(np.median(np.abs(x - med))) if len(x) else 0.0
    return med, mad


def detect_cuts(frames, fps):
    """[{"t", "kind": "hard"|"gradual"|"jump", "dur_f"}] sorted by time (see the module docstring).
    "jump" entries are edit points inside a shot, not shot boundaries."""
    return detect_cuts_ex(frames, fps)[0]


def detect_cuts_ex(frames, fps):
    """(cuts, ui_changes): ui_changes are spikes confined to a small part of the frame (a screen
    recording repaint); they are never cuts."""
    n = len(frames)
    if n < 2:
        return [], []
    hist, pix, changed = frame_distances(frames)
    ui = []
    return _cuts_from_distance(np.maximum(hist, pix), frames, fps, pix, changed, ui), ui


JUMP_MIN = 0.03   # a jump cut changes at least this much of the picture in one frame
JUMP_X = 5.0      # ... and at least this many times the local median change (+-15 frames)
GRAD_MAX_S = 2.0  # a longer run of change is a slow move or fade, not a transition


UI_AREA = 0.15    # a spike whose changed pixels fit in less than this share of the frame may be a repaint
UI_FILL = 0.6     # ... if they also fill this much of their bounding box (a rectangle)
JUMP_MARGIN = 2.0  # a jump is at least this many times the largest ordinary moving frame around it


def _content_bbox(region):
    """Bounding box (x0, y0, x1, y1) of what differs from the region's border colour, or None."""
    if region.size == 0:
        return None
    r = region.astype(np.int16)
    border = np.concatenate([r[0], r[-1], r[:, 0], r[:, -1]])
    m = np.abs(r - int(np.median(border))) > 24
    if m.sum() < 0.01 * m.size:
        return None
    ys, xs = np.where(m)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0.0


def _is_ui_repaint(prev, cur, mask):
    """A local rectangular repaint: the changed pixels are < 15 % of the frame, fill >= 60 % of their
    bounding box, and what is drawn inside is not a subject that moved or scaled (its content box is
    the same before and after, or there is none in either)."""
    ys, xs = np.where(mask)
    if not len(ys):
        return False, 0.0
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    box = (y1 - y0) * (x1 - x0)
    area = box / mask.size
    if area >= UI_AREA or len(ys) / box < UI_FILL:
        return False, float(area)
    cp, cc = _content_bbox(prev[y0:y1, x0:x1]), _content_bbox(cur[y0:y1, x0:x1])
    if cp is None and cc is None:
        return True, float(area)
    if cp is None or cc is None:
        return False, float(area)
    return _iou(cp, cc) >= 0.5, float(area)


def _jump_cuts(pix, changed, frames, skip, fps):
    """(jumps, ui_changes): single-frame spikes of the pixel change.

    The reference is the ordinary MOVING frames within +-15 frames (at least 4): not hold frames of
    stepped or frame-rate-converted footage, not hard/gradual cuts (+-2 frames), and not other
    candidate spikes (so two jumps ten frames apart do not hide each other). A spike must be >= 5x
    the median of that reference, >= 0.03, >= 2x its largest frame, and alone (not the edge of
    sustained motion). A rectangular local repaint is a ui_change; a subject that moved or scaled is a jump."""
    n = len(pix)
    near_cut = {k + d for k in skip for d in range(-2, 3)}

    def window(i):
        return [k for k in range(max(1, i - 15), min(n, i + 16))
                if k != i and k not in near_cut and changed[k] >= REPEAT_FRAC]

    cand = set()
    for i in range(1, n):
        if pix[i] < JUMP_MIN or i in near_cut:
            continue
        idx = window(i)
        if len(idx) >= 4 and pix[i] >= JUMP_X * float(np.median(pix[idx])):
            cand.add(i)
    jumps, uis = [], []
    for i in sorted(cand):
        idx = [k for k in window(i) if k not in cand]
        if len(idx) < 4:
            continue
        win = pix[idx]
        if pix[i] < JUMP_X * float(np.median(win)) or pix[i] < JUMP_MARGIN * float(win.max()):
            continue
        nb = max(pix[i - 1] if i > 1 else 0.0, pix[i + 1] if i + 1 < n else 0.0)
        if nb > pix[i] / 3:  # a single-frame spike, not the edge of sustained motion
            continue
        mask = np.abs(frames[i].astype(np.int16) - frames[i - 1].astype(np.int16)) > CHANGE_PX
        if i >= 2:  # drop what was already changing a frame earlier (steady motion elsewhere)
            prev = np.abs(frames[i - 1].astype(np.int16) - frames[i - 2].astype(np.int16)) > CHANGE_PX
            dil = prev.copy()
            for dy in range(-3, 4):
                for dx in range(-3, 4):
                    dil |= np.roll(np.roll(prev, dy, axis=0), dx, axis=1)
            mask &= ~dil
        ui, area = _is_ui_repaint(frames[i - 1], frames[i], mask)
        rec = {"t": round(i / fps, 4), "kind": "ui_change" if ui else "jump", "dur_f": 1}
        if ui:
            rec["area"] = round(area, 4)
            uis.append(rec)
        else:
            jumps.append(rec)
    return jumps, uis


def _cuts_from_distance(d, frames, fps, pix=None, changed=None, ui_out=None):
    n = len(d)
    med, mad = _med_mad(d[1:])
    hard_thr = max(med + HARD_MAD_K * mad, HARD_MIN)
    hard = [i for i in range(1, n) if d[i] > hard_thr]
    merged = []  # two flagged frames within 2 frames of each other are one cut (keep the larger)
    for i in hard:
        if merged and i - merged[-1] <= 2:
            if d[i] > d[merged[-1]]:
                merged[-1] = i
        else:
            merged.append(i)
    cuts = [{"t": round(i / fps, 4), "kind": "hard", "dur_f": 1} for i in merged]

    moderate = max(med + 3 * mad, 0.02)
    dd = d.copy()
    dd[0] = 0.0
    for i in merged:
        dd[i] = 0.0
    i = 1
    while i < n:
        if dd[i] > moderate:
            j = i
            while j + 1 < n and dd[j + 1] > moderate:
                j += 1
            run = j - i + 1
            if GRAD_MIN_RUN <= run <= GRAD_MAX_S * fps and dd[i:j + 1].sum() > GRAD_SUM_MIN:
                a, b = frames[i - 1], frames[j]
                ends = max(_hist_emd(a, b), np.abs(a.astype(np.int16) - b.astype(np.int16)).mean() / 255.0)
                if ends > GRAD_ENDS_MIN:
                    cuts.append({"t": round(i / fps, 4), "kind": "gradual", "dur_f": run})
            i = j + 1
        else:
            i += 1
    if pix is not None:
        skip = list(merged)
        for c in cuts:
            if c["kind"] == "gradual":
                f0 = int(round(c["t"] * fps))
                skip.extend(range(f0, f0 + c["dur_f"] + 1))
        if changed is None:
            changed = frame_distances(frames)[2]
        jumps, uis = _jump_cuts(pix, changed, frames, skip, fps)
        cuts += jumps
        if ui_out is not None:
            ui_out.extend(uis)
    return sorted(cuts, key=lambda c: c["t"])


def _hist_emd(a, b):
    ca = np.cumsum(np.bincount((a >> 2).ravel(), minlength=64)) / a.size
    cb = np.cumsum(np.bincount((b >> 2).ravel(), minlength=64)) / b.size
    return float(np.abs(ca - cb).mean())


def cadence(frames, fps):
    """{"repeat_fraction", "unique_fps" (None when under 4 frames move), "on_twos", "moving_frames"},
    over the frames where something moves."""
    n = len(frames)
    if n < 4:
        return {"repeat_fraction": 0.0, "unique_fps": None, "on_twos": False, "moving_frames": 0}
    _, _, ch = frame_distances(frames)
    return _cadence_from_changed(ch, frames, fps)


def _cadence_from_changed(ch, frames, fps):
    n = len(frames)
    out = {"repeat_fraction": 0.0, "unique_fps": None, "on_twos": False, "moving_frames": 0}
    if n < 4:
        return out
    # frame i moves when it differs from the frame two back (so a held frame between two moving ones counts)
    two = np.zeros(n)
    for i in range(2, n):
        two[i] = (np.abs(frames[i].astype(np.int16) - frames[i - 2].astype(np.int16)) > CHANGE_PX).mean()
    moving = [i for i in range(2, n) if two[i] >= REPEAT_FRAC]
    out["moving_frames"] = len(moving)
    if len(moving) < 4:  # nothing (or too little) moves: the true frame rate is unknown, not "fps"
        return out
    flags = np.array([1 if ch[i] < REPEAT_FRAC else 0 for i in moving])
    rf = float(flags.mean())
    out["repeat_fraction"] = round(rf, 4)
    out["unique_fps"] = round(float(fps) * (1 - rf), 2)
    # on twos: about half the frames repeat and the pattern alternates (same flag two frames apart)
    ok = bool((np.diff(flags) != 0).mean() > 0.9) if len(flags) > 3 and 0.4 <= rf <= 0.6 else False  # moving frames only
    out["on_twos"] = ok
    return out


# ---------------------------------------------------------------- palette (OKLab, palette.py converters)

def _rgb_to_oklab(rgb):
    """One (r, g, b) 0..255 triple -> OKLab, via palette.py's converters."""
    return palette._lin_to_oklab(*(palette._lin(c / 255.0) for c in rgb))


def _oklab_to_hex(L, a, b):
    lin = palette._oklab_to_lin(L, a, b)
    return palette._rgb_hex(tuple(palette._gam(max(0.0, min(1.0, c))) for c in lin))


def shot_palette(rgb_frames, k=5):
    """[{"hex", "share"}] largest first: weighted k-means in OKLab on the frames' distinct colours."""
    px = np.asarray(rgb_frames, dtype=np.uint8).reshape(-1, 3)
    if len(px) == 0:
        return []
    q = (px >> 3).astype(np.int32)  # 5 bits per channel
    key = (q[:, 0] << 10) | (q[:, 1] << 5) | q[:, 2]
    uniq, counts = np.unique(key, return_counts=True)
    cols = np.stack([(uniq >> 10) & 31, (uniq >> 5) & 31, uniq & 31], axis=1) * 8 + 4
    lab = np.array([_rgb_to_oklab(c) for c in cols])
    w = counts.astype(float)
    k = max(1, min(k, len(uniq)))
    centers = [lab[int(np.argmax(w))]]  # deterministic farthest-point start
    while len(centers) < k:
        dist = np.min([((lab - c) ** 2).sum(axis=1) for c in centers], axis=0) * w
        centers.append(lab[int(np.argmax(dist))])
    centers = np.array(centers)
    for _ in range(25):
        assign = np.argmin(((lab[:, None, :] - centers[None]) ** 2).sum(axis=2), axis=1)
        new = centers.copy()
        for c in range(k):
            m = assign == c
            if m.any():
                new[c] = (lab[m] * w[m, None]).sum(axis=0) / w[m].sum()
        if np.allclose(new, centers, atol=1e-5):
            break
        centers = new
    assign = np.argmin(((lab[:, None, :] - centers[None]) ** 2).sum(axis=2), axis=1)
    total = w.sum()
    out = [{"hex": _oklab_to_hex(*centers[c]), "share": round(float(w[assign == c].sum() / total), 4)}
           for c in range(k) if (assign == c).any()]
    return sorted(out, key=lambda e: -e["share"])


# ---------------------------------------------------------------- faces and text (cv2)

def _haar():
    """The OpenCV Haar frontal-face classifier, or None when cv2 or its cascade file is missing
    (some cv2 builds ship without cv2/data)."""
    cv2 = _cv2()
    if cv2 is None:
        return None
    try:
        path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    except AttributeError:
        return None
    if not os.path.isfile(path):
        return None
    casc = cv2.CascadeClassifier(path)
    return None if casc.empty() else casc


def _mediapipe_detector():
    """Legacy mediapipe face detection (mp.solutions.face_detection) using only the model inside the
    installed wheel, or None. Never downloads: any failure means "not available"."""
    try:
        import mediapipe as mp
        fd = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
    except Exception:
        return None

    def detect(gray):
        h, w = gray.shape
        rgb = np.repeat(gray[:, :, None], 3, axis=2)
        res = fd.process(rgb)
        out = []
        for d in (res.detections or []):
            bb = d.location_data.relative_bounding_box
            out.append((bb.xmin * w, bb.ymin * h, bb.width * w, bb.height * h))
        return out
    return detect


def _face_detector():
    """A callable gray frame -> [(x, y, w, h) pixels]: Haar if its cascade exists, else mediapipe's
    bundled model, else None (faces are then "not_run")."""
    casc = _haar()
    if casc is not None:
        def detect(gray):
            w = gray.shape[1]
            return [tuple(b) for b in casc.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
                                                             minSize=(max(24, w // 16),) * 2)]
        return detect
    return _mediapipe_detector()


def face_track(frames, fps, every=3):
    """[{"t", "box": [x, y, w, h]}] (fractions of the frame) for every `every`-th gray frame that
    shows a face (the largest one): Haar frontal cascade, else mediapipe's bundled model."""
    det = _face_detector()
    if det is None:
        raise ImportError("no face detector available (Haar cascade or mediapipe)")
    out = []
    for i in range(0, len(frames), max(1, every)):
        f = frames[i]
        h, w = f.shape
        found = det(f)
        if len(found):
            x, y, bw, bh = max(found, key=lambda b: b[2] * b[3])
            out.append({"t": round(i / fps, 4), "box": [round(x / w, 4), round(y / h, 4),
                                                       round(bw / w, 4), round(bh / h, 4)]})
    return out


def text_boxes(frame):
    """[[x, y, w, h]] (fractions of the frame) of text-like regions: morphological gradient, Otsu,
    horizontal closing, connected components of aspect >= 2 and height 2-30 % of the frame."""
    cv2 = _cv2()
    if cv2 is None:
        raise ImportError("cv2 is not installed")
    h, w = frame.shape
    grad = cv2.morphologyEx(frame, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    _, bw = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (max(9, w // 32), 1)))
    nlab, _, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    out = []
    for i in range(1, nlab):
        x, y, bwid, bh, area = (int(v) for v in stats[i])
        if bh <= 0 or bwid / bh < 2.0 or not (0.02 * h <= bh <= 0.30 * h) or bwid < 0.03 * w:
            continue
        if area < 0.3 * bwid * bh:  # a sparse smear is not a line of text
            continue
        out.append([round(x / w, 4), round(y / h, 4), round(bwid / w, 4), round(bh / h, 4)])
    return out


def text_box_records(frame, t):
    """[{"t", "box": [x, y, w, h] (fractions), "h_pct": box height as % of the frame}] for one frame."""
    return [{"t": round(float(t), 4), "box": b, "h_pct": round(b[3] * 100, 2)} for b in text_boxes(frame)]


def _stats(vals):
    if not len(vals):
        return {"median": None, "p10": None, "p90": None, "n": 0}
    a = np.asarray(vals, float)
    return {"median": round(float(np.median(a)), 4), "p10": round(float(np.percentile(a, 10)), 4),
            "p90": round(float(np.percentile(a, 90)), 4), "n": int(len(a))}


def _run_ocr(engine, frames_idx, frames, fps):
    cv2 = _cv2()
    kind, eng = engine
    out = []
    for i in frames_idx:
        img = cv2.resize(frames[i], None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        try:
            if kind == "rapidocr":
                res, _ = eng(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
                text = " ".join(r[1] for r in (res or []))
            else:
                with tempfile.TemporaryDirectory() as td:
                    p = Path(td) / "f.png"
                    cv2.imwrite(str(p), img)
                    r = subprocess.run([eng, str(p), "stdout"], capture_output=True, text=True,
                                       encoding="utf-8", errors="replace")
                    text = r.stdout.strip()
        except Exception:
            continue
        if text.strip():
            out.append({"t": round(i / fps, 3), "text": text.strip()})
    return out


# ---------------------------------------------------------------- shots and events

def build_shots(cuts, duration_s, rgb, fps, faces):
    # shot k runs from the end of cut k-1 (after a dissolve) to the start of cut k
    starts, ends = [0.0], []
    cuts = [c for c in cuts if c["kind"] != "jump"]  # a jump cut stays inside its shot
    for c in cuts:
        ends.append(c["t"])
        starts.append(c["t"] + (c["dur_f"] / fps if c["kind"] == "gradual" else 0.0))
    ends.append(duration_s)
    shots = []
    for t0, t1 in zip(starts, ends):
        if t1 <= t0 and shots:
            continue
        pal = []
        if rgb is not None and len(rgb):
            a, b = int(t0 * fps), max(int(t0 * fps) + 1, int(math.ceil(t1 * fps)))
            sel = rgb[a:b:max(1, (b - a) // 12)]
            if len(sel):
                pal = shot_palette(sel)
        inside = [f for f in faces["tracks"] if t0 <= f["t"] < t1] if faces["ran"] else []
        sampled = [t for t in faces["sample_t"] if t0 <= t < t1] if faces["ran"] else []
        share = round(len(inside) / len(sampled), 4) if sampled else None
        head = round(float(np.median([f["box"][1] for f in inside])), 4) if inside else None
        shots.append({"t0": round(t0, 4), "t1": round(t1, 4), "dur_s": round(max(0.0, t1 - t0), 4),
                      "palette": pal, "face_share": share, "headroom": head})
    return shots


def event_rate(cuts, changed, fps, duration_s):
    """Visual events per 1 s bin: cuts plus motion onsets (change after >= 3 still frames)."""
    nbin = max(1, int(math.ceil(duration_s)))
    bins = [0] * nbin
    for c in cuts:
        bins[min(nbin - 1, int(c["t"]))] += 1
    still = 3
    at_cut = {int(round(c["t"] * fps)) for c in cuts}
    for i in range(1, len(changed)):
        if i in at_cut:
            continue  # a cut is already counted once
        if changed[i] >= REPEAT_FRAC and (i - 1 < still or all(changed[i - still:i] < REPEAT_FRAC)):
            bins[min(nbin - 1, int(i / fps))] += 1
    return bins


def longest_hold(changed, fps):
    best = run = 1
    for i in range(1, len(changed)):
        run = run + 1 if changed[i] < REPEAT_FRAC else 1
        best = max(best, run)
    return round(best / fps, 3) if len(changed) else 0.0


# ---------------------------------------------------------------- imports

def _num(v):
    """A finite float or None (bad values are skipped, never a traceback)."""
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def load_words(path):
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    ws = doc.get("words", []) if isinstance(doc, dict) else doc
    if not isinstance(ws, list):
        return []
    out = []
    for w in ws:
        if not isinstance(w, dict):
            continue
        if "start" in w:
            a, b = _num(w.get("start")), _num(w.get("end", w.get("start")))
        elif "start_ms" in w:
            a, b = _num(w["start_ms"]), _num(w.get("end_ms", w["start_ms"]))
            a, b = (None if a is None else a / 1000), (None if b is None else b / 1000)
        else:
            continue
        if a is None:
            continue
        out.append((a, a if b is None else b))
    return out


def _first(d, keys):
    for k in keys:
        v = d.get(k)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return float(v)
    return None


def import_analysis(adir, fps):
    """Fields of a video-analysis folder that exist: {cuts?, pacing?, loudness?, words?, imported}."""
    adir = Path(adir)
    got, imported = {}, {}

    def load(name):
        p = adir / name
        if not p.is_file():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
    m = load("measurements.json")
    if isinstance(m, dict):
        eps = m.get("edit_points")
        if isinstance(eps, list):
            cuts = []
            for e in eps:
                if not isinstance(e, dict) or not isinstance(e.get("t_s"), (int, float)):
                    continue
                typ = str(e.get("type") or e.get("kind") or "").lower()
                grad = any(s in typ for s in ("dissolve", "fade", "gradual", "crossfade", "wipe"))
                cuts.append({"t": round(float(e["t_s"]), 4), "kind": "gradual" if grad else "hard",
                             "dur_f": int(e["dur_f"]) if isinstance(e.get("dur_f"), (int, float)) else (
                                 max(2, int(round(fps * 0.4))) if grad else 1)})
            got["cuts"] = sorted(cuts, key=lambda c: c["t"])
            imported["cuts"] = len(cuts)
        if "pacing" in m:
            got["pacing"] = m["pacing"]
            imported["pacing"] = True
    a = load("audio.json")
    if isinstance(a, dict) and isinstance(a.get("loudness"), dict):
        lo = a["loudness"]
        lufs = _first(lo, ("lufs", "integrated_lufs", "integrated", "I", "i"))
        tp = _first(lo, ("tp", "true_peak", "true_peak_dbtp", "peak", "peak_dbfs"))
        if lufs is not None:
            got["loudness"] = {"lufs": round(lufs, 1), "tp": None if tp is None else round(tp, 1)}
            imported["loudness"] = True
    t = load("transcript.json")
    if isinstance(t, dict) and isinstance(t.get("words"), list):
        ws = [(_num(w["start"]), _num(w.get("end", w["start"]))) for w in t["words"]
              if isinstance(w, dict) and _num(w.get("start")) is not None]
        if ws:
            got["words"] = ws
            imported["words"] = len(ws)
    got["imported"] = imported
    return got


# ---------------------------------------------------------------- the measurement

def measure(video, out_dir, *, words=None, from_analysis=None, faces="auto", text="auto", analysis_width=320):
    """Measure one video and write measure.json + frames.csv into out_dir. Returns the document."""
    video, out_dir = Path(video), Path(out_dir)
    if not video.is_file():
        raise FileNotFoundError(f"video not found: {video}")
    sha = sha256_of(video)
    info = probe(video)
    frames, fps = decode_gray(video, analysis_width)
    n = len(frames)
    duration = info["duration_s"] if info["duration_s"] else (n / fps if fps else 0.0)
    not_run = []

    hist, pix, changed = frame_distances(frames)
    imp = import_analysis(from_analysis, fps) if from_analysis else {"imported": {}}
    cuts = imp.get("cuts")
    ui_changes = []
    own = _cuts_from_distance(np.maximum(hist, pix), frames, fps, pix, changed, ui_changes) if n >= 2 else []
    if cuts is None:
        cuts = own
    cad = _cadence_from_changed(changed, frames, fps) if n >= 4 else cadence(frames, fps)

    cv2 = _cv2()
    sample = list(range(0, n, 3))
    faces_info = {"ran": False, "tracks": [], "sample_t": [t / fps for t in sample]}
    faces_summary = {"ran": False, "frames_sampled": 0, "frames_with_face": 0, "share": None,
                     "box_median": None, "headroom": _stats([])}
    if faces == "off" or n == 0 or _face_detector() is None:
        not_run.append("faces")
    else:
        tr = face_track(frames, fps, every=3)
        faces_info.update(ran=True, tracks=tr)
        boxes = np.array([f["box"] for f in tr]) if tr else None
        faces_summary.update(ran=True, frames_sampled=len(sample), frames_with_face=len(tr),
                             share=round(len(tr) / len(sample), 4) if sample else None,
                             box_median=[round(float(v), 4) for v in np.median(boxes, axis=0)] if tr else None,
                             headroom=_stats([b[1] for b in boxes] if tr else []))

    heat = np.zeros((HEATMAP, HEATMAP))
    heights, ocr, tboxes = [], "not_run", []
    if text == "off" or cv2 is None or n == 0:
        not_run.append("text")
    else:
        idx = sorted(set(np.linspace(0, n - 1, min(n, 40)).astype(int).tolist()))
        with_boxes = []
        for i in idx:
            recs = text_box_records(frames[i], i / fps)
            tboxes.extend(recs)
            bxs = [r["box"] for r in recs]
            if bxs:
                with_boxes.append(i)
            for x, y, w, h in bxs:
                heights.append(h * 100)
                x0, x1 = x * HEATMAP, (x + w) * HEATMAP
                y0, y1 = y * HEATMAP, (y + h) * HEATMAP
                for gy in range(HEATMAP):
                    for gx in range(HEATMAP):
                        ox = max(0.0, min(gx + 1, x1) - max(gx, x0))
                        oy = max(0.0, min(gy + 1, y1) - max(gy, y0))
                        heat[gy, gx] += ox * oy
        heat = heat / max(1, len(idx))
        if not _ocr_available():
            if text == "ocr" or with_boxes:
                not_run.append("ocr")
        else:
            ocr = []  # OCR could run: an empty list means it found nothing
            if text == "ocr" or with_boxes:
                eng = _ocr_engine()
                if eng is not None:
                    ocr = _run_ocr(eng, with_boxes[:6] if with_boxes else idx[:6], frames, fps)

    rgb = None
    try:
        rgb = decode_rgb(video, 64)
    except RuntimeError:
        not_run.append("palette")
    shots = build_shots(cuts, duration, rgb, fps, faces_info)

    lo = imp.get("loudness")
    if lo is None:
        lo = loudness(video) if info["has_audio"] else None
    if lo is None:
        not_run.append("loudness")
        lo = {"lufs": None, "tp": None}

    wl = imp.get("words")
    if words:
        wl = load_words(words)
    words_per_s = None
    if wl is not None:
        words_per_s = round(len(wl) / duration, 3) if duration > 0 else None

    doc = {
        "schema": SCHEMA,
        "video": {"id": sha[:12], "sha256": sha, "duration_s": round(float(duration), 4),
                  "fps": round(float(fps), 4), "width": info["width"], "height": info["height"],
                  "vfr": info["vfr"]},
        "notes": info["notes"],
        "cuts": cuts,
        "ui_changes": ui_changes,
        "shots": shots,
        "cadence": cad,
        "faces_summary": faces_summary,
        "text": {"boxes_heatmap": [[round(float(v), 4) for v in row] for row in heat],
                 "boxes": tboxes, "size_pct_h": _stats(heights), "ocr": ocr},
        "loudness": lo,
        "words_per_s": words_per_s,
        "visual_event_rate": event_rate(cuts, changed, fps, duration) if n else [],
        "longest_hold_s": longest_hold(changed, fps),
        "not_run": not_run,
        "imported": imp.get("imported", {}),
    }
    if "pacing" in imp:
        doc["pacing"] = imp["pacing"]

    problems = validate(doc)
    if problems:
        raise RuntimeError("internal: measure.json failed validation: " + "; ".join(problems))
    if sha256_of(video) != sha:
        raise RuntimeError("the source video changed while it was measured")
    out_dir.mkdir(parents=True, exist_ok=True)
    store.write_json(out_dir / "measure.json", doc)
    _write_csv(out_dir / "frames.csv", fps, hist, pix, changed)
    return doc


def _write_csv(path, fps, hist, pix, changed):
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["frame", "t_s", "hist_dist", "pix_change", "changed_frac", "repeat"])
            for i in range(len(hist)):
                w.writerow([i, round(i / fps, 4), round(float(hist[i]), 5), round(float(pix[i]), 5),
                            round(float(changed[i]), 5), 1 if i and changed[i] < REPEAT_FRAC else 0])
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.unlink(tmp)
            except OSError:
                pass


def validate(doc):
    """Problems with a measure/1 document (empty list = valid)."""
    p = []
    if not isinstance(doc, dict) or doc.get("schema") != SCHEMA:
        return ["schema is not measure/1"]
    need = {"video": dict, "cuts": list, "shots": list, "cadence": dict, "faces_summary": dict, "text": dict,
            "loudness": dict, "visual_event_rate": list, "longest_hold_s": (int, float), "not_run": list}
    for k, t in need.items():
        if not isinstance(doc.get(k), t):
            p.append(f"{k} missing or wrong type")
    if "words_per_s" not in doc:
        p.append("words_per_s missing")
    if p:
        return p
    for k in ("id", "sha256", "duration_s", "fps", "width", "height"):
        if k not in doc["video"]:
            p.append(f"video.{k} missing")
    for c in doc["cuts"]:
        if not (isinstance(c.get("t"), (int, float)) and c.get("kind") in ("hard", "gradual", "jump") and "dur_f" in c):
            p.append(f"bad cut {c!r}")
    if not doc["shots"]:
        p.append("shots is empty (a video is at least one shot)")
    for s in doc["shots"]:
        for k in ("t0", "t1", "dur_s", "palette", "face_share", "headroom"):
            if k not in s:
                p.append(f"shot.{k} missing")
    for k in ("repeat_fraction", "unique_fps", "on_twos"):
        if k not in doc["cadence"]:
            p.append(f"cadence.{k} missing")
    t = doc["text"]
    hm = t.get("boxes_heatmap")
    if not (isinstance(hm, list) and len(hm) == HEATMAP and all(isinstance(r, list) and len(r) == HEATMAP for r in hm)):
        p.append("text.boxes_heatmap is not 8x8")
    if not isinstance(t.get("boxes"), list):
        p.append("text.boxes missing")
    if "size_pct_h" not in t or not (t.get("ocr") == "not_run" or isinstance(t.get("ocr"), list)):
        p.append("text.size_pct_h / text.ocr missing")
    return p


# ---------------------------------------------------------------- CLI

def main(argv=None):
    store.print_utf8()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--out", required=True, help="output directory (measure.json, frames.csv)")
    ap.add_argument("--words", help="words.json (list or {words: [...]}), for words_per_s")
    ap.add_argument("--from-analysis", help="a video-analysis output folder to import from")
    ap.add_argument("--faces", choices=("auto", "off"), default="auto")
    ap.add_argument("--text", choices=("auto", "ocr", "off"), default="auto")
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
    for label, p in (("--words", a.words), ("--from-analysis", a.from_analysis)):
        if p and not Path(p).exists():
            print(f"error: {label} path not found: {p}", file=sys.stderr)
            return 2
    try:
        doc = measure(a.video, a.out, words=a.words, from_analysis=a.from_analysis, faces=a.faces,
                      text=a.text, analysis_width=a.analysis_width)
    except MissingTool as e:
        print(str(e), file=sys.stderr)
        return 3
    except InputError as e:
        print(f"error: {str(e).splitlines()[0]}", file=sys.stderr)
        return 2
    except MemoryError:
        print("error: too long for memory: use --analysis-width 160 or less", file=sys.stderr)
        return 1
    except (RuntimeError, ValueError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    v = doc["video"]
    print(f"{v['duration_s']:.2f}s {v['width']}x{v['height']} @ {v['fps']:.2f} fps: {len(doc['cuts'])} cuts, "
          f"{len(doc['shots'])} shots, unique fps {doc['cadence']['unique_fps']}, longest hold "
          f"{doc['longest_hold_s']}s, not run: {doc['not_run'] or 'none'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

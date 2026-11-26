"""Synthetic media for the cut tests: tiny clips with words at known times.

The audio is built in numpy so every level is exact: a steady low room tone (a 100 Hz hum,
one full period per 10 ms slice, so every slice has the same level) and a 400 Hz tone for each
"word" with 5 ms ramps. The video is ffmpeg's testsrc2 at a chosen frame rate and a small size.
The matching aligned-words document has the shape align.py writes.
"""
import json
import subprocess
import wave
from pathlib import Path

import numpy as np

SR = 48000


def db_to_amp(db):
    return 10 ** (db / 20.0) * np.sqrt(2)  # peak of a sine whose RMS is `db` dBFS


def tone_track(dur_ms, spans, *, word_db=-12.0, room_db=-55.0, left_db=0.0,
               continuous=None, dips=()):
    """Stereo float32 audio: room tone everywhere, a word tone on each (start_ms, end_ms).

    continuous: (a_ms, b_ms) one unbroken word tone (words that run together).
    dips: [(a_ms, b_ms, drop_db)] lower the word tone by drop_db inside that range.
    left_db: extra gain on the left channel only.
    """
    n = int(dur_ms * SR / 1000)
    t = np.arange(n) / SR
    room = db_to_amp(room_db) * np.sin(2 * np.pi * 100 * t)
    env = np.zeros(n)
    ramp = int(0.005 * SR)
    ranges = list(spans) if continuous is None else [continuous]
    for a, b in ranges:
        i0, i1 = int(a * SR / 1000), int(b * SR / 1000)
        seg = np.ones(i1 - i0)
        seg[:ramp] = np.linspace(0, 1, ramp)
        seg[-ramp:] = np.linspace(1, 0, ramp)
        env[i0:i1] = np.maximum(env[i0:i1], seg)
    for a, b, drop in dips:
        i0, i1 = int(a * SR / 1000), int(b * SR / 1000)
        env[i0:i1] *= 10 ** (-drop / 20.0)
    voice = db_to_amp(word_db) * env * np.sin(2 * np.pi * 400 * t)
    mono = (room + voice).astype(np.float32)
    left = mono * np.float32(10 ** (left_db / 20.0))
    return np.stack([left, mono], axis=1)


def write_wav(path, audio):
    pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes(pcm.tobytes())


def make_clip(path, spans, dur_ms, *, fps=25, size="160x90", video="testsrc2", **audio_kw):
    """Write a small clip (video + the tone track, PCM audio in MKV). Returns the path.

    video: "testsrc2" (ffmpeg test pattern) or "ramp" (flat frames whose luma is 16 + frame
    number, stored lossless, so a cut frame can be traced back to its source frame)."""
    path = Path(path)
    wav = path.with_suffix(".src.wav")
    write_wav(wav, tone_track(dur_ms, spans, **audio_kw))
    d = dur_ms / 1000
    if video == "ramp":
        src, enc = f"nullsrc=s={size}:r={fps}:d={d},geq=lum='16+N':cb=128:cr=128", ["-qp", "0"]
    else:
        src, enc = f"testsrc2=s={size}:r={fps}:d={d}", []
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", src, "-i", str(wav),
                    "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-preset", "ultrafast", *enc,
                    "-g", str(fps), "-pix_fmt", "yuv420p", "-c:a", "pcm_s16le", str(path)],
                   check=True)
    wav.unlink()
    return path


def words_doc(spans, flags=None, asr=None, texts=None, lang="he"):
    """Aligned-words document for word spans [(start_ms, end_ms)]; flags/asr: {index: ...}."""
    flags, asr, texts = flags or {}, asr or {}, texts or {}
    out = []
    for i, (a, b) in enumerate(spans):
        aa, ab = asr.get(i, (a, b))
        w = {"i": i, "w": texts.get(i, f"w{i}"), "start_ms": a, "end_ms": b, "p": 0.9,
             "asr_start_ms": aa, "asr_end_ms": ab, "ctc_start_ms": a, "ctc_end_ms": b,
             "score": 0.05 if flags.get(i) == "low_score" else 0.9}
        if flags.get(i):
            w["flag"] = flags[i]
        out.append(w)
    return {"language": lang, "words": out}


def write_json(path, doc):
    Path(path).write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    return Path(path)


# ---------------------------------------------------------------- motion clips (measure_video tests)

_BG_CYCLE = [(20, 20, 30), (210, 200, 180), (70, 30, 110), (230, 120, 40), (30, 150, 90)]
_MOVE_DEFAULT = {"x": 0.5, "y": 0.5, "scale": 1.0, "opacity": 1.0}


def _ease(name, p):
    """GSAP names: powerN.in/out/inOut use exponent N+1 (power1 = quad, power2 = cubic, power3 = quart),
    back.out is the standard back ease (overshoot 1.70158)."""
    p = min(1.0, max(0.0, p))
    n = name.lower()
    if n in ("linear", "none"):
        return p
    if n == "back.out":
        c1 = 1.70158
        return 1 + (c1 + 1) * (p - 1) ** 3 + c1 * (p - 1) ** 2
    if n.startswith("power") and "." in n:
        k, kind = n[5:].split(".", 1)
        e = int(k) + 1
        if kind == "in":
            return p ** e
        if kind == "out":
            return 1 - (1 - p) ** e
        if kind == "inout":
            return 2 ** (e - 1) * p ** e if p < 0.5 else 1 - (-2 * p + 2) ** e / 2
    raise ValueError(f"unknown ease {name!r}")


def _prop_at(moves, prop, t):
    ms = sorted((m for m in moves if m[2] == prop), key=lambda m: m[0])
    if not ms:
        return _MOVE_DEFAULT[prop]
    val = ms[0][3]
    for t0, t1, _p, v0, v1, ease in ms:
        if t < t0:
            return val
        if t >= t1:
            val = v1
        else:
            val = v0 + (v1 - v0) * _ease(ease, (t - t0) / (t1 - t0))
            return val
    return val


def _bg_at(i, fps, cuts):
    """Background colour of frame i: cuts are float times (hard) or (time, dissolve_frames)."""
    norm = sorted((c, 0) if not isinstance(c, (tuple, list)) else (c[0], int(c[1])) for c in cuts)
    seg, mix_to, alpha = 0, None, 0.0
    for k, (ct, dn) in enumerate(norm):
        f0 = int(round(ct * fps))
        if i >= f0 + dn:
            seg = k + 1
        elif i >= f0 and dn:
            seg, mix_to, alpha = k, k + 1, (i - f0 + 1) / dn
    a = np.array(_BG_CYCLE[seg % len(_BG_CYCLE)], float)
    if mix_to is not None:
        b = np.array(_BG_CYCLE[mix_to % len(_BG_CYCLE)], float)
        a = a * (1 - alpha) + b * alpha
    return a


def make_motion_clip(path, *, dur_s, fps, moves=(), cuts=(), size="320x180", stepped=1):
    """Silent clip: a white box on a dark background, lossless H.264, so motion is exact.

    moves: [(t0, t1, prop, v0, v1, ease)], prop x|y (frame fractions of the box centre) |scale
    (1 = box side 20% of the height) | opacity; ease linear|power2.out|power3.inout|back.out.
    cuts: hard-cut times (they change the background colour); (time, n_frames) is an n-frame
    cross-dissolve of the background. stepped=2 holds every frame twice ("on twos")."""
    path = Path(path)
    w, h = (int(v) for v in size.split("x"))
    n = int(round(dur_s * fps))
    cols, rows = np.arange(w)[None, :], np.arange(h)[:, None]
    proc = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", size, "-r", str(fps),
         "-i", "-", "-c:v", "libx264", "-preset", "ultrafast", "-qp", "0", "-g", str(fps),
         "-pix_fmt", "yuv420p", str(path)], stdin=subprocess.PIPE)
    try:
        for i in range(n):
            bg = _bg_at(i, fps, cuts)
            t = ((i // stepped) * stepped) / fps
            cx, cy = _prop_at(moves, "x", t) * w, _prop_at(moves, "y", t) * h
            side = _prop_at(moves, "scale", t) * 0.2 * h
            op = _prop_at(moves, "opacity", t)
            covx = np.clip(np.minimum(cols + 1, cx + side / 2) - np.maximum(cols, cx - side / 2), 0, 1)
            covy = np.clip(np.minimum(rows + 1, cy + side / 2) - np.maximum(rows, cy - side / 2), 0, 1)
            alpha = (covy * covx * op)[..., None]
            frame = bg[None, None, :] * (1 - alpha) + 255.0 * alpha
            proc.stdin.write(np.clip(np.rint(frame), 0, 255).astype(np.uint8).tobytes())
        proc.stdin.close()
        if proc.wait() != 0:
            raise RuntimeError("ffmpeg failed writing the motion clip")
    finally:
        if proc.poll() is None:
            proc.kill()
    return path

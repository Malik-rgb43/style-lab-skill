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

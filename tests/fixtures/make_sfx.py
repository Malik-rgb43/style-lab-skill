"""Synthetic sound-design tracks for the sound_events tests.

Everything is built in numpy at make_media.SR, so every onset time and level is exact:
  - the base is make_media.tone_track: a steady 100 Hz room hum plus a 400 Hz tone "word" on each
    voice span, plus a very low white-noise floor so no band is digital zero;
  - SFX synthesisers (each scaled so its loudest 10 ms slice has RMS `db` dBFS):
      whoosh  band-passed noise (400-5000 Hz) with a rise-fall envelope, 0.3-0.6 s
      hit     a 60 Hz exponentially decaying sine with a short noise attack
      click   a 5 ms high-passed (>= 3 kHz) noise burst
      riser   an exponential sine sweep 300 -> 5000 Hz over 1.5 s, level rising 18 dB
      typing  4 ms high-passed clicks at 9 Hz
  - an optional music bed: a kick on every beat plus a steady three-note pad at `music_bpm`,
    scaled to RMS `music_db` dBFS, and ducked by `duck_db` under the voice spans (60 ms ramps).
"""
import subprocess
from pathlib import Path

import numpy as np

from make_media import SR, tone_track, write_wav

NOISE_FLOOR_DB = -80.0


def _peak_rms_scale(x, db):
    """Scale x so its loudest 10 ms slice has RMS `db` dBFS."""
    sl = SR // 100
    n = max(1, len(x) // sl)
    xx = np.zeros(n * sl)
    xx[: min(len(x), n * sl)] = x[: n * sl]
    peak = np.sqrt((xx.reshape(n, sl) ** 2).mean(axis=1)).max()
    return x * (10 ** (db / 20.0) / max(peak, 1e-12))


def _band(x, lo, hi):
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / SR)
    X[(f < lo) | (f > hi)] = 0
    return np.fft.irfft(X, len(x))


def whoosh(dur=0.4, db=-18.0, seed=1):
    rng = np.random.default_rng(seed)
    n = int(dur * SR)
    t = np.arange(n) / n
    x = _band(rng.standard_normal(n), 400, 5000) * np.sin(np.pi * t) ** 1.5
    return _peak_rms_scale(x, db)


def hit(db=-10.0, seed=2):
    rng = np.random.default_rng(seed)
    n = int(0.6 * SR)
    t = np.arange(n) / SR
    body = np.sin(2 * np.pi * 60 * t) * np.exp(-t / 0.12)
    attack = rng.standard_normal(n) * np.exp(-t / 0.004) * 0.5
    return _peak_rms_scale(body + attack, db)


def click(db=-20.0, seed=3, ms=5.0, lo=3000):
    rng = np.random.default_rng(seed)
    n = int(ms / 1000 * SR)
    x = _band(rng.standard_normal(n * 4), lo, SR / 2)[:n]
    r = max(1, int(0.0005 * SR))
    x[:r] *= np.linspace(0, 1, r)
    x[-r:] *= np.linspace(1, 0, r)
    # scale on the burst itself (it is shorter than a 10 ms slice)
    return x * (10 ** (db / 20.0) / max(np.sqrt((x ** 2).mean()), 1e-12))


def riser(dur=1.5, db=-14.0, f0=300.0, f1=5000.0):
    n = int(dur * SR)
    t = np.arange(n) / SR
    k = f1 / f0
    phase = 2 * np.pi * f0 * dur / np.log(k) * (k ** (t / dur) - 1)
    amp = 10 ** ((-18.0 + 18.0 * t / dur) / 20.0)
    x = np.sin(phase) * amp
    r = int(0.01 * SR)
    x[:r] *= np.linspace(0, 1, r)
    x[-r:] *= np.linspace(1, 0, r)
    return _peak_rms_scale(x, db)


def typing(dur=1.0, db=-22.0, rate_hz=9.0, seed=4):
    n = int(dur * SR) + int(0.01 * SR)
    x = np.zeros(n)
    rng = np.random.default_rng(seed)
    k = 0
    while k / rate_hz < dur:
        c = click(db + rng.uniform(-2, 2), seed=seed + 10 + k, ms=4.0, lo=2000)
        i0 = int(k / rate_hz * SR)
        x[i0:i0 + len(c)] += c
        k += 1
    return x


SYNTHS = {"whoosh": whoosh, "hit": hit, "click": click, "riser": riser, "typing": typing}


def music_bed(dur_s, bpm, db=-30.0, *, first_beat_s=0.25, duck_spans=(), duck_db=0.0):
    """Kick on every beat + steady pad, RMS `db` dBFS before ducking. Returns (audio, beat_times)."""
    n = int(dur_s * SR)
    t = np.arange(n) / SR
    pad = sum(np.sin(2 * np.pi * f * t) for f in (220.0, 277.18, 329.63)) * 0.08
    kick = np.zeros(n)
    kn = int(0.4 * SR)
    kt = np.arange(kn) / SR
    freq = 50 + 60 * np.exp(-kt / 0.03)
    one = np.sin(2 * np.pi * np.cumsum(freq) / SR) * np.exp(-kt / 0.08)
    beats = []
    b = first_beat_s
    while b < dur_s:
        i0 = int(round(b * SR))
        m = min(kn, n - i0)
        kick[i0:i0 + m] += one[:m]
        beats.append(b)
        b += 60.0 / bpm
    bed = kick + pad
    bed *= 10 ** (db / 20.0) / max(np.sqrt((bed ** 2).mean()), 1e-12)
    if duck_db and duck_spans:
        gain_db = np.zeros(n)
        ramp = int(0.06 * SR)
        for a, bb in duck_spans:
            i0, i1 = int(a * SR / 1000), int(bb * SR / 1000)
            g = np.full(i1 - i0 + 2 * ramp, -duck_db)
            g[:ramp] = np.linspace(0, -duck_db, ramp)
            g[-ramp:] = np.linspace(-duck_db, 0, ramp)
            lo = i0 - ramp
            s0, s1 = max(0, lo), min(n, i1 + ramp)
            gain_db[s0:s1] = np.minimum(gain_db[s0:s1], g[s0 - lo:s1 - lo])
        bed *= 10 ** (gain_db / 20.0)
    return bed, beats


def sfx_audio(dur_s, events, *, voice_spans=(), music_bpm=None, music_db=-30.0, duck_db=0.0,
              word_db=-12.0, room_db=-60.0, seed=0):
    """Stereo float32 track. events: [(t_s, kind) or (t_s, kind, {synth kwargs})];
    voice_spans: [(start_ms, end_ms)] tone words (make_media.tone_track)."""
    base = tone_track(int(dur_s * 1000), list(voice_spans), word_db=word_db, room_db=room_db)
    mono = base[:, 1].astype(np.float64)
    n = len(mono)
    rng = np.random.default_rng(seed)
    mono += rng.standard_normal(n) * 10 ** (NOISE_FLOOR_DB / 20.0)
    for ev in events:
        t0, kind = ev[0], ev[1]
        kw = ev[2] if len(ev) > 2 else {}
        x = SYNTHS[kind](**kw)
        i0 = int(round(t0 * SR))
        m = min(len(x), n - i0)
        mono[i0:i0 + m] += x[:m]
    if music_bpm:
        bed, _ = music_bed(dur_s, music_bpm, music_db, duck_spans=voice_spans, duck_db=duck_db)
        mono[: len(bed)] += bed[:n]
    mono = mono.astype(np.float32)
    return np.stack([mono, mono], axis=1)


def make_sfx_track(path, dur_s, events, *, voice_spans=(), music_bpm=None, music_db=-30.0,
                   duck_db=0.0, fps=25, size="160x90", **kw):
    """Write the track: a .wav file, or (any other suffix) a small testsrc2 clip with PCM audio."""
    path = Path(path)
    audio = sfx_audio(dur_s, events, voice_spans=voice_spans, music_bpm=music_bpm,
                      music_db=music_db, duck_db=duck_db, **kw)
    if path.suffix.lower() == ".wav":
        write_wav(path, audio)
        return path
    wav = path.with_suffix(".src.wav")
    write_wav(wav, audio)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                    f"testsrc2=s={size}:r={fps}:d={dur_s}", "-i", str(wav), "-map", "0:v", "-map", "1:a",
                    "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
                    "-c:a", "pcm_s16le", str(path)], check=True)
    wav.unlink()
    return path


def make_silent_video(path, dur_s=2.0, fps=25, size="160x90"):
    """A small clip with a video stream and no audio stream."""
    path = Path(path)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                    f"testsrc2=s={size}:r={fps}:d={dur_s}", "-c:v", "libx264", "-preset", "ultrafast",
                    "-pix_fmt", "yuv420p", "-an", str(path)], check=True)
    return path

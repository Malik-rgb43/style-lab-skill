#!/usr/bin/env python3
"""Sound events of one reference video: SFX onsets with class guesses, music and voice, sync.

Reads the source (never modifies it: it is hashed before and after), decodes its audio once
through ffmpeg (stereo, 24 kHz) and writes one sound.json (schema "sound/1").

Usage:
  sound_events.py <video|audio> --out <dir>/sound.json [--words w.json] [--measure m.json]
                  [--motion mo.json] [--no-librosa] [--allow-download]

What is measured (all local, numpy only; librosa is used for the onset envelope when it imports):
  - sfx: onsets by spectral flux over 12 log-spaced bands (10 ms hops) with adaptive peak picking
    (local median + delta), each confirmed by a level jump on a 1 ms trailing envelope and moved
    to where that envelope first rises. Each onset gets features (duration above peak - 20 dB,
    attack, level spread, bursts, centroid and its slope, flatness, tonal peak share, low-band
    share) and a class guess from the THRESHOLDS / RULES tables below, or "unknown". Runs of
    clicks become one "typing" event, runs of very short bursts one "glitch".
    With --words, onsets at word starts are voice, not SFX; onsets on the music's beat or half
    beat that are no louder than the beat itself are music, not SFX.
  - music: BPM from the onset-envelope autocorrelation (log-normal prior at 120 BPM, one octave;
    the half and double candidates are recorded in bpm_octave_check), energy per second, ducking
    (music level under the voice vs outside it, measured in the spectral bins the voice leaves
    alone), and the share of cuts that land on a beat (+-70 ms).
  - voice: integrated loudness (ITU-R BS.1770 K-weighting and gating) of the word spans, words per
    second, median pause between words.
  - coupling: share of cuts (--measure, cuts[].t) and of motion onsets (--motion, every moves[].t0)
    that have an SFX onset within 0.3 s, and the offsets SFX - cut / SFX - move / SFX - word onset.
Class labels are guesses (labels_are_guesses: true). Nothing is uploaded; no song is identified.
A part that cannot run is listed in "not_run" (with the reason in "not_run_why"), never a crash;
a source without an audio stream still gives a valid file with "sound" in not_run.

Exit: 0 done, 1 check failed (the source changed while it was read), 2 bad input,
3 ffmpeg missing (one line, with an install hint).
"""
import argparse
import hashlib
import json
import math
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import store  # noqa: E402

SCHEMA = "sound/1"
AR = 24000          # analysis sample rate
HOP_S = 0.01        # 10 ms feature hop
ENV_WIN_S = 0.01    # envelope window (10 ms: covers a 60 Hz period well enough for 6 dB decisions)
ENV_HOP_S = 0.002   # envelope hop for temporal features (a 5 ms click needs finer than 10 ms)
MAX_EVENT_S = 3.0   # longest span one event is measured over
CLASSES = ("whoosh", "swish", "riser", "hit", "pop", "click", "typing", "shutter", "glitch",
           "ambience", "unknown")

# Every decision threshold lives here (onset picking, filters, grouping, music, coupling).
THRESHOLDS = {
    # onsets
    "onset_bands": 12,             # log-spaced bands, 40 Hz .. Nyquist
    "onset_delta": 6.0,            # novelty (summed band dB rise per hop) above the local median
    "onset_median_s": 0.5,         # half-width of the local median window
    "onset_peak_s": 0.03,          # a novelty peak is the maximum within +- this
    "onset_min_gap_s": 0.05,       # two onsets closer than this are one (the stronger is kept)
    "onset_jump_db": 6.0,          # level rise (max of the next 80 ms vs the 20-100 ms before)
    "onset_min_level_db": -65.0,   # an onset whose peak stays below this is ignored (room tone)
    "refine_s": 0.04,              # the onset moves to the first envelope rise within +- this
    "refine_rise_db": 6.0,         # ... rise = trailing envelope this far above the level before
    "inside_sharp_db": 6.0,        # an onset inside the previous event's body needs this sharp an attack
    # voice / music filters
    "voice_onset_s": 0.06,         # an onset this close to a word start is the voice
    "over_voice_sharp_db": 6.0,    # an onset inside a word is an SFX if the whole mix jumps this sharply ...
    "over_voice_sub_db": 9.0,      # ... or the sub band (below sub_hz, where speech has nothing) rises this much
    "sub_hz": 100.0,
    "speech_pad_s": 0.12,          # speech = the word spans widened by this (ASR word edges are 0.1-0.3 s off)
    "speech_local_s": 0.08,        # an onset in speech must rise event_min_rise_db over the median level
                                   # of the speech in the 80 ms before it, and show a non-voice signature:
    "nv_hi_hz": 6000.0,            #   air: >= 30 % of its excess energy in 6-12 kHz, flat there (>= 0.3),
    "nv_hi_top_hz": 12000.0,
    "nv_hi_share": 0.3,            #   and longer than a fricative (>= 0.25 s) ...
    "nv_hi_flat": 0.3,
    "nv_hi_min_dur_s": 0.25,
    "air_hold_s": 0.06,            # an air sound's start: its 6-12 kHz rise holds this long (not a plosive)
    "nv_low_share": 0.6,           #   ... or a thump: >= 60 % of its excess energy below low_hz, centre
    "nv_low_centroid_hz": 250.0,   #   under 250 Hz, instant (<= 30 ms) attack (voiced speech also has 40-60 %
    "nv_low_attack_s": 0.03,       #   below 150 Hz, but its centroid sits on the formants, 300-900 Hz) ...
    "nv_sweep_hz_s": 800.0,        #   ... or a tonal sweep: |centroid slope| >= 800 Hz/s, tonal (peak share
    "nv_sweep_tonal": 0.5,         #   >= 0.5), >= 0.5 s long (speech glides are slower and shorter)
    "nv_sweep_min_dur_s": 0.5,
    # voice-like: tonal, centred on the formants, no air; an "unknown" event like this is dropped
    # anywhere (speech the words missed), it is never listed as an SFX
    "voice_tonal": 0.7,            # (it also vetoes a "sweep" signature and a voice-like "glitch")
    "voice_centroid_hz": (150.0, 1500.0),
    "voice_hi_share": 0.1,
    "infra_hz": 25.0,              # spectral features ignore energy below this (rumble, DC)
    "min_low_event_s": 0.02,       # an event shorter than 20 ms centred under 500 Hz is a level ripple of
    "min_low_event_hz": 500.0,     # a low note (a 5 ms click needs high frequencies), not an event
    "over_voice_per_s": 0.5,       # at most this many over-voice SFX per second of speech (strongest kept)
    "beat_tol_s": 0.05,            # an onset this close to a beat or half beat may be the music
    "beat_louder_db": 6.0,         # ... it stays an SFX only if its attack is this much sharper than the beats'
    "bed_margin_db": 15.0,         # an event less than this above the sound before it is measured on the
                                   # spectral excess over that sound (the bed), not on the mix level
    "event_min_rise_db": 6.0,      # an event must rise this much over the bed (level or one band), else unknown
    # event features
    "dur_db": 20.0,                # duration = time above peak - 20 dB (and above the level before + 3)
    "dur_gap_s": 0.03,             # dips shorter than this do not end the event
    "burst_floor_db": 12.0,        # a burst peak is within 12 dB of the event peak ...
    "burst_dip_db": 6.0,           # ... and separated from the next by a dip of at least 6 dB
    "low_hz": 150.0,               # the low band for low_share
    "peak_bins": 8,                # tonal peak share = energy in the 8 strongest bins / total
    # grouping
    "typing_min_n": 4,             # >= 4 clicks ...
    "typing_gap_s": (0.07, 0.25),  # ... 4-14 per second -> one typing event
    "glitch_min_n": 3,             # >= 3 bursts of <= 60 ms ...
    "glitch_max_dur_s": 0.06,
    "glitch_gap_s": 0.07,          # ... less than 70 ms apart -> one glitch event
    # music
    "bpm_range": (50.0, 220.0),
    "bpm_prior_bpm": 120.0,        # log-normal tempo prior (as in librosa), centre ...
    "bpm_prior_oct": 1.0,          # ... and width in octaves
    "bpm_min_conf": 0.09,          # harmonic sum of the mean windowed acf below this: no BPM (a beat-driven
                                   # reference under narration measured 0.10, narration over a soft bed
                                   # 0.02, synthetic speech alone <= 0.05; a clean bed 0.3-0.9)
    "tempo_win_s": 8.0,            # tempogram: autocorrelation of 8 s windows ...
    "tempo_step_s": 1.0,           # ... every 1 s, each normalised, then averaged
    "tempo_low_hz": (30.0, 150.0), # kick band
    "tempo_hat_hz": (2000.0, 8000.0),  # hat band
    "tempo_speech_weight": 0.5,    # envelope weight inside speech (from the words)
    "grid_min": 0.15,              # a tempo's beat grid must beat its off-beat grid by this (in env std) on
                                   # the low band, else half or double tempo is tried
    "bpm_min_s": 4.0,              # shorter audio: no BPM
    "beat_hit_s": 0.07,            # a cut within 70 ms of a beat is on the beat
    "music_bed_min_db": -45.0,     # 25th-percentile music-alone level above this: a bed plays (needed for
                                   # a BPM); with under 2 s of music-alone audio a found tempo is enough
    "music_alone_min_s": 2.0,
    "phrase_gap_s": 0.35,          # words closer than this form one phrase (for ducking)
    "duck_inner_s": 0.15,          # voice frames: at least this far inside a phrase
    "duck_outer_s": 0.25,          # music-alone frames: at least this far from any phrase
    "duck_min_s": 0.5,             # both sides need this much audio
    "duck_bins_db": 30.0,          # music bins: within 30 dB of the strongest music-alone bin
    "duck_percentile": 25.0,       # bins the voice leaves alone have the lowest ratio
    # voice
    "tilt_lo_hz": 125.0,           # voice spectral tilt: dB per octave fitted over these octave bands
    "tilt_hi_hz": 8000.0,
    "crest_pct": 99.9,             # crest = this percentile of |x| over the RMS, in the word audio
    "rt60_min_gap_s": 0.2,         # a word end is read for the room only when this much quiet follows
    "rt60_window_s": 0.8,          # ... over at most this long
    "rt60_drop_db": 20.0,          # T20: 3 x the time the level takes to fall this far after the end
    "rt60_min_ends": 2,
    "sib_band_hz": (5000.0, 9000.0),  # de-ess: sibilance band over 100 Hz..5 kHz, dB
    "section_win_s": 1.0,          # music sections: features per window, Foote novelty kernel half-width
    "section_kernel_w": 4,
    "section_min_s": 4.0,
    "section_db_scale": 6.0,       # windows this many dB apart (RMS over bands) are ~37 % similar
    "section_min_novelty": 0.3,    # 0..1; a boundary needs at least this
    "pause_min_s": 0.1,            # gaps between words shorter than this are not pauses
    # coupling
    "couple_window_s": 0.3,        # an SFX this close to a cut / move / word onset is coupled to it
}

# Class guesses: a class matches when every condition of one of its alternatives holds.
# Condition = (feature, op, threshold, scale); the margin (distance / scale, capped at 1) of the
# weakest condition sets the confidence (0.5 .. 0.95). Checked in this order; the first match wins.
RULES = {
    # glitch: several bursts packed tightly (a stutter, a digital break-up).
    "glitch": [[("dur_s", "<=", 0.8, 0.3), ("bursts", ">=", 3.5, 2), ("burst_rate_hz", ">=", 15.0, 10.0)]],
    # shutter: two quick noisy bursts (curtain open + close) within a quarter second.
    "shutter": [[("dur_s", "<=", 0.25, 0.1), ("bursts", ">=", 1.5, 1), ("bursts", "<=", 2.5, 1),
                 ("peak_share", "<=", 0.35, 0.2), ("centroid_hz", ">=", 1500.0, 1000.0)]],
    # click: one very short, bright burst (key, mouse, UI tick).
    "click": [[("dur_s", "<=", 0.03, 0.02), ("centroid_hz", ">=", 2000.0, 1500.0), ("bursts", "<=", 1.5, 1)]],
    # hit: instant attack with the energy in the sub/bass (impact, boom); a tight thump can be 40 ms.
    "hit": [[("attack_s", "<=", 0.03, 0.02), ("low_share", ">=", 0.4, 0.3), ("dur_s", ">=", 0.04, 0.04),
             ("dur_s", "<=", 1.5, 0.5)]],
    # pop: short, instant and mid-range (bubble, UI pop): not low like a hit, not bright like a click.
    "pop": [[("dur_s", "<=", 0.15, 0.05), ("attack_s", "<=", 0.02, 0.02), ("centroid_hz", "<=", 2500.0, 1000.0),
             ("low_share", "<=", 0.4, 0.3)]],
    # riser: long and building: the brightness climbs, or the level climbs to a late peak.
    "riser": [[("dur_s", ">=", 0.8, 0.4), ("centroid_slope_hz_s", ">=", 400.0, 600.0)],
              [("dur_s", ">=", 0.8, 0.4), ("attack_frac", ">=", 0.6, 0.2), ("level_rise_db", ">=", 6.0, 6.0)]],
    # whoosh: noise that swells and fades over 0.25-1 s (peak in the middle).
    "whoosh": [[("peak_share", "<=", 0.35, 0.2), ("dur_s", ">=", 0.25, 0.1), ("dur_s", "<=", 1.0, 0.3),
                ("attack_frac", ">=", 0.15, 0.15), ("attack_frac", "<=", 0.8, 0.15)]],
    # swish: the same noisy swell, short (under 0.25 s) and bright.
    "swish": [[("peak_share", "<=", 0.35, 0.2), ("dur_s", ">=", 0.05, 0.03), ("dur_s", "<=", 0.25, 0.1),
               ("centroid_hz", ">=", 1500.0, 1000.0)]],
    # ambience: a long steady noisy bed (room, crowd, rain).
    "ambience": [[("dur_s", ">=", 2.0, 1.0), ("level_std_db", "<=", 4.0, 2.0), ("peak_share", "<=", 0.35, 0.2)]],
}


class MissingTool(Exception):
    """ffmpeg/ffprobe not found: the CLI prints one line and exits 3."""


class BadInput(Exception):
    """unreadable source or json: the CLI prints one line and exits 2."""


# ---------------------------------------------------------------- small helpers

def _db(p):
    return 10.0 * np.log10(np.maximum(p, 1e-12))


def _r(x, nd=3):
    if x is None:
        return None
    x = float(x)
    return None if not math.isfinite(x) else round(x, nd)


def stats(values):
    v = np.asarray([x for x in values if x is not None and math.isfinite(x)], dtype=float)
    if not len(v):
        return {"median": None, "p10": None, "p90": None, "n": 0}
    return {"median": _r(np.median(v)), "p10": _r(np.percentile(v, 10)), "p90": _r(np.percentile(v, 90)),
            "n": int(len(v))}


def _mono(y):
    y = np.asarray(y, dtype=np.float64)
    return y.mean(axis=1) if y.ndim == 2 else y


def _frames(y, n_fft, hop, center=True):
    """Frame matrix (n_frames, n_fft); frame i is centred at i * hop when center=True."""
    y = np.asarray(y, dtype=np.float32)
    if center:
        y = np.concatenate([np.zeros(n_fft // 2, np.float32), y, np.zeros(n_fft // 2, np.float32)])
    if len(y) < n_fft:
        y = np.concatenate([y, np.zeros(n_fft - len(y), np.float32)])
    n = 1 + (len(y) - n_fft) // hop
    return np.lib.stride_tricks.as_strided(y, shape=(n, n_fft), strides=(y.strides[0] * hop, y.strides[0]))


def _power(y, sr, n_fft, hop, center=True, chunk=4096):
    fr = _frames(y, n_fft, hop, center)
    win = np.hanning(n_fft).astype(np.float32)
    out = np.empty((len(fr), n_fft // 2 + 1), np.float32)
    for i in range(0, len(fr), chunk):
        X = np.fft.rfft(fr[i:i + chunk] * win, axis=1)
        out[i:i + chunk] = (X.real ** 2 + X.imag ** 2).astype(np.float32)
    return out, np.fft.rfftfreq(n_fft, 1.0 / sr)


def _pow2(x):
    return 1 << int(math.ceil(math.log2(max(2, x))))


def _env(y, sr, a_s, b_s, *, win_s=ENV_WIN_S, hop_s=ENV_HOP_S, trailing=False):
    """Level envelope (dBFS RMS) on [a_s, b_s]; window centred on each time, or ending at it."""
    n = len(y)
    w = max(1, int(round(win_s * sr)))
    h = max(1, int(round(hop_s * sr)))
    i0 = int(round(a_s * sr))
    i1 = int(round(b_s * sr))
    if i1 <= i0:
        return np.zeros(0), np.zeros(0)
    idx = np.arange(i0, i1, h)
    lo = idx - w if trailing else idx - w // 2
    hi = lo + w
    lo_c, hi_c = np.clip(lo, 0, n), np.clip(hi, 0, n)
    s0, s1 = max(0, int(lo_c.min())), max(0, int(hi_c.max()))
    seg = np.asarray(y[s0:s1], dtype=np.float64)
    cs = np.concatenate([[0.0], np.cumsum(seg * seg)])
    e = cs[hi_c - s0] - cs[lo_c - s0]
    return idx / sr, _db(e / w)


# ---------------------------------------------------------------- onsets

def _band_matrix(freqs, sr, n_bands):
    edges = np.geomspace(40.0, sr / 2.0, n_bands + 1)
    m = np.zeros((len(freqs), n_bands), np.float32)
    for b in range(n_bands):
        m[(freqs >= edges[b]) & (freqs < edges[b + 1]), b] = 1.0
    return m


def _band_levels(y, sr):
    """(band dB levels per 10 ms hop (floored at each band's 10th percentile), hop_s, band edges)."""
    y = _mono(y)
    hop = max(1, int(round(HOP_S * sr)))
    n_fft = _pow2(0.02 * sr)
    P, freqs = _power(y, sr, n_fft, hop)
    edges = np.geomspace(40.0, sr / 2.0, THRESHOLDS["onset_bands"] + 1)
    L = _db(P @ _band_matrix(freqs, sr, THRESHOLDS["onset_bands"]))
    if len(L):
        L = np.maximum(L, np.maximum(np.percentile(L, 10, axis=0), -110.0))
    return L, hop / sr, edges


def novelty(y, sr):
    """Band spectral flux: per 10 ms hop, the summed positive dB rise over log-spaced bands."""
    L, hop_s, _ = _band_levels(y, sr)
    nov = np.zeros(len(L))
    if len(L) > 1:
        nov[1:] = np.maximum(L[1:] - L[:-1], 0).sum(axis=1)
    return nov, hop_s


def _rolling_median(x, half):
    if not len(x):
        return x
    pad = np.concatenate([np.full(half, np.median(x)), x, np.full(half, np.median(x))])
    w = np.lib.stride_tricks.sliding_window_view(pad, 2 * half + 1)
    out = np.empty(len(x))
    for i in range(0, len(x), 20000):
        out[i:i + 20000] = np.median(w[i:i + 20000], axis=1)
    return out


def _candidates(y, sr):
    """Onset candidates [{t, jump_db, sharp_db, sharp_bb_db, sub_rise_db, peak_db, pre_db}] with
    refined times, level-checked, de-duplicated."""
    T = THRESHOLDS
    y = _mono(y)
    dur = len(y) / sr if sr else 0.0
    if dur < 0.05:
        return []
    L, hop_s, edges = _band_levels(y, sr)
    nov = np.zeros(len(L))
    if len(L) > 1:
        nov[1:] = np.maximum(L[1:] - L[:-1], 0).sum(axis=1)
    med = _rolling_median(nov, max(1, int(T["onset_median_s"] / hop_s)))
    k = max(1, int(round(T["onset_peak_s"] / hop_s)))
    out = []
    for i in range(1, len(nov)):
        v = nov[i]
        if v <= 0 or v < med[i] + T["onset_delta"]:
            continue
        lo, hi = max(0, i - k), min(len(nov), i + k + 1)
        if v < nov[lo:hi].max() or (np.argmax(nov[lo:hi]) + lo) != i:
            continue
        rise = L[i:i + 8].max(axis=0) - np.median(L[max(0, i - 10):max(1, i - 1)], axis=0)
        b = int(np.argmax(rise))
        c = _confirm(y, sr, i * hop_s, (edges[b] / 1.2, edges[b + 1] * 1.2))
        if c:
            c["sub_rise_db"] = float(rise[0]) if edges[1] <= T["sub_hz"] else 0.0
            out.append(c)
    out.sort(key=lambda c: c["t"])
    merged = []
    for c in out:
        if merged and c["t"] - merged[-1]["t"] < T["onset_min_gap_s"]:
            if c["jump_db"] > merged[-1]["jump_db"]:
                merged[-1] = c
            continue
        merged.append(c)
    return merged


def _bandpass(x, sr, lo, hi):
    """Zero-phase band-pass by an FFT mask with raised-cosine edges (short ringing)."""
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1.0 / sr)
    m = np.zeros(len(f))
    m[(f >= lo) & (f <= hi)] = 1.0
    lo2, hi2 = lo / 1.3, hi * 1.3
    a = (f >= lo2) & (f < lo)
    m[a] = 0.5 - 0.5 * np.cos(np.pi * (f[a] - lo2) / (lo - lo2))
    b = (f > hi) & (f <= hi2)
    m[b] = 0.5 + 0.5 * np.cos(np.pi * (f[b] - hi) / (hi2 - hi))
    return np.fft.irfft(X * m, len(x))


def _confirm(y, sr, t, band=None):
    """Level check and refinement of one novelty peak; None when the level does not rise there.

    Checked on the broadband signal, else on the band that rose most (so a sound under a louder
    one in other bands, e.g. a boom under the voice, is still found). pre = median trailing level
    20-100 ms before the peak; the onset is the first time within +- refine_s where the trailing
    10 ms level stays refine_rise_db above pre for 5 ms. jump = the loudest level of the next
    80 ms - pre (jump_db on the broadband signal, band_jump_db on the signal that rose);
    sharp = the loudest level of the next 20 ms - the mean level 5-25 ms before (small inside a
    swell, large for a new attack; sharp_bb_db on the broadband signal)."""
    T = THRESHOLDS
    dur = len(y) / sr
    a_s = max(0.0, t - 0.2)
    seg = np.asarray(y[int(a_s * sr):int(min(dur, t + 0.15) * sr)], dtype=np.float64)
    if len(seg) < int(0.01 * sr):
        return None
    sdur = len(seg) / sr
    tl = t - a_s  # local time

    def level(x, a, b, how):
        if b <= 0 or b - a < 0.001:
            return -120.0
        _, e = _env(x, sr, max(0.0, a), min(sdur, b), hop_s=0.001, trailing=True)
        if not len(e):
            return -120.0
        return float({"median": np.median, "max": np.max, "mean": np.mean}[how](e))

    def rise_at(x, pre):
        ts, e = _env(x, sr, max(0.0, tl - T["refine_s"]), min(sdur, tl + T["refine_s"]), hop_s=0.001,
                     trailing=True)
        for j in np.nonzero(e >= pre + T["refine_rise_db"])[0]:
            if np.all(e[j:j + 5] >= pre + T["refine_rise_db"]):
                return max(0.0, float(ts[j]) - 0.001)
        return None

    def sharp_of(x, tr):
        return level(x, tr, tr + 0.02, "max") - (level(x, tr - 0.025, tr - 0.005, "mean")
                                                 if a_s + tr >= 0.025 else -120.0)
    at_start = t < 0.02
    pre_bb = -120.0 if at_start else level(seg, tl - 0.1, tl - 0.02, "median")
    sig, pre, t_ref = seg, pre_bb, rise_at(seg, pre_bb)
    if t_ref is None and band is not None:
        sig = _bandpass(seg, sr, *band)
        pre = -120.0 if at_start else level(sig, tl - 0.1, tl - 0.02, "median")
        t_ref = rise_at(sig, pre)
        # a band rise just before a broadband rise is the band-pass ringing ahead of that onset,
        # which its own novelty peak finds
        if t_ref is not None and level(seg, t_ref, t_ref + 0.1, "max") - pre_bb >= T["refine_rise_db"]:
            return None
    if t_ref is None:
        return None
    peak = level(sig, t_ref, t_ref + 0.08, "max")
    if peak - pre < T["onset_jump_db"] or peak < T["onset_min_level_db"]:
        return None
    peak_bb = level(seg, t_ref, t_ref + 0.08, "max")
    return {"t": a_s + t_ref, "jump_db": peak_bb - pre_bb, "band_jump_db": peak - pre,
            "sharp_db": sharp_of(sig, t_ref), "sharp_bb_db": sharp_of(seg, t_ref), "peak_db": peak_bb,
            "pre_db": pre_bb, "via": "broadband" if sig is seg else "band"}


def onsets(y, sr):
    """Onset times (s): spectral flux with adaptive peak picking, refined on the level envelope."""
    return [round(c["t"], 4) for c in _candidates(y, sr)]


# ---------------------------------------------------------------- features + classification

def features(y, sr, t0, t_end=None):
    """Features of the event starting at t0, measured until t_end (the next onset) at most."""
    T = THRESHOLDS
    y = _mono(y)
    dur_all = len(y) / sr
    limit = min(dur_all, t0 + MAX_EVENT_S,
                t_end - ENV_WIN_S / 2 - ENV_HOP_S if t_end is not None else 1e9)  # keep the next onset out
    f = {"dur_s": 0.0, "attack_s": 0.0, "attack_frac": 0.0, "peak_db": -120.0, "pre_db": -120.0,
         "level_std_db": 0.0, "level_rise_db": 0.0, "bursts": 0, "burst_rate_hz": 0.0,
         "centroid_hz": 0.0, "centroid_slope_hz_s": 0.0, "flatness": 0.0, "peak_share": 1.0,
         "low_share": 0.0, "band_rise_db": -120.0, "bed_subtracted": False, "mix_peak_db": -120.0,
         "hi_share": 0.0, "hi_flatness": 0.0}
    if limit - t0 < ENV_HOP_S:
        return f
    if t0 >= 0.02:
        _, pe = _env(y, sr, max(0.0, t0 - 0.08), t0 - 0.01)
        pre = float(np.median(pe)) if len(pe) else -120.0
    else:
        pre = -120.0
    ts, e = _env(y, sr, t0, limit)
    if not len(e):
        return f
    peak_mix = float(e.max())
    floor = pre + 3.0
    bed = peak_mix - pre < T["bed_margin_db"] and t0 >= 0.1
    if bed:
        ts, e, floor = _excess_env(y, sr, t0, limit)
        if not len(e):
            return f
    pk = int(np.argmax(e))
    peak = float(e[pk])
    thr = max(peak - T["dur_db"], floor)
    gap = max(1, int(round(T["dur_gap_s"] / ENV_HOP_S)))
    end, below = pk, 0
    for j in range(pk, len(e)):
        if e[j] >= thr:
            end, below = j, 0
        else:
            below += 1
            if below > gap:
                break
    dur = max(ENV_HOP_S, ts[end] - t0 + ENV_HOP_S)
    span = e[: end + 1]
    # bursts: peaks within burst_floor_db of the event peak, separated by burst_dip_db dips
    bursts, in_b, b_max = 0, False, -1e9
    for v in span:
        if not in_b:
            if v >= peak - T["burst_floor_db"]:
                in_b, b_max, bursts = True, v, bursts + 1
        else:
            b_max = max(b_max, v)
            if v < b_max - T["burst_dip_db"]:
                in_b = False
    own = float(_db(max(10 ** (peak_mix / 10) - 10 ** (pre / 10), 1e-12)))  # the event alone, bed removed
    f.update({"dur_s": dur, "attack_s": max(0.0, ts[pk] - t0), "peak_db": own, "mix_peak_db": peak_mix,
              "pre_db": pre,
              "bed_subtracted": bool(bed),
              "level_std_db": float(np.std(span)) if len(span) > 1 else 0.0,
              "bursts": int(bursts), "burst_rate_hz": bursts / dur})
    f["attack_frac"] = f["attack_s"] / dur
    q = max(1, len(span) // 5)
    f["level_rise_db"] = float(np.mean(span[-q:]) - np.mean(span[:q]))
    f.update(_spectral(y, sr, t0, dur, ts[pk]))
    return f


def _excess_env(y, sr, t0, limit):
    """Envelope (dB) of the spectral excess over the bed: per 2 ms hop, the summed power above the
    mean spectrum of 20-100 ms before t0. Returns (times, dB, floor) where floor = the residual of
    the bed itself (the 90th percentile before t0) + 3 dB."""
    n_fft = _pow2(0.02 * sr)
    hop = max(1, int(round(ENV_HOP_S * sr)))
    a = max(0, int((t0 - 0.1) * sr) - n_fft // 2)
    b = min(len(y), int(limit * sr) + n_fft // 2)
    seg = y[a:b]
    if len(seg) < n_fft:
        return np.zeros(0), np.zeros(0), 0.0
    P, _ = _power(seg, sr, n_fft, hop, center=False)
    tc = (a + np.arange(len(P)) * hop + n_fft / 2) / sr
    pre_m = (tc >= t0 - 0.1) & (tc <= t0 - 0.02)
    if not pre_m.any():
        return np.zeros(0), np.zeros(0), 0.0
    bed = P[pre_m].mean(axis=0)
    ex = _db(np.maximum(P - bed, 0).sum(axis=1).astype(np.float64) / (n_fft * n_fft / 2.0))
    floor = float(np.percentile(ex[pre_m], 90)) + 3.0
    m = (tc >= t0) & (tc <= limit)
    return tc[m], ex[m], floor


def _spectral(y, sr, t0, dur, t_peak):
    T = THRESHOLDS
    n_fft = _pow2(0.04 * sr)
    hop = max(1, int(round(HOP_S * sr)))
    a = max(0, int((t0 - 0.15) * sr))
    b = min(len(y), int((t0 + dur) * sr) + n_fft)
    seg = y[a:b]
    out = {"centroid_hz": 0.0, "centroid_slope_hz_s": 0.0, "flatness": 0.0, "peak_share": 1.0, "low_share": 0.0,
           "band_rise_db": -120.0, "hi_share": 0.0, "hi_flatness": 0.0}
    if len(seg) < n_fft:
        return out
    P, freqs = _power(seg, sr, n_fft, hop, center=False)
    starts = (a + np.arange(len(P)) * hop) / sr
    centres = starts + n_fft / 2 / sr
    pre_m = (starts + n_fft / sr) <= t0 - 0.005
    pre = P[pre_m].mean(axis=0) if pre_m.any() else np.zeros(P.shape[1], np.float32)
    ev_m = (centres >= t0) & (centres <= t0 + dur)
    if not ev_m.any():
        ev_m = np.zeros(len(P), bool)
        ev_m[int(np.argmin(np.abs(centres - t_peak)))] = True
    E = np.maximum(P[ev_m] - pre, 0.0).astype(np.float64)
    E[:, freqs < T["infra_hz"]] = 0.0
    tc = centres[ev_m]
    tot = E.sum(axis=1)
    if tot.max() <= 0:
        return out
    keep = tot >= tot.max() * 10 ** (-25 / 10)
    E, tc, tot = E[keep], tc[keep], tot[keep]
    cen = (E * freqs).sum(axis=1) / tot
    out["centroid_hz"] = float((cen * tot).sum() / tot.sum())
    if len(tc) >= 3 and np.ptp(tc) > 0:
        out["centroid_slope_hz_s"] = float(np.polyfit(tc, cen, 1)[0])
    band = (freqs >= 50) & (freqs <= min(12000.0, sr / 2 * 0.95))
    Eb = E[:, band] + tot.max() * 1e-9
    flat = np.exp(np.log(Eb).mean(axis=1)) / Eb.mean(axis=1)
    out["flatness"] = float((flat * tot).sum() / tot.sum())
    kb = min(THRESHOLDS["peak_bins"], E.shape[1])
    top = np.sort(E, axis=1)[:, -kb:].sum(axis=1)
    out["peak_share"] = float(top.sum() / tot.sum())  # power-weighted over the frames
    out["low_share"] = float(E[:, freqs < T["low_hz"]].sum() / E.sum())
    hi = (freqs >= T["nv_hi_hz"]) & (freqs <= min(T["nv_hi_top_hz"], sr / 2 * 0.95))
    if hi.sum() >= 4:
        out["hi_share"] = float(E[:, hi].sum() / E.sum())
        Eh = E[:, hi] + tot.max() * 1e-9
        fh = np.exp(np.log(Eh).mean(axis=1)) / Eh.mean(axis=1)
        wh = E[:, hi].sum(axis=1) + 1e-30
        out["hi_flatness"] = float((fh * wh).sum() / wh.sum())
    bm = _band_matrix(freqs, sr, T["onset_bands"]).astype(np.float64)
    eb = E.mean(axis=0) @ bm
    pb = pre.astype(np.float64) @ bm
    out["band_rise_db"] = float(np.max(_db(eb) - _db(pb + max(pb.sum(), 1e-12) * 1e-6)))
    return out


def _rule_margin(feat, rule):
    worst = 1e9
    for name, op, thr, scale in rule:
        v = feat[name]
        m = (v - thr) / scale if op == ">=" else (thr - v) / scale
        worst = min(worst, m)
    return worst


def class_of(feat):
    """(class_guess, confidence) from the RULES table; ("unknown", 0.0) when no rule matches or the
    event does not stand out of the sound before it (neither its level nor any band rises)."""
    if max(feat["peak_db"] - feat["pre_db"], feat.get("band_rise_db", -120.0)) < THRESHOLDS["event_min_rise_db"]:
        return "unknown", 0.0
    for cls, alts in RULES.items():
        best = max(_rule_margin(feat, r) for r in alts)
        if best >= 0:
            return cls, round(0.5 + 0.45 * min(1.0, float(best)), 2)
    return "unknown", 0.0


def classify(y, sr, t0, t_end=None, voice_db=None):
    """{class_guess, confidence, dur_s, level_vs_voice_db} of the event at t0 (+ its features)."""
    feat = features(y, sr, t0, t_end)
    cls, conf = class_of(feat)
    return {"class_guess": cls, "confidence": conf, "dur_s": _r(feat["dur_s"]),
            "level_vs_voice_db": None if voice_db is None else _r(feat["peak_db"] - voice_db, 1),
            "features": {k: _r(v, 3) if isinstance(v, float) else v for k, v in feat.items()}}


# ---------------------------------------------------------------- words

def word_spans(words):
    """[(start_s, end_s)] from a words document: a list or {"words": [...]}, seconds or *_ms."""
    if words is None:
        return None
    ws = words.get("words", words.get("segments")) if isinstance(words, dict) else words
    if not isinstance(ws, list):
        raise BadInput("words json: expected a list or {\"words\": [...]}")
    out = []
    for w in ws:
        if not isinstance(w, dict):
            continue
        try:
            if "start_ms" in w and "end_ms" in w:
                a, b = float(w["start_ms"]) / 1000.0, float(w["end_ms"]) / 1000.0
            elif "start" in w and "end" in w:
                a, b = float(w["start"]), float(w["end"])
            else:
                continue
        except (TypeError, ValueError):
            raise BadInput(f"words json: a word has a non-numeric time: {w!r}"[:200])
        if b > a:
            out.append((float(a), float(b)))
    out.sort()
    return out


def _spans(words):
    """None, [(start_s, end_s)] pairs, or a words document -> spans or None."""
    if words is None:
        return None
    if isinstance(words, list) and all(isinstance(w, (tuple, list)) and len(w) == 2 for w in words):
        return sorted((float(a), float(b)) for a, b in words)
    return word_spans(words)


def _phrases(spans, gap):
    out = []
    for a, b in spans:
        if out and a - out[-1][1] < gap:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def speech_regions(spans, pad=None):
    """Word spans widened by speech_pad_s and merged: where the voice may be."""
    pad = THRESHOLDS["speech_pad_s"] if pad is None else pad
    out = []
    for a, b in spans or []:
        a, b = max(0.0, a - pad), b + pad
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def voice_level_db(y, sr, spans):
    """Median 10 ms level (dBFS) inside the word spans, or None."""
    if not spans:
        return None
    y = _mono(y)
    vals = []
    for a, b in spans:
        _, e = _env(y, sr, a, b, hop_s=HOP_S)
        vals.extend(e.tolist())
    return float(np.median(vals)) if vals else None


# ---------------------------------------------------------------- SFX detection

def _voice_like(feat):
    """Tonal, centred on the formants, no air: what speech looks like to these features."""
    T = THRESHOLDS
    lo, hi = T["voice_centroid_hz"]
    return feat["peak_share"] >= T["voice_tonal"] and lo <= feat["centroid_hz"] <= hi \
        and feat["hi_share"] < T["voice_hi_share"]


def _non_voice(feat):
    """Which non-voice signature an event inside speech shows ("air", "thump", "sweep") or None."""
    T = THRESHOLDS
    if feat["hi_share"] >= T["nv_hi_share"] and feat["hi_flatness"] >= T["nv_hi_flat"] \
            and feat["dur_s"] >= T["nv_hi_min_dur_s"]:
        return "air"
    if feat["low_share"] >= T["nv_low_share"] and feat["centroid_hz"] <= T["nv_low_centroid_hz"] \
            and feat["attack_s"] <= T["nv_low_attack_s"]:
        return "thump"
    if abs(feat["centroid_slope_hz_s"]) >= T["nv_sweep_hz_s"] and feat["peak_share"] >= T["nv_sweep_tonal"] \
            and feat["dur_s"] >= T["nv_sweep_min_dur_s"] and not _voice_like(feat):
        return "sweep"
    return None


def _air_start(y, sr, t0, dur_s):
    """Where the 6-12 kHz band first rises event_min_rise_db over its level in the 100 ms before
    t0, inside [t0, t0 + dur_s]; None when it never does."""
    T = THRESHOLDS
    lo, hi = T["nv_hi_hz"], min(T["nv_hi_top_hz"], sr / 2 - 1)
    if hi <= lo or dur_s <= 0:
        return None
    a = max(0.0, t0 - 0.1)
    b = min(len(y) / sr, t0 + dur_s)
    seg = _bandpass(np.asarray(y[int(a * sr):int(b * sr)], np.float64), sr, lo, hi)
    ts, e = _env(seg, sr, 0.0, len(seg) / sr, hop_s=HOP_S)
    if not len(e):
        return None
    ts = ts + a
    pre = e[ts < t0]
    if not len(pre):
        return None
    ref = float(np.median(pre)) + T["event_min_rise_db"]
    # the rise must hold for air_hold_s, so a plosive or fricative burst at a word start (a few ms
    # of high band) does not count as the air sound's start
    hold = max(1, int(round(T["air_hold_s"] / HOP_S)))
    above = (e >= ref) & (ts >= t0)
    run = 0
    for k, ok in enumerate(above):
        run = run + 1 if ok else 0
        if run >= hold:
            return float(ts[k - hold + 1])
    return None


def detect_sfx(y, sr, spans=None, beats=None, voice_db=None, log=None):
    """SFX events [{t, class_guess, confidence, dur_s, level_vs_voice_db, ...}] after the voice and
    music filters, classification and grouping.

    Voice filter (with word spans): an onset within voice_onset_s of a word start is the voice. An
    onset inside speech (the spans widened by speech_pad_s) stays only if it rises event_min_rise_db
    over the speech level around it, shows a non-voice signature (_non_voice) and gets a class; the
    over-voice SFX are then capped at over_voice_per_s per second of speech, strongest first.
    level_vs_voice_db is null without words (no voice level)."""
    T = THRESHOLDS
    y = _mono(y)
    cands = _candidates(y, sr)
    speech = speech_regions(spans) if spans else []
    kept = []
    for c in cands:
        t = c["t"]
        c["over_voice"] = False
        if spans:
            if any(abs(t - a) <= T["voice_onset_s"] for a, _ in spans):
                continue
            c["over_voice"] = any(a <= t <= b for a, b in speech)
        kept.append(c)
    if beats is not None and len(beats) >= 2:
        beats = np.asarray(beats, float)
        half = (beats[:-1] + beats[1:]) / 2
        grid = np.sort(np.concatenate([beats, half]))

        def on_grid(t):
            return float(np.min(np.abs(grid - t))) <= T["beat_tol_s"]
        on = [c for c in kept if on_grid(c["t"])]
        if on:
            ref = float(np.median([c["sharp_bb_db"] for c in on]))
            kept = [c for c in kept if not on_grid(c["t"]) or c["sharp_bb_db"] >= ref + T["beat_louder_db"]]
    dur_all = len(y) / sr if sr else 0.0
    events = []
    dropped = {"no_rise": 0, "voice_like": 0, "in_speech": 0, "over_voice_cap": 0}
    i = 0
    while i < len(kept):
        c = kept[i]
        # an onset inside this event's body without a sharp attack is part of it (riser steps,
        # the flutter of a swell); the body is measured without the next-onset limit. Not inside
        # speech: there the "body" runs on through the words and would swallow an SFX's onset
        body = 0.0 if c["over_voice"] else features(y, sr, c["t"])["dur_s"]
        while i + 1 < len(kept) and kept[i + 1]["t"] < c["t"] + body and \
                kept[i + 1]["sharp_bb_db"] < T["inside_sharp_db"]:
            kept.pop(i + 1)
        nxt = kept[i + 1]["t"] if i + 1 < len(kept) else None
        ev = classify(y, sr, c["t"], nxt, voice_db)
        ev = {"t": _r(c["t"]), **ev, "peak_db": _r(ev["features"]["peak_db"], 1), "over_voice": c["over_voice"]}
        f = ev["features"]
        if max(f["peak_db"] - f["pre_db"], f["band_rise_db"]) < T["event_min_rise_db"]:
            dropped["no_rise"] += 1  # does not stand out of what was sounding: not an event
            i += 1
            continue
        if f["dur_s"] < T["min_low_event_s"] and f["centroid_hz"] < T["min_low_event_hz"]:
            dropped["no_rise"] += 1  # shorter than one period of its own frequency: a level ripple
            i += 1
            continue
        if ev["class_guess"] in ("unknown", "glitch") and _voice_like(f):
            dropped["voice_like"] += 1  # tonal syllable bursts the words missed
            i += 1
            continue
        if c["over_voice"]:
            f = ev["features"]
            ls = T["speech_local_s"]
            # the speech level just before the onset (the window after it holds the event itself)
            _, le = _env(y, sr, max(0.0, c["t"] - ls), max(0.0, c["t"] - 0.005), hop_s=HOP_S)
            rise = f["mix_peak_db"] - (float(np.median(le)) if len(le) else -120.0)
            sig = _non_voice(f)
            ev["rise_over_speech_db"] = _r(rise, 1)
            ev["non_voice"] = sig
            if rise < T["event_min_rise_db"] or sig is None or ev["class_guess"] == "unknown":
                dropped["in_speech"] += 1
                i += 1
                continue
            if sig == "air":
                # a soft-attack air sound (a whoosh) has no onset of its own inside speech: the
                # nearest speech onset before it carries it. Its true start is where the air band
                # (6-12 kHz) first rises event_min_rise_db over the level before the speech onset.
                t_air = _air_start(y, sr, c["t"], f["dur_s"])
                if t_air is not None and t_air - c["t"] > T["voice_onset_s"]:
                    # measure and classify the sound from its own start, not the speech before it
                    re_ev = classify(y, sr, t_air, nxt, voice_db)
                    if re_ev["class_guess"] != "unknown":
                        keep = {k: ev[k] for k in ("over_voice", "rise_over_speech_db", "non_voice")}
                        ev = {"t": _r(t_air), **re_ev, "peak_db": _r(re_ev["features"]["peak_db"], 1),
                              **keep, "anchored": "air_band"}
                    else:
                        ev["t"] = _r(t_air)
                        ev["anchored"] = "air_band"
        events.append(ev)
        i += 1
    if spans:
        speech_s = sum(b - a for a, b in spans)
        cap = int(math.floor(T["over_voice_per_s"] * speech_s + 1e-9))
        ov = sorted((e for e in events if e["over_voice"]), key=lambda e: -e["rise_over_speech_db"])
        drop = {id(e) for e in ov[cap:]}
        dropped["over_voice_cap"] = len(drop)
        events = [e for e in events if id(e) not in drop]
    if log is not None:
        log.update(dropped)
    return _group(events)


def _group(events):
    T = THRESHOLDS
    out, i = [], 0
    while i < len(events):
        j = i
        if events[i]["class_guess"] == "click":
            while j + 1 < len(events) and events[j + 1]["class_guess"] == "click" and \
                    T["typing_gap_s"][0] <= events[j + 1]["t"] - events[j]["t"] <= T["typing_gap_s"][1]:
                j += 1
            if j - i + 1 >= T["typing_min_n"]:
                out.append(_merge(events[i:j + 1], "typing"))
                i = j + 1
                continue
        j = i
        if events[i]["dur_s"] <= T["glitch_max_dur_s"]:
            while j + 1 < len(events) and events[j + 1]["dur_s"] <= T["glitch_max_dur_s"] and \
                    events[j + 1]["t"] - events[j]["t"] < T["glitch_gap_s"]:
                j += 1
            if j - i + 1 >= T["glitch_min_n"]:
                out.append(_merge(events[i:j + 1], "glitch"))
                i = j + 1
                continue
        out.append(events[i])
        i += 1
    return out


def _merge(group, cls):
    t0 = group[0]["t"]
    t1 = group[-1]["t"] + group[-1]["dur_s"]
    gaps = np.diff([g["t"] for g in group])
    lv = [g["level_vs_voice_db"] for g in group if g["level_vs_voice_db"] is not None]
    return {"t": t0, "class_guess": cls,
            "confidence": round(float(np.mean([g["confidence"] for g in group])), 2),
            "dur_s": _r(t1 - t0), "level_vs_voice_db": _r(max(lv), 1) if lv else None,
            "count": len(group), "rate_hz": _r(1.0 / float(np.median(gaps)), 2) if len(gaps) else None,
            "peak_db": max(g["peak_db"] for g in group), "over_voice": any(g["over_voice"] for g in group)}


# ---------------------------------------------------------------- music

USE_LIBROSA = True  # --no-librosa switches it off (the numpy envelope is then used)


def _get_librosa():
    if not USE_LIBROSA:
        return None
    try:
        import librosa  # noqa: F401
        return librosa
    except Exception:
        return None


def _detrend(d, k=51):
    """Subtract a 0.5 s running mean (slow level changes such as speech phrases), half-wave."""
    if len(d) < k:
        return np.maximum(d - (d.mean() if len(d) else 0.0), 0.0)
    tr = np.convolve(d, np.ones(k) / k, mode="same")
    return np.maximum(d - tr, 0.0)


def _flux(P, freqs, lo, hi):
    m = (freqs >= lo) & (freqs < hi)
    d = np.zeros(len(P))
    if m.any() and len(P) > 1:
        L = _db(P[:, m])
        d[1:] = np.maximum(L[1:] - L[:-1], 0).mean(axis=1)
    return _detrend(d)


def tempo_envelopes(y, sr, spans=None):
    """({name: onset envelope at 10 ms hops}, hop_s, method): the low (kick) band and the hat band
    flux, detrended, down-weighted inside speech; plus librosa's onset strength when it imports."""
    T = THRESHOLDS
    y = _mono(y)
    hop = max(1, int(round(HOP_S * sr)))
    P, freqs = _power(y, sr, _pow2(0.04 * sr), hop)
    envs = {"low": _flux(P, freqs, *T["tempo_low_hz"]), "hat": _flux(P, freqs, *T["tempo_hat_hz"])}
    names = ["low", "hat"]
    lr = _get_librosa()
    if lr is not None:
        try:
            e = np.asarray(lr.onset.onset_strength(y=y.astype(np.float32), sr=sr, hop_length=hop), float)
            e = np.pad(e, (0, max(0, len(P) - len(e))))[: len(P)]
            envs["librosa"] = _detrend(e)
            names.append("librosa")
        except Exception:
            pass
    if spans:
        tc = np.arange(len(P)) * hop / sr
        w = np.ones(len(P))
        for a, b in speech_regions(spans):
            w[(tc >= a) & (tc <= b)] = T["tempo_speech_weight"]
        envs = {k: v * w for k, v in envs.items()}
    return envs, hop / sr, "tempogram:" + "+".join(names)


def tempogram(env, hop_s):
    """Mean of the normalised autocorrelations of tempo_win_s Hann windows (one window when shorter)."""
    T = THRESHOLDS
    n = len(env)
    win = min(n, int(round(T["tempo_win_s"] / hop_s)))
    step = max(1, int(round(T["tempo_step_s"] / hop_s)))
    if win < 4:
        return np.zeros(max(1, win))
    w = np.hanning(win)
    acc, cnt = np.zeros(win), 0
    for s0 in range(0, n - win + 1, step):
        e = env[s0:s0 + win]
        e = (e - e.mean()) * w
        F = np.fft.rfft(e, 2 * win)
        a = np.fft.irfft(F * np.conj(F))[:win]
        if a[0] > 0:
            acc += a / a[0]
            cnt += 1
    return acc / cnt if cnt else acc


def grid_fit(env, period):
    """How much better the beat grid of `period` frames lands on the envelope than the off-beat grid
    half a period later (best phase), in units of the envelope's std."""
    n = len(env)
    sd = float(env.std())
    if sd <= 0 or period < 2 or n < 2 * period:
        return 0.0
    best = -1e9
    for ph in np.arange(0.0, period, 1.0):
        on = np.round(np.arange(ph, n - 1, period)).astype(int)
        off = np.round(np.arange(ph + period / 2, n - 1, period)).astype(int)
        if len(on) and len(off):
            best = max(best, float(env[on].mean() - env[off].mean()))
    return best / sd


def tempo(envs, hop_s):
    """(bpm|None, conf, octave_check, period_frames|None) from the averaged tempogram of the
    envelopes; half and double tempo are audited on the low band's beat grid."""
    T = THRESHOLDS
    if not isinstance(envs, dict):
        envs = {"low": np.asarray(envs, float)}
    envs = {k: v for k, v in envs.items() if len(v) and float(np.std(v)) > 0}
    n = min((len(v) for v in envs.values()), default=0)
    lo_l = int(math.floor(60.0 / T["bpm_range"][1] / hop_s))
    hi_l = int(math.ceil(60.0 / T["bpm_range"][0] / hop_s))
    if not envs or n * hop_s < T["bpm_min_s"] or min(n, int(T["tempo_win_s"] / hop_s)) < hi_l + 2:
        return None, 0.0, None, None
    tg = np.mean([tempogram(v[:n], hop_s) for v in envs.values()], axis=0)
    lags = np.arange(lo_l, min(hi_l, len(tg) - 2) + 1)
    bpms = 60.0 / (lags * hop_s)

    def prior(b):
        return float(np.exp(-0.5 * (math.log2(b / T["bpm_prior_bpm"]) / T["bpm_prior_oct"]) ** 2))
    def hsum(p):
        """Harmonic sum: mean tempogram at 1, 2 and 3 periods. A true beat repeats at every multiple;
        a 2:3 or off-beat sub-period (a dotted rhythm, a half beat) does not."""
        vals = [float(tg[int(round(k * p))]) for k in (1, 2, 3) if 1 <= int(round(k * p)) < len(tg)]
        return float(np.mean(vals)) if vals else 0.0
    prior_v = np.exp(-0.5 * (np.log2(bpms / T["bpm_prior_bpm"]) / T["bpm_prior_oct"]) ** 2)
    score = np.maximum([hsum(l) for l in lags], 0) * prior_v
    li = int(lags[int(np.argmax(score))])

    def interp(l):
        a, b, c = tg[l - 1], tg[l], tg[l + 1]
        den = a - 2 * b + c
        d = 0.5 * (a - c) / den if den < 0 else 0.0
        return l + max(-0.5, min(0.5, d))

    def tg_at(p):
        k = int(round(p))
        return float(tg[k]) if 1 <= k < len(tg) else None
    low = envs.get("low", next(iter(envs.values())))[:n]
    period = interp(li)
    cands = {"picked": period, "half_bpm": period * 2, "double_bpm": period / 2}
    audit = {}
    for name, p in cands.items():
        b = 60.0 / (p * hop_s)
        audit[name] = {"bpm": _r(b, 2), "acf": _r(tg_at(p), 3), "hsum": _r(hsum(p), 3), "prior": _r(prior(b), 3),
                       "grid_fit": _r(grid_fit(low, p), 3)}
    choice = "picked"
    if (audit["picked"]["grid_fit"] or 0) < T["grid_min"]:
        alts = [k for k in ("half_bpm", "double_bpm")
                if (audit[k]["grid_fit"] or 0) >= T["grid_min"] and audit[k]["acf"] is not None
                and T["bpm_range"][0] <= audit[k]["bpm"] <= T["bpm_range"][1]]
        if alts:
            choice = max(alts, key=lambda k: max(audit[k]["hsum"], 0) * audit[k]["prior"])
    if choice != "picked":
        k = int(round(cands[choice]))
        period = interp(k) if 1 <= k < len(tg) - 1 else cands[choice]
    bpm = 60.0 / (period * hop_s)
    conf = float(max(0.0, min(1.0, hsum(period))))
    octave = {**audit, "chosen": choice, "bpm": _r(bpm, 2),
              "rule": f"harmonic sum (1-3 periods) of the windowed acf ({T['tempo_win_s']:.0f} s) x log-normal prior ({T['bpm_prior_bpm']:.0f} BPM, "
                      f"{T['bpm_prior_oct']} octave), range {T['bpm_range'][0]:.0f}-{T['bpm_range'][1]:.0f} BPM; "
                      f"half/double tried when the low-band grid fit is under {T['grid_min']}"}
    if conf < T["bpm_min_conf"]:
        return None, conf, octave, None
    return bpm, conf, octave, period


def beat_times(env, hop_s, period):
    """Beat grid: the phase that collects the most envelope, then every period."""
    n = len(env)
    best, best_s = 0, -1.0
    for ph in range(int(math.ceil(period))):
        idx = np.round(np.arange(ph, n, period)).astype(int)
        idx = idx[idx < n]
        s = float(env[idx].sum())
        if s > best_s:
            best, best_s = ph, s
    return (np.arange(best, n, period) * hop_s).tolist()


def _music(y, sr, spans=None, cuts=None, tempo_y=None):
    T = THRESHOLDS
    y = _mono(y)
    tsrc = _mono(tempo_y) if tempo_y is not None else y
    envs, hop_s, method = tempo_envelopes(tsrc, sr, None if tempo_y is not None else spans)
    if tempo_y is not None:
        method = "stem-" + method
    bpm, conf, octave, period = tempo(envs, hop_s)
    comb = sum(v / (float(v.std()) or 1.0) for v in envs.values() if len(v))
    beats = beat_times(comb, hop_s, period) if period else []
    dur = len(y) / sr
    # music-alone frames (10 ms): away from every phrase, or everything when there are no words
    _, lv = _env(y, sr, 0.0, dur, hop_s=HOP_S)
    tt = np.arange(len(lv)) * HOP_S
    phrases = _phrases(spans, T["phrase_gap_s"]) if spans else []
    alone = np.ones(len(lv), bool)
    for a, b in phrases:
        alone &= ~((tt > a - T["duck_outer_s"]) & (tt < b + T["duck_outer_s"]))
    # a bed keeps sounding between the onsets, so even its quiet quarter stays up; a train of SFX
    # over room tone does not (its 25th percentile is the room)
    bed_db = float(np.percentile(lv[alone], 25)) if alone.any() else None
    # music = a bed that keeps sounding between the onsets; a regular train of SFX over room tone
    # is periodic too, so without a bed its tempo is only recorded as onset_periodicity_bpm. With
    # hardly any music-alone audio (wall-to-wall narration) a found tempo is taken as music.
    alone_s = float(alone.sum()) * HOP_S
    present = (bed_db is not None and bed_db >= T["music_bed_min_db"]) or \
        (alone_s < T["music_alone_min_s"] and bpm is not None)
    periodicity = bpm
    if not present:
        bpm, period, beats = None, None, []
    energy = []
    if present and len(lv):
        for s in range(int(math.ceil(dur))):
            m = alone & (tt >= s) & (tt < s + 1)
            energy.append(_r(float(_db(np.mean(10 ** (lv[m] / 10)))), 1) if m.any() else None)
    ducking = _ducking(y, sr, phrases) if (present and phrases) else None
    share = None
    if beats and cuts:
        bt = np.asarray(beats)
        share = _r(np.mean([float(np.min(np.abs(bt - c))) <= T["beat_hit_s"] for c in cuts]), 3)
    prof = {"present": bool(present), "bpm": _r(bpm, 2), "bpm_conf": _r(conf, 3), "bpm_method": method,
            "onset_periodicity_bpm": _r(periodicity, 2),
            "bpm_octave_check": octave, "beats_n": len(beats), "energy_per_s": energy,
            "bed_level_db": _r(bed_db, 1), "ducking_db": _r(ducking, 1), "beat_cut_share": share,
            "sections": music_sections(y, sr, alone) if present else []}
    return prof, beats


def _ducking(y, sr, phrases):
    """Music level outside the voice minus under it (dB), in the bins the voice leaves alone."""
    T = THRESHOLDS
    n_fft = _pow2(0.08 * sr)
    hop = max(1, int(round(0.04 * sr)))
    P, freqs = _power(y, sr, n_fft, hop)
    tc = np.arange(len(P)) * hop / sr
    under = np.zeros(len(P), bool)
    alone = np.ones(len(P), bool)
    for a, b in phrases:
        under |= (tc >= a + T["duck_inner_s"]) & (tc <= b - T["duck_inner_s"])
        alone &= ~((tc > a - T["duck_outer_s"]) & (tc < b + T["duck_outer_s"]))
    if under.sum() * hop / sr < T["duck_min_s"] or alone.sum() * hop / sr < T["duck_min_s"]:
        return None
    Pv = P[under].mean(axis=0).astype(np.float64)
    Pn = P[alone].mean(axis=0).astype(np.float64)
    band = (freqs >= 30) & (freqs <= min(16000.0, sr / 2 * 0.95))
    if not band.any() or Pn[band].max() <= 0:
        return None
    music = band & (Pn >= Pn[band].max() * 10 ** (-T["duck_bins_db"] / 10))
    ratio = _db(Pv[music] + 1e-20) - _db(Pn[music] + 1e-20)
    return float(-np.percentile(ratio, T["duck_percentile"]))


def music_profile(y, sr, words=None, cuts=None, tempo_y=None):
    """{bpm|None, bpm_conf, energy_per_s, ducking_db, beat_cut_share|None, ...}.
    words: a words document or [(start_s, end_s)]; cuts: [t_s]; tempo_y: a non-vocal stem to take
    the tempo from (else the mix, with speech down-weighted)."""
    return _music(y, sr, _spans(words), cuts, tempo_y)[0]


# ---------------------------------------------------------------- voice + loudness

def _k_coeffs(sr):
    """BS.1770 K-weighting biquads (pre-filter shelf, RLB high-pass) for any sample rate, from the
    analog prototype (the libebur128 derivation; at 48 kHz it gives the ITU coefficients)."""
    f0, G, Q = 1681.974450955533, 3.999843853973347, 0.7071752369554196
    K = math.tan(math.pi * f0 / sr)
    Vh = 10 ** (G / 20.0)
    Vb = Vh ** 0.4996667741545416
    a0 = 1.0 + K / Q + K * K
    b1 = [(Vh + Vb * K / Q + K * K) / a0, 2.0 * (K * K - Vh) / a0, (Vh - Vb * K / Q + K * K) / a0]
    a1 = [1.0, 2.0 * (K * K - 1.0) / a0, (1.0 - K / Q + K * K) / a0]
    f0, Q = 38.13547087602444, 0.5003270373238773
    K = math.tan(math.pi * f0 / sr)
    d = 1.0 + K / Q + K * K
    b2 = [1.0, -2.0, 1.0]
    a2 = [1.0, 2.0 * (K * K - 1.0) / d, (1.0 - K / Q + K * K) / d]
    return (np.array(b1), np.array(a1)), (np.array(b2), np.array(a2))


def _k_power_response(sr, n):
    f = np.fft.rfftfreq(n, 1.0 / sr)
    z = np.exp(-1j * 2 * np.pi * f / sr)
    h = np.ones(len(f), complex)
    for b, a in _k_coeffs(sr):
        h *= (b[0] + b[1] * z + b[2] * z * z) / (a[0] + a[1] * z + a[2] * z * z)
    return np.abs(h) ** 2


def _block_ms(x, sr):
    """Mean square of the K-weighted signal per 100 ms sub-block: Parseval per sub-block with the
    K power response (numpy only; scipy.signal is not used because its import alone can take
    tens of seconds on a slow machine)."""
    sub = int(round(0.1 * sr))
    nb = len(x) // sub
    if nb == 0:
        return np.zeros(0)
    H2 = _k_power_response(sr, sub)
    w = np.full(len(H2), 2.0)
    w[0] = 1.0
    if sub % 2 == 0:
        w[-1] = 1.0
    out = np.empty(nb)
    blocks = x[: nb * sub].reshape(nb, sub)
    for i in range(0, nb, 600):
        X = np.fft.rfft(blocks[i:i + 600], axis=1)
        out[i:i + 600] = ((X.real ** 2 + X.imag ** 2) * H2 * w).sum(axis=1) / (sub * sub)
    return out


def lufs(y, sr):
    """Integrated loudness (LUFS, BS.1770: 400 ms blocks, 75 % overlap, -70 / -10 gates) or None."""
    y = np.asarray(y, dtype=np.float64)
    chans = [y] if y.ndim == 1 else [y[:, c] for c in range(y.shape[1])]
    subs = [_block_ms(c, sr) for c in chans]
    nb = min(len(s) for s in subs)
    if nb < 4:
        return None
    tot = sum(s[:nb] for s in subs)
    z = np.convolve(tot, np.ones(4) / 4, mode="valid")  # 400 ms blocks, 100 ms step
    lk = -0.691 + _db(z)
    g = z[lk > -70.0]
    if not len(g):
        return None
    rel = -0.691 + _db(g.mean()) - 10.0
    g2 = z[(lk > -70.0) & (lk > rel)]
    return float(-0.691 + _db(g2.mean())) if len(g2) else None


def voice_profile(y, sr, words=None):
    """{lufs, words_per_s, pause_median_s, lufs_scope, mix_lufs}; words: a document or spans."""
    T = THRESHOLDS
    spans = _spans(words)
    y = np.asarray(y, dtype=np.float64)
    mix = lufs(y, sr) if len(y) else None
    out = {"lufs": None, "lufs_scope": None, "mix_lufs": _r(mix, 2), "words_per_s": None,
           "pause_median_s": None, "words_n": len(spans) if spans else 0}
    if spans:
        idx = np.concatenate([np.arange(int(a * sr), min(len(y), int(b * sr))) for a, b in spans]) \
            if len(y) else np.zeros(0, int)
        if len(idx):
            out["lufs"], out["lufs_scope"] = _r(lufs(y[idx], sr), 2), "words"
        span = spans[-1][1] - spans[0][0]
        out["words_per_s"] = _r(len(spans) / span, 3) if span > 0 else None
        gaps = [b0 - a1 for (_, a1), (b0, _) in zip(spans[:-1], spans[1:]) if b0 - a1 >= T["pause_min_s"]]
        out["pause_median_s"] = _r(np.median(gaps), 3) if gaps else None
    elif mix is not None:
        out["lufs"], out["lufs_scope"] = _r(mix, 2), "mix"
    out.update(timbre(y, sr, spans))
    return out


def _span_idx(n, sr, spans):
    if not spans:
        return np.arange(n)
    parts = [np.arange(max(0, int(a * sr)), min(n, int(b * sr))) for a, b in spans]
    parts = [p for p in parts if len(p)]
    return np.concatenate(parts) if parts else np.zeros(0, int)


def spectral_tilt(x, sr):
    """dB per octave: the slope of the octave-band levels from tilt_lo_hz to tilt_hi_hz (None if silent)."""
    T = THRESHOLDS
    if len(x) < 2048:
        return None
    P, freqs = _power(x, sr, 2048, 1024)
    spec = P.mean(axis=0)
    lo, hi = T["tilt_lo_hz"], min(T["tilt_hi_hz"], sr / 2 * 0.95)
    centres, lv = [], []
    f = lo
    while f * 2 ** 0.5 <= hi:
        m = (freqs >= f / 2 ** 0.5) & (freqs < f * 2 ** 0.5)
        if m.any() and spec[m].sum() > 0:
            centres.append(math.log2(f))
            lv.append(float(_db(spec[m].sum())))
        f *= 2
    if len(centres) < 3:
        return None
    return float(np.polyfit(centres, lv, 1)[0])


def crest_db(x):
    """Peak (crest_pct percentile of |x|) over RMS in dB; lower = more compressed."""
    x = np.asarray(x, dtype=np.float64)
    rms = float(np.sqrt(np.mean(x ** 2))) if len(x) else 0.0
    if rms <= 1e-9:
        return None
    return float(20 * math.log10(max(float(np.percentile(np.abs(x), THRESHOLDS["crest_pct"])), 1e-12) / rms))


def rt60_estimate(y, sr, spans):
    """Room decay, estimated from the level after each phrase end that has quiet after it (T20 x 3):
    median seconds, or None with fewer than rt60_min_ends readable ends."""
    T = THRESHOLDS
    if not spans:
        return None
    dur = len(y) / sr
    vals = []
    for i, (_, end) in enumerate(spans):
        nxt = spans[i + 1][0] if i + 1 < len(spans) else dur
        gap = nxt - end
        if gap < T["rt60_min_gap_s"]:
            continue
        a, b = max(0.0, end - 0.03), min(dur, end + min(gap, T["rt60_window_s"]))
        tt, e = _env(y, sr, a, b)
        if len(e) < 10:
            continue
        head = tt <= end + 0.02
        if not head.any():
            continue
        k = int(np.argmax(np.where(head, e, -np.inf)))
        L0, t0 = float(e[k]), float(tt[k])
        below = np.nonzero(e[k:] < L0 - T["rt60_drop_db"])[0]
        if len(below):
            vals.append(3.0 * (float(tt[k + below[0]]) - t0))
            continue
        seg_t, seg_e = tt[k:], e[k:]
        if len(seg_t) >= 5 and seg_t[-1] - seg_t[0] > 0.05:
            slope = float(np.polyfit(seg_t, seg_e, 1)[0])
            if slope < -1e-3:
                vals.append(min(10.0, 60.0 / -slope))
    return float(np.median(vals)) if len(vals) >= T["rt60_min_ends"] else None


def de_ess(x, sr):
    """Sibilance band energy over 100 Hz..5 kHz, in dB (higher = more sibilant / less de-essed)."""
    lo_s, hi_s = THRESHOLDS["sib_band_hz"]
    if len(x) < 1024 or sr / 2 <= lo_s:
        return None
    P, freqs = _power(x, sr, 1024, 512)
    spec = P.mean(axis=0)
    sib = spec[(freqs >= lo_s) & (freqs < min(hi_s, sr / 2))].sum()
    rest = spec[(freqs >= 100) & (freqs < lo_s)].sum()
    if sib <= 0 or rest <= 0:
        return None
    return float(_db(sib) - _db(rest))


def timbre(y, sr, spans=None):
    """{spectral_tilt_db_per_oct, crest_db, rt60_est_s, de_ess_db}, from the word audio (the mix without words)."""
    y = _mono(np.asarray(y, dtype=np.float64))
    x = y[_span_idx(len(y), sr, spans)] if len(y) else y
    return {"spectral_tilt_db_per_oct": _r(spectral_tilt(x, sr), 2), "crest_db": _r(crest_db(x), 2),
            "rt60_est_s": _r(rt60_estimate(y, sr, spans), 3), "de_ess_db": _r(de_ess(x, sr), 2),
            "timbre_scope": "words" if spans else "mix"}


def music_sections(y, sr, mask_s=None):
    """[{t0, t1, energy}] from Foote novelty over per-window band levels (checkerboard kernel on the
    self-similarity matrix). mask_s: per-10 ms booleans of music-alone time; energy = mean dBFS there."""
    T = THRESHOLDS
    L, hop_s, _ = _band_levels(y, sr)
    dur = len(_mono(y)) / sr
    w = max(1, int(round(T["section_win_s"] / hop_s)))
    n = len(L) // w
    if n < 2:
        return [{"t0": 0.0, "t1": _r(dur, 2), "energy": None}] if dur > 0 else []
    F = L[: n * w].reshape(n, w, -1).mean(axis=1)
    # similarity from the RMS dB distance between windows (a steady bed is 1 everywhere, so a tiny
    # wobble never becomes a boundary); Foote novelty normalised to 0..1
    d = np.sqrt(((F[:, None, :] - F[None, :, :]) ** 2).mean(axis=2))
    S = np.exp(-(d / T["section_db_scale"]) ** 2)
    kw = T["section_kernel_w"]
    sign = np.concatenate([-np.ones(kw), np.ones(kw)])
    K = np.outer(sign, sign)
    nov = np.zeros(n)
    for i in range(1, n):
        k = min(kw, i, n - i)
        sub = S[i - k:i + k, i - k:i + k]
        kk = K[kw - k:kw + k, kw - k:kw + k]
        nov[i] = max(0.0, float((sub * kk).sum()) / (2 * k * k))
    min_w = max(1, int(round(T["section_min_s"] / T["section_win_s"])))
    bounds = []
    for i in np.argsort(-nov):
        if nov[i] < T["section_min_novelty"]:
            break
        if all(abs(i - b) >= min_w for b in bounds) and min_w <= i <= n - min_w:
            bounds.append(int(i))
    edges = [0] + sorted(bounds) + [n]
    _, lv = _env(_mono(y), sr, 0.0, dur, hop_s=hop_s)  # dBFS, same scale as energy_per_s
    lin = 10 ** (lv / 10)
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        seg = lin[a * w:b * w]
        if mask_s is not None:
            m = np.asarray(mask_s[a * w:b * w], bool)
            k = min(len(m), len(seg))
            seg = seg[:k][m[:k]]
        out.append({"t0": _r(a * w * hop_s, 2), "t1": _r(dur if b == n else b * w * hop_s, 2),
                    "energy": _r(float(_db(seg.mean())), 1) if len(seg) else None})
    return out


# ---------------------------------------------------------------- coupling

def sync_offsets(events, targets, window=None):
    """For each event onset, the offset to the nearest target within the window (event - target):
    {median, p10, p90, n}. Negative = the event comes first."""
    window = THRESHOLDS["couple_window_s"] if window is None else window
    tg = np.asarray(sorted(targets or []), float)
    offs = []
    if len(tg):
        for e in events or []:
            d = e - tg
            k = int(np.argmin(np.abs(d)))
            if abs(d[k]) <= window:
                offs.append(float(d[k]))
    return stats(offs)


def coupled_share(targets, events, window=None):
    """Share of targets that have an event within the window, or None without targets."""
    window = THRESHOLDS["couple_window_s"] if window is None else window
    if not targets:
        return None
    ev = np.asarray(events or [], float)
    if not len(ev):
        return 0.0
    return _r(np.mean([float(np.min(np.abs(ev - t))) <= window for t in targets]), 3)


# ---------------------------------------------------------------- whole analysis

def analyse(y, sr, words=None, cuts=None, moves=None, stem=None, stems="none"):
    """sound.json body (without the source block) for audio y (mono or (n, channels)).

    stem: the non-vocal stem (same rate and length), when a separator ran; SFX and tempo are then
    measured on it, the voice and ducking on the mix. The coupling shares and offsets leave out
    "unknown" events."""
    not_run, why = [], {}
    spans = _spans(words)
    if spans is None:
        not_run += ["voice_filter", "voice_rate", "ducking", "coupling_words"]
        why.update({k: "no --words" for k in ("voice_filter", "voice_rate", "ducking", "coupling_words")})
    music, beats = _music(y, sr, spans, cuts, stem)
    voice = voice_profile(y, sr, spans)
    vdb = voice_level_db(y, sr, spans)
    sfx_dropped = {}
    sfx = detect_sfx(y if stem is None else stem, sr, spans, beats if music["bpm"] is not None else None, vdb,
                     log=sfx_dropped)
    ts = [e["t"] for e in sfx if e["class_guess"] != "unknown"]
    if cuts is None:
        not_run.append("coupling_cuts")
        why["coupling_cuts"] = "no --measure"
    if moves is None:
        not_run.append("coupling_moves")
        why["coupling_moves"] = "no --motion"
    coupling = {"window_s": THRESHOLDS["couple_window_s"],
                "cut_with_sfx_share": coupled_share(cuts, ts) if cuts is not None else None,
                "move_with_sfx_share": coupled_share(moves, ts) if moves is not None else None,
                "offsets": {"sfx_vs_cuts": sync_offsets(ts, cuts or []),
                            "sfx_vs_moves": sync_offsets(ts, moves or []),
                            "sfx_vs_words": sync_offsets(ts, [a for a, _ in spans] if spans else [])}}
    return {"stems": stems, "sfx": sfx, "sfx_dropped": sfx_dropped, "music": music, "voice": voice, "coupling": coupling,
            "labels_are_guesses": True, "not_run": not_run, "not_run_why": why}


def empty_doc(spans, cuts, moves, reason):
    """Valid sound.json body when the audio cannot be measured."""
    music = {"present": None, "bpm": None, "bpm_conf": None, "bpm_method": None, "bpm_octave_check": None,
             "onset_periodicity_bpm": None, "beats_n": 0, "energy_per_s": [], "bed_level_db": None,
             "ducking_db": None, "beat_cut_share": None}
    voice = {"lufs": None, "lufs_scope": None, "mix_lufs": None, "words_per_s": None, "pause_median_s": None,
             "words_n": len(spans) if spans else 0}
    if spans:
        span = spans[-1][1] - spans[0][0]
        voice["words_per_s"] = _r(len(spans) / span, 3) if span > 0 else None
    empty = stats([])
    return {"stems": "none", "sfx": [], "music": music, "voice": voice,
            "coupling": {"window_s": THRESHOLDS["couple_window_s"], "cut_with_sfx_share": None,
                         "move_with_sfx_share": None,
                         "offsets": {"sfx_vs_cuts": empty, "sfx_vs_moves": dict(empty), "sfx_vs_words": dict(empty)}},
            "labels_are_guesses": True, "not_run": ["sound"], "not_run_why": {"sound": reason}}


# ---------------------------------------------------------------- inputs

def require_ffmpeg():
    for tool in ("ffmpeg", "ffprobe"):
        if not shutil.which(tool):
            raise MissingTool(f"error: {tool} not found on PATH; install ffmpeg "
                              "(winget install Gyan.FFmpeg / brew install ffmpeg / apt install ffmpeg)")


DEMUCS_MODEL = "htdemucs"


def _torch_hub_dir():
    import os
    if os.environ.get("TORCH_HOME"):
        return Path(os.environ["TORCH_HOME"]) / "hub"
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "torch" / "hub"


def demucs_status(model=DEMUCS_MODEL):
    """("missing" | "not_cached" | "cached", detail). Never imports torch and never downloads:
    reads the model's signatures from demucs' own remote yaml and looks for their weight files
    in the torch hub checkpoint cache."""
    import importlib.util
    spec = importlib.util.find_spec("demucs")
    if spec is None or not spec.submodule_search_locations:
        return "missing", "demucs is not installed"
    yml = Path(list(spec.submodule_search_locations)[0]) / "remote" / f"{model}.yaml"
    try:
        sigs = re.findall(r"\b([0-9a-f]{8})\b", yml.read_text(encoding="utf-8"))
    except OSError:
        return "not_cached", f"no {model}.yaml in the demucs package"
    ck = _torch_hub_dir() / "checkpoints"
    have = [g for g in sigs if ck.is_dir() and any(ck.glob(f"{g}-*.th"))]
    if sigs and len(have) == len(sigs):
        return "cached", str(ck)
    return "not_cached", f"{model} weights not in the torch hub cache ({len(have)}/{len(sigs)})"


def separate_no_vocals(y, sr, allow_download=False, model=DEMUCS_MODEL):
    """(non-vocal stem at sr, same length | None, "demucs" | "none", reason). Runs demucs two-stem
    separation in a subprocess, only when the weights are cached or allow_download is set."""
    import tempfile
    state, detail = demucs_status(model)
    if state == "missing" or (state == "not_cached" and not allow_download):
        return None, "none", detail
    with tempfile.TemporaryDirectory(prefix="sound_events_") as td:
        wav = Path(td) / "mix.wav"
        pcm = (np.clip(np.asarray(y, np.float64), -1, 1) * 32767).astype("<i2")
        if pcm.ndim == 1:
            pcm = np.stack([pcm, pcm], axis=1)
        import wave
        with wave.open(str(wav), "wb") as wf:
            wf.setnchannels(2)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(pcm.tobytes())
        r = subprocess.run([sys.executable, "-m", "demucs.separate", "--two-stems", "vocals", "-n", model,
                            "-o", td, str(wav)], capture_output=True)
        out = Path(td) / model / "mix" / "no_vocals.wav"
        if r.returncode != 0 or not out.is_file():
            msg = r.stderr.decode("utf-8", "replace").strip().splitlines()
            return None, "none", "demucs failed: " + (msg[-1] if msg else "no output")
        stem = load_audio(out, sr)
    n = len(y)
    stem = stem[:n] if len(stem) >= n else np.concatenate([stem, np.zeros((n - len(stem), 2), np.float32)])
    return stem, "demucs", detail


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def probe(src):
    """(has_audio, duration_s) via ffprobe."""
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type:format=duration",
                        "-of", "json", str(src)], capture_output=True)
    if r.returncode != 0:
        msg = r.stderr.decode("utf-8", "replace").strip().splitlines()
        raise BadInput("ffprobe could not read the source: " + (msg[-1] if msg else "unknown error"))
    doc = json.loads(r.stdout.decode("utf-8", "replace") or "{}")
    has_audio = any(s.get("codec_type") == "audio" for s in doc.get("streams", []))
    try:
        dur = float(doc.get("format", {}).get("duration"))
    except (TypeError, ValueError):
        dur = None
    return has_audio, dur


def load_audio(src, sr=AR):
    """Stereo float32 in [-1, 1] at sr through ffmpeg (a mono source is copied to both channels)."""
    r = subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-i", str(src), "-vn", "-ac", "2",
                        "-ar", str(sr), "-f", "s16le", "-"], capture_output=True)
    if r.returncode != 0:
        msg = r.stderr.decode("utf-8", "replace").strip().splitlines()
        raise BadInput("ffmpeg could not read the source audio: " + (msg[-1] if msg else "unknown error"))
    raw = r.stdout[: len(r.stdout) // 4 * 4]  # whole stereo int16 frames only
    return np.frombuffer(raw, dtype=np.int16).reshape(-1, 2).astype(np.float32) / 32768.0


def _read_json(path, what):
    try:
        return json.loads(Path(path).read_bytes().decode("utf-8-sig"))
    except FileNotFoundError:
        raise BadInput(f"{what} not found: {path}")
    except (ValueError, UnicodeDecodeError) as e:
        raise BadInput(f"{what} is not valid JSON ({path}): {e}")


def cut_times(doc):
    """cuts[].t (seconds) of a measure.json."""
    cuts = doc.get("cuts") if isinstance(doc, dict) else None
    if not isinstance(cuts, list):
        raise BadInput("measure json: expected a \"cuts\" list")
    return sorted(float(c["t"]) for c in cuts if isinstance(c, dict) and isinstance(c.get("t"), (int, float)))


def move_onsets(doc):
    """Every numeric t0 of an item of any "moves" list, at any depth, sorted and de-duplicated."""
    found = []

    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if k == "moves" and isinstance(v, list):
                    for m in v:
                        if isinstance(m, dict) and isinstance(m.get("t0"), (int, float)) \
                                and not isinstance(m.get("t0"), bool):
                            found.append(float(m["t0"]))
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
    walk(doc)
    out = []
    for t in sorted(found):
        if not out or t - out[-1] > 0.001:
            out.append(t)
    return out


def run(src, out, words_path=None, measure_path=None, motion_path=None, allow_download=False):
    src = Path(src)
    if not src.is_file():
        raise BadInput(f"source not found: {src}")
    words = _read_json(words_path, "words json") if words_path else None
    spans = word_spans(words) if words is not None else None
    cuts = cut_times(_read_json(measure_path, "measure json")) if measure_path else None
    moves = move_onsets(_read_json(motion_path, "motion json")) if motion_path else None
    h0 = sha256(src)
    has_audio, dur = probe(src)
    if not has_audio:
        body = empty_doc(spans, cuts, moves, "the source has no audio stream")
        if cuts is None:
            body["not_run"].append("coupling_cuts")
        if moves is None:
            body["not_run"].append("coupling_moves")
    else:
        y = load_audio(src)
        if len(y) == 0:
            body = empty_doc(spans, cuts, moves, "the audio stream decodes to no samples")
        else:
            dur = len(y) / AR
            stem, stems, why_stem = separate_no_vocals(y, AR, allow_download)
            body = analyse(y, AR, spans, cuts, moves, stem=stem, stems=stems)
            if stem is None:
                body["not_run"].append("stems")
                body["not_run_why"]["stems"] = why_stem
    h1 = sha256(src)
    if h1 != h0:
        raise RuntimeError("the source changed while it was read (sha256 before and after differ)")
    doc = {"schema": SCHEMA, "source": {"name": src.name, "sha256": h0, "duration_s": _r(dur, 3),
                                        "has_audio": has_audio, "analysis_sr": AR},
           "inputs": {"words": words_path is not None, "measure": measure_path is not None,
                      "motion": motion_path is not None}, **body}
    store.write_json(Path(out), doc)
    return doc


def main(argv=None):
    store.print_utf8()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", help="video or audio file (read only)")
    ap.add_argument("--out", required=True, help="output json, e.g. <dir>/sound.json")
    ap.add_argument("--words", help="words json (a list or {\"words\": [...]}, seconds or start_ms/end_ms)")
    ap.add_argument("--measure", help="measure.json (its cuts[].t are used)")
    ap.add_argument("--motion", help="motion.json (every moves[].t0 is used)")
    ap.add_argument("--no-librosa", action="store_true", help="use the numpy onset envelope even if librosa imports")
    ap.add_argument("--allow-download", action="store_true",
                    help="let demucs download its model when it is not cached (otherwise no stems)")
    a = ap.parse_args(argv)
    global USE_LIBROSA
    if a.no_librosa:
        USE_LIBROSA = False
    try:
        require_ffmpeg()
        doc = run(a.source, a.out, a.words, a.measure, a.motion, a.allow_download)
    except MissingTool as e:
        print(str(e), file=sys.stderr)
        return 3
    except BadInput as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except OSError as e:
        print(f"error: cannot read or write a file: {e}", file=sys.stderr)
        return 2
    except RuntimeError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except Exception as e:  # never a traceback
        print(f"error: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    kinds = {}
    for e in doc["sfx"]:
        kinds[e["class_guess"]] = kinds.get(e["class_guess"], 0) + 1
    m = doc["music"]
    print(f"{len(doc['sfx'])} sfx {kinds}; music bpm {m['bpm']} ducking {m['ducking_db']} dB; "
          f"voice {doc['voice']['lufs']} LUFS; not run {doc['not_run']}")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Sound events of one reference video: SFX onsets with class guesses, music and voice, sync.

Reads the source (never modifies it: it is hashed before and after), decodes its audio once
through ffmpeg (stereo, 24 kHz) and writes one sound.json (schema "sound/1").

Usage:
  sound_events.py <video|audio> --out <dir>/sound.json [--words w.json] [--measure m.json]
                  [--motion mo.json] [--no-librosa]

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
    "bpm_min_conf": 0.3,           # weaker periodicity than this: no BPM
    "bpm_min_s": 4.0,              # shorter audio: no BPM
    "beat_hit_s": 0.07,            # a cut within 70 ms of a beat is on the beat
    "music_bed_min_db": -45.0,     # median music-alone level above this: a bed plays (needed for a BPM)
    "phrase_gap_s": 0.35,          # words closer than this form one phrase (for ducking)
    "duck_inner_s": 0.15,          # voice frames: at least this far inside a phrase
    "duck_outer_s": 0.25,          # music-alone frames: at least this far from any phrase
    "duck_min_s": 0.5,             # both sides need this much audio
    "duck_bins_db": 30.0,          # music bins: within 30 dB of the strongest music-alone bin
    "duck_percentile": 25.0,       # bins the voice leaves alone have the lowest ratio
    # voice
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
    # hit: instant attack with the energy in the sub/bass (impact, boom).
    "hit": [[("attack_s", "<=", 0.03, 0.02), ("low_share", ">=", 0.4, 0.3), ("dur_s", ">=", 0.06, 0.05),
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
         "low_share": 0.0, "band_rise_db": -120.0, "bed_subtracted": False, "mix_peak_db": -120.0}
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
           "band_rise_db": -120.0}
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

def detect_sfx(y, sr, spans=None, beats=None, voice_db=None):
    """SFX events [{t, class_guess, confidence, dur_s, level_vs_voice_db, ...}] after the voice and
    music filters, classification and grouping."""
    T = THRESHOLDS
    y = _mono(y)
    cands = _candidates(y, sr)
    kept = []
    for c in cands:
        t = c["t"]
        c["over_voice"] = False
        if spans:
            if any(abs(t - a) <= T["voice_onset_s"] for a, _ in spans):
                continue
            inside = any(a <= t <= b for a, b in spans)
            if inside and c["sharp_bb_db"] < T["over_voice_sharp_db"] and c["sub_rise_db"] < T["over_voice_sub_db"]:
                continue
            c["over_voice"] = inside
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
    events = []
    i = 0
    while i < len(kept):
        c = kept[i]
        # an onset inside this event's body without a sharp attack is part of it (riser steps,
        # the flutter of a swell); the body is measured without the next-onset limit
        body = features(y, sr, c["t"])["dur_s"]
        while i + 1 < len(kept) and kept[i + 1]["t"] < c["t"] + body and                 kept[i + 1]["sharp_bb_db"] < T["inside_sharp_db"]:
            kept.pop(i + 1)
        nxt = kept[i + 1]["t"] if i + 1 < len(kept) else None
        ev = classify(y, sr, c["t"], nxt, voice_db)
        ev = {"t": _r(c["t"]), **ev, "peak_db": _r(ev["features"]["peak_db"], 1), "over_voice": c["over_voice"]}
        events.append(ev)
        i += 1
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


def onset_envelope(y, sr):
    """(envelope at 10 ms hops, hop_s, method): librosa's onset strength when it imports, else
    the numpy band flux."""
    y = _mono(y)
    lr = _get_librosa()
    hop = max(1, int(round(HOP_S * sr)))
    if lr is not None:
        try:
            env = lr.onset.onset_strength(y=y.astype(np.float32), sr=sr, hop_length=hop)
            return np.asarray(env, float), hop / sr, "librosa-onset-acf"
        except Exception:
            pass
    nov, hop_s = novelty(y, sr)
    return nov, hop_s, "numpy-flux-acf"


def tempo(env, hop_s):
    """(bpm|None, conf, octave_check, period_frames|None) from the envelope autocorrelation."""
    T = THRESHOLDS
    n = len(env)
    lo_l = int(math.floor(60.0 / T["bpm_range"][1] / hop_s))
    hi_l = int(math.ceil(60.0 / T["bpm_range"][0] / hop_s))
    if n * hop_s < T["bpm_min_s"] or n < 2 * hi_l + 2:
        return None, 0.0, None, None
    e = env - env.mean()
    if not np.any(e):
        return None, 0.0, None, None
    F = np.fft.rfft(e, 2 * n)
    acf = np.fft.irfft(F * np.conj(F))[:n]
    acf = acf / acf[0] * n / (n - np.arange(n))  # unbiased, normalised
    lags = np.arange(lo_l, hi_l + 1)
    bpms = 60.0 / (lags * hop_s)
    prior = np.exp(-0.5 * (np.log2(bpms / T["bpm_prior_bpm"]) / T["bpm_prior_oct"]) ** 2)
    score = np.maximum(acf[lags], 0) * prior
    li = int(lags[int(np.argmax(score))])
    a, b, c = acf[li - 1], acf[li], acf[li + 1]
    den = a - 2 * b + c
    d = 0.5 * (a - c) / den if den < 0 else 0.0
    period = li + max(-0.5, min(0.5, d))
    bpm = 60.0 / (period * hop_s)
    conf = float(max(0.0, min(1.0, b)))

    def cand(p):
        p_i = int(round(p))
        if p_i < 1 or p_i >= n:
            return {"bpm": _r(60.0 / (p * hop_s), 2), "acf": None}
        return {"bpm": _r(60.0 / (p * hop_s), 2), "acf": _r(acf[p_i], 3)}
    octave = {"picked": {"bpm": _r(bpm, 2), "acf": _r(b, 3)}, "half_bpm": cand(period * 2),
              "double_bpm": cand(period / 2),
              "rule": f"acf x log-normal prior ({T['bpm_prior_bpm']:.0f} BPM, {T['bpm_prior_oct']} octave), "
                      f"range {T['bpm_range'][0]:.0f}-{T['bpm_range'][1]:.0f} BPM"}
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


def _music(y, sr, spans=None, cuts=None):
    T = THRESHOLDS
    y = _mono(y)
    env, hop_s, method = onset_envelope(y, sr)
    bpm, conf, octave, period = tempo(env, hop_s)
    beats = beat_times(env, hop_s, period) if period else []
    dur = len(y) / sr
    # music-alone frames (10 ms): away from every phrase, or everything when there are no words
    _, lv = _env(y, sr, 0.0, dur, hop_s=HOP_S)
    tt = np.arange(len(lv)) * HOP_S
    phrases = _phrases(spans, T["phrase_gap_s"]) if spans else []
    alone = np.ones(len(lv), bool)
    for a, b in phrases:
        alone &= ~((tt > a - T["duck_outer_s"]) & (tt < b + T["duck_outer_s"]))
    bed_db = float(np.median(lv[alone])) if alone.any() else None
    # music = a bed that keeps sounding between the onsets; a regular train of SFX over room tone
    # is periodic too, so without a bed its tempo is only recorded as onset_periodicity_bpm
    present = bed_db is not None and bed_db >= T["music_bed_min_db"]
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
            "bed_level_db": _r(bed_db, 1), "ducking_db": _r(ducking, 1), "beat_cut_share": share}
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


def music_profile(y, sr, words=None, cuts=None):
    """{bpm|None, bpm_conf, energy_per_s, ducking_db, beat_cut_share|None, ...}.
    words: a words document or [(start_s, end_s)]; cuts: [t_s]."""
    return _music(y, sr, _spans(words), cuts)[0]


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

def analyse(y, sr, words=None, cuts=None, moves=None):
    """sound.json body (without the source block) for audio y (mono or (n, channels))."""
    not_run, why = [], {}
    spans = _spans(words)
    if spans is None:
        not_run += ["voice_filter", "voice_rate", "ducking", "coupling_words"]
        why.update({k: "no --words" for k in ("voice_filter", "voice_rate", "ducking", "coupling_words")})
    music, beats = _music(y, sr, spans, cuts)
    voice = voice_profile(y, sr, spans)
    vdb = voice_level_db(y, sr, spans)
    sfx = detect_sfx(y, sr, spans, beats if music["bpm"] is not None else None, vdb)
    ts = [e["t"] for e in sfx]
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
    return {"sfx": sfx, "music": music, "voice": voice, "coupling": coupling, "labels_are_guesses": True,
            "not_run": not_run, "not_run_why": why}


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
    return {"sfx": [], "music": music, "voice": voice,
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
    return np.frombuffer(r.stdout, dtype=np.int16).reshape(-1, 2).astype(np.float32) / 32768.0


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


def run(src, out, words_path=None, measure_path=None, motion_path=None):
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
            body = analyse(y, AR, spans, cuts, moves)
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
    a = ap.parse_args(argv)
    global USE_LIBROSA
    if a.no_librosa:
        USE_LIBROSA = False
    try:
        require_ffmpeg()
        doc = run(a.source, a.out, a.words, a.measure, a.motion)
    except MissingTool as e:
        print(str(e), file=sys.stderr)
        return 3
    except BadInput as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except RuntimeError as e:
        print(f"error: {e}", file=sys.stderr)
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

#!/usr/bin/env python3
"""Precise, frame-accurate cut of kept word spans, rendered in one ffmpeg pass.

Every in/out point is chosen from the real audio: inside the pause around the boundary, on a
frame boundary of --fps, at the quietest 10 ms slices, never inside a word (it uses the
force-aligned word edges from align.py). Rules:
  - in-point in the silence BEFORE a sentence, at least --inside-ms inside the silence
  - --lead-ms before the first word, --tail-ms after the last word
  - spans whose gap is < --merge-gap-ms are merged; pieces shorter than --sliver-frames are dropped
  - pauses inside a kept span longer than --pause-max-ms (and voiced gaps: fillers, hidden
    repeats) are shortened to about 2 x --pause-keep-ms, cut only on quiet frames (local room
    tone + 10 dB); a gap with no quiet frame is kept and listed in cuts.json long_gaps_kept
  - an in/out point with no quiet frame in reach lands on the quietest frame and is listed in
    cuts.json soft_joins (never a silent failure)
  - a word flagged low_score is never cut inside the union of its ASR and refined spans; a word
    flagged unaligned / unaligned_script is never cut inside its ASR span
  - 15 ms equal-power audio crossfade at every join (both sides are room tone, so the track
    never drops to digital silence)
  - both output channels at the same level: a source whose channels differ by more than 0.5 dB
    is balanced (cuts.json channel_balance_db = the source difference, left minus right)

Usage:
  cut.py <source> <words.aligned.json> <keep.json> --out <dir/stem> [--preset reels|lesson]
         [--fps N] [--lead-ms N] ... [--no-render]

Presets: lesson (default) = --fps 25 --lead-ms 80 --tail-ms 120 --merge-gap-ms 250
--pause-max-ms 450 --pause-keep-ms 150, no end air; reels = --fps 30 --lead-ms 60 --tail-ms 80
--merge-gap-ms 180 --pause-max-ms 260 --pause-keep-ms 70 --end-air-ms 1000.
Explicit flags override the preset.

keep.json: {"keep": [[i0, i1], ...]} (inclusive word indices of the raw transcript).
Optional "retime": {"<word index>": [start_ms, end_ms]} pins a word the aligner put on the wrong
take (the transcript merged two takes); the false take then sits in a voiced gap and is cut out.
Optional "exclude_ms": [[a, b], ...] source ranges that must never be in the cut (a hidden
fragment the transcript does not list); a piece overlapping one is split around it.
Optional "placeholders": [{"id": "...", "after_word": i, "dur_s": 20.0, "note": "..."[, "tone_ms": [a, b]]}]
inserts black video + looped room tone (quietest steady 3 s of the same recording within 2 min of
the join, or tone_ms) after word i, which must end a piece (a join between sentences); its start/end
in the cut are written to placeholders.json next to the output and to cuts.json "placeholders".
Optional "fade_out_ms": [[src_out_ms, ms], ...] / "fade_in_ms": [[src_in_ms, ms], ...] give the piece
that ends (starts) at that source time (+-20 ms) a longer equal-power fade of ms ending at its
out-point (starting at its in-point), for an out-point on a soft voiced tail.
Optional "set_in_ms" / "set_out_ms": [[old_src_ms, new_src_ms], ...] move the in (out) point of the
piece that starts (ends) at old_src_ms (+-20 ms) to the frame nearest new_src_ms.
--sentence-keep-ms keeps more air at shortened pauses after . ? ! : (sentence ends).
--end-air-ms N places the out-point of the LAST word of the whole cut about N ms after the word's
end (quietest frame within -20/+40 ms of that).

Writes <stem>.mp4, <stem>.cuts.json (pieces with src_in/src_out/dst_in, joins, long_gaps_kept,
soft_joins, placeholders) and <stem>.words.json (kept words remapped to the cut timeline).

Exit: 0 done, 1 error (bad input, ffmpeg failed), 2 usage, 3 ffmpeg missing.
"""
import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import store  # noqa: E402

AR = 48000
SL = AR // 100  # 10 ms

DEFAULTS = {
    "fps": 25, "lead_ms": 80, "tail_ms": 120, "inside_ms": 60, "merge_gap_ms": 250,
    "pause_max_ms": 450, "pause_keep_ms": 150, "sentence_keep_ms": None, "pause_min_cut_frames": 3,
    "pause_snap_ms": 100, "end_air_ms": None, "sliver_frames": 6, "xfade_ms": 15.0, "crf": 18,
    "excl_overlap": False,
}
PRESETS = {
    "lesson": {"fps": 25, "lead_ms": 80, "tail_ms": 120, "merge_gap_ms": 250,
               "pause_max_ms": 450, "pause_keep_ms": 150, "end_air_ms": None},
    "reels": {"fps": 30, "lead_ms": 60, "tail_ms": 80, "merge_gap_ms": 180,
              "pause_max_ms": 260, "pause_keep_ms": 70, "end_air_ms": 1000},
}
SETTING_KEYS = ("lead_ms", "tail_ms", "inside_ms", "merge_gap_ms", "pause_max_ms", "pause_keep_ms",
                "sentence_keep_ms", "pause_min_cut_frames", "pause_snap_ms", "sliver_frames", "xfade_ms",
                "end_air_ms")
PROTECT_FLAGS = ("low_score", "unaligned", "unaligned_script")
QUIET_DB = 10.0        # quiet = local room-tone floor + 10 dB
BALANCE_TOL_DB = 0.5   # channels further apart than this are balanced
DEAD_CHANNEL_DB = 20.0  # a channel this far below the other is replaced, not boosted
SENTENCE_END = (".", "?", "!", ":")
DIGITAL_SILENCE_DB = -90.0  # a slice below this is digital silence, never used as room tone


class MissingTool(Exception):
    """ffmpeg/ffprobe not found: the CLI prints one line and exits 3."""


def resolve_opts(preset=None, overrides=None):
    """Course defaults, then the preset, then explicit overrides (None values are ignored)."""
    if preset is not None and preset not in PRESETS:
        raise ValueError(f"unknown preset {preset!r}; choose one of {', '.join(sorted(PRESETS))}")
    o = dict(DEFAULTS)
    o.update(PRESETS[preset or "lesson"])
    o.update({k: v for k, v in (overrides or {}).items() if v is not None})
    if o.get("end_air_ms") == 0:  # --end-air-ms 0 switches the end air off (also under --preset reels)
        o["end_air_ms"] = None
    return o


# ---------------------------------------------------------------- audio helpers

def require_ffmpeg():
    for tool in ("ffmpeg", "ffprobe"):
        if not shutil.which(tool):
            raise MissingTool(f"error: {tool} not found on PATH; install ffmpeg "
                              "(winget install Gyan.FFmpeg / brew install ffmpeg / apt install ffmpeg)")


def load_audio(src):
    """Stereo 48 kHz float32 in [-1, 1] through ffmpeg (a mono source is copied to both channels)."""
    r = subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-i", str(src), "-vn", "-ac", "2",
                        "-ar", str(AR), "-f", "s16le", "-"], capture_output=True)
    if r.returncode != 0:
        msg = r.stderr.decode("utf-8", "replace").strip().splitlines()
        raise RuntimeError("ffmpeg could not read the source audio: " + (msg[-1] if msg else "unknown error"))
    return np.frombuffer(r.stdout, dtype=np.int16).reshape(-1, 2).astype(np.float32) / 32768.0


def slice_db(mono):
    """RMS level of every 10 ms slice, dBFS."""
    n = len(mono) // SL
    x = np.asarray(mono[: n * SL], dtype=np.float64).reshape(n, SL)
    return 20 * np.log10(np.sqrt((x * x).mean(axis=1)) + 1e-9)


def probe_video(src):
    """(width, height, duration_ms or None) of the first video stream."""
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                        "stream=width,height,duration:stream_tags=DURATION:format=duration", "-of", "json",
                        str(src)],
                       capture_output=True, text=True, encoding="utf-8")
    if r.returncode != 0:
        raise RuntimeError("ffprobe could not read the source: " + (r.stderr.strip().splitlines() or ["?"])[-1])
    doc = json.loads(r.stdout or "{}")
    streams = doc.get("streams") or []
    if not streams:
        raise RuntimeError("the source has no video stream")
    s = streams[0]
    dur_ms = _seconds(s.get("duration"))
    if dur_ms is None:  # Matroska/WebM keep the stream duration in a tag, HH:MM:SS.fraction
        dur_ms = _seconds((s.get("tags") or {}).get("DURATION"))
    if dur_ms is None:
        dur_ms = _seconds((doc.get("format") or {}).get("duration"))
    return int(s["width"]), int(s["height"]), dur_ms


def _seconds(v):
    """ffprobe duration ("12.34" or "00:00:12.340000000") -> ms; None when absent."""
    if v in (None, "", "N/A"):
        return None
    try:
        if ":" in str(v):
            h, m, sec = str(v).split(":")
            return (int(h) * 3600 + int(m) * 60 + float(sec)) * 1000
        return float(v) * 1000
    except ValueError:
        return None


def channel_balance(audio, db=None):
    """Measure the source channel difference on voiced slices and choose a fix.

    Returns {"channel_balance_db": L-R dB, "channel_balance": none|gain|copy_left|copy_right,
    "gains": [gL, gR]}."""
    n = len(audio) // SL
    if n == 0:
        return {"channel_balance_db": 0.0, "channel_balance": "none", "gains": [1.0, 1.0]}
    x = np.asarray(audio[: n * SL], dtype=np.float64).reshape(n, SL, 2)
    if db is None:
        db = slice_db(audio.mean(axis=1))
    db = db[:n]
    voiced = db > float(np.percentile(db, 10)) + QUIET_DB
    sel = x[voiced] if voiced.any() else x
    rms = np.sqrt((sel ** 2).mean(axis=(0, 1))) + 1e-12
    diff = float(20 * np.log10(rms[0] / rms[1]))
    out = {"channel_balance_db": round(diff, 2), "channel_balance": "none", "gains": [1.0, 1.0]}
    if abs(diff) <= BALANCE_TOL_DB:
        return out
    if abs(diff) >= DEAD_CHANNEL_DB:
        out["channel_balance"] = "copy_left" if diff > 0 else "copy_right"
    else:
        # raise the quieter channel to the louder one (the voice level stays what it was)
        g = 10 ** (abs(diff) / 20)
        out["channel_balance"] = "gain"
        out["gains"] = [1.0, g] if diff > 0 else [g, 1.0]
    return out


def apply_balance(audio, bal):
    mode = bal.get("channel_balance", "none")
    if mode == "none":
        return audio
    if mode == "copy_left":
        return np.repeat(audio[:, :1], 2, axis=1)
    if mode == "copy_right":
        return np.repeat(audio[:, 1:], 2, axis=1)
    out = audio * np.asarray(bal["gains"], dtype=np.float32)[None, :]
    peak = float(np.abs(out).max()) if len(out) else 0.0
    if peak > 0.999:  # keep the raised channel out of clipping: lower both equally
        out *= np.float32(0.999 / peak)
    return out


# ---------------------------------------------------------------- frame picking

class Picker:
    """Frame-boundary energy and quietest-frame search on 10 ms slice levels."""

    def __init__(self, db, fps, protected=(), dur_ms=None):
        self.db, self.fms = db, 1000.0 / fps
        self.gfloor = float(np.percentile(db, 10)) if len(db) else -96.0
        self.protected = list(protected)
        dur_ms = len(db) * 10 if dur_ms is None else min(dur_ms, len(db) * 10)
        self.end_k = int(dur_ms // 10)                    # slices from here on are past the end
        self.fmax = math.floor(dur_ms / self.fms + 1e-9)  # last frame boundary inside the source

    def e(self, f):
        """energy at frame boundary f: loudest 10 ms slice within +-30 ms (the crossfade plus margin), dB.
        Slices past the end of the source count as loud (0 dB): there is nothing there to cut into."""
        k = int((f * self.fms) // 10)
        lo, hi = max(0, k - 3), k + 3
        vals = self.db[lo:min(hi, self.end_k)]
        m = float(vals.max()) if len(vals) else -96.0
        return max(m, 0.0) if hi > self.end_k else m

    def allowed(self, f):
        """a frame boundary strictly inside a flagged word's protected span is never a cut point."""
        t = f * self.fms
        return not any(a < t < b for a, b in self.protected)

    def pick(self, lo_ms, hi_ms, target_ms):
        f0, f1 = math.ceil(lo_ms / self.fms), min(math.floor(hi_ms / self.fms), self.fmax)
        cand = [f for f in range(max(0, f0), f1 + 1) if self.allowed(f)]
        if not cand:
            return None
        return min(cand, key=lambda f: (round(self.e(f), 0), abs(f * self.fms - target_ms)))

    def pick_outward(self, edge_ms, direction, reach_ms=5000):
        """last resort: quietest allowed frame in growing windows past edge_ms (direction +1 / -1), then
        the other way (an edge past the end of the source can only move back); always inside the source."""
        for d in (direction, -direction):
            for span in range(200, reach_ms + 1, 200):
                lo, hi = (edge_ms, edge_ms + span) if d > 0 else (edge_ms - span, edge_ms)
                f = self.pick(lo, hi, edge_ms)
                if f is not None:
                    return f
        return min(self.fmax, max(0, round(edge_ms / self.fms)))

    def quiet_thr(self, t_ms):
        """quiet threshold near t: local room-tone floor (10th percentile of +-3 s) + 10 dB."""
        k = int(t_ms // 10)
        loc = self.db[max(0, k - 300): k + 300]
        if not len(loc):
            return self.gfloor + QUIET_DB
        return min(float(np.percentile(loc, 10)) + QUIET_DB, self.gfloor + 20)


# ---------------------------------------------------------------- planning

def _cut_view(w):
    """word edges used for cut decisions: a flagged word occupies its ASR span too."""
    flag = w.get("flag")
    if flag not in PROTECT_FLAGS:
        return w
    s, e = w["start_ms"], w["end_ms"]
    if w.get("asr_start_ms") is not None:
        s = min(s, w["asr_start_ms"])
    if w.get("asr_end_ms") is not None:
        e = max(e, w["asr_end_ms"])
    return dict(w, start_ms=s, end_ms=e)


def merge_spans(spans, cut, merge_gap_ms):
    """merge keep spans: index-adjacent, or source gap < merge gap."""
    merged = []
    for s in sorted([list(x) for x in spans]):
        if merged:
            p = merged[-1]
            gap = cut[s[0]]["start_ms"] - cut[p[1]]["end_ms"]
            if s[0] <= p[1] + 1 or gap < merge_gap_ms:
                p[1] = max(p[1], s[1])
                continue
        merged.append(s)
    return merged


class _Planner:
    def __init__(self, words, keep, db, fps, o, dur_ms):
        self.o, self.fps, self.fms = o, fps, 1000.0 / fps
        self.keep, self.db = keep, db
        self.byi = {w["i"]: dict(w) for w in words}
        for i, (rs, re_) in (keep.get("retime") or {}).items():
            i = int(i)
            if i not in self.byi:
                raise ValueError(f"retime: no word {i}")
            self.byi[i] = dict(self.byi[i], start_ms=int(rs), end_ms=int(re_), flag="retimed")
        self.order = sorted(self.byi)
        self.pos = {i: n for n, i in enumerate(self.order)}
        self.cut = {i: _cut_view(self.byi[i]) for i in self.order}
        protected = [(c["start_ms"], c["end_ms"]) for i, c in self.cut.items()
                     if self.byi[i].get("flag") in PROTECT_FLAGS]
        self.P = Picker(db, fps, protected, dur_ms)
        self.dur_ms = dur_ms
        self.voiced_log, self.kept_long = [], []

    def nb(self, i, d):
        n = self.pos[i] + d
        return self.cut[self.order[n]] if 0 <= n < len(self.order) else None

    # -- pauses inside a kept span

    def voiced(self, g0, g1):
        """a gap >= 0.35 s inside a kept span that holds speech energy = hidden repeat / filler."""
        if g1 - g0 < 350:
            return False
        seg = self.db[int((g0 + 80) // 10): int((g1 - 80) // 10)]
        v = len(seg) > 0 and float((seg > self.P.gfloor + 18).mean()) > 0.4
        if v:
            self.voiced_log.append([round(g0 / 1000, 2), round(g1 / 1000, 2)])
        return v

    def split_points(self, wa, wb):
        """frames (out, in) that remove the middle of a pause, both on QUIET frame boundaries; None = keep it."""
        o, P, fms = self.o, self.P, self.fms
        thr = P.quiet_thr((wa["end_ms"] + wb["start_ms"]) / 2)
        f0 = math.ceil((wa["end_ms"] + o.inside_ms) / fms)
        f1 = min(math.floor((wb["start_ms"] - o.inside_ms) / fms), P.fmax)
        Q =[f for f in range(max(0, f0), f1 + 1) if P.e(f) < thr and P.allowed(f)]
        if not Q:
            return None
        k = o.pause_keep_ms
        if o.sentence_keep_ms is not None and wa["w"].rstrip().endswith(SENTENCE_END):
            k = o.sentence_keep_ms
        near_out = [f for f in Q if f * fms <= wa["end_ms"] + k + o.pause_snap_ms]
        near_in = [f for f in Q if f * fms >= wb["start_ms"] - k - o.pause_snap_ms]
        fo = min(near_out, key=lambda f: (round(P.e(f)), abs(f * fms - wa["end_ms"] - k))) if near_out else min(Q)
        fi = min(near_in, key=lambda f: (round(P.e(f)), abs(f * fms - wb["start_ms"] + k))) if near_in else max(Q)
        if fi - fo < o.pause_min_cut_frames:
            return None
        return fo, fi

    def shorten_pauses(self, idx):
        """split a kept span's words into groups at every pause that can be shortened."""
        groups, cur, splits = [], [idx[0]], []
        for i in idx[1:]:
            wa, wb = self.cut[cur[-1]], self.cut[i]
            g0, g1 = wa["end_ms"], wb["start_ms"]
            v = self.voiced(g0, g1)
            long_gap = g1 - g0 > self.o.pause_max_ms
            sp = self.split_points(wa, wb) if (long_gap or v) else None
            if sp:
                groups.append(cur)
                splits.append(sp)
                cur = [i]
            else:
                if v or long_gap:
                    self.kept_long.append({"after_word": cur[-1], "src": [round(g0 / 1000, 2), round(g1 / 1000, 2)],
                                           "why": "no quiet frame to cut on (filler / drawn-out sound / repeat)"})
                cur.append(i)
        groups.append(cur)
        return groups, splits

    # -- span edges

    def out_point(self, w, nxt, keep_ms, final=False):
        o, P = self.o, self.P
        lim = (nxt["start_ms"] - 40) if nxt else self.dur_ms
        if final and o.end_air_ms is not None:
            air = o.end_air_ms
            f = P.pick(w["end_ms"] + air - 20, min(w["end_ms"] + air + 40, lim), w["end_ms"] + air)
            if f is not None:
                return f, False
            # no room for the end air (the next word or the source end is closer): normal tail rule
        lo = w["end_ms"] + o.inside_ms
        hi = w["end_ms"] + keep_ms + 80
        f = P.pick(lo, min(hi, lim), w["end_ms"] + keep_ms)
        if f is not None:
            return f, False
        # words run together (aligned edges touch): take the deepest energy dip near the word end
        f = P.pick(w["end_ms"] - 40, min(w["end_ms"] + 160, self.dur_ms), w["end_ms"] + 40)
        if f is None:
            f = P.pick_outward(w["end_ms"], +1)
        return f, True

    def in_point(self, w, prv, keep_ms):
        P = self.P
        hi = w["start_ms"] - self.o.inside_ms
        lo = w["start_ms"] - keep_ms - 80
        lim = (prv["end_ms"] + 40) if prv else 0
        f = P.pick(max(lo, lim), hi, w["start_ms"] - keep_ms)
        if f is not None:
            return f, False
        f = P.pick(max(0, w["start_ms"] - 160), w["start_ms"] + 40, w["start_ms"] - 40)
        if f is None:
            f = P.pick_outward(w["start_ms"], -1)
        return f, True

    def span_pieces(self, spans):
        o, pieces = self.o, []
        for si, (s0, s1) in enumerate(spans):
            idx = [i for i in self.order if s0 <= i <= s1]
            if not idx:
                continue
            groups, splits = self.shorten_pauses(idx)
            for g, grp in enumerate(groups):
                first, last = self.cut[grp[0]], self.cut[grp[-1]]
                ti = to = False
                if g == 0:
                    fi, ti = self.in_point(first, self.nb(grp[0], -1), o.lead_ms)
                else:
                    fi = splits[g - 1][1]
                if g == len(groups) - 1:
                    fo, to = self.out_point(last, self.nb(grp[-1], +1), o.tail_ms, final=(si == len(spans) - 1))
                else:
                    fo = splits[g][0]
                pieces.append({"f_in": fi, "f_out": fo, "w0": grp[0], "w1": grp[-1],
                               "kind_in": "cut" if g == 0 else "pause", "tight": ti or to,
                               "why_in": "tight" if ti else ("cut" if g == 0 else "pause"),
                               "why_out": "tight" if to else ("cut" if g == len(groups) - 1 else "pause")})
        return pieces

    # -- forced edits from keep.json

    def apply_exclusions(self, pieces):
        """split pieces around forced-out source ranges."""
        byi, order, fms = self.byi, self.order, self.fms
        for ea, eb in [tuple(x) for x in self.keep.get("exclude_ms", [])]:
            nxt = []
            for p in pieces:
                pa, pb = p["f_in"] * fms, p["f_out"] * fms
                if eb <= pa or ea >= pb:
                    nxt.append(p)
                    continue
                inp = [i for i in order if p["w0"] <= i <= p["w1"]]
                if self.o.excl_overlap:
                    # a word belongs to the side that holds most of its aligned span,
                    # so it is never lost from words.json
                    def left_side(i):
                        w = byi[i]
                        return min(w["end_ms"], ea) - w["start_ms"] >= w["end_ms"] - max(w["start_ms"], eb)
                    inp = [i for i in inp if not (byi[i]["start_ms"] >= ea and byi[i]["end_ms"] <= eb)]
                    lw = [i for i in inp if left_side(i)]
                    rw = [i for i in inp if not left_side(i)]
                    left_w1, right_w0 = max(lw or [p["w0"]]), min(rw or [p["w1"]])
                else:
                    left_w1 = max([i for i in inp if byi[i]["end_ms"] <= ea] or [p["w0"]])
                    right_w0 = min([i for i in inp if byi[i]["start_ms"] >= eb] or [p["w1"]])
                if ea > pa:
                    nxt.append(dict(p, f_out=math.floor(ea / fms), w1=left_w1, why_out="exclude_ms"))
                if eb < pb:
                    nxt.append(dict(p, f_in=math.ceil(eb / fms), w0=right_w0, kind_in="cut", why_in="exclude_ms"))
            pieces = nxt
        return pieces

    def apply_manual_points(self, pieces):
        """"set_in_ms" / "set_out_ms": move the in (out) point at old_src_ms (+-20 ms) to the frame nearest new."""
        for key, fk, wk in (("set_in_ms", "f_in", "why_in"), ("set_out_ms", "f_out", "why_out")):
            for old, new in self.keep.get(key, []):
                for p in pieces:
                    if abs(p[fk] * self.fms - old) <= 20:
                        p[fk] = int(round(new / self.fms))
                        p[wk] = key
        return pieces

    def drop_and_merge(self, pieces):
        sl = self.o.sliver_frames
        dropped = [p for p in pieces if p["f_out"] - p["f_in"] < sl]
        pieces = [p for p in pieces if p["f_out"] - p["f_in"] >= sl]
        clean = []  # overlapping neighbours (two pauses snapped across each other) -> merge
        for p in pieces:
            if clean and p["f_in"] <= clean[-1]["f_out"]:
                c = clean[-1]
                if p["f_out"] > c["f_out"]:
                    c["f_out"], c["why_out"] = p["f_out"], p["why_out"]
                c["w1"] = p["w1"]
                c["tight"] = c["tight"] or p["tight"]
                continue
            clean.append(p)
        return clean, dropped

    # -- placeholders

    def find_tone(self, t_ms, need_ms=3000):
        """quietest steady window of need_ms near t_ms (+-120 s): room tone of the same recording."""
        db = self.db
        k0, n = int(t_ms // 10), need_ms // 10
        best = fallback = None
        for k in range(max(0, k0 - 12000), max(1, min(len(db) - n, k0 + 12000)), 10):
            win = db[k: k + n]
            sc = float(np.percentile(win, 98))
            if fallback is None or sc < fallback[0]:
                fallback = (sc, k)
            if len(win) and float(win.min()) < DIGITAL_SILENCE_DB:
                continue  # digital silence (an edit gap, a muted stretch) is not room tone
            if best is None or sc < best[0]:
                best = (sc, k)
        best = best or fallback  # only digital silence near: take it rather than fail
        return best[1] * 10, min(best[1] * 10 + need_ms, len(db) * 10), round(best[0], 1)

    def insert_placeholders(self, pieces):
        for ph in self.keep.get("placeholders", []):
            k = next((n for n, p in enumerate(pieces) if "placeholder" not in p and p["w1"] == ph["after_word"]), None)
            if k is None:
                raise ValueError(f"placeholder {ph['id']}: word {ph['after_word']} does not end a piece")
            if "tone_ms" in ph:
                (t0, t1), tdb = ph["tone_ms"], None
            else:
                t0, t1, tdb = self.find_tone(pieces[k]["f_out"] * self.fms)
            pieces.insert(k + 1, {"placeholder": ph["id"], "frames": int(round(ph["dur_s"] * self.fps)),
                                  "kind_in": "placeholder", "tone": [t0, t1], "tone_db": tdb,
                                  "note": ph.get("note", ""), "tight": False})
        return pieces

    # -- output records

    def pwords(self, p):
        if p.get("w0") is None:
            return ["[" + str(p.get("placeholder")) + "]"]
        return [self.byi[i]["w"] for i in self.order if p["w0"] <= i <= p["w1"]]

    def assemble(self, pieces):
        fms, byi, order = self.fms, self.byi, self.order
        dst = 0
        out_pieces, joins, out_words, ph_out = [], [], [], []
        for n, p in enumerate(pieces):
            if "placeholder" in p:
                frames = p["frames"]
                rec = {"id": n, "placeholder": p["placeholder"], "src_in": round(p["tone"][0] / 1000, 3),
                       "src_out": round(p["tone"][1] / 1000, 3), "dst_in": round(dst * fms / 1000, 3),
                       "src_in_ms": p["tone"][0], "src_out_ms": p["tone"][1], "dst_in_ms": round(dst * fms, 3),
                       "dst_f": dst, "frames": frames, "w0": None, "w1": None,
                       "text": f"[{p['placeholder']}: black + room tone]", "tone_db": p["tone_db"]}
                ph_out.append({"id": p["placeholder"], "start": rec["dst_in"],
                               "end": round((dst + frames) * fms / 1000, 3), "note": p["note"],
                               "room_tone_src_s": [rec["src_in"], rec["src_out"]]})
            else:
                frames = p["f_out"] - p["f_in"]
                src_in_ms, src_out_ms = p["f_in"] * fms, p["f_out"] * fms
                rec = {"id": n, "src_in": round(src_in_ms / 1000, 3), "src_out": round(src_out_ms / 1000, 3),
                       "dst_in": round(dst * fms / 1000, 3), "src_in_ms": round(src_in_ms, 3),
                       "src_out_ms": round(src_out_ms, 3), "dst_in_ms": round(dst * fms, 3), "dst_f": dst,
                       "frames": frames, "f_in": p["f_in"], "f_out": p["f_out"], "w0": p["w0"], "w1": p["w1"],
                       "text": " ".join(self.pwords(p))}
                if p["tight"]:
                    rec["tight"] = True
                off = dst * fms - src_in_ms
                for i in order:
                    if self.o.excl_overlap and any(x["src_i"] == i for x in out_words[-8:]):
                        continue  # a word-less remainder of an excluded split repeats its neighbour's label
                    if p["w0"] <= i <= p["w1"]:
                        w = byi[i]
                        out_words.append({"i": len(out_words), "src_i": i, "w": w["w"],
                                          "start_ms": int(round(max(w["start_ms"], src_in_ms) + off)),
                                          "end_ms": int(round(min(w["end_ms"], src_out_ms) + off)),
                                          "p": w.get("p"), "score": w.get("score")})
            out_pieces.append(rec)
            if n:
                prev = out_pieces[n - 1]
                isph = rec["w0"] is None or prev["w0"] is None
                joins.append({"n": n, "dst": rec["dst_in"], "src_out": prev["src_out"], "src_in": rec["src_in"],
                              "type": "placeholder" if isph else p["kind_in"],
                              "removed_s": 0.0 if isph else round(rec["src_in"] - prev["src_out"], 3),
                              "removed_words": [] if isph else [byi[i]["w"] for i in order if prev["w1"] < i < rec["w0"]],
                              "before": " ".join(self.pwords(pieces[n - 1])[-6:]),
                              "after": " ".join(self.pwords(p)[:6])})
            dst += frames
        return out_pieces, joins, out_words, ph_out, dst

    def soft_joins(self, pieces, out_pieces, joins):
        """every in/out point that is not on a quiet frame, with the reason it could not be."""
        why_text = {"tight": "words run together: no frame in the pause, quietest frame near the word edge",
                    "cut": "no quiet frame within reach of the word edge: quietest frame taken",
                    "pause": "no quiet frame within reach of the pause: quietest frame taken",
                    "exclude_ms": "edge forced by keep.json exclude_ms",
                    "set_in_ms": "edge set by keep.json set_in_ms", "set_out_ms": "edge set by keep.json set_out_ms"}
        soft = []
        for p, rec in zip(pieces, out_pieces):
            if "placeholder" in p:
                continue
            for edge, f, why in (("in", p["f_in"], p["why_in"]), ("out", p["f_out"], p["why_out"])):
                t = f * self.fms
                e, thr = self.P.e(f), self.P.quiet_thr(t)
                if e >= thr:
                    soft.append({"piece": rec["id"], "edge": edge, "frame": f, "src_ms": round(t, 3),
                                 "db": round(e, 1), "quiet_thr_db": round(thr, 1), "why": why_text.get(why, why),
                                 "words": rec["text"][:60]})
        soft_ids = {(s["piece"], s["edge"]) for s in soft}
        for j in joins:
            if (j["n"] - 1, "out") in soft_ids or (j["n"], "in") in soft_ids:
                j["soft"] = True
        return soft


def plan_pieces(words, keep, audio_db, fps, opts, dur_ms=None):
    """Pure planning: kept words + keep.json + 10 ms slice levels -> pieces, joins, remapped words.

    Returns a dict with pieces (src_in_ms/src_out_ms/dst_in_ms and the course keys src_in/src_out/
    dst_in in seconds, f_in/f_out frames), joins, long_gaps_kept, soft_joins, placeholders,
    voiced_gaps, dropped_slivers, words (remapped), frames, duration_s and settings."""
    o = SimpleNamespace(**resolve_opts(None, dict(opts or {}, fps=fps)))
    db = np.asarray(audio_db, dtype=np.float64)
    full_ms = len(db) * 10
    dur_ms = full_ms if dur_ms is None else min(dur_ms, full_ms)
    pl = _Planner(words, keep, db, fps, o, dur_ms)
    for s0, s1 in keep.get("keep", []):
        if s0 not in pl.byi or s1 not in pl.byi:
            raise ValueError(f"keep span [{s0}, {s1}]: no such word index")
    spans = merge_spans(keep.get("keep", []), pl.cut, o.merge_gap_ms)
    pieces = pl.span_pieces(spans)
    pieces = pl.apply_exclusions(pieces)
    pieces = pl.apply_manual_points(pieces)
    pieces, dropped = pl.drop_and_merge(pieces)
    pieces = pl.insert_placeholders(pieces)
    out_pieces, joins, out_words, ph_out, total = pl.assemble(pieces)
    soft = pl.soft_joins(pieces, out_pieces, joins)
    fms = pl.fms
    return {
        "fps": fps, "frames": total, "duration_s": round(total / fps, 3),
        "settings": {k: getattr(o, k) for k in SETTING_KEYS},
        "pieces": out_pieces, "joins": joins, "placeholders": ph_out, "voiced_gaps": pl.voiced_log,
        "long_gaps_kept": pl.kept_long, "soft_joins": soft,
        "dropped_slivers": [{"src_in": p["f_in"] * fms / 1000, "src_out": p["f_out"] * fms / 1000,
                             "w0": p["w0"], "w1": p["w1"]} for p in dropped],
        "words": out_words,
        "fade_out_ms": [list(x) for x in keep.get("fade_out_ms", [])],
        "fade_in_ms": [list(x) for x in keep.get("fade_in_ms", [])],
    }


# ---------------------------------------------------------------- render

def tone_audio(audio, t0_ms, t1_ms, nsamp, xf_ms=200):
    """room tone of nsamp samples: the source range looped back and forth with 200 ms crossfades."""
    base = audio[int(t0_ms * AR / 1000): int(t1_ms * AR / 1000)]
    if len(base) == 0:
        return np.zeros((nsamp, 2), dtype=np.float32)
    xf = max(1, min(int(xf_ms * AR / 1000), len(base) // 2))
    fi_ = np.sin(np.linspace(0, math.pi / 2, xf))[:, None]
    fo_ = np.cos(np.linspace(0, math.pi / 2, xf))[:, None]
    out = base.copy()
    flip = True
    while len(out) < nsamp:
        nxt = (base[::-1] if flip else base).copy()
        flip = not flip
        out = np.concatenate([out[:-xf], out[-xf:] * fo_ + nxt[:xf] * fi_, nxt[xf:]])
    return out[:nsamp].astype(np.float32)


def build_audio(audio, plan):
    """the cut audio: pieces concatenated with equal-power crossfades centred on each join."""
    fps = plan["fps"]
    xfade_ms = plan["settings"]["xfade_ms"]
    half = int(round(xfade_ms / 2 * AR / 1000))
    spf = AR // fps
    pieces = plan["pieces"]
    outa = np.zeros((plan["frames"] * spf, 2), dtype=np.float32)
    for n, p in enumerate(pieces):
        d0 = p["dst_f"] * spf
        pre = half if n else 0
        post = half if n < len(pieces) - 1 else 0
        if p["w0"] is None:
            seg = tone_audio(audio, p["src_in_ms"], p["src_out_ms"], p["frames"] * spf + pre + post)
        else:
            s0, s1 = p["f_in"] * spf, p["f_out"] * spf
            seg = audio[max(0, s0 - pre): min(len(audio), s1 + post)].copy()
            if s0 - pre < 0:  # the first piece at source 0 has no audio before it
                seg = np.concatenate([np.zeros((pre - s0, 2), np.float32), seg])
        env = np.ones(len(seg), dtype=np.float32)
        if p["w0"] is not None:
            for t_ms, ms in plan.get("fade_out_ms", []):
                if abs(p["src_out_ms"] - t_ms) <= 20:
                    m = int(ms * AR / 1000)
                    e1 = len(seg) - post
                    env[max(0, e1 - m): e1] = np.cos(np.linspace(0, math.pi / 2, e1 - max(0, e1 - m)))
                    env[e1:] = 0
            for t_ms, ms in plan.get("fade_in_ms", []):
                if abs(p["src_in_ms"] - t_ms) <= 20:
                    m = int(ms * AR / 1000)
                    env[:pre] = 0
                    env[pre: pre + m] = np.sin(np.linspace(0, math.pi / 2, len(env[pre: pre + m])))
        if pre:
            env[: 2 * pre] = np.minimum(env[: 2 * pre], np.sin(np.linspace(0, math.pi / 2, 2 * pre)))
        if post:
            env[-2 * post:] = np.minimum(env[-2 * post:], np.cos(np.linspace(0, math.pi / 2, 2 * post)))
        seg = seg * env[:, None]
        a0 = d0 - pre
        b0 = min(len(outa), a0 + len(seg))
        outa[max(0, a0):b0] += seg[max(0, -a0): b0 - a0]
    return outa


def select_expr(pieces):
    """frame-select expression; terms grouped by 20 in parentheses, because ffmpeg's expression
    parser fails (ENOMEM) on a flat sum of more than 100 between() terms."""
    t = [f"between(n,{p['f_in']},{p['f_out'] - 1})" for p in pieces]
    if len(t) <= 20:
        return "+".join(t)
    return "+".join("(" + "+".join(t[i:i + 20]) + ")" for i in range(0, len(t), 20))


def video_graph(pieces, fps, size=None):
    """one filter graph: the source resampled to --fps (frame n = time n/fps, so frame numbers
    match the plan whatever the source rate), the kept frames selected, black placeholders
    concatenated in place."""
    head = f"[0:v]setpts=PTS-STARTPTS,fps={fps}"
    if all(p["w0"] is not None for p in pieces):
        return f"{head},select='{select_expr(pieces)}',setpts=N/{fps}/TB[v]"
    runs, cur = [], []
    for p in pieces:
        if p["w0"] is None:
            if cur:
                runs.append(("src", cur))
                cur = []
            runs.append(("ph", p))
        else:
            cur.append(p)
    if cur:
        runs.append(("src", cur))
    nsrc = sum(1 for k, _ in runs if k == "src")
    lines = [f"{head},split={nsrc}" + "".join(f"[s{j}]" for j in range(nsrc))] if nsrc else []
    labels, j = [], 0
    vw, vh = size
    for r, (kind, x) in enumerate(runs):
        if kind == "src":
            nf = sum(p["frames"] for p in x)
            lines.append(f"[s{j}]select='{select_expr(x)}',setpts=N/{fps}/TB,trim=end_frame={nf},setsar=1[r{r}]")
            j += 1
        else:
            nf = x["frames"]
            lines.append(f"color=c=black:s={vw}x{vh}:r={fps}:d={nf / fps + 1},trim=end_frame={nf},setsar=1[r{r}]")
        labels.append(f"[r{r}]")
    lines.append("".join(labels) + f"concat=n={len(runs)}:v=1:a=0,setpts=N/{fps}/TB[v]")
    return ";\n".join(lines)


def _graph_flag():
    """ffmpeg 7+ reads a filter graph from a file with -/filter_complex; older builds use -filter_complex_script."""
    r = subprocess.run(["ffmpeg", "-hide_banner", "-version"], capture_output=True, text=True)
    m = re.search(r"ffmpeg version n?(\d+)", r.stdout or "")
    return "-/filter_complex" if not m or int(m.group(1)) >= 7 else "-filter_complex_script"


def write_plan(plan, stem, meta=None):
    """write <stem>.cuts.json, <stem>.words.json (and placeholders.json next to them)."""
    stem = str(stem)
    os.makedirs(os.path.dirname(os.path.abspath(stem)) or ".", exist_ok=True)
    meta = meta or {}
    cuts = {k: meta[k] for k in ("source", "words", "keep", "preset") if k in meta}
    cuts.update({k: v for k, v in plan.items() if k not in ("words", "fade_out_ms", "fade_in_ms")})
    _dump(stem + ".cuts.json", cuts)
    fms = 1000.0 / plan["fps"]
    _dump(stem + ".words.json", {"source": os.path.abspath(stem + ".mp4"), "language": meta.get("language"),
                                 "duration_ms": int(plan["frames"] * fms), "words": plan["words"]})
    if plan["placeholders"]:
        _dump(os.path.join(os.path.dirname(os.path.abspath(stem)), "placeholders.json"),
              [{k: v for k, v in x.items() if k != "room_tone_src_s"} for x in plan["placeholders"]])


def _dump(path, doc):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def render(source, plan, out_stem, opts=None, audio=None, meta=None):
    """write the plan files and render <stem>.mp4 in one ffmpeg pass (video select + cut audio)."""
    opts = resolve_opts(None, opts or {})
    stem = str(out_stem)
    stem = stem[:-4] if stem.endswith(".mp4") else stem
    if audio is None:
        audio = load_audio(source)
    if "channel_balance" not in plan:  # plan_pieces is pure: measure the balance here if the caller did not
        plan.update(channel_balance(audio))
    write_plan(plan, stem, meta)
    bal = {k: plan[k] for k in ("channel_balance", "gains") if k in plan}
    outa = build_audio(apply_balance(audio, bal), plan)
    fps = plan["fps"]
    size = None
    if any(p["w0"] is None for p in plan["pieces"]):
        w, h, _ = probe_video(source)
        size = (w, h)
    with tempfile.TemporaryDirectory(prefix="slcut_") as tmp:
        wav = os.path.join(tmp, "cut.wav")
        pcm = (np.clip(outa, -1, 1) * 32767).astype("<i2")
        with wave.open(wav, "wb") as wf:
            wf.setnchannels(2)
            wf.setsampwidth(2)
            wf.setframerate(AR)
            wf.writeframes(pcm.tobytes())
        fg = os.path.join(tmp, "fg.txt")
        with open(fg, "w", encoding="ascii") as f:
            f.write(video_graph(plan["pieces"], fps, size))
        cmd = ["ffmpeg", "-v", "error", "-nostdin", "-y", "-i", str(source), "-i", wav, _graph_flag(), fg,
               "-map", "[v]", "-map", "1:a", "-c:v", "libx264", "-preset", "fast", "-crf", str(opts["crf"]),
               "-pix_fmt", "yuv420p", "-r", str(fps), "-g", str(fps), "-c:a", "aac", "-b:a", "192k",
               "-frames:v", str(plan["frames"]), "-movflags", "+faststart", stem + ".tmp.mp4"]
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if r.returncode != 0:
            try:
                os.remove(stem + ".tmp.mp4")
            except OSError:
                pass
            msg = r.stderr.strip().splitlines()
            raise RuntimeError("ffmpeg render failed: " + (msg[-1] if msg else "unknown error"))
        os.replace(stem + ".tmp.mp4", stem + ".mp4")


# ---------------------------------------------------------------- CLI

def _read_json(path, what):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError) as e:
        raise ValueError(f"cannot read {what} {path}: {e}")


def main(argv=None):
    store.print_utf8()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source")
    ap.add_argument("words", help="words.aligned.json from align.py")
    ap.add_argument("keep", help="keep.json")
    ap.add_argument("--out", required=True, help="output stem, e.g. 1.1/edit_ready.v3")
    ap.add_argument("--preset", choices=sorted(PRESETS), default=None,
                    help="lesson (default, the course tool values) or reels; explicit flags override it")
    ap.add_argument("--fps", type=int)
    ap.add_argument("--lead-ms", type=int)
    ap.add_argument("--tail-ms", type=int)
    ap.add_argument("--inside-ms", type=int)
    ap.add_argument("--merge-gap-ms", type=int)
    ap.add_argument("--pause-max-ms", type=int)
    ap.add_argument("--pause-keep-ms", type=int, help="air kept on each side of a shortened pause")
    ap.add_argument("--sentence-keep-ms", type=int,
                    help="air kept on each side of a shortened pause after a sentence end (. ? ! :); "
                         "default = --pause-keep-ms")
    ap.add_argument("--pause-min-cut-frames", type=int, help="shorten a pause only if this many frames go")
    ap.add_argument("--pause-snap-ms", type=int,
                    help="how far past the kept air a shortened pause may snap to a quieter frame")
    ap.add_argument("--end-air-ms", type=int,
                    help="air after the last word of the whole cut (default: same rule as every out-point)")
    ap.add_argument("--sliver-frames", type=int)
    ap.add_argument("--xfade-ms", type=float)
    ap.add_argument("--crf", type=int)
    ap.add_argument("--excl-overlap", action="store_true", default=None,
                    help="exclude_ms splits assign a word to the side holding most of its aligned span (default: "
                         "a word must lie fully outside the excluded range)")
    ap.add_argument("--no-render", action="store_true")
    a = ap.parse_args(argv)

    try:
        require_ffmpeg()
    except MissingTool as e:
        print(str(e), file=sys.stderr)
        return 3
    overrides = {k: getattr(a, k) for k in DEFAULTS}
    opts = resolve_opts(a.preset, overrides)
    if opts["fps"] <= 0 or AR % opts["fps"]:
        print(f"error: --fps must divide {AR} (e.g. 24, 25, 30, 50, 60)", file=sys.stderr)
        return 2
    try:
        doc = _read_json(a.words, "words file")
        keep = _read_json(a.keep, "keep file")
        if not isinstance(doc, dict) or not isinstance(doc.get("words"), list):
            raise ValueError(f"{a.words}: no 'words' list")
        if not isinstance(keep, dict) or not isinstance(keep.get("keep"), list):
            raise ValueError(f"{a.keep}: no 'keep' list")
        if not os.path.isfile(a.source):
            raise ValueError(f"source not found: {a.source}")
        _, _, video_ms = probe_video(a.source)
        audio = load_audio(a.source)
        db = slice_db(audio.mean(axis=1))
        dur_ms = len(audio) * 1000 / AR
        if video_ms:
            dur_ms = min(dur_ms, video_ms)
        plan = plan_pieces(doc["words"], keep, db, opts["fps"], opts, dur_ms=dur_ms)
        if not plan["frames"]:
            raise ValueError("nothing to cut: the keep spans give no piece")
        plan.update(channel_balance(audio, db))
        meta = {"source": os.path.abspath(a.source), "words": os.path.abspath(a.words),
                "keep": os.path.abspath(a.keep), "preset": a.preset or "lesson", "language": doc.get("language")}
        stem = a.out[:-4] if a.out.endswith(".mp4") else a.out
        ph = [(x["id"], x["start"], x["end"], x["room_tone_src_s"]) for x in plan["placeholders"]]
        print(f"{len(plan['pieces'])} pieces, {len(plan['joins'])} joins, {plan['frames']} frames = "
              f"{plan['frames'] / opts['fps']:.2f}s, {len(plan['dropped_slivers'])} sliver(s) dropped, "
              f"{sum(1 for p in plan['pieces'] if p.get('tight'))} tight, placeholders {ph}")
        if plan["soft_joins"]:
            print(f"warning: {len(plan['soft_joins'])} edge(s) not on a quiet frame: see cuts.json soft_joins")
        if plan["long_gaps_kept"]:
            print(f"note: {len(plan['long_gaps_kept'])} long gap(s) kept: see cuts.json long_gaps_kept")
        if plan["channel_balance"] != "none":
            print(f"note: source channels differ by {plan['channel_balance_db']} dB (L-R): "
                  f"balanced ({plan['channel_balance']})")
        if a.no_render:
            write_plan(plan, stem, meta)
            return 0
        render(a.source, plan, stem, opts, audio=audio, meta=meta)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except RuntimeError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(f"rendered -> {stem}.mp4")
    return 0


if __name__ == "__main__":
    sys.exit(main())

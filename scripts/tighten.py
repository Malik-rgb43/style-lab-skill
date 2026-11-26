#!/usr/bin/env python3
"""Tighten the joins of a precise cut for talking heads and VSLs: no dead air, nothing swallowed.

Every pause of at least --min-pause-ms between two kept phrases (inside a piece or at a join
between pieces) is closed so the next phrase's first sound starts right where the previous
phrase's voice has died away (+ --gap-ms), like a clip stacked on the track above. The edges
are measured on the audio energy (10 ms slices against the local room tone), not taken from
the aligner: the start is walked back and the end walked forward until 3 slices in a row are
below the room tone + --soft-db (4 dB), so a soft breathy onset or a long decay is kept.

A pause is only closed when it is safe to close:
  - both edges were found within the 400 ms search (a voice that never dies away is never cut
    into; the join is kept as the cut made it and listed in cuts.json voice_not_died_away)
  - nothing in the pause is sound (3 slices in a row >= the soft threshold, the same rule as the
    edges, so a single noisy slice is not sound; or any single slice >= room tone + --loud-db,
    10 dB, a click or a short consonant): an unlisted word, a laugh or a breath there keeps the
    join as cut (listed in cuts.json sound_in_pause). When more than half the joins are held only
    by sound near the room-noise level, a warning is printed and stored (room_noise_warning).

The room tone is local: the median of the quiet slices within +-1.5 s, and the soft threshold is
room + max(--soft-db, the room's own spread + 1.5 dB), so a room whose noise drifts still closes
its joins while a soft word clearly above the room still holds them.
    At a join between pieces the cut's own air on both sides is checked too.

  - lip sync is exact: picture and sound of every clip share one offset in whole frames
  - the two clips meet with an equal-power crossfade inside the quiet room tone between the
    previous voice end and the next voice start, within both pieces
  - a join that is already tighter than the target is left as the cut made it (never widened);
    inside a piece, a pause shorter than --min-pause-ms on the true edges is left alone
  - placeholders keep their length and the air on both sides (--no-placeholders drops them and
    closes the join instead); the cut's own lead and end air are kept

Checks (every time, on the in-memory output mix before encoding):
  edge check: for every clip, the level of its first and last 120 ms and 30 ms of voice (true
    start/end, measured without the cut's in/out points) vs the same span in the source; off by
    more than --check-db (0.05 dB) on 120 ms or 0.5 dB on 30 ms, or more than 2.5 ms of that voice
    not played at full level by the clip (uncovered_ms; overlap_ms = ms under another clip, both
    reported apart from the dB), goes into edge_bad. A cut that clipped a word shows here.
  missing audit: every stretch of the cut's timeline that is not in the output at full level and
    holds sound (the same 3-slice rule, overlapping it by >= 5 ms) goes into edge_bad (edge "missing").
  added audit: any output audio read from the source outside every kept piece (beyond the cut's
    own 7.5 ms join crossfade) goes into edge_bad (edge "added").
Any edge_bad item is printed with the piece and the CLI exits 1 (the files are still written so
the join can be inspected).

Usage:
  tighten.py <stem> --out <out_stem> [--min-pause-ms 250] [--no-placeholders] [--gap-ms 0]

Reads <stem>.mp4, <stem>.cuts.json, <stem>.words.json (from cut.py; the source is cuts.json
"source"). Writes <out>.mp4 (atomically), <out>.cuts.json (pieces, joins, placeholders,
edge_check, missing_check, edge_bad, voice_not_died_away, sound_in_pause) and <out>.words.json
(words on the new timeline). <out> must differ from <stem>. Run check_cut.py on <out> afterwards.

Exit: 0 clean, 1 edge deviation / missing sound or error, 2 usage, 3 ffmpeg missing.
"""
import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cut  # noqa: E402
import store  # noqa: E402
from cut import AR, SL, MissingTool, _dump, _graph_flag, apply_balance, load_audio, require_ffmpeg, slice_db  # noqa: E402,F401

XF_MIN_MS = 4.0       # shortest crossfade where two clips meet
BRIDGE_XF_MS = 10.0   # linear crossfade into/out of the cut's own audio around placeholders
JOIN_XF_MS = 15.0     # crossfade at a join kept as the cut made it (same as cut.py)
QUIET_RUN = 3         # slices in a row below the soft threshold = the voice has died away
ROOM_WIN_MS = 1500    # the room level is measured on the quiet slices within +-1.5 s
ROOM_CAND_DB = 8.0    # quiet slices: at most this far above the window's 10th percentile
ROOM_CAP_DB = 6.0     # a local room level is never taken more than this above the whole file's
SPREAD_MAD = 1.2      # spread = this x the median absolute deviation of the quiet slices
REFINE_MIN_DB = 2.0   # the room level is re-measured on the slices at most max(this, spread) above it
SPREAD_MARGIN_DB = 1.5  # the soft threshold sits this far above the room's spread
NOISE_NEAR_DB = 3.0   # a held join whose "sound" is within this of the soft threshold is noise-like
SHORT_WIN_MS = 30     # the short edge-check window at a clip's first/last voice
SHORT_DB = 0.5        # its tolerance (dB)
EDGE_UNCOVERED_MS = XF_MIN_MS / 2 + 0.5  # voice not played at full level beyond this fails the edge


# ---------------------------------------------------------------- audio helpers

def level_db(audio, t0_ms, t1_ms):
    """RMS level (dBFS) of audio between two times; outside the array counts as digital silence."""
    i0, i1 = int(round(t0_ms * AR / 1000)), int(round(t1_ms * AR / 1000))
    if i1 <= i0:
        return -180.0
    x = np.zeros(i1 - i0, dtype=np.float64)
    a, b = max(0, i0), min(len(audio), i1)
    if b > a:
        x[a - i0: b - i0] = audio[a:b].mean(axis=1)
    return float(20 * np.log10(np.sqrt((x * x).mean()) + 1e-9))


def take(audio, i0, n):
    """n samples from i0 (zero padded outside the array)."""
    out = np.zeros((n, 2), dtype=np.float32)
    a, b = max(0, i0), min(len(audio), i0 + n)
    if b > a:
        out[a - i0: b - i0] = audio[a:b]
    return out


class Energy:
    """true edges on the 10 ms slice levels of the source, against the LOCAL room tone.

    room(ms): the quiet slices within +-ROOM_WIN_MS (the slices at most ROOM_CAND_DB above the
    window's 10th percentile): their median (the room level) and their spread (SPREAD_MAD x the
    median absolute deviation, about the room's own 90th percentile above its median; about 0 in a
    steady room, a few dB in a room whose noise drifts, and robust to a soft sound in the pause).
    soft = room + max(soft_db, spread + SPREAD_MARGIN_DB): an edge is where QUIET_RUN slices in a
      row are below it; QUIET_RUN slices in a row at or above it are sound.
    loud = room + max(loud_db, spread + soft_db): any single slice at or above it is sound (a click,
      a short consonant)."""

    def __init__(self, db, search_ms, soft_db, loud_db=10.0):
        self.db, self.search_ms, self.soft_db, self.loud_db = db, search_ms, soft_db, loud_db
        self._room = {}
        g = self.db[self.db <= np.percentile(self.db, 10) + ROOM_CAND_DB] if len(self.db) else np.zeros(1)
        self.global_room = float(np.median(g))

    def at(self, j):
        return self.db[j] if 0 <= j < len(self.db) else -200.0

    def room(self, ms):
        """(room level dB, spread dB) of the quiet slices around ms (cached per 50 ms)."""
        key = int(ms // 50)
        if key not in self._room:
            k, h = key * 5, ROOM_WIN_MS // 10
            w = self.db[max(0, k - h): k + h]
            if len(w) < 10:
                w = self.db
            q = w[w <= np.percentile(w, 10) + ROOM_CAND_DB]
            med = float(np.median(q))
            # robust spread: a soft word or breath that is a minority of the quiet slices must not
            # raise it (a 90th percentile would, and then hide that very sound)
            spread = SPREAD_MAD * float(np.median(np.abs(q - med)))
            # the level is re-measured on the slices within the room's own spread of it, so a soft
            # sound in the window (a few dB above the room) does not lift the room level either
            q2 = q[q < med + max(REFINE_MIN_DB, spread)]
            if len(q2) >= 10:
                med = float(np.median(q2))
            # a window with no room tone in it (continuous speech) must not raise the room level
            self._room[key] = (min(med, self.global_room + ROOM_CAP_DB), spread)
        return self._room[key]

    def floor(self, ms):
        return self.room(ms)[0]

    def soft(self, ms):
        med, spread = self.room(ms)
        return med + max(self.soft_db, spread + SPREAD_MARGIN_DB)

    def loud(self, ms):
        med, spread = self.room(ms)
        return med + max(self.loud_db, spread + self.soft_db)

    def soft_at(self, j):
        """the soft threshold of slice j, from the room around that slice (edges and the sound
        check use the same per-slice thresholds, so a slice quiet for one is quiet for the other)"""
        return self.soft(j * 10.0 + 5)

    def loud_at(self, j):
        return self.loud(j * 10.0 + 5)

    def _quiet(self, j, step):
        return all(self.at(j + step * i) < self.soft_at(j + step * i) for i in range(QUIET_RUN))

    def true_end(self, ms):
        """(end_ms, died): walk forward from the aligned word end until the voice has died away;
        died is False when it is still sounding at the search limit (end_ms is that limit)."""
        k = int(ms // 10)
        for j in range(k, k + self.search_ms // 10):
            if self._quiet(j, 1):
                return j * 10.0, True
        return k * 10.0 + self.search_ms, False

    def true_start(self, ms):
        """(start_ms, found): walk back from the aligned word start to the first sound."""
        k = int(ms // 10)
        if self._quiet(k, 1):  # aligned start in room tone (a pinned retime): go to the first sound
            for j in range(k, k + 2 * self.search_ms // 10):
                if all(self.at(j + i) >= self.soft_at(j + i) for i in range(QUIET_RUN)):
                    k = j
                    break
            else:
                return float(ms), False
        for j in range(k, k - self.search_ms // 10, -1):
            if self._quiet(j, -1):
                return (j + 1) * 10.0, True
        return k * 10.0 - self.search_ms, False

    def quiet_after(self, ms, limit):
        t = 0.0
        while t < limit and self.at(int((ms + t) // 10)) < self.soft_at(int((ms + t) // 10)):
            t += 10
        return t

    def quiet_before(self, ms, limit):
        u = 0.0
        while u < limit and self.at(int((ms - u - 10) // 10)) < self.soft_at(int((ms - u - 10) // 10)):
            u += 10
        return u

    def loud_runs(self, a_ms, b_ms, min_overlap_ms=5.0):
        """sound = QUIET_RUN or more slices in a row >= the soft threshold (the same rule as the
        edges, so a single noisy slice is not sound), or any run holding a slice >= the loud
        threshold (a short click or consonant). Returns the runs that overlap [a, b) by at least
        min_overlap_ms: [(run_start_ms, run_end_ms, loudest_db)]."""
        if b_ms - a_ms < min_overlap_ms:
            return []
        j0, j1 = int(a_ms // 10) - QUIET_RUN, int(math.ceil(b_ms / 10)) + QUIET_RUN
        runs, start = [], None
        for j in range(j0, j1 + 1):
            loud = j < j1 and self.at(j) >= self.soft_at(j)
            if loud and start is None:
                start = j
            elif not loud and start is not None:
                top = float(max(self.at(k) for k in range(start, j)))
                hit = any(self.at(k) >= self.loud_at(k) for k in range(start, j))
                if ((j - start >= QUIET_RUN or hit)
                        and min(b_ms, j * 10.0) - max(a_ms, start * 10.0) >= min_overlap_ms):
                    runs.append((start * 10.0, j * 10.0, top))
                start = None
        return runs

    def sound_in(self, a_ms, b_ms):
        """the loudest sound run overlapping [a, b): (slice_ms, db) inside [a, b), or None."""
        runs = self.loud_runs(a_ms, b_ms)
        if not runs:
            return None
        r0, r1, db = max(runs, key=lambda r: r[2])
        return max(r0, a_ms), db

    def sound_span(self, lo, hi):
        """first sound and true end inside [lo, hi): (on, found, end, died), None when only room tone."""
        runs = self.loud_runs(lo, hi)
        if not runs:
            return None
        on, found = self.true_start(max(runs[0][0], lo) + 5)
        last = min(runs[-1][1], hi)
        end, died = self.true_end(last)
        return on, found, max(last, end), died


# ---------------------------------------------------------------- plan

def _ms(p, key):
    if key + "_ms" in p:
        return float(p[key + "_ms"])
    return float(p[key]) * 1000


def _pieces(cuts, fms):
    out = []
    for idx, p in enumerate(cuts["pieces"]):
        q = {"idx": idx, "id": p.get("id", idx), "raw": p, "frames": int(p["frames"]),
             "ph": p.get("w0") is None or "placeholder" in p}
        q["dst_f"] = int(p["dst_f"]) if "dst_f" in p else int(round(_ms(p, "dst_in") / fms))
        if not q["ph"]:
            q["f_in"] = int(p["f_in"]) if "f_in" in p else int(round(_ms(p, "src_in") / fms))
            q["f_out"] = q["f_in"] + q["frames"]
            q["old_off"] = q["f_in"] - q["dst_f"]
            q["lo"], q["hi"] = q["f_in"] * fms, q["f_out"] * fms
        out.append(q)
    return out


class Seg:
    def __init__(self, piece, words):
        self.P, self.words, self.ph_after, self.parts = piece, list(words), [], [piece["id"]]


def _words_text(ws, last=False):
    ws = ws[-3:] if last else ws[:3]
    return " ".join(w["w"] for w in ws)


def _segments(pieces, words, E, min_pause_ms, keep_ph, log):
    segs = []
    for P in pieces:
        if P["ph"]:
            continue
        ws = [w for w in words if w["P"] is P]
        if not ws:
            span = E.sound_span(P["lo"], P["hi"])
            if span is None:
                log.append({"cut_piece": P["id"], "src_ms": [P["lo"], P["hi"]], "why": "no words and only room tone"})
                continue
            s = Seg(P, [])
            s.on_true, s.on_found, s.end_true, s.end_died = span
            segs.append(s)
            continue
        cur = Seg(P, ws[:1])
        for x, y in zip(ws, ws[1:]):
            if y["start_ms"] - x["end_ms"] >= min_pause_ms:
                segs.append(cur)
                cur = Seg(P, [y])
            else:
                cur.words.append(y)
        segs.append(cur)
    for s in segs:
        if s.words:
            s.on_true, s.on_found = E.true_start(s.words[0]["s0"])
            e, s.end_died = E.true_end(s.words[-1]["s1"])
            s.end_true = max(s.words[-1]["s1"], e)
        s.on = min(max(s.on_true, s.P["lo"]), s.P["hi"])
        s.end = max(min(s.end_true, s.P["hi"]), s.on)
    # the pause is measured on the true edges: inside a piece, one shorter than min_pause_ms (or
    # with no quiet room in it) is not closed, the two phrases stay one clip
    merged = []
    for s in segs:
        if merged and merged[-1].P is s.P and s.on_true - merged[-1].end_true < max(min_pause_ms, 20):
            _absorb(merged[-1], s)
        else:
            merged.append(s)
    segs = merged
    if keep_ph:
        for a, b in zip(segs, segs[1:]):
            a.ph_after = [p for p in pieces if p["ph"] and a.P["idx"] < p["idx"] < b.P["idx"]]
        if segs:
            segs[-1].ph_after = [p for p in pieces if p["ph"] and p["idx"] > segs[-1].P["idx"]]
    return segs


def _absorb(pv, s):
    pv.words += s.words
    pv.end, pv.end_true, pv.end_died = s.end, s.end_true, s.end_died
    pv.ph_after = s.ph_after
    pv.parts += [x for x in s.parts if x not in pv.parts]


def _sound_between(E, pv, s):
    """sound that closing the pause pv -> s would drop: [(a_ms, b_ms, slice_ms, db)] in the source."""
    if s.P is pv.P:
        spans = [(pv.end_true, s.on_true)]
    else:  # the rest of the pause and the cut's own air on both sides of the join
        spans = [(pv.end_true, pv.P["hi"]), (s.P["lo"], s.on_true)]
    out = []
    for a, b in spans:
        hit = E.sound_in(a, b)
        if hit:
            out.append((a, b) + hit)
    return out


def plan(cuts, words_doc, src_db, opts):
    """Pure placement: cut.py cuts.json + words.json + source slice levels -> timeline items."""
    fps = int(cuts["fps"])
    fms = 1000.0 / fps
    E = Energy(np.asarray(src_db, dtype=np.float64), opts["search_ms"], opts.get("soft_db", 4.0),
               opts.get("loud_db", 10.0))
    pieces = _pieces(cuts, fms)
    keep_ph = opts["keep_placeholders"]
    words = []
    for w in sorted(words_doc.get("words", []), key=lambda w: (w["start_ms"], w["end_ms"])):
        mid = (w["start_ms"] + w["end_ms"]) / 2
        P = next((p for p in pieces if not p["ph"] and p["dst_f"] * fms - 1 <= mid < (p["dst_f"] + p["frames"]) * fms + 1),
                 None)
        if P is None:
            raise ValueError(f"word {w.get('i')} {w.get('w')!r} at {mid:.0f} ms lies in no source piece of the cut")
        off = P["old_off"] * fms
        words.append(dict(w, P=P, s0=w["start_ms"] + off, s1=w["end_ms"] + off))
    dropped = []
    segs = _segments(pieces, words, E, opts["min_pause_ms"], keep_ph, dropped)
    if not segs:
        raise ValueError("nothing to tighten: the cut has no words and no sound")
    lead_phs = [p for p in pieces if p["ph"] and p["idx"] < segs[0].P["idx"]] if keep_ph else []
    voice_held, sound_held = [], []

    def held_voice(s, edge):
        ms = s.end_true if edge == "end" else s.on_true
        # noise-like: the level over the search stays near the soft threshold (a noisy room, not a voice)
        a, b = (ms - opts["search_ms"], ms) if edge == "end" else (ms, ms + opts["search_ms"])
        lv = [E.at(j) for j in range(int(a // 10), int(b // 10))]
        noise = bool(lv) and float(np.median(lv)) < E.soft(ms) + NOISE_NEAR_DB
        voice_held.append({"cut_piece": s.P["id"], "edge": edge, "src_ms": round(ms, 1),
                           "words": _words_text(s.words, edge == "end"), "noise_like": noise,
                           "why": f"the voice has not died away within {opts['search_ms']} ms of the word {edge}"})
        return noise

    # ---- offsets (source frame - timeline frame), whole frames
    first = segs[0]
    first.off = first.P["old_off"] if lead_phs else first.P["f_in"]
    out, joins = [first], []
    considered, noise_holds = 0, 0
    for s in segs[1:]:
        pv = out[-1]
        if pv.ph_after:
            D = pv.P["old_off"] - pv.off
            s.off = s.P["old_off"] - D
            joins.append(["placeholder", pv, s])
            out.append(s)
            continue
        same, consecutive = s.P is pv.P, s.P["idx"] == pv.P["idx"] + 1
        end_tl = pv.end - pv.off * fms
        unchanged = s.P["old_off"] - (pv.P["old_off"] - pv.off)
        considered += 1
        hold, noise = False, []
        if not pv.end_died:
            noise.append(held_voice(pv, "end"))
            hold = True
        if not s.on_found:
            noise.append(held_voice(s, "start"))
            hold = True
        if not hold:
            for a, b, j_ms, db in _sound_between(E, pv, s):
                nl = db < E.soft(j_ms) + NOISE_NEAR_DB and db < E.loud(j_ms)
                sound_held.append({"cut_piece": sorted({pv.P["id"], s.P["id"]}), "src_ms": [round(a, 1), round(b, 1)],
                                   "sound_at_ms": j_ms, "db": round(db, 1), "thr_db": round(E.soft(j_ms), 1),
                                   "room_db": round(E.floor(j_ms), 1), "noise_like": bool(nl),
                                   "before": _words_text(pv.words, True), "after": _words_text(s.words)})
                noise.append(bool(nl))
                hold = True
        if hold and all(noise):
            noise_holds += 1
        # pieces that touch on the timeline: s starts where pv's piece ends (a dropped room-tone
        # piece between them leaves no hole)
        touching = s.P["f_in"] - (pv.P["f_out"] - pv.off)
        if hold:  # the join as the cut made it: pv's piece out-point, s's piece in-point
            if same:
                _absorb(pv, s)
                continue
            off, kind = touching, "keep"
        else:
            off = math.floor((s.on - (end_tl + opts["gap_ms"])) / fms + 1e-6)
            if not same and not consecutive:
                off = max(off, touching)
            kind = "stack"
            if off <= unchanged:  # already as tight as wanted: never widen, never go back in the source
                off = unchanged
                if same:
                    _absorb(pv, s)
                    continue
                if consecutive:
                    kind = "keep"
        s.off = off
        joins.append([kind, pv, s])
        out.append(s)
    segs = out
    room_warning = None
    if considered and noise_holds * 2 > considered:
        room_warning = (f"{noise_holds} of {considered} joins were held only by sound near the room-noise level: "
                        "the room may be noisy or drifting; check sound_in_pause / voice_not_died_away, "
                        "or raise --soft-db")
    if not segs[0].on_found:
        held_voice(segs[0], "start")
    if not segs[-1].end_died:
        held_voice(segs[-1], "end")

    # ---- audio bounds, fades and picture switches
    for s in segs:
        s.fi = s.fo = None
        s.a0_tl = s.P["lo"] - s.off * fms
        s.a1_tl = s.P["hi"] - s.off * fms
        s.v0, s.v1 = s.P["f_in"] - s.off, s.P["f_out"] - s.off
    items, bridges = [], []
    old_total_ms = int(cuts["frames"]) * fms
    if lead_phs:
        for p in lead_phs:
            items.append({"kind": "ph", "P": p, "v0": p["dst_f"], "v1": p["dst_f"] + p["frames"]})
        w1 = first.on - first.off * fms  # D = 0 before the first clip
        w0 = max(w1 - BRIDGE_XF_MS, first.P["lo"] - first.off * fms)
        bridges.append({"old0": 0.0, "old1": w1, "new0": 0.0, "fi": None, "fo": ("lin", w0, w1)})
        first.a0_tl, first.fi = w0, ("lin", w0, w1)
    else:
        first.v0 = 0
    jinfo = {}
    for kind, pv, s in joins:
        end_tl, on_tl = pv.end - pv.off * fms, s.on - s.off * fms
        pv_hi_tl, s_lo_tl = pv.P["hi"] - pv.off * fms, s.P["lo"] - s.off * fms
        if kind == "stack":
            g = max(on_tl - end_tl, 0.0)
            t = E.quiet_after(pv.end, g + 20)
            u = E.quiet_before(s.on, g + 20)
            x1 = min(on_tl, end_tl + t, pv_hi_tl)
            x0 = max(end_tl, on_tl - u, s_lo_tl)
            if x0 > x1:
                x0, x1 = x1, x0
            if x1 - x0 < XF_MIN_MS:
                c = (x0 + x1) / 2
                x0, x1 = c - XF_MIN_MS / 2, c + XF_MIN_MS / 2
            # never read source outside the kept pieces (s's piece starts no later than pv's ends)
            x0, x1 = max(x0, s_lo_tl), min(x1, pv_hi_tl)
            if x1 < x0:
                x0 = x1 = (x0 + x1) / 2
            pv.a1_tl, pv.fo, s.a0_tl, s.fi = x1, ("pow", x0, x1), x0, ("pow", x0, x1)
            v = math.ceil(x0 / fms - 1e-6)
            if v * fms > on_tl + 1e-6:
                v = math.floor(on_tl / fms + 1e-6)
            v_lo, v_hi = s.P["f_in"] - s.off, pv.P["f_out"] - pv.off
            v = min(max(v, v_lo), v_hi) if v_lo <= v_hi else min(max(v, v_hi), v_lo)
            pv.v1 = s.v0 = v
        elif kind == "keep":  # the cut's own join, its own crossfade
            hj = pv.P["f_out"] - pv.off
            x0, x1 = hj * fms - JOIN_XF_MS / 2, hj * fms + JOIN_XF_MS / 2
            pv.a1_tl, pv.fo, s.a0_tl, s.fi = x1, ("cut", x0, x1), x0, ("cut", x0, x1)
            pv.v1 = s.v0 = hj
        old_gap = (s.on - s.P["old_off"] * fms) - (pv.end - pv.P["old_off"] * fms)
        jinfo[id(s)] = {"type": kind, "closed_ms": round(old_gap - (on_tl - end_tl), 1)}

    def place_phs(pv, s):
        """placeholders after pv (and before s, or at the very end): original air on both sides."""
        D = pv.P["old_off"] - pv.off  # new timeline = old timeline + D frames, on both sides
        z1 = min(pv.end - pv.off * fms + BRIDGE_XF_MS, pv.P["hi"] - pv.off * fms)
        z0 = min(pv.end - pv.off * fms, z1)
        pv.a1_tl, pv.fo = z1, ("lin", z0, z1)
        phi = [{"kind": "ph", "P": p, "v0": p["dst_f"] + D, "v1": p["dst_f"] + D + p["frames"]} for p in pv.ph_after]
        if s is not None:
            w1 = s.on - s.off * fms
            w0 = max(w1 - BRIDGE_XF_MS, s.P["lo"] - s.off * fms)
            bridges.append({"old0": z0 - D * fms, "old1": w1 - D * fms, "new0": z0, "fi": ("lin", z0, z1),
                            "fo": ("lin", w0, w1)})
            s.a0_tl, s.fi = w0, ("lin", w0, w1)
            s.v0 = phi[-1]["v1"]
        else:
            bridges.append({"old0": z0 - D * fms, "old1": old_total_ms, "new0": z0, "fi": ("lin", z0, z1), "fo": None})
        pv.v1 = phi[0]["v0"]
        return phi

    for k, s in enumerate(segs):
        items.append({"kind": "seg", "S": s})
        if s.ph_after:
            items.extend(place_phs(s, segs[k + 1] if k + 1 < len(segs) else None))
    for it in items:
        if it["kind"] == "seg":
            it["v0"], it["v1"] = it["S"].v0, it["S"].v1
    for a, b in zip(items, items[1:]):  # pictures tile the timeline
        if a["v1"] != b["v0"]:
            if a["kind"] == "seg":
                a["v1"] = a["S"].v1 = b["v0"]
            elif b["kind"] == "seg":
                b["v0"] = b["S"].v0 = a["v1"]
            else:
                raise RuntimeError(f"placeholders do not tile the timeline at frame {a['v1']}")
    if items[0]["v0"] != 0:
        raise RuntimeError("the tightened timeline does not start at frame 0")
    for it in items:
        if it["v1"] <= it["v0"]:
            raise RuntimeError(f"empty clip on the tightened timeline at frame {it['v0']}")
    return {"fps": fps, "fms": fms, "items": items, "bridges": bridges, "jinfo": jinfo, "pieces": pieces,
            "words": words, "frames": items[-1]["v1"], "old_frames": int(cuts["frames"]),
            "dropped_air_pieces": dropped, "voice_not_died_away": voice_held, "sound_in_pause": sound_held, "E": E,
            "room_noise_warning": room_warning}


# ---------------------------------------------------------------- mix and checks

def _env(n, s0, fi, fo):
    t = (s0 + np.arange(n)) * 1000.0 / AR
    env = np.ones(n, dtype=np.float64)
    for f, up in ((fi, True), (fo, False)):
        if f is None:
            continue
        kind, x0, x1 = f
        m = np.clip(((t - x0) if up else (x1 - t)) / max(x1 - x0, 1e-9), 0.0, 1.0)
        env *= m if kind == "lin" else np.sin(m * np.pi / 2)
    return env.astype(np.float32)


def mix(pl, src, old):
    spf = AR // pl["fps"]
    n = pl["frames"] * spf
    out = np.zeros((n, 2), dtype=np.float32)

    def add(arr, i0_src, t0_tl, t1_tl, fi, fo):
        s0, s1 = int(round(t0_tl * AR / 1000)), int(round(t1_tl * AR / 1000))
        s0c, s1c = max(0, s0), min(n, s1)
        if s1c <= s0c:
            return
        seg = take(arr, i0_src + (s0c - s0), s1c - s0c)
        out[s0c:s1c] += seg * _env(s1c - s0c, s0c, fi, fo)[:, None]

    for it in pl["items"]:
        if it["kind"] == "seg":
            s = it["S"]
            add(src, int(round(s.a0_tl * AR / 1000)) + s.off * spf, s.a0_tl, s.a1_tl, s.fi, s.fo)
    for b in pl["bridges"]:
        add(old, int(round(b["old0"] * AR / 1000)), b["new0"], b["new0"] + (b["old1"] - b["old0"]), b["fi"], b["fo"])
    return out


def _full_span(s):
    """timeline span where clip s plays at full level (an equal-power or the cut's crossfade is not
    full level; a linear crossfade against the cut's own audio of the same source is)."""
    c0 = s.a0_tl if s.fi is None else (s.fi[1] if s.fi[0] == "lin" else s.fi[2])
    c1 = s.a1_tl if s.fo is None else (s.fo[2] if s.fo[0] == "lin" else s.fo[1])
    return c0, c1


def _len_in(spans, t0, t1):
    return sum(max(0.0, min(b, t1) - max(a, t0)) for a, b in _union(spans))


def edge_check(pl, src, out, check_ms, check_db, ids, short_ms=SHORT_WIN_MS, short_db=SHORT_DB):
    """every clip's first/last voice, output mix vs the same span of the source, in two windows:
    check_ms (tolerance check_db) and short_ms (tolerance short_db; catches a soft onset or the last
    decay clipped under a loud word, which the long window averages away).

    Per row also: uncovered_ms = ms of the window where this clip does NOT play at full level
    (clipped by its own in/out point or a crossfade; > EDGE_UNCOVERED_MS fails the row whatever
    the dB), overlap_ms = ms of the window where another clip also sounds (stacked)."""
    fms = pl["fms"]
    src_ms = len(src) * 1000.0 / AR
    segs = [it for it in pl["items"] if it["kind"] == "seg"]
    rows = []
    for it in segs:
        s = it["S"]
        c0, c1 = _full_span(s)
        others = [(o["S"].a0_tl, o["S"].a1_tl) for o in segs if o is not it]
        for win, tol in ((check_ms, check_db), (short_ms, short_db)):
            for edge, t0, t1 in (("start", s.on_true, s.on_true + win), ("end", s.end_true - win, s.end_true)):
                t0, t1 = max(t0, 0.0), min(t1, src_ms)
                if t1 <= t0:
                    continue
                T0, T1 = t0 - s.off * fms, t1 - s.off * fms
                lo, ls = level_db(out, T0, T1), level_db(src, t0, t1)
                d = lo - ls
                unc = (T1 - T0) - _len_in([(c0, c1)], T0, T1)
                ovl = _len_in(others, T0, T1)
                rows.append({"piece": ids[id(it)], "edge": edge, "win_ms": win,
                             "dst_s": round(T0 / 1000 if edge == "start" else T1 / 1000, 3),
                             "src_ms": round(t0 if edge == "start" else t1, 1), "out_db": round(lo, 2),
                             "src_db": round(ls, 2), "diff_db": round(d, 2), "uncovered_ms": round(unc, 1),
                             "overlap_ms": round(ovl, 1), "ok": abs(d) <= tol and unc <= EDGE_UNCOVERED_MS,
                             "words": _words_text(s.words, edge == "end"), "cut_piece": s.parts})
    return rows


def _union(iv):
    out = []
    for a, b in sorted(x for x in iv if x[1] > x[0]):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def missing_audit(pl):
    """independent of the planner: which stretches of the cut's timeline are not in the output at
    full level (an equal-power crossfade counts as missing, the cut's own join crossfade and the
    linear crossfades against the cut's own audio do not), and is there any sound in them?"""
    fms, E = pl["fms"], pl["E"]
    real = [p for p in pl["pieces"] if not p["ph"]]
    covered = []
    for it in pl["items"]:
        if it["kind"] != "seg":
            continue
        s = it["S"]
        c0 = s.fi[2] if s.fi and s.fi[0] == "pow" else s.a0_tl
        c1 = s.fo[1] if s.fo and s.fo[0] == "pow" else s.a1_tl
        if c1 <= c0:
            continue
        a, b = c0 + s.off * fms, c1 + s.off * fms  # source span at full level
        for q in real:
            lo, hi = max(a, q["lo"]), min(b, q["hi"])
            if hi > lo:
                covered.append((lo - q["old_off"] * fms, hi - q["old_off"] * fms))
    for b in pl["bridges"]:
        covered.append((b["old0"], b["old1"]))
    covered = _union(covered)
    rows = []
    for q in real:
        t0, t1 = q["dst_f"] * fms, (q["dst_f"] + q["frames"]) * fms
        gaps, cur = [], t0
        for a, b in covered:
            if b <= cur or a >= t1:
                continue
            if a > cur:
                gaps.append((cur, min(a, t1)))
            cur = max(cur, b)
        if cur < t1:
            gaps.append((cur, t1))
        for a, b in gaps:
            sa, sb = a + q["old_off"] * fms, b + q["old_off"] * fms
            hit = E.sound_in(sa, sb)
            if hit is None:
                continue
            ws = [w for w in pl["words"] if w["P"] is q and w["s1"] > sa - 300 and w["s0"] < sb + 300]
            rows.append({"edge": "missing", "cut_piece": q["id"], "cut_ms": [round(a, 1), round(b, 1)],
                         "src_ms": [round(sa, 1), round(sb, 1)], "sound_at_ms": hit[0], "db": round(hit[1], 1),
                         "thr_db": round(E.soft(hit[0]), 1), "ok": False,
                         "words": " ".join(w["w"] for w in ws)})
    return rows


def added_audit(pl, ids):
    """the output's audio may come only from the kept pieces of the source (plus the cut's own
    audio around placeholders): any clip reading source outside every kept piece is listed.
    The cut's own join crossfade reaches JOIN_XF_MS / 2 past a piece edge and is allowed."""
    fms = pl["fms"]
    tol = JOIN_XF_MS / 2 + 0.5
    kept = _union([(p["lo"] - tol, p["hi"] + tol) for p in pl["pieces"] if not p["ph"]])
    rows = []
    for it in pl["items"]:
        if it["kind"] != "seg":
            continue
        s = it["S"]
        a, b = s.a0_tl + s.off * fms, s.a1_tl + s.off * fms
        cur = a
        outside = []
        for k0, k1 in kept:
            if k1 <= cur or k0 >= b:
                continue
            if k0 > cur:
                outside.append((cur, min(k0, b)))
            cur = max(cur, k1)
        if cur < b:
            outside.append((cur, b))
        for x0, x1 in outside:
            if x1 - x0 > 0.5:
                rows.append({"edge": "added", "piece": ids[id(it)], "src_ms": [round(x0, 1), round(x1, 1)],
                             "cut_piece": s.parts, "ok": False, "words": _words_text(s.words)})
    return rows


# ---------------------------------------------------------------- video

def _run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")


def _select(ranges):
    t = [f"between(n,{a},{b - 1})" for a, b in ranges]
    if len(t) <= 20:
        return "+".join(t)
    return "+".join("(" + "+".join(t[i:i + 20]) + ")" for i in range(0, len(t), 20))


def render(pl, source, old_mp4, out_path, audio, crf=18):
    """write the tightened video + mixed audio to out_path (an mp4; the caller moves it into place)."""
    fps = pl["fps"]
    runs = []  # [input, [(f0, f1), ...]]
    for it in pl["items"]:
        if it["kind"] == "seg":
            inp, f0, f1 = str(source), it["v0"] + it["S"].off, it["v1"] + it["S"].off
        else:
            inp, f0, f1 = str(old_mp4), it["P"]["dst_f"], it["P"]["dst_f"] + it["P"]["frames"]
        if runs and runs[-1][0] == inp and f0 >= runs[-1][1][-1][1]:
            runs[-1][1].append((f0, f1))
        else:
            runs.append([inp, [(f0, f1)]])
    flag = _graph_flag()
    enc = ["-c:v", "libx264", "-preset", "fast", "-crf", str(crf), "-pix_fmt", "yuv420p", "-r", str(fps), "-g", str(fps)]
    with tempfile.TemporaryDirectory(prefix="sltight_") as tmp:
        parts = []
        for r, (inp, ranges) in enumerate(runs):
            nf = sum(b - a for a, b in ranges)
            fg = os.path.join(tmp, f"fg{r}.txt")
            with open(fg, "w", encoding="ascii") as f:
                f.write(f"[0:v]setpts=PTS-STARTPTS,fps={fps},select='{_select(ranges)}',setpts=N/{fps}/TB,"
                        f"trim=end_frame={nf},setsar=1,format=yuv420p[v]")
            part = os.path.join(tmp, f"p{r:03d}.mp4")
            res = _run(["ffmpeg", "-v", "error", "-nostdin", "-y", "-i", inp, flag, fg, "-map", "[v]", *enc, "-an",
                        "-frames:v", str(nf), part])
            if res.returncode != 0:
                msg = res.stderr.strip().splitlines()
                raise RuntimeError("ffmpeg video part failed: " + (msg[-1] if msg else "unknown error"))
            parts.append(part)
        wav = os.path.join(tmp, "mix.wav")
        with wave.open(wav, "wb") as wf:
            wf.setnchannels(2)
            wf.setsampwidth(2)
            wf.setframerate(AR)
            wf.writeframes((np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes())
        lst = os.path.join(tmp, "parts.txt")
        with open(lst, "w", encoding="utf-8") as f:
            f.write("".join("file '" + p.replace("\\", "/").replace("'", "'\\''") + "'\n" for p in parts))
        res = _run(["ffmpeg", "-v", "error", "-nostdin", "-y", "-f", "concat", "-safe", "0", "-i", lst, "-i", wav,
                    "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                    "-movflags", "+faststart", "-f", "mp4", out_path])
        if res.returncode != 0:
            msg = res.stderr.strip().splitlines()
            raise RuntimeError("ffmpeg mux failed: " + (msg[-1] if msg else "unknown error"))


# ---------------------------------------------------------------- outputs

def _strip(stem):
    stem = str(stem)
    return stem[:-4] if stem.endswith(".mp4") else stem


def _read_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError) as e:
        raise ValueError(f"cannot read {path}: {e}")


def _records(pl, cuts):
    fms = pl["fms"]
    notes = {x.get("id"): x.get("note") for x in cuts.get("placeholders", [])}
    pieces, words, phs, ids = [], [], [], {}
    for n, it in enumerate(pl["items"]):
        ids[id(it)] = n
        v0, v1 = it["v0"], it["v1"]
        if it["kind"] == "ph":
            raw = it["P"]["raw"]
            rec = {"id": n, "placeholder": raw.get("placeholder"), "src_in_ms": raw.get("src_in_ms"),
                   "src_out_ms": raw.get("src_out_ms"), "dst_in": round(v0 * fms / 1000, 3),
                   "dst_in_ms": round(v0 * fms, 3), "dst_f": v0, "frames": v1 - v0, "w0": None, "w1": None,
                   "text": raw.get("text", f"[{raw.get('placeholder')}]"), "cut_piece": it["P"]["id"]}
            if "tone_db" in raw:
                rec["tone_db"] = raw["tone_db"]
            phs.append({"id": raw.get("placeholder"), "start": rec["dst_in"], "end": round(v1 * fms / 1000, 3),
                        "note": notes.get(raw.get("placeholder"))})
        else:
            s = it["S"]
            f_in, f_out = v0 + s.off, v1 + s.off
            ws = s.words
            rec = {"id": n, "src_in": round(f_in * fms / 1000, 3), "src_out": round(f_out * fms / 1000, 3),
                   "dst_in": round(v0 * fms / 1000, 3), "src_in_ms": round(f_in * fms, 3),
                   "src_out_ms": round(f_out * fms, 3), "dst_in_ms": round(v0 * fms, 3), "dst_f": v0,
                   "frames": v1 - v0, "f_in": f_in, "f_out": f_out,
                   "w0": ws[0].get("src_i", ws[0].get("i")) if ws else s.P["raw"].get("w0"),
                   "w1": ws[-1].get("src_i", ws[-1].get("i")) if ws else s.P["raw"].get("w1"),
                   "text": " ".join(w["w"] for w in ws) if ws else s.P["raw"].get("text", ""),
                   "voice_src_ms": [round(s.on_true, 1), round(s.end_true, 1)],
                   "audio_src_ms": [round(s.a0_tl + s.off * fms, 3), round(s.a1_tl + s.off * fms, 3)],
                   "audio_dst_ms": [round(s.a0_tl, 3), round(s.a1_tl, 3)], "cut_piece": s.parts}
            for w in ws:
                nw = {k: v for k, v in w.items() if k not in ("P", "s0", "s1", "start_ms", "end_ms")}
                nw.update({"i": len(words), "start_ms": int(round(w["s0"] - s.off * fms)),
                           "end_ms": int(round(w["s1"] - s.off * fms))})
                words.append(nw)
        pieces.append(rec)
    joins = []
    for n in range(1, len(pl["items"])):
        a, b = pl["items"][n - 1], pl["items"][n]
        if a["kind"] == "ph" or b["kind"] == "ph":
            info = {"type": "placeholder", "closed_ms": 0.0}
        else:
            info = pl["jinfo"].get(id(b["S"]), {"type": "stack", "closed_ms": 0.0})
        t = b["v0"] * fms
        joins.append({"n": n, "dst": round(t / 1000, 3), **info,
                      "before": " ".join(w["w"] for w in words if t - 2500 <= w["end_ms"] <= t + 100),
                      "after": " ".join(w["w"] for w in words if t - 100 <= w["start_ms"] <= t + 2500)})
    return pieces, joins, words, phs, ids


def tighten(stem, out_stem, min_pause_ms=250, keep_placeholders=True, *, gap_ms=0, soft_db=4.0, loud_db=10.0,
            search_ms=400, check_ms=120, check_db=0.05, crf=18):
    """Close the join pauses of a cut.py cut; write <out>.mp4/.cuts.json/.words.json; return a summary.

    The summary (and cuts.json) carries edge_check (every clip start and end, output vs source
    level), missing_check (sound in a stretch of the cut that is not in the output), added_check
    (output audio read from outside every kept piece), edge_bad (all failures),
    voice_not_died_away and sound_in_pause (joins kept as the cut made them)."""
    stem, out_stem = _strip(stem), _strip(out_stem)
    if os.path.normcase(os.path.abspath(stem)) == os.path.normcase(os.path.abspath(out_stem)):
        raise ValueError("--out must differ from the input stem (the cut is read while the output is written)")
    if gap_ms < 0:
        raise ValueError("--gap-ms must be >= 0 (an overlap would play two voices at once)")
    cuts = _read_json(stem + ".cuts.json")
    wdoc = _read_json(stem + ".words.json")
    if not isinstance(cuts, dict) or not isinstance(cuts.get("pieces"), list) or "fps" not in cuts:
        raise ValueError(f"{stem}.cuts.json: no pieces/fps (not a cut.py cuts.json)")
    if not isinstance(wdoc, dict) or not isinstance(wdoc.get("words"), list):
        raise ValueError(f"{stem}.words.json: no 'words' list")
    fps = int(cuts["fps"])
    if fps <= 0 or AR % fps:
        raise ValueError(f"fps {fps} must divide {AR}")
    source = cuts.get("source")
    if not source or not os.path.isfile(source):
        raise ValueError(f"source of the cut not found: {source}")
    if not os.path.isfile(stem + ".mp4"):
        raise ValueError(f"cut not found: {stem}.mp4")
    if os.path.normcase(os.path.abspath(out_stem)) == os.path.normcase(os.path.abspath(os.path.splitext(source)[0])):
        raise ValueError("--out must differ from the raw source's stem (it would overwrite the recording)")
    mode = cuts.get("channel_balance", "none")
    if mode not in ("none", "gain", "copy_left", "copy_right"):
        raise ValueError(f"{stem}.cuts.json: unknown channel_balance {mode!r}")
    g = cuts.get("gains")
    if mode == "gain" and not (isinstance(g, list) and len(g) == 2 and all(isinstance(x, (int, float)) and x > 0 for x in g)):
        raise ValueError(f"{stem}.cuts.json: channel_balance is 'gain' but gains is missing or not two positive numbers")
    src = apply_balance(load_audio(source), cuts)
    old = load_audio(stem + ".mp4")
    opts = {"min_pause_ms": min_pause_ms, "keep_placeholders": keep_placeholders, "gap_ms": gap_ms,
            "soft_db": soft_db, "loud_db": loud_db, "search_ms": search_ms}
    pl = plan(cuts, wdoc, slice_db(src.mean(axis=1)), opts)
    audio = mix(pl, src, old)
    pieces, joins, words, phs, ids = _records(pl, cuts)
    edges = edge_check(pl, src, audio, check_ms, check_db, ids)
    missing = missing_audit(pl)
    added = added_audit(pl, ids)
    bad = [e for e in edges if not e["ok"]] + missing + added
    os.makedirs(os.path.dirname(os.path.abspath(out_stem)) or ".", exist_ok=True)
    fms = pl["fms"]
    frames = pl["frames"]
    settings = dict(opts, check_ms=check_ms, check_db=check_db, crf=crf)
    doc = {"source": source, "cut": os.path.abspath(stem + ".mp4"), "fps": fps, "frames": frames,
           "duration_s": round(frames / fps, 3), "duration_before_s": round(int(cuts["frames"]) / fps, 3),
           "settings": settings, "pieces": pieces, "joins": joins, "placeholders": phs,
           "dropped_air_pieces": pl["dropped_air_pieces"], "voice_not_died_away": pl["voice_not_died_away"],
           "sound_in_pause": pl["sound_in_pause"], "room_noise_warning": pl["room_noise_warning"],
           "edge_check": edges, "missing_check": missing,
           "added_check": added, "edge_bad": bad}
    for k in ("channel_balance", "channel_balance_db", "gains", "preset"):
        if k in cuts:
            doc[k] = cuts[k]
    wd = {"source": os.path.abspath(out_stem + ".mp4"), "language": wdoc.get("language"),
          "duration_ms": int(frames * fms), "words": words}
    # all three files are written under temp names and moved into place together
    final = [out_stem + ".mp4", out_stem + ".cuts.json", out_stem + ".words.json"]
    temp = [out_stem + ".tmp.mp4", out_stem + ".tmp.cuts.json", out_stem + ".tmp.words.json"]
    try:
        render(pl, source, stem + ".mp4", temp[0], audio, crf)
        _dump(temp[1], doc)
        _dump(temp[2], wd)
        for a, b in zip(temp, final):
            os.replace(a, b)
    except BaseException:
        for t in temp:
            if os.path.exists(t):
                os.remove(t)
        raise
    return {"out": out_stem, "frames": frames, "duration_s": doc["duration_s"],
            "duration_before_s": doc["duration_before_s"], "pieces": len(pieces), "joins": joins,
            "placeholders": phs, "edge_check": edges, "missing_check": missing, "edge_bad": bad,
            "voice_not_died_away": pl["voice_not_died_away"], "sound_in_pause": pl["sound_in_pause"],
            "room_noise_warning": pl["room_noise_warning"]}


# ---------------------------------------------------------------- CLI

def main(argv=None):
    store.print_utf8()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stem", help="cut.py output stem (<stem>.mp4, .cuts.json, .words.json)")
    ap.add_argument("--out", required=True, help="output stem (must differ from the input stem)")
    ap.add_argument("--min-pause-ms", type=int, default=250, help="close every pause at least this long")
    ap.add_argument("--no-placeholders", action="store_true", help="drop placeholders and close their joins")
    ap.add_argument("--gap-ms", type=int, default=0, help="room tone left between a voice end and the next start")
    ap.add_argument("--soft-db", type=float, default=4.0,
                    help="edge = 3 slices below room tone + this, sound = 3 slices at or above it (dB)")
    ap.add_argument("--loud-db", type=float, default=10.0,
                    help="any single slice this far above the room tone is sound (a click, a consonant) (dB)")
    ap.add_argument("--check-db", type=float, default=0.05, help="edge check tolerance, 120 ms window (dB)")
    a = ap.parse_args(argv)
    try:
        require_ffmpeg()
    except cut.MissingTool as e:
        print(str(e), file=sys.stderr)
        return 3
    try:
        res = tighten(a.stem, a.out, a.min_pause_ms, not a.no_placeholders, gap_ms=a.gap_ms,
                      soft_db=a.soft_db, loud_db=a.loud_db, check_db=a.check_db)
    except cut.MissingTool as e:
        print(str(e), file=sys.stderr)
        return 3
    except (ValueError, RuntimeError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    closed = [j for j in res["joins"] if j["type"] == "stack"]
    print(f"{res['pieces']} clips, {len(closed)} pauses closed, {res['duration_before_s']:.2f}s -> "
          f"{res['duration_s']:.2f}s, placeholders {[(p['id'], p['start'], p['end']) for p in res['placeholders']]}")
    for v in res["voice_not_died_away"]:
        print(f"  voice not died away: cut piece {v['cut_piece']} {v['edge']} at {v['src_ms'] / 1000:.2f}s src "
              f"({v['words']}): pause kept as cut")
    for v in res["sound_in_pause"]:
        print(f"  sound in pause: cut piece {v['cut_piece']} at {v['sound_at_ms'] / 1000:.2f}s src ({v['db']:.1f} dB, "
              f"between '{v['before']}' and '{v['after']}'): pause kept as cut")
    if res["room_noise_warning"]:
        print(f"warning: {res['room_noise_warning']}")
    bad = res["edge_bad"]
    print(f"edge check: {len(res['edge_check'])} clip edges + missing audit, {len(bad)} problem(s)")
    for e in bad:
        if e["edge"] == "added":
            print(f"  ADDED: piece {e['piece']} plays source {e['src_ms'][0] / 1000:.2f}-{e['src_ms'][1] / 1000:.2f}s "
                  f"that is in no kept piece of the cut (cut piece {e['cut_piece']})")
        elif e["edge"] == "missing":
            print(f"  MISSING: cut piece {e['cut_piece']} src {e['src_ms'][0] / 1000:.2f}-{e['src_ms'][1] / 1000:.2f}s "
                  f"is not in the output but has sound ({e['db']:.1f} dB at {e['sound_at_ms'] / 1000:.2f}s; {e['words']})")
        else:
            print(f"  piece {e['piece']} {e['edge']} at {e['dst_s']:.2f}s ({e['words']}), {e['win_ms']} ms window: "
                  f"{e['diff_db']:+.2f} dB vs the source, {e['uncovered_ms']:.0f} ms of voice not played at full "
                  f"level, {e['overlap_ms']:.0f} ms under another clip (cut piece {e['cut_piece']})")
    print(f"wrote {res['out']}.mp4; next: check_cut.py {res['out']}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

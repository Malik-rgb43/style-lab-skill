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
  - nothing in the pause is sound (any 10 ms slice >= room tone + --voice-db, 8 dB): an unlisted
    word, a laugh or a breath there keeps the join as cut (listed in cuts.json sound_in_pause).
    At a join between pieces the cut's own air on both sides is checked too.

  - lip sync is exact: picture and sound of every clip share one offset in whole frames
  - the two clips meet with an equal-power crossfade inside the quiet room tone between the
    previous voice end and the next voice start, within both pieces
  - a join that is already tighter than the target is left as the cut made it (never widened);
    inside a piece, a pause shorter than --min-pause-ms on the true edges is left alone
  - placeholders keep their length and the air on both sides (--no-placeholders drops them and
    closes the join instead); the cut's own lead and end air are kept

Checks (every time, on the in-memory output mix before encoding):
  edge check: for every clip, the level of its first and last 120 ms of voice (true start/end,
    measured without the cut's in/out points) vs the same span in the source; off by more than
    --check-db (0.05 dB, i.e. not 0.0 dB) goes into edge_bad. A cut that clipped a word shows here.
  missing audit: every stretch of the cut's timeline that is not in the output at full level and
    holds any slice >= room tone + --voice-db goes into edge_bad (edge "missing").
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
    """true edges on the 10 ms slice levels of the source.

    voice = local room tone (10th percentile over +-3 s) + voice_db: what counts as sound.
    soft  = room tone + soft_db: an edge is where QUIET_RUN slices in a row are below it."""

    def __init__(self, db, voice_db, search_ms, soft_db):
        self.db, self.voice_db, self.search_ms, self.soft_db = db, voice_db, search_ms, soft_db

    def at(self, j):
        return self.db[j] if 0 <= j < len(self.db) else -200.0

    def floor(self, ms):
        k = int(ms // 10)
        w = self.db[max(0, k - 300): k + 300]
        if not len(w):
            w = self.db
        return float(np.percentile(w, 10))

    def thr(self, ms):
        return self.floor(ms) + self.voice_db

    def soft(self, ms):
        return self.floor(ms) + self.soft_db

    def _quiet(self, j, thr, step):
        return all(self.at(j + step * i) < thr for i in range(QUIET_RUN))

    def true_end(self, ms):
        """(end_ms, died): walk forward from the aligned word end until the voice has died away;
        died is False when it is still sounding at the search limit (end_ms is that limit)."""
        thr, k = self.soft(ms), int(ms // 10)
        for j in range(k, k + self.search_ms // 10):
            if self._quiet(j, thr, 1):
                return j * 10.0, True
        return k * 10.0 + self.search_ms, False

    def true_start(self, ms):
        """(start_ms, found): walk back from the aligned word start to the first sound."""
        thr_v, thr_s, k = self.thr(ms), self.soft(ms), int(ms // 10)
        if self.at(k) < thr_v and self.at(k + 1) < thr_v:  # aligned start in room tone (a pinned retime)
            for j in range(k, k + 2 * self.search_ms // 10):
                if self.at(j) >= thr_v and self.at(j + 1) >= thr_v:
                    k = j
                    break
            else:
                return float(ms), False
        for j in range(k, k - self.search_ms // 10, -1):
            if self._quiet(j, thr_s, -1):
                return (j + 1) * 10.0, True
        return k * 10.0 - self.search_ms, False

    def quiet_after(self, ms, limit):
        thr, t = self.soft(ms), 0.0
        while t < limit and self.at(int((ms + t) // 10)) < thr:
            t += 10
        return t

    def quiet_before(self, ms, limit):
        thr, u = self.soft(ms), 0.0
        while u < limit and self.at(int((ms - u - 10) // 10)) < thr:
            u += 10
        return u

    def sound_in(self, a_ms, b_ms):
        """loudest slice >= the voice threshold fully inside [a, b): (slice_ms, db), or None."""
        j0, j1 = int(math.ceil(a_ms / 10 - 1e-9)), int(math.floor(b_ms / 10 + 1e-9))
        if j1 <= j0:
            return None
        thr = self.thr((a_ms + b_ms) / 2)
        js = [j for j in range(j0, j1) if self.at(j) >= thr]
        if not js:
            return None
        j = max(js, key=self.at)
        return j * 10.0, float(self.at(j))

    def sound_span(self, lo, hi):
        """first sound and true end inside [lo, hi): (on, found, end, died), None when only room tone."""
        thr = self.thr((lo + hi) / 2)
        loud = [j for j in range(int(lo // 10), int(hi // 10)) if self.at(j) >= thr and self.at(j + 1) >= thr]
        if not loud:
            return None
        on, found = self.true_start(loud[0] * 10.0 + 5)
        end, died = self.true_end((loud[-1] + 1) * 10.0)
        return on, found, max((loud[-1] + 1) * 10.0, end), died


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
    E = Energy(np.asarray(src_db, dtype=np.float64), opts["voice_db"], opts["search_ms"], opts.get("soft_db", 4.0))
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
        voice_held.append({"cut_piece": s.P["id"], "edge": edge, "src_ms": round(ms, 1),
                           "words": _words_text(s.words, edge == "end"),
                           "why": f"the voice has not died away within {opts['search_ms']} ms of the word {edge}"})

    # ---- offsets (source frame - timeline frame), whole frames
    first = segs[0]
    first.off = first.P["old_off"] if lead_phs else first.P["f_in"]
    out, joins = [first], []
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
        hold = False
        if not pv.end_died:
            held_voice(pv, "end")
            hold = True
        if not s.on_found:
            held_voice(s, "start")
            hold = True
        if not hold:
            for a, b, j_ms, db in _sound_between(E, pv, s):
                sound_held.append({"cut_piece": sorted({pv.P["id"], s.P["id"]}), "src_ms": [round(a, 1), round(b, 1)],
                                   "sound_at_ms": j_ms, "db": round(db, 1), "thr_db": round(E.thr(j_ms), 1),
                                   "before": _words_text(pv.words, True), "after": _words_text(s.words)})
                hold = True
        off = unchanged if hold else math.floor((s.on - (end_tl + opts["gap_ms"])) / fms + 1e-6)
        if not same and not consecutive:  # a dropped room-tone piece between: no hole in the timeline
            off = max(off, math.ceil((s.P["lo"] - (pv.P["hi"] - pv.off * fms)) / fms - 1e-6))
        kind = "stack"
        if off <= unchanged:  # already as tight as wanted (or held): never widen, never go back in the source
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
        on_tl = first.on - first.off * fms
        bridges.append({"old0": 0.0, "old1": first.on - first.P["old_off"] * fms, "new0": 0.0,
                        "fi": None, "fo": ("lin", on_tl - BRIDGE_XF_MS, on_tl)})
        first.a0_tl, first.fi = on_tl - BRIDGE_XF_MS, ("lin", on_tl - BRIDGE_XF_MS, on_tl)
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
        D = pv.P["old_off"] - pv.off
        end_tl = pv.end - pv.off * fms
        pv.a1_tl, pv.fo = end_tl + BRIDGE_XF_MS, ("lin", end_tl, end_tl + BRIDGE_XF_MS)
        phi = [{"kind": "ph", "P": p, "v0": p["dst_f"] + D, "v1": p["dst_f"] + D + p["frames"]} for p in pv.ph_after]
        old0 = pv.end - pv.P["old_off"] * fms
        new0 = old0 + D * fms
        if s is not None:
            on_tl = s.on - s.off * fms
            old1 = s.on - s.P["old_off"] * fms
            bridges.append({"old0": old0, "old1": old1, "new0": new0, "fi": ("lin", new0, new0 + BRIDGE_XF_MS),
                            "fo": ("lin", on_tl - BRIDGE_XF_MS, on_tl)})
            s.a0_tl, s.fi = on_tl - BRIDGE_XF_MS, ("lin", on_tl - BRIDGE_XF_MS, on_tl)
            s.v0 = phi[-1]["v1"]
        else:
            bridges.append({"old0": old0, "old1": old_total_ms, "new0": new0,
                            "fi": ("lin", new0, new0 + BRIDGE_XF_MS), "fo": None})
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
            "dropped_air_pieces": dropped, "voice_not_died_away": voice_held, "sound_in_pause": sound_held, "E": E}


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


def edge_check(pl, src, out, check_ms, check_db, ids):
    """level of every clip's first/last check_ms of voice: output mix vs the same span of the source."""
    fms = pl["fms"]
    src_ms = len(src) * 1000.0 / AR
    rows = []
    for it in pl["items"]:
        if it["kind"] != "seg":
            continue
        s = it["S"]
        for edge, t0, t1 in (("start", s.on_true, s.on_true + check_ms), ("end", s.end_true - check_ms, s.end_true)):
            t0, t1 = max(t0, 0.0), min(t1, src_ms)
            if t1 <= t0:
                continue
            T0, T1 = t0 - s.off * fms, t1 - s.off * fms
            lo, ls = level_db(out, T0, T1), level_db(src, t0, t1)
            d = lo - ls
            rows.append({"piece": ids[id(it)], "edge": edge, "dst_s": round(T0 / 1000 if edge == "start" else T1 / 1000, 3),
                         "src_ms": round(t0 if edge == "start" else t1, 1), "out_db": round(lo, 2),
                         "src_db": round(ls, 2), "diff_db": round(d, 2), "ok": abs(d) <= check_db,
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
                         "thr_db": round(E.thr(hit[0]), 1), "ok": False,
                         "words": " ".join(w["w"] for w in ws)})
    return rows


# ---------------------------------------------------------------- video

def _run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")


def _select(ranges):
    t = [f"between(n,{a},{b - 1})" for a, b in ranges]
    if len(t) <= 20:
        return "+".join(t)
    return "+".join("(" + "+".join(t[i:i + 20]) + ")" for i in range(0, len(t), 20))


def render(pl, source, old_mp4, out_stem, audio, crf=18):
    """<out>.mp4, written to <out>.tmp.mp4 first and moved into place (no half file on failure)."""
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
    final, part_out = out_stem + ".mp4", out_stem + ".tmp.mp4"
    try:
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
                        "-movflags", "+faststart", "-f", "mp4", part_out])
            if res.returncode != 0:
                msg = res.stderr.strip().splitlines()
                raise RuntimeError("ffmpeg mux failed: " + (msg[-1] if msg else "unknown error"))
        os.replace(part_out, final)
    except BaseException:
        if os.path.exists(part_out):
            os.remove(part_out)
        raise


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


def tighten(stem, out_stem, min_pause_ms=250, keep_placeholders=True, *, gap_ms=0, voice_db=8.0, soft_db=4.0,
            search_ms=400, check_ms=120, check_db=0.05, crf=18):
    """Close the join pauses of a cut.py cut; write <out>.mp4/.cuts.json/.words.json; return a summary.

    The summary (and cuts.json) carries edge_check (every clip start and end, output vs source
    level), missing_check (sound in a stretch of the cut that is not in the output), edge_bad (both
    kinds of failure), voice_not_died_away and sound_in_pause (joins kept as the cut made them)."""
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
    src = apply_balance(load_audio(source), cuts)
    old = load_audio(stem + ".mp4")
    opts = {"min_pause_ms": min_pause_ms, "keep_placeholders": keep_placeholders, "gap_ms": gap_ms,
            "voice_db": voice_db, "soft_db": soft_db, "search_ms": search_ms}
    pl = plan(cuts, wdoc, slice_db(src.mean(axis=1)), opts)
    audio = mix(pl, src, old)
    pieces, joins, words, phs, ids = _records(pl, cuts)
    edges = edge_check(pl, src, audio, check_ms, check_db, ids)
    missing = missing_audit(pl)
    bad = [e for e in edges if not e["ok"]] + missing
    os.makedirs(os.path.dirname(os.path.abspath(out_stem)) or ".", exist_ok=True)
    render(pl, source, stem + ".mp4", out_stem, audio, crf)
    fms = pl["fms"]
    frames = pl["frames"]
    settings = dict(opts, check_ms=check_ms, check_db=check_db, crf=crf)
    doc = {"source": source, "cut": os.path.abspath(stem + ".mp4"), "fps": fps, "frames": frames,
           "duration_s": round(frames / fps, 3), "duration_before_s": round(int(cuts["frames"]) / fps, 3),
           "settings": settings, "pieces": pieces, "joins": joins, "placeholders": phs,
           "dropped_air_pieces": pl["dropped_air_pieces"], "voice_not_died_away": pl["voice_not_died_away"],
           "sound_in_pause": pl["sound_in_pause"], "edge_check": edges, "missing_check": missing, "edge_bad": bad}
    for k in ("channel_balance", "channel_balance_db", "gains", "preset"):
        if k in cuts:
            doc[k] = cuts[k]
    _dump(out_stem + ".cuts.json", doc)
    _dump(out_stem + ".words.json", {"source": os.path.abspath(out_stem + ".mp4"), "language": wdoc.get("language"),
                                     "duration_ms": int(frames * fms), "words": words})
    return {"out": out_stem, "frames": frames, "duration_s": doc["duration_s"],
            "duration_before_s": doc["duration_before_s"], "pieces": len(pieces), "joins": joins,
            "placeholders": phs, "edge_check": edges, "missing_check": missing, "edge_bad": bad,
            "voice_not_died_away": pl["voice_not_died_away"], "sound_in_pause": pl["sound_in_pause"]}


# ---------------------------------------------------------------- CLI

def main(argv=None):
    store.print_utf8()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stem", help="cut.py output stem (<stem>.mp4, .cuts.json, .words.json)")
    ap.add_argument("--out", required=True, help="output stem (must differ from the input stem)")
    ap.add_argument("--min-pause-ms", type=int, default=250, help="close every pause at least this long")
    ap.add_argument("--no-placeholders", action="store_true", help="drop placeholders and close their joins")
    ap.add_argument("--gap-ms", type=int, default=0, help="room tone left between a voice end and the next start")
    ap.add_argument("--voice-db", type=float, default=8.0, help="sound = room tone + this (dB)")
    ap.add_argument("--soft-db", type=float, default=4.0, help="an edge = 3 slices below room tone + this (dB)")
    ap.add_argument("--check-db", type=float, default=0.05, help="edge check tolerance (dB)")
    a = ap.parse_args(argv)
    try:
        require_ffmpeg()
    except cut.MissingTool as e:
        print(str(e), file=sys.stderr)
        return 3
    try:
        res = tighten(a.stem, a.out, a.min_pause_ms, not a.no_placeholders, gap_ms=a.gap_ms, voice_db=a.voice_db,
                      soft_db=a.soft_db, check_db=a.check_db)
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
    bad = res["edge_bad"]
    print(f"edge check: {len(res['edge_check'])} clip edges + missing audit, {len(bad)} problem(s)")
    for e in bad:
        if e["edge"] == "missing":
            print(f"  MISSING: cut piece {e['cut_piece']} src {e['src_ms'][0] / 1000:.2f}-{e['src_ms'][1] / 1000:.2f}s "
                  f"is not in the output but has sound ({e['db']:.1f} dB at {e['sound_at_ms'] / 1000:.2f}s; {e['words']})")
        else:
            print(f"  piece {e['piece']} {e['edge']} at {e['dst_s']:.2f}s ({e['words']}): {e['diff_db']:+.2f} dB vs the "
                  f"source (cut piece {e['cut_piece']})")
    print(f"wrote {res['out']}.mp4; next: check_cut.py {res['out']}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

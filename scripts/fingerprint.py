#!/usr/bin/env python3
"""Fingerprint of a reference set: per-layer metrics with spreads, traits, duplicates, held-out ids.

Reads <pack>/analysis/<id>/ for every video (all files optional except measure.json):
  measure.json (measure/1), motion.json, sound.json (sound/1), look.json (look/1), words.json ({language, words}),
  events.manual.json ([{t, type, label, attrs}], written by Claude from the sheets),
  thumbs.json ([aHash hex strings], see --thumbs). The source video path, when known, is the "source"
  field of measure.json (or of an optional analysis/<id>/source.json {"source": path}).
Also <pack>/layers.json and <pack>/tokens.json (Task 4, code_tokens.py).

Usage:
  fingerprint.py <pack> [--train V01,V02 | --all] [--exclude V03,...]   write <pack>/fingerprint.json
  fingerprint.py <pack> --thumbs                                        write thumbs.json (10 aHash
                         values per video, ffmpeg + numpy) for every analysis that has a source path

Output: {schema "fingerprint/1", trained_on, held_out, dupes [[kept, dropped]], layers {layer: {source,
metrics}}, traits [{name, seen_in, share, status}], not_run}. A number is {median, p10, p90, n, seen_in}.
A trait seen in under 30 % of the training videos is an "option", otherwise a "rule-candidate".
A layer whose source is not "refs" takes its metrics from tokens.json (code) and never from the
reference measurements; a code layer whose tokens.json comes from another project is refused (exit 2).
Layer "look" (grain, vignette, sharpness, tone, split toning, halation, chromatic aberration, letterbox,
flicker, text finish; from look.json) is measured from the references only: its source is always "refs".

Duplicates: two videos whose 10 aHash thumbnails are at a mean Hamming distance of at most 6 (or, with no
thumbs.json, that share the measure.json video.sha256) are one video; the later one is dropped from
training and listed in "dupes".

Exit: 0 ok, 2 bad input, 3 ffmpeg missing (only for --thumbs).
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import code_tokens  # noqa: E402
import store  # noqa: E402

SCHEMA = "fingerprint/1"
RULE_SHARE = 0.30          # a trait seen in at least this share of the training videos is a rule candidate
DUP_HAMMING = 6            # mean Hamming distance (of 64 bits) at or below which two videos are duplicates
THUMBS_N = 10
BIG_TEXT_PCT = 8.0         # a text box at least this tall (percent of the frame) is "big text"
REF_ONLY_LAYERS = ("look",)   # measured from the references only; layers.json cannot point them elsewhere
LAYER_NAMES = tuple(code_tokens.LAYERS) + REF_ONLY_LAYERS
# layers.look.metrics keys (cover.py and the recipes read them): look.json summary.<key>.median ...
LOOK_SUMMARY_KEYS = ("grain_std", "grain_size_px", "vignette_falloff", "sharp_centre_edge_ratio", "tone_black_pt",
                     "tone_white_pt", "tone_contrast", "shadow_tint_a", "shadow_tint_b", "highlight_tint_a",
                     "highlight_tint_b", "sat", "halation", "chroma_ab_px", "letterbox_top", "letterbox_bottom",
                     "flicker")
# ... and from look.json text_finish
LOOK_TEXT_KEYS = {"text_stroke_px": ("stroke_px", "median"), "text_shadow_share": ("shadow_share",),
                  "text_glow_share": ("glow_share",)}
LOOK_METRICS = LOOK_SUMMARY_KEYS + tuple(LOOK_TEXT_KEYS)
VOICE_TIMBRE_KEYS = ("spectral_tilt_db_per_oct", "crest_db", "rt60_est_s", "de_ess_db")  # sound.json voice


class BadInput(ValueError):
    """Bad pack, bad id list or a refused layer source: the CLI prints one line and exits 2."""


# ---------------------------------------------------------------- numbers

def _num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and x == x and abs(x) != float("inf")


def stat(values, n_videos=None):
    """{median, p10, p90, n, seen_in} of the numbers in `values` (one per video that shows the trait;
    None and NaN are skipped). With n_videos, also {share} = seen_in / n_videos."""
    vals = [float(v) for v in (values or []) if _num(v)]
    if vals:
        a = np.asarray(vals, float)
        out = {"median": round(float(np.median(a)), 4), "p10": round(float(np.percentile(a, 10)), 4),
               "p90": round(float(np.percentile(a, 90)), 4), "n": len(vals), "seen_in": len(vals)}
    else:
        out = {"median": None, "p10": None, "p90": None, "n": 0, "seen_in": 0}
    if n_videos:
        out["share"] = round(out["seen_in"] / n_videos, 4)
    return out


def trait_status(seen_in, n_videos):
    """"rule-candidate" when the trait is seen in at least 30 % of the training videos, else "option"."""
    if n_videos <= 0 or seen_in <= 0:
        return "option"
    return "rule-candidate" if seen_in / n_videos >= RULE_SHARE - 1e-9 else "option"


# ---------------------------------------------------------------- loading

def _read(p, default):
    try:
        return store.read_json(p, default)
    except (OSError, ValueError):
        return default


def load_analysis(d):
    """One analysis folder -> {id, measure, motion, sound, look, words, manual, thumbs, source}. Missing files
    are None (manual: [])."""
    d = Path(d)
    m = _read(d / "measure.json", None)
    if not isinstance(m, dict):
        raise BadInput(f"{d.name}: measure.json missing or not an object")
    src = m.get("source")
    sj = _read(d / "source.json", None)
    if isinstance(sj, dict) and isinstance(sj.get("source"), str):
        src = sj["source"]
    manual = []
    if (d / "events.manual.json").exists():
        manual = _read(d / "events.manual.json", None)
        if not isinstance(manual, list):
            raise BadInput(f"{d.name}: events.manual.json is not a list")
        for k, e in enumerate(manual):
            why = None
            if not isinstance(e, dict):
                why = "not an object"
            elif not _num(e.get("t")):
                why = "t must be a number"
            elif not (isinstance(e.get("type"), str) and e["type"].strip()):
                why = "type must be a non-empty string"
            elif "attrs" in e and e["attrs"] is not None and not isinstance(e["attrs"], dict):
                why = "attrs must be an object"
            elif "label" in e and e["label"] is not None and not isinstance(e["label"], str):
                why = "label must be a string"
            if why:
                raise BadInput(f"{d.name}: events.manual.json entry {k}: {why}")
    th = _read(d / "thumbs.json", None)
    return {"id": d.name, "measure": m, "motion": _read(d / "motion.json", None),
            "sound": _read(d / "sound.json", None), "look": _read(d / "look.json", None),
            "words": _read(d / "words.json", None),
            "manual": manual,
            "thumbs": th if isinstance(th, list) and all(h is None or isinstance(h, str) for h in th) else None,
            "source": src if isinstance(src, str) and src else None, "dir": str(d)}


def load_all(pack):
    root = Path(pack) / "analysis"
    if not root.is_dir():
        raise BadInput(f"no analysis folder in {Path(pack).name}")
    out = {}
    for d in sorted(p for p in root.iterdir() if p.is_dir() and (p / "measure.json").exists()):
        out[d.name] = load_analysis(d)
    if not out:
        raise BadInput("no analysed videos (no analysis/<id>/measure.json)")
    return out


def select(pack_ids, train=None, exclude=None):
    """(trained candidates, held_out) from the available ids. Unknown ids are bad input."""
    avail = list(pack_ids)
    for label, lst in (("--train", train), ("--exclude", exclude)):
        bad = [i for i in (lst or []) if i not in avail]
        if bad:
            raise BadInput(f"{label}: unknown video id {', '.join(bad)}")
    held = [i for i in avail if i in (exclude or [])]
    cand = [i for i in (train if train else avail) if i not in held]
    return cand, held


# ---------------------------------------------------------------- duplicates

LOW_BITS, HIGH_BITS = 8, 56     # an aHash with fewer / more set bits is flat (blank frame), not evidence
MIN_USABLE_THUMBS = 3


def _bits(h):
    try:
        return bin(int(h, 16)).count("1")
    except (ValueError, TypeError):
        return None


def _hamming_hex(a, b):
    try:
        return bin(int(a, 16) ^ int(b, 16)).count("1")
    except (ValueError, TypeError):
        return 64


def _sha(a):
    v = (a.get("measure") or {}).get("video")
    return v.get("sha256") if isinstance(v, dict) and isinstance(v.get("sha256"), str) else None


def is_dupe(a, b):
    ta, tb = a.get("thumbs"), b.get("thumbs")
    if ta and tb:
        n = min(len(ta), len(tb))
        ds = []
        for x, y in zip(ta[:n], tb[:n]):
            bx, by = _bits(x), _bits(y)
            if bx is None or by is None or not (LOW_BITS <= bx <= HIGH_BITS and LOW_BITS <= by <= HIGH_BITS):
                continue   # missing or low-entropy slot: never a duplicate on its own
            ds.append(_hamming_hex(x, y))
        if ds and len(ds) >= min(MIN_USABLE_THUMBS, n) and sum(ds) / len(ds) <= DUP_HAMMING:
            return True
    sa, sb = _sha(a), _sha(b)
    return bool(sa and sa == sb)


def dedupe(ids, analyses):
    """(kept_ids, dupes [(kept, dropped)]). The first of two near-identical videos is kept."""
    kept, dupes = [], []
    for i in ids:
        a = analyses[i]
        hit = next((k for k in kept if is_dupe(analyses[k], a)), None)
        if hit is None:
            kept.append(i)
        else:
            dupes.append((hit, i))
    return kept, dupes


def thumbs_for(video_path, n=THUMBS_N):
    """n aHash values (16 hex chars: 8x8 gray, pixel above the mean) at n evenly spaced frames."""
    from cut import MissingTool, require_ffmpeg
    require_ffmpeg()
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of",
                        "default=nw=1:nk=1", str(video_path)], capture_output=True, text=True, encoding="utf-8")
    try:
        dur = float(r.stdout.strip())
    except ValueError:
        raise BadInput(f"cannot read the duration of {Path(video_path).name}")
    out = []
    for k in range(n):
        t = dur * (k + 0.5) / n
        p = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t:.3f}", "-i", str(video_path), "-frames:v", "1",
                            "-vf", "scale=8:8:flags=area,format=gray", "-f", "rawvideo", "-"],
                           capture_output=True)
        px = np.frombuffer(p.stdout, np.uint8)
        if px.size < 64:
            out.append(None)   # keep the slot so positions stay aligned
            continue
        bits = (px[:64] > px[:64].mean()).astype(np.uint8)
        out.append(format(int("".join(map(str, bits)), 2), "016x"))
    return out


# ---------------------------------------------------------------- events

def _t(x):
    return float(x) if _num(x) else None


def _words_of(words_doc):
    if isinstance(words_doc, dict):
        lst = words_doc.get("words")
    else:
        lst = words_doc
    out = []
    for w in lst if isinstance(lst, list) else []:
        if not isinstance(w, dict):
            continue
        txt = w.get("w", w.get("word"))
        st = w.get("start")
        if isinstance(txt, str) and _num(st):
            out.append((txt, float(st)))
    return out


def word_language(analysis):
    d = analysis.get("words")
    if isinstance(d, dict) and isinstance(d.get("language"), str):
        return d["language"]
    return None


def word_events(analysis):
    """(events, reason). reason is None when the word triggers ran, else why they did not."""
    import lang
    ws = _words_of(analysis.get("words"))
    if not ws:
        return [], "no words"
    code = word_language(analysis)
    if not code:
        return [], "no language in words.json"
    try:
        prof = lang.load_profile(code)
    except KeyError:
        return [], f"no language profile for {code}"
    found = lang.find_triggers([w for w, _ in ws], prof)
    ev = []
    for f in found:
        kind = "date" if f["kind"] == "month" else f["kind"]
        ev.append({"t": round(ws[f["i"]][1], 4), "kind": "word", "type": f"word:{kind}",
                   "attrs": {"lang": code, "text": f["text"]}})
    return ev, None


def _box_bucket(box):
    """Text position key: the box centre (box = [x, y, w, h] as fractions of the frame) in 10 % buckets."""
    try:
        x, y, w, h = (float(v) for v in box[:4])
        return f"{int((x + w / 2) * 10)},{int((y + h / 2) * 10)}"
    except (TypeError, ValueError, IndexError):
        return "?"


def layer_of(type_):
    """The layer an event type belongs to: word, cut, motion, text, sound or manual."""
    if type_.startswith("word:"):
        return "word"
    if type_ in ("cut", "transition"):
        return "cut"
    if type_.startswith("camera:") or type_ == "entrance":
        return "motion"
    if type_ == "big_text":
        return "text"
    if type_.startswith("sfx:"):
        return "sound"
    return "manual"


def measured_layers(analysis, words_ok=True):
    """Set of layers that were really measured for one video (file present and not in its not_run)."""
    out = {"manual"}
    m = analysis.get("measure") or {}
    nr = m.get("not_run") if isinstance(m.get("not_run"), list) else []
    if isinstance(m.get("cuts"), list) and "cuts" not in nr:
        out.add("cut")
    mo = analysis.get("motion") if isinstance(analysis.get("motion"), dict) else None
    mnr = mo.get("not_run") if mo and isinstance(mo.get("not_run"), list) else []
    if mo is not None and not ("camera" in mnr and "elements" in mnr):
        out.add("motion")
    if (isinstance(m.get("text"), dict) and "text" not in nr) or (mo is not None and "elements" not in mnr):
        out.add("text")
    so = analysis.get("sound") if isinstance(analysis.get("sound"), dict) else None
    if so is not None and "sound" not in (so.get("not_run") if isinstance(so.get("not_run"), list) else []):
        out.add("sound")
    if words_ok and word_events(analysis)[1] is None:
        out.add("word")
    return out


def events(analysis, words=True):
    """Unified, time-sorted event stream [{t, kind, type, src, attrs}] of one analysis.

    kind "word" (type word:number|date|question|name), "visual" (cut, transition, camera:<prop>, entrance,
    big_text), "sound" (sfx:<class>) and "manual" (type as written in events.manual.json). `src` is the
    source id of the thing that emitted the event (an element id, a text box, a cut...): two events of the
    same source never make a law. Every field of motion.json / sound.json is read defensively.
    Big text has one source: motion elements with h_pct (or size_pct_h) >= 8 first, then measure.json
    text.boxes [{t, box, h_pct}] only where no element's big text is on screen (1 s after its t0, or its
    dur_s). Consecutive samples of one box (gaps under 1 s) are one event. words=False skips word triggers."""
    ev = []
    m = analysis.get("measure") or {}
    for k, c in enumerate(m.get("cuts") or []):
        t = _t(c.get("t")) if isinstance(c, dict) else None
        if t is not None:
            ev.append({"t": t, "kind": "visual", "type": "transition" if c.get("kind") == "gradual" else "cut",
                       "src": f"cut:{k}", "attrs": {}})
    mo = analysis.get("motion") if isinstance(analysis.get("motion"), dict) else {}
    for k, c in enumerate(mo.get("camera_moves") or []):
        t = _t(c.get("t0")) if isinstance(c, dict) else None
        if t is not None:
            ev.append({"t": t, "kind": "visual", "type": f"camera:{c.get('prop') or 'move'}", "src": f"cam:{k}",
                       "attrs": {k2: c[k2] for k2 in ("dur_f", "ease_name") if k2 in c}})
    el_big = []
    for k, e in enumerate(mo.get("elements") or []):
        if not isinstance(e, dict):
            continue
        t = _t(e.get("t0"))
        if t is None:
            continue
        src = f"el:{e.get('id') if e.get('id') is not None else k}"
        ev.append({"t": t, "kind": "visual", "type": "entrance", "src": src, "attrs": {"id": e.get("id")}})
        h = _t(e.get("h_pct", e.get("size_pct_h")))
        if h is not None and h >= BIG_TEXT_PCT:
            ev.append({"t": t, "kind": "visual", "type": "big_text", "src": src, "attrs": {"id": e.get("id")}})
            el_big.append((t, t + (_t(e.get("dur_s")) or 1.0)))
    txt = m.get("text") if isinstance(m.get("text"), dict) else {}
    last = {}
    for b in sorted((b for b in txt.get("boxes") or [] if isinstance(b, dict) and _t(b.get("t")) is not None),
                    key=lambda b: b["t"]):
        if (_t(b.get("h_pct")) or 0) < BIG_TEXT_PCT:
            continue
        key = _box_bucket(b.get("box"))
        t = float(b["t"])
        new = key not in last or t - last[key] > 1.0
        last[key] = t
        if new and not any(a <= t <= z for a, z in el_big):
            ev.append({"t": t, "kind": "visual", "type": "big_text", "src": f"box:{key}", "attrs": {}})
    so = analysis.get("sound") if isinstance(analysis.get("sound"), dict) else {}
    for k, sf in enumerate(so.get("sfx") or []):
        t = _t(sf.get("t")) if isinstance(sf, dict) else None
        if t is not None:
            ev.append({"t": t, "kind": "sound", "type": f"sfx:{sf.get('class_guess') or 'unknown'}",
                       "src": f"sfx:{k}",
                       "attrs": {k2: sf[k2] for k2 in ("confidence", "level_vs_voice_db") if k2 in sf}})
    for k, e in enumerate(analysis.get("manual") or []):
        t = _t(e.get("t")) if isinstance(e, dict) else None
        if t is not None and isinstance(e.get("type"), str):
            ev.append({"t": t, "kind": "manual", "type": e["type"], "src": f"man:{k}",
                       "attrs": dict(e.get("attrs") or {}, **({"label": e["label"]} if e.get("label") else {}))})
    if words:
        for k, w in enumerate(word_events(analysis)[0]):
            w["src"] = f"word:{k}"
            ev.append(w)
    for e in ev:
        e["layer"] = layer_of(e["type"])
    return sorted(ev, key=lambda e: (e["t"], e["type"], e["src"]))


def word_plan(analyses, ids):
    """(allowed, notes). notes are not_run entries: "word_laws (why)" when word laws cannot run at all,
    "word_laws:<id> (why)" for a video with words that is left out (no language, no profile).
    Needs words in at least one video and one single language among the videos that can run."""
    langs, notes, any_words = set(), [], False
    for i in ids:
        if not _words_of(analyses[i].get("words")):
            continue
        any_words = True
        _, why = word_events(analyses[i])
        if why:
            notes.append(f"word_laws:{i} ({why})")
        else:
            langs.add(word_language(analyses[i]))
    if not any_words:
        return False, ["word_laws (no transcript in any training video)"]
    if not langs:
        return False, ["word_laws (no video with a usable language profile)"] + notes
    if len(langs) > 1:
        return False, ["word_laws (mixed languages: " + ", ".join(sorted(langs)) + ")"]
    return True, notes


# ---------------------------------------------------------------- metrics

def _per_min(n, dur):
    return round(n / (dur / 60.0), 3) if _num(dur) and dur > 0 else None


def _dur(a):
    v = (a.get("measure") or {}).get("video")
    return v.get("duration_s") if isinstance(v, dict) else None


def _vals(analyses, ids, fn):
    out = []
    for i in ids:
        try:
            v = fn(analyses[i])
        except (AttributeError, KeyError, TypeError, IndexError, ValueError, ZeroDivisionError):
            v = None
        if _num(v):
            out.append(v)
    return out


def _median(xs):
    xs = [x for x in xs if _num(x)]
    return float(np.median(xs)) if xs else None


def _dominant_lch(a):
    import palette
    Ls, Cs = [], []
    for s in (a["measure"].get("shots") or []):
        pal = s.get("palette") or []
        if pal and isinstance(pal[0], dict) and isinstance(pal[0].get("hex"), str):
            try:
                L, C, _ = palette.hex_to_oklch(pal[0]["hex"])
                Ls.append(L)
                Cs.append(C)
            except (ValueError, TypeError):
                pass
    return _median(Ls), _median(Cs)


def _moves(a):
    """Every camera move and every element move of one analysis (dicts), read defensively."""
    mo = a.get("motion") if isinstance(a.get("motion"), dict) else {}
    out = [c for c in mo.get("camera_moves") or [] if isinstance(c, dict)]
    for e in mo.get("elements") or []:
        if isinstance(e, dict):
            out.extend(m for m in e.get("moves") or [] if isinstance(m, dict))
    return out


def motion_eases(analyses, ids):
    """[{name, bezier (median per name), share, seen_in}] over every move's ease_name / bezier."""
    seen, bez = {}, {}
    for i in ids:
        for m in _moves(analyses[i]):
            nm = m.get("ease_name")
            if not isinstance(nm, str) or not nm:
                continue
            seen.setdefault(nm, set()).add(i)
            b = m.get("bezier")
            if isinstance(b, (list, tuple)) and len(b) == 4 and all(_num(x) for x in b):
                bez.setdefault(nm, []).append([float(x) for x in b])
    n = len(ids)
    out = []
    for nm, v in sorted(seen.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        b = [round(float(np.median([x[k] for x in bez[nm]])), 3) for k in range(4)] if nm in bez else None
        out.append({"name": nm, "bezier": b, "share": round(len(v) / n, 4) if n else 0.0, "seen_in": len(v)})
    return out


def look_value(a, key):
    """One video's value of a layers.look metric from its look.json (None when absent)."""
    lk = a.get("look") if isinstance(a.get("look"), dict) else None
    if lk is None:
        return None
    if key in LOOK_TEXT_KEYS:
        v = lk.get("text_finish")
        for part in LOOK_TEXT_KEYS[key]:
            v = v.get(part) if isinstance(v, dict) else None
        return v
    v = (lk.get("summary") or {}).get(key)
    return v.get("median") if isinstance(v, dict) else None


def ref_metrics(layer, analyses, ids):
    """Metrics of one layer measured from the reference videos (source "refs")."""
    n = len(ids)
    A = analyses
    M = lambda a: a["measure"]  # noqa: E731
    mo = lambda a: a["motion"] if isinstance(a.get("motion"), dict) else {}  # noqa: E731
    so = lambda a: a["sound"] if isinstance(a.get("sound"), dict) else {}  # noqa: E731
    sp = lambda vals: stat(vals, n)  # noqa: E731
    if layer == "type":
        return {"text_height_pct": sp(_vals(A, ids, lambda a: M(a)["text"]["size_pct_h"]["median"])),
                "big_text_per_min": sp(_vals(A, ids, lambda a: _per_min(
                    sum(1 for e in events(a, words=False) if e["type"] == "big_text"), _dur(a))))}
    if layer == "motion":
        return {"camera_moves_per_min": sp(_vals(A, ids, lambda a: _per_min(len(mo(a)["camera_moves"]), _dur(a)))),
                "camera_move_dur_f": sp(_vals(A, ids, lambda a: _median(
                    [c.get("dur_f") for c in mo(a)["camera_moves"]]))),
                "dur_f": sp(_vals(A, ids, lambda a: _median([m.get("dur_f") for m in _moves(a)]))),
                "peak_at": sp(_vals(A, ids, lambda a: _median([m.get("peak_at") for m in _moves(a)]))),
                "entrances_per_min": sp(_vals(A, ids, lambda a: _per_min(len(mo(a)["elements"]), _dur(a))))}
    if layer == "colour":
        return {"dominant_L": sp(_vals(A, ids, lambda a: _dominant_lch(a)[0])),
                "dominant_chroma": sp(_vals(A, ids, lambda a: _dominant_lch(a)[1]))}
    if layer == "broll":
        return {"shot_dur_s": sp(_vals(A, ids, lambda a: _median([s["dur_s"] for s in M(a)["shots"]]))),
                "face_share": sp(_vals(A, ids, lambda a: _median([s["face_share"] for s in M(a)["shots"]])))}
    if layer == "structure":
        return {"cuts_per_min": sp(_vals(A, ids, lambda a: _per_min(len(M(a)["cuts"]), _dur(a)))),
                "longest_hold_s": sp(_vals(A, ids, lambda a: M(a)["longest_hold_s"])),
                "words_per_s": sp(_vals(A, ids, lambda a: M(a)["words_per_s"])),
                "duration_s": sp(_vals(A, ids, _dur)),
                "unique_fps": sp(_vals(A, ids, lambda a: M(a)["cadence"]["unique_fps"]))}
    if layer == "look":
        return {k: sp(_vals(A, ids, lambda a, k=k: look_value(a, k))) for k in LOOK_METRICS}
    if layer == "sound":
        return {"lufs": sp(_vals(A, ids, lambda a: M(a)["loudness"]["lufs"])),
                "sfx_per_min": sp(_vals(A, ids, lambda a: _per_min(len(so(a)["sfx"]), _dur(a)))),
                "sfx_level_vs_voice_db": sp(_vals(A, ids, lambda a: _median(
                    [s.get("level_vs_voice_db") for s in so(a)["sfx"]]))),
                "bpm": sp(_vals(A, ids, lambda a: so(a)["music"]["bpm"])),
                "music_ducking_db": sp(_vals(A, ids, lambda a: so(a)["music"]["ducking_db"])),
                "music_sections_per_min": sp(_vals(A, ids, lambda a: _per_min(len(so(a)["music"]["sections"]), _dur(a))
                                                   if so(a)["music"]["present"] else None)),
                **{k: sp(_vals(A, ids, lambda a, k=k: so(a)["voice"][k])) for k in VOICE_TIMBRE_KEYS}}
    return {}


def code_metrics(layer, tokens):
    """Metrics of a code layer: the tokens read from the project, verbatim (no spreads: one project)."""
    sect = tokens.get(layer if layer != "colour" else "colour")
    if not isinstance(sect, dict):
        return {}
    return json.loads(json.dumps(sect))


def build_layers(pack, analyses, ids, not_run):
    layers = code_tokens.read_layers(pack)
    tokens = _read(Path(pack) / "tokens.json", None)
    out = {}
    for name in LAYER_NAMES:
        entry = layers.get(name) or {"source": "refs", "ref": ""}
        src = "refs" if name in REF_ONLY_LAYERS else entry.get("source", "refs")
        if src == "refs":
            out[name] = {"source": "refs", "metrics": ref_metrics(name, analyses, ids)}
            if name == "motion":
                out[name]["eases"] = motion_eases(analyses, ids)
        elif src == "code":
            if not isinstance(tokens, dict):
                raise BadInput(f"layer {name} has source code but tokens.json is missing (run code_tokens.py)")
            if not code_tokens.tokens_match_layer(entry, tokens):
                raise BadInput(f"layer {name}: tokens.json does not come from the project set for this layer "
                               f"({entry.get('path_hint')!r}); re-run code_tokens.py on it")
            out[name] = {"source": "code", "metrics": code_metrics(name, tokens)}
        else:
            out[name] = {"source": src, "metrics": {}}
            not_run.append(f"layer_{name}_metrics ({src} source has no measurements)")
    return out


def build_traits(analyses, ids, word_ok):
    n = len(ids)
    seen = {}
    for i in ids:
        a = analyses[i]
        names = {e["type"] for e in events(a, words=word_ok)}
        for c in (a.get("motion") or {}).get("camera_moves") or [] if isinstance(a.get("motion"), dict) else []:
            if isinstance(c, dict) and c.get("ease_name"):
                names.add(f"ease:{c['ease_name']}")
        fs = [s.get("face_share") for s in (a["measure"].get("shots") or []) if isinstance(s, dict)]
        if any(_num(f) and f > 0 for f in fs):
            names.add("face_on_screen")
        if (a["measure"].get("cadence") or {}).get("on_twos") is True:
            names.add("on_twos")
        for nm in names:
            seen.setdefault(nm, set()).add(i)
    return [{"name": nm, "seen_in": len(v), "share": round(len(v) / n, 4) if n else 0.0,
             "status": trait_status(len(v), n)} for nm, v in sorted(seen.items(), key=lambda kv: (-len(kv[1]), kv[0]))]


# ---------------------------------------------------------------- orchestration

def prepare(pack, train=None, exclude=None):
    """Shared by fingerprint.py and laws.py: {analyses, trained, held_out, dupes, not_run, word_ok}."""
    analyses = load_all(pack)
    cand, held = select(analyses.keys(), train, exclude)
    if not cand:
        raise BadInput("no videos left to train on")
    trained, dupes = dedupe(cand, analyses)
    not_run = []
    if any(analyses[i].get("thumbs") is None for i in cand) and len(cand) > 1:
        not_run.append("dedupe_thumbs (no thumbs.json for some videos: only identical files are caught; "
                       "run fingerprint.py <pack> --thumbs)")
    ok, notes = word_plan(analyses, trained)
    not_run.extend(notes)
    return {"analyses": analyses, "trained": trained, "held_out": held, "dupes": [list(d) for d in dupes],
            "not_run": not_run, "word_ok": ok}


def build(pack, train=None, exclude=None):
    ctx = prepare(pack, train, exclude)
    not_run = list(ctx["not_run"])
    a, ids = ctx["analyses"], ctx["trained"]
    layers = build_layers(pack, a, ids, not_run)
    return {"schema": SCHEMA, "trained_on": ids, "held_out": ctx["held_out"], "dupes": ctx["dupes"],
            "layers": layers, "traits": build_traits(a, ids, ctx["word_ok"]), "not_run": not_run}


def write_thumbs(pack):
    from cut import MissingTool
    analyses = load_all(pack)
    done, skipped = 0, []
    for i, a in analyses.items():
        if not a["source"] or not Path(a["source"]).is_file():
            skipped.append(i)
            continue
        th = thumbs_for(a["source"])
        store.write_json(Path(a["dir"]) / "thumbs.json", th)
        done += 1
    return done, skipped


def parse_ids(s):
    return [x.strip() for x in (s or "").split(",") if x.strip()]


def add_selection_args(ap):
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--train", help="comma list of video ids to train on")
    g.add_argument("--all", action="store_true", help="train on every analysed video (default)")
    ap.add_argument("--exclude", help="comma list of video ids held out of training")


def main(argv=None):
    from cut import MissingTool
    store.print_utf8()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pack")
    add_selection_args(ap)
    ap.add_argument("--thumbs", action="store_true", help="write thumbs.json for every analysis with a source")
    a = ap.parse_args(argv)
    try:
        if not Path(a.pack).is_dir():
            raise BadInput(f"pack folder not found: {Path(a.pack).name}")
        if a.thumbs:
            done, skipped = write_thumbs(a.pack)
            print(f"thumbs.json written for {done} video(s)" +
                  (f"; no source path for {', '.join(skipped)}" if skipped else ""))
            return 0
        fp = build(a.pack, parse_ids(a.train), parse_ids(a.exclude))
        store.write_json(Path(a.pack) / "fingerprint.json", fp)
        print(f"fingerprint.json: {len(fp['trained_on'])} trained, {len(fp['held_out'])} held out, "
              f"{len(fp['dupes'])} duplicate(s), {len(fp['traits'])} traits"
              + (f"; not run: {'; '.join(fp['not_run'])}" if fp["not_run"] else ""))
        return 0
    except MissingTool as e:
        print(str(e), file=sys.stderr)
        return 3
    except (BadInput, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except OSError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(store.exit_on_lock_timeout(main))

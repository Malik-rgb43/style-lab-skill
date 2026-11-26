#!/usr/bin/env python3
"""Think like the creator: why each visual, cut and beat was made, synthesised into a creator mind.

The measurements (measure, motion, sound, look) say WHAT the creator did. This layer records WHY, so a new
video is made with the creator's judgement and not by copying their moves. The centre is the visual ideas:
for every visual beat, what line it sits on, what the viewer should get, how the creator translated that
message into a picture (literal, metaphor, scale shift, ...), and why that beats the literal shot. Story
structure, decisions, retention devices and the toolchain come second.

Claude writes one <pack>/analysis/<id>/creator.json per training video, following
references/creator-mind.md. The controlled vocabularies (parts, methods, message types, principles, elements,
tools, the method x message-type fit table) live in assets/creator-vocab.json:

  {schema "creator/2" (creator/1 files are reported with an upgrade message),
   story: {structure: [{part, t0, t1, words}], hook: {type, first_words_s, visual_device},
           open_loops: [{opened_t, closed_t|null, text}], audience_promise},
   visual_ideas: [{t0, t1, line, visual_only?, message, message_type, emotion, idea, translation, principle,
                   principle_note? (required when principle is other), why_this, alternatives: [2+ str], craft: {framing, motion, text_role, sound_role}, pattern?}],
   decisions: [{t, what, why, problem_solved, principle, principle_note?, alternative_rejected?, evidence}],
   retention: [{t, device, gap_s_since_last?}],
   toolchain: [{element, likely_tool (one tool, or a list = a pipeline, order-insensitive), how, confidence 0..1,
               evidence}],
   rules_of_thumb: [str]}

`line` may be "" only with visual_only: true (a visual before any speech). gap_s_since_last is informative
only: synth computes every gap from the device times and the video's duration.

Usage:
  creator.py check <pack>                       validate every analysis/<id>/creator.json
  creator.py candidates <pack> <id>             candidate device times (cuts, jumps, sfx, big text), JSON
  creator.py merge-rules <pack> [--kind rules|never|pattern|promise|device|all] [--apply map.json]
                                                near-duplicate wordings (token Jaccard >= 0.6) of rules of thumb,
                                                never-do items, patterns, audience promises or hook devices;
                                                --apply rewrites them from {"old wording": "new wording"}
  creator.py synth <pack> [--train V01,V02 | --all] [--exclude V03]   write <pack>/creator-mind.json
  creator.py ideate <pack> --line "<line>" --type <message_type> [--used plan-used.json]   brief, JSON

synth writes {schema "creator-mind/1", trained_on, held_out, dupes, ideation, story_template, hooks,
open_loops, audience_promise, retention, principles, problems, never_do, principle_notes, toolchain,
rules_of_thumb, checklist}. Numbers are {median, p10, p90, n, seen_in}; seen_in always counts videos. Shares are computed per
video and macro-averaged (the pooled value is kept as pooled_share). Counted free text is normalised first:
lowercase, punctuation stripped, whitespace collapsed. Videos with the same source sha256 (measure.json
video.sha256) or near-identical thumbs (thumbs.json, mean Hamming <= 6 of 64 bits) count once.

Python: validate(doc), synth({id: doc}, durations={id: s}), checklist(mind), ideate(mind, line, message_type,
used=None), candidates(measure, sound), similar_rules(docs), score_structure(pred, actual, tol_s=1.0) (the
hold-out scorer prove.py calls), retention_events(retention) -> [{t, type}] for prove.score_placement.

Exit: 0 ok, 1 schema errors (each printed as "<id>/creator.json: <field>: <problem>"), 2 bad input.
No third-party packages are needed.
"""
import argparse
import hashlib
import json
import math
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import store  # noqa: E402

SCHEMA = "creator/2"
OLD_SCHEMAS = ("creator/1",)
MIND_SCHEMA = "creator-mind/1"
VOCAB_PATH = Path(__file__).resolve().parent.parent / "assets" / "creator-vocab.json"
RULE_SHARE = 0.30
RARE_SHARE = 0.10          # a method under this macro share of the creator's ideas is "rarely used"
DUP_HAMMING = 6
STRUCTURE_TOL_S = 1.0
STRUCTURE_START_S = 0.5     # boundaries this close to t=0 are the opening, never scored
DEVICE_SHOT_RATIO = (0.5, 2.0)  # a cut is a device candidate when its shot is < 0.5x or > 2x the median shot
NEVER_DO_MIN_SEEN = 2
MERGE_JACCARD = 0.6
BIG_TEXT_PCT = 8.0
EXAMPLES_PER_METHOD = 6
EXAMPLES_PER_VIDEO = 2
CRAFT_KEYS = ("framing", "motion", "text_role", "sound_role")
NOT_WILD = ("literal", "other")


class BadInput(ValueError):
    """Bad pack, id list or vocab: the CLI prints one line and exits 2."""


def _load_vocab():
    try:
        v = json.loads(VOCAB_PATH.read_text(encoding="utf-8"))
        return v, None
    except (OSError, ValueError) as e:
        return {}, f"{VOCAB_PATH.name} missing or unreadable ({type(e).__name__})"


VOCAB, VOCAB_ERROR = _load_vocab()
PARTS = tuple(VOCAB.get("parts", ()))
HOOK_TYPES = tuple(VOCAB.get("hook_types", ()))
TRANSLATIONS = tuple(VOCAB.get("translations", ()))
MESSAGE_TYPES = tuple(VOCAB.get("message_types", ()))
PROBLEMS = tuple(VOCAB.get("problems", ()))
DEVICES = tuple(VOCAB.get("devices", ()))
EVIDENCE = tuple(VOCAB.get("evidence", ()))
ELEMENTS = tuple(VOCAB.get("elements", ()))
TOOLS = tuple(VOCAB.get("tools", ()))
PRINCIPLES = tuple(VOCAB.get("principles", {}))
FIT = {k: tuple(v) for k, v in VOCAB.get("fit", {}).items() if k in MESSAGE_TYPES}


# ---------------------------------------------------------------- helpers

def _num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _str(x):
    return isinstance(x, str) and bool(x.strip())


def norm_text(s):
    """Lowercase, punctuation stripped (letters, digits and spaces kept, any script), whitespace collapsed."""
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFKC", s).lower()
    s = "".join(ch if (ch.isalnum() or ch.isspace()) else ("" if ch in "'’`׳" else " ") for ch in s)
    return re.sub(r"\s+", " ", s).strip()


def _pct(sorted_vals, q):
    """Percentile with linear interpolation (the same as numpy's default)."""
    pos = (len(sorted_vals) - 1) * q / 100.0
    lo, hi = math.floor(pos), math.ceil(pos)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (pos - lo)


def stat(values, seen_in=None):
    """{median, p10, p90, n, seen_in}. seen_in counts videos: pass it when the values are pooled over videos
    (several values per video); by default every value comes from its own video."""
    vals = sorted(float(v) for v in (values or []) if _num(v))
    if not vals:
        return {"median": None, "p10": None, "p90": None, "n": 0, "seen_in": 0}
    return {"median": round(_pct(vals, 50), 4), "p10": round(_pct(vals, 10), 4),
            "p90": round(_pct(vals, 90), 4), "n": len(vals),
            "seen_in": len(vals) if seen_in is None else seen_in}


def _status(seen_in, n):
    return "rule-candidate" if n and seen_in / n >= RULE_SHARE - 1e-9 else "option"


def _share(k, n):
    return round(k / n, 4) if n else None


def _mean(xs):
    return sum(xs) / len(xs) if xs else None


def _r4(x):
    return round(x, 4) if _num(x) else None


def _lst(doc, key):
    v = doc.get(key) if isinstance(doc, dict) else None
    return [e for e in v if isinstance(e, dict)] if isinstance(v, list) else []


def _story_of(doc):
    s = doc.get("story") if isinstance(doc, dict) else None
    return s if isinstance(s, dict) else {}


def tool_key(likely_tool):
    """The tool itself, or a pipeline list joined with '+'. Order-insensitive: the items are sorted in the
    vocab's tool order, so ["ai_i2v", "ai_image"] and ["ai_image", "ai_i2v"] are both 'ai_image+ai_i2v'."""
    if isinstance(likely_tool, list):
        rank = {t: i for i, t in enumerate(TOOLS)}
        items = sorted({str(t) for t in likely_tool}, key=lambda t: (rank.get(t, len(rank)), t))
        return "+".join(items)
    return str(likely_tool) if likely_tool is not None else ""


# ---------------------------------------------------------------- validation

def _req_str(d, key, path, errs, allow_empty=False):
    v = d.get(key)
    if key not in d:
        errs.append(f"{path}.{key}: missing")
    elif not isinstance(v, str) or (not allow_empty and not v.strip()):
        errs.append(f"{path}.{key}: must be a non-empty string")


def _req_num(d, key, path, errs, nullable=False):
    if key not in d:
        errs.append(f"{path}.{key}: missing")
    elif d[key] is None and nullable:
        return
    elif not _num(d[key]):
        errs.append(f"{path}.{key}: must be a number")


def _req_enum(d, key, allowed, path, errs):
    if key not in d:
        errs.append(f"{path}.{key}: missing")
    elif d[key] not in allowed:
        errs.append(f"{path}.{key}: {d[key]!r} is not one of {'|'.join(allowed)}")


def _list_of_dicts(doc, key, path, errs):
    v = doc.get(key)
    if key not in doc:
        errs.append(f"{path}{key}: missing")
        return []
    if not isinstance(v, list):
        errs.append(f"{path}{key}: must be a list")
        return []
    out = []
    for i, e in enumerate(v):
        if isinstance(e, dict):
            out.append((f"{path}{key}[{i}]", e))
        else:
            errs.append(f"{path}{key}[{i}]: must be an object")
    return out


def _span(d, p, errs):
    _req_num(d, "t0", p, errs)
    _req_num(d, "t1", p, errs)
    if _num(d.get("t0")) and _num(d.get("t1")) and d["t1"] < d["t0"]:
        errs.append(f"{p}.t1: ends before t0")


def _principle(d, p, errs, old=False):
    """principle: an id from the vocab; principle_note: optional free text, required when principle is other."""
    if "principle" not in d and old:
        errs.append(f"{p}.principle: missing (upgrade: add principle)")
    else:
        _req_enum(d, "principle", PRINCIPLES, p, errs)
    note = d.get("principle_note")
    if note is not None and not isinstance(note, str):
        errs.append(f"{p}.principle_note: must be a string")
    elif d.get("principle") == "other" and not _str(note):
        errs.append(f"{p}.principle_note: required when principle is 'other' (say what the principle is)")


def validate(doc):
    """Schema errors of one creator.json as 'field: problem' strings (empty list = valid)."""
    if VOCAB_ERROR:
        raise BadInput(VOCAB_ERROR)
    errs = []
    if not isinstance(doc, dict):
        return ["(root): must be a JSON object"]
    old = doc.get("schema") in OLD_SCHEMAS
    if old:
        errs.append(f"schema: {doc['schema']!r} is the old format. Upgrade: add principle (an id from "
                    f"assets/creator-vocab.json) to every field listed below, then set schema to {SCHEMA!r}")
    elif doc.get("schema") != SCHEMA:
        errs.append(f"schema: must be {SCHEMA!r}")

    story = doc.get("story")
    if not isinstance(story, dict):
        errs.append("story: missing or not an object")
    else:
        for p, s in _list_of_dicts(story, "structure", "story.", errs):
            _req_enum(s, "part", PARTS, p, errs)
            _span(s, p, errs)
            _req_str(s, "words", p, errs, allow_empty=True)
        hook = story.get("hook")
        if not isinstance(hook, dict):
            errs.append("story.hook: missing or not an object")
        else:
            _req_enum(hook, "type", HOOK_TYPES, "story.hook", errs)
            _req_num(hook, "first_words_s", "story.hook", errs, nullable=True)
            _req_str(hook, "visual_device", "story.hook", errs, allow_empty=True)
        for p, lp in _list_of_dicts(story, "open_loops", "story.", errs):
            _req_num(lp, "opened_t", p, errs)
            _req_num(lp, "closed_t", p, errs, nullable=True)
            _req_str(lp, "text", p, errs)
        _req_str(story, "audience_promise", "story", errs, allow_empty=True)

    for p, v in _list_of_dicts(doc, "visual_ideas", "", errs):
        _span(v, p, errs)
        if "line" not in v:
            errs.append(f"{p}.line: missing")
        elif not isinstance(v["line"], str):
            errs.append(f"{p}.line: must be a string")
        elif not v["line"].strip() and v.get("visual_only") is not True:
            errs.append(f"{p}.line: empty (allowed only with visual_only: true, a visual before any speech)")
        if "visual_only" in v and not isinstance(v["visual_only"], bool):
            errs.append(f"{p}.visual_only: must be true or false")
        for k in ("message", "emotion", "idea", "why_this"):
            _req_str(v, k, p, errs)
        _req_enum(v, "translation", TRANSLATIONS, p, errs)
        _req_enum(v, "message_type", MESSAGE_TYPES, p, errs)
        _principle(v, p, errs, old)
        alts = v.get("alternatives")
        if "alternatives" not in v:
            errs.append(f"{p}.alternatives: missing")
        elif not isinstance(alts, list) or len([a for a in alts if _str(a)]) < 2:
            errs.append(f"{p}.alternatives: needs at least 2 non-empty ideas")
        craft = v.get("craft")
        if not isinstance(craft, dict):
            errs.append(f"{p}.craft: missing or not an object")
        else:
            for k in CRAFT_KEYS:
                _req_str(craft, k, f"{p}.craft", errs, allow_empty=True)
        if "pattern" in v and not isinstance(v["pattern"], str):
            errs.append(f"{p}.pattern: must be a string")

    for p, d in _list_of_dicts(doc, "decisions", "", errs):
        _req_num(d, "t", p, errs)
        for k in ("what", "why"):
            _req_str(d, k, p, errs)
        _req_enum(d, "problem_solved", PROBLEMS, p, errs)
        _principle(d, p, errs, old)
        if d.get("alternative_rejected") is not None and not isinstance(d["alternative_rejected"], str):
            errs.append(f"{p}.alternative_rejected: must be a string")
        ev = d.get("evidence")
        evs = ev if isinstance(ev, list) else [ev]
        if "evidence" not in d:
            errs.append(f"{p}.evidence: missing")
        elif not evs or any(e not in EVIDENCE for e in evs):
            errs.append(f"{p}.evidence: {ev!r} is not one (or a list) of {'|'.join(EVIDENCE)}")

    for p, r in _list_of_dicts(doc, "retention", "", errs):
        _req_num(r, "t", p, errs)
        _req_enum(r, "device", DEVICES, p, errs)
        if r.get("gap_s_since_last") is not None and not _num(r["gap_s_since_last"]):
            errs.append(f"{p}.gap_s_since_last: must be a number (informative only)")

    for p, tc in _list_of_dicts(doc, "toolchain", "", errs):
        _req_enum(tc, "element", ELEMENTS, p, errs)
        lt = tc.get("likely_tool")
        if "likely_tool" not in tc:
            errs.append(f"{p}.likely_tool: missing")
        else:
            tools = lt if isinstance(lt, list) else [lt]
            if not tools or any(t not in TOOLS for t in tools):
                errs.append(f"{p}.likely_tool: {lt!r} is not one (or a pipeline list) of {'|'.join(TOOLS)}")
        _req_str(tc, "how", p, errs)
        _req_num(tc, "confidence", p, errs)
        if _num(tc.get("confidence")) and not 0 <= tc["confidence"] <= 1:
            errs.append(f"{p}.confidence: must be within 0..1")
        ev = tc.get("evidence")
        if "evidence" not in tc:
            errs.append(f"{p}.evidence: missing")
        elif not (_str(ev) or (isinstance(ev, list) and ev and all(_str(e) for e in ev))):
            errs.append(f"{p}.evidence: name what you saw (no evidence, no toolchain entry)")

    rules = doc.get("rules_of_thumb")
    if "rules_of_thumb" not in doc:
        errs.append("rules_of_thumb: missing")
    elif not isinstance(rules, list) or not all(isinstance(r, str) for r in rules):
        errs.append("rules_of_thumb: must be a list of strings")
    return errs


# ---------------------------------------------------------------- counting free text

class _Counted:
    """Counts normalised free text: occurrences, the videos it appears in, and the first original wording."""

    def __init__(self):
        self.items = {}

    def add(self, text, vid, extra=None):
        key = norm_text(text)
        if not key:
            return None
        it = self.items.setdefault(key, {"key": key, "text": text.strip(), "count": 0, "videos": []})
        it["count"] += 1
        if vid not in it["videos"]:
            it["videos"].append(vid)
        for k, v in (extra or {}).items():
            it.setdefault(k, Counter())[v] += 1
        return key

    def ranked(self, n_videos):
        out = []
        for it in self.items.values():
            d = {k: (dict(v.most_common()) if isinstance(v, Counter) else v) for k, v in it.items()}
            d["seen_in"] = len(it["videos"])
            d["status"] = _status(d["seen_in"], n_videos)
            out.append(d)
        return sorted(out, key=lambda d: (-d["seen_in"], -d["count"], d["key"]))


# ---------------------------------------------------------------- synthesis

def _as_dict(analyses):
    if isinstance(analyses, dict):
        return dict(analyses)
    return {f"V{i + 1:02d}": a for i, a in enumerate(analyses or [])}


def _segs(doc):
    return sorted((s for s in _lst(_story_of(doc), "structure")
                   if _num(s.get("t0")) and _num(s.get("t1")) and s.get("part") in PARTS), key=lambda s: s["t0"])


def video_duration(doc, measured=None):
    """The measured duration (measure.json video.duration_s) when known, else the structure's end, else the
    latest time named anywhere in the file."""
    if _num(measured) and measured > 0:
        return float(measured)
    segs = _segs(doc)
    if segs:
        return max(s["t1"] for s in segs)
    ends = [v.get("t1") for v in _lst(doc, "visual_ideas")] + [r.get("t") for r in _lst(doc, "retention")] \
        + [d.get("t") for d in _lst(doc, "decisions")]
    ends = [e for e in ends if _num(e)]
    return max(ends) if ends else None


def _story(docs):
    """Consensus order: parts seen in >= 30 % of the videos, ordered by their median normalised position."""
    n = len(docs)
    exact = Counter()
    durs, pos, seen = {}, {}, Counter()
    for doc in docs.values():
        segs = _segs(doc)
        if not segs:
            continue
        order = []
        for s in segs:
            if not order or order[-1] != s["part"]:
                order.append(s["part"])
        exact[tuple(order)] += 1
        s0, s1 = segs[0]["t0"], max(s["t1"] for s in segs)
        span = max(s1 - s0, 1e-9)
        per, first = {}, {}
        for s in segs:
            per[s["part"]] = per.get(s["part"], 0.0) + (s["t1"] - s["t0"])
            first.setdefault(s["part"], ((s["t0"] + s["t1"]) / 2 - s0) / span)
        for p in per:
            durs.setdefault(p, []).append(per[p])
            pos.setdefault(p, []).append(first[p])
            seen[p] += 1
    parts = {}
    for p in PARTS:
        if p not in seen:
            continue
        parts[p] = {"duration_s": stat(durs[p]), "position": stat(pos[p]), "seen_in": seen[p],
                    "share": _share(seen[p], n), "in_template": seen[p] / n >= RULE_SHARE - 1e-9}
    order = sorted((p for p in parts if parts[p]["in_template"]),
                   key=lambda p: (parts[p]["position"]["median"], PARTS.index(p)))
    ranked = sorted(exact.items(), key=lambda kv: (-kv[1], len(kv[0]), kv[0]))
    most = {"order": list(ranked[0][0]), "count": ranked[0][1], "share": _share(ranked[0][1], n)} if ranked \
        else {"order": [], "count": 0, "share": _share(0, n)}
    return {"order": order, "parts": parts, "most_common_exact": most,
            "orders": [{"order": list(o), "count": c} for o, c in ranked]}


def _hooks(docs):
    n = len(docs)
    types, first, devs = Counter(), [], _Counted()
    for vid, doc in docs.items():
        h = _story_of(doc).get("hook")
        if isinstance(h, dict) and h.get("type") in HOOK_TYPES:
            types[h["type"]] += 1
            first.append(h.get("first_words_s"))
            if _str(h.get("visual_device")):
                devs.add(h["visual_device"], vid)
    return {"types": dict(types.most_common()), "first_words_s": stat(first),
            "visual_devices": [{"device": d["text"], "key": d["key"], "count": d["count"], "seen_in": d["seen_in"],
                                "status": d["status"]} for d in devs.ranked(n)]}


def _open_loops(docs):
    counts, spans, closed, total, with_closed = [], [], 0, 0, 0
    for doc in docs.values():
        loops = [lp for lp in _lst(_story_of(doc), "open_loops") if _num(lp.get("opened_t"))]
        counts.append(len(loops))
        got = False
        for lp in loops:
            total += 1
            if _num(lp.get("closed_t")) and lp["closed_t"] >= lp["opened_t"]:
                closed += 1
                spans.append(lp["closed_t"] - lp["opened_t"])
                got = True
        with_closed += got
    return {"per_video": stat(counts), "open_to_close_s": stat(spans, seen_in=with_closed),
            "closed_share": _share(closed, total)}


def device_gaps(times, duration):
    """Gaps from the start to the first device, between devices and from the last device to the end. Zero
    devices = one gap of the whole duration. Without a duration the end gap is left out."""
    ts = sorted(float(t) for t in times if _num(t))
    if not ts:
        return [float(duration)] if _num(duration) and duration > 0 else []
    gaps = [ts[0]] + [b - a for a, b in zip(ts, ts[1:])]
    if _num(duration) and duration >= ts[-1]:
        gaps.append(duration - ts[-1])
    return gaps


def _retention(docs, durations):
    rates, maxes, all_gaps, devices, vids_with_gaps = [], [], [], Counter(), 0
    for vid, doc in docs.items():
        rets = [r for r in _lst(doc, "retention") if _num(r.get("t"))]
        for r in rets:
            if r.get("device") in DEVICES:
                devices[r["device"]] += 1
        dur = video_duration(doc, (durations or {}).get(vid))
        if dur and dur > 0:
            rates.append(len(rets) / dur * 10.0)
        gaps = device_gaps([r["t"] for r in rets], dur)
        if gaps:
            maxes.append(max(gaps))
            all_gaps += gaps
            vids_with_gaps += 1
    mg = stat(maxes)
    return {"devices_per_10s": stat(rates), "gap_s": stat(all_gaps, seen_in=vids_with_gaps), "max_gap_s": mg,
            "never_more_than_s": mg["p90"], "devices": dict(devices.most_common()),
            "gaps_from": "device times + video duration (gap_s_since_last is ignored)"}


def _decisions(docs):
    n = len(docs)
    count, vids, probs, notes = Counter(), {}, {}, {}
    problems, never = Counter(), _Counted()
    for vid, doc in docs.items():
        for d in _lst(doc, "decisions"):
            if d.get("problem_solved") in PROBLEMS:
                problems[d["problem_solved"]] += 1
            p = d.get("principle")
            if p in PRINCIPLES:
                count[p] += 1
                vids.setdefault(p, set()).add(vid)
                probs.setdefault(p, Counter())[d.get("problem_solved", "other")] += 1
                if _str(d.get("principle_note")) and len(notes.setdefault(p, [])) < 3:
                    notes[p].append(d["principle_note"].strip())
            if _str(d.get("alternative_rejected")):
                never.add(d["alternative_rejected"], vid, {"problems": d.get("problem_solved", "other")})
    principles = [{"principle": p, "means": VOCAB.get("principles", {}).get(p, ""), "count": c,
                   "seen_in": len(vids[p]), "problems": dict(probs[p].most_common()), "notes": notes.get(p, []),
                   "status": _status(len(vids[p]), n)}
                  for p, c in sorted(count.items(), key=lambda kv: (-len(vids[kv[0]]), -kv[1], kv[0]))]
    never_do = [{"avoid": x["text"], "key": x["key"], "count": x["count"], "seen_in": x["seen_in"],
                 "videos": x["videos"], "problems": x.get("problems", {}), "status": x["status"]}
                for x in never.ranked(n)]
    return principles, dict(problems.most_common()), never_do


def _principle_notes(docs):
    """Notes written where principle is "other": candidates Claude may promote into the vocab."""
    pc = _Counted()
    for vid, doc in docs.items():
        for key in ("decisions", "visual_ideas"):
            for d in _lst(doc, key):
                if d.get("principle") == "other" and _str(d.get("principle_note")):
                    pc.add(d["principle_note"], vid, {"from": key})
    return [{"note": x["text"], "key": x["key"], "count": x["count"], "seen_in": x["seen_in"],
             "videos": x["videos"], "from": x.get("from", {}), "status": x["status"]} for x in pc.ranked(len(docs))]


def _toolchain(docs):
    """Per element: votes weighted by the stated confidence; consensus = the heaviest tool (pipeline),
    confidence = its weight / the element's total weight."""
    n = len(docs)
    by_el = {}
    for vid, doc in docs.items():
        seen = {}
        for tc in _lst(doc, "toolchain"):
            el, tool = tc.get("element"), tool_key(tc.get("likely_tool"))
            if el not in ELEMENTS or not tool:
                continue
            c = float(tc["confidence"]) if _num(tc.get("confidence")) else 0.0
            if el not in seen or c > seen[el][1]:
                seen[el] = (tool, c, tc.get("how") if isinstance(tc.get("how"), str) else "")
        for el, (tool, c, how) in seen.items():
            e = by_el.setdefault(el, {"videos": [], "votes": {}})
            e["videos"].append(vid)
            v = e["votes"].setdefault(tool, {"tool": tool, "videos": 0, "weight": 0.0, "how": []})
            v["videos"] += 1
            v["weight"] += c
            if how and how not in v["how"] and len(v["how"]) < 3:
                v["how"].append(how)
    out = {}
    for el, e in sorted(by_el.items()):
        k = len(e["videos"])
        total = sum(v["weight"] for v in e["votes"].values())
        votes = sorted(e["votes"].values(), key=lambda v: (-v["weight"], -v["videos"], v["tool"]))
        for v in votes:
            v["weight"] = round(v["weight"], 4)
        top = votes[0]
        conf = top["weight"] / total if total > 0 else top["videos"] / k
        out[el] = {"consensus": top["tool"], "confidence": round(conf, 4), "seen_in": k, "share": _share(k, n),
                   "status": _status(k, n), "how": top["how"], "votes": votes}
    return out


def _rules(docs):
    rc = _Counted()
    for vid, doc in docs.items():
        rules = doc.get("rules_of_thumb") if isinstance(doc.get("rules_of_thumb"), list) else []
        for r in rules:
            if isinstance(r, str):
                rc.add(r, vid)
    return rc.ranked(len(docs))


def _promises(docs):
    pc = _Counted()
    for vid, doc in docs.items():
        if _str(_story_of(doc).get("audience_promise")):
            pc.add(_story_of(doc)["audience_promise"], vid)
    return [{"promise": p["text"], "key": p["key"], "count": p["count"], "seen_in": p["seen_in"],
             "status": p["status"]} for p in pc.ranked(len(docs))]


def _macro(per_video_counts, keys):
    """Per-video shares macro-averaged over the videos that have any item, with seen_in and pooled_share.
    per_video_counts: [Counter]."""
    vids = [c for c in per_video_counts if sum(c.values())]
    pooled_n = sum(sum(c.values()) for c in vids)
    out = {}
    for k in keys:
        shares = [c.get(k, 0) / sum(c.values()) for c in vids]
        cnt = sum(c.get(k, 0) for c in vids)
        if not cnt:
            continue
        out[k] = {"share": _r4(_mean(shares)), "pooled_share": _share(cnt, pooled_n), "count": cnt,
                  "seen_in": sum(1 for c in vids if c.get(k, 0))}
    return dict(sorted(out.items(), key=lambda kv: (-kv[1]["share"], -kv[1]["count"], kv[0])))


def _top_text(counter):
    return counter.most_common(1)[0][0] if counter else ""


def _ideation(docs):
    n = len(docs)
    per_vid_methods, per_vid_mt = [], {}
    examples, craft, princ_m, princ_mtm, emo_mt = {}, {}, {}, {}, {}
    patterns = _Counted()
    total = 0
    for vid, doc in docs.items():
        vm, vmt = Counter(), {}
        for v in _lst(doc, "visual_ideas"):
            m, mt = v.get("translation"), v.get("message_type")
            if m not in TRANSLATIONS or mt not in MESSAGE_TYPES:
                continue
            total += 1
            vm[m] += 1
            vmt.setdefault(mt, Counter())[m] += 1
            if _str(v.get("emotion")):
                emo_mt.setdefault(mt, Counter())[norm_text(v["emotion"])] += 1
            p = v.get("principle")
            if p in PRINCIPLES:
                princ_m.setdefault(m, Counter())[p] += 1
                princ_mtm.setdefault((mt, m), Counter())[p] += 1
            ex = examples.setdefault((mt, m), [])
            if len(ex) < EXAMPLES_PER_METHOD and sum(e["video"] == vid for e in ex) < EXAMPLES_PER_VIDEO:
                t0 = v.get("t0")
                ex.append({"example_id": f"{vid}@{t0:g}" if _num(t0) else f"{vid}@{len(ex)}", "video": vid,
                           "line": v.get("line", ""), "message": v.get("message", ""), "idea": v.get("idea", ""),
                           "why_this": v.get("why_this", ""), "principle": p})
            c = v.get("craft") if isinstance(v.get("craft"), dict) else {}
            cm = craft.setdefault(m, {k: Counter() for k in CRAFT_KEYS})
            for k in CRAFT_KEYS:
                if _str(c.get(k)):
                    cm[k][c[k].strip()] += 1
            if _str(v.get("pattern")):
                patterns.add(v["pattern"], vid, {"methods": m, "message_types": mt})
        per_vid_methods.append(vm)
        for mt, cnt in vmt.items():
            per_vid_mt.setdefault(mt, []).append(cnt)
    methods = _macro(per_vid_methods, TRANSLATIONS)
    for m, d in methods.items():
        d["status"] = _status(d["seen_in"], n)
    playbook = {}
    for mt in MESSAGE_TYPES:
        if mt not in per_vid_mt:
            continue
        dist = _macro(per_vid_mt[mt], TRANSLATIONS)
        ms = [{"method": m, **d, "principles": dict(princ_mtm.get((mt, m), Counter()).most_common()),
               "examples": examples.get((mt, m), [])} for m, d in dist.items()]
        emo = emo_mt.get(mt, Counter())
        k = sum(emo.values())
        playbook[mt] = {"n": sum(sum(c.values()) for c in per_vid_mt[mt]), "seen_in": len(per_vid_mt[mt]),
                        "methods": ms, "top": ms[0]["method"],
                        "summary": f"{mt} → " + ", ".join(f"{e['method']} {round(e['share'] * 100)} %" for e in ms),
                        "emotions": {e: _share(c, k) for e, c in emo.most_common()}}
    lit = [c.get("literal", 0) / sum(c.values()) for c in per_vid_methods if sum(c.values())]
    pooled_lit = sum(c.get("literal", 0) for c in per_vid_methods)
    pats = [{"pattern": p["text"], "key": p["key"], "count": p["count"], "seen_in": p["seen_in"],
             "videos": p["videos"], "methods": p.get("methods", {}), "message_types": p.get("message_types", {}),
             "status": p["status"]} for p in patterns.ranked(n)]
    return {"n_ideas": total, "methods": methods, "playbook": playbook, "patterns": pats,
            "literal_share": _r4(_mean(lit)), "figurative_share": _r4(1 - _mean(lit)) if lit else None,
            "pooled_literal_share": _share(pooled_lit, total),
            "principles_by_method": {m: dict(c.most_common()) for m, c in princ_m.items()},
            "craft": {m: {k: _top_text(cm[k]) for k in CRAFT_KEYS} for m, cm in craft.items()}}


def synth(analyses, held_out=None, dupes=None, durations=None):
    """Aggregate {video id: creator.json doc} into the creator mind. durations: {id: measured seconds}."""
    docs = {k: v for k, v in _as_dict(analyses).items() if isinstance(v, dict)}
    principles, problems, never_do = _decisions(docs)
    mind = {"schema": MIND_SCHEMA, "trained_on": list(docs), "held_out": list(held_out or []),
            "dupes": [list(d) for d in (dupes or [])],
            "ideation": _ideation(docs), "story_template": _story(docs), "hooks": _hooks(docs),
            "open_loops": _open_loops(docs), "audience_promise": _promises(docs),
            "retention": _retention(docs, durations), "principles": principles, "problems": problems,
            "never_do": never_do, "principle_notes": _principle_notes(docs), "toolchain": _toolchain(docs),
            "rules_of_thumb": _rules(docs)}
    mind["checklist"] = checklist(mind)
    return mind


# ---------------------------------------------------------------- planner questions

def _fmt(x):
    return str(round(float(x), 2))


def checklist(mind):
    """The "what would this creator do here?" questions Plan 5's beat planner asks for every beat. Generic
    wording; only numbers and the fixed vocabularies (methods, parts, principles) come from the mind."""
    mind = mind if isinstance(mind, dict) else {}
    ide = mind.get("ideation") or {}
    ret = mind.get("retention") or {}
    st = mind.get("story_template") or {}
    loops = mind.get("open_loops") or {}
    items = ["What should the viewer understand or feel on this line (the message, not the noun), and which "
             "message type is it (number before contrast before claim before fact; emotion only when the line "
             "has no factual content)?"]
    play = ide.get("playbook") or {}
    if play:
        tops = "; ".join(p["summary"] for p in play.values())
        items.append(f"Which translation method would the creator reach for on this message type? Their playbook: "
                     f"{tops}. Run ideate for 3 ideas (2 in their methods, 1 wild card) before choosing.")
    else:
        items.append("Which translation method turns this message into a picture (literal, metaphor, scale shift, "
                     "contrast, ...)? Write 3 ideas before choosing one.")
    ls = ide.get("literal_share")
    if _num(ls):
        items.append(f"Does the literal-vs-figurative balance hold? The creator shows {round(ls * 100)} % of "
                     "visual beats literally; is this beat earning a literal shot or does it need a figure?")
    items.append("Why does this idea serve the message better than the literal shot, and what two other ideas "
                 "did you reject?")
    items.append("Do framing, motion, text and sound all point at the same idea, or is one of them decoration?")
    gap = ret.get("never_more_than_s")
    if _num(gap):
        items.append(f"Is there a retention device within the max gap of {_fmt(gap)} s since the last one "
                     "(the creator rarely lets more pass)?")
    else:
        items.append("Is there a retention device close enough to the last one that attention cannot dip?")
    per, span = (loops.get("per_video") or {}).get("median"), (loops.get("open_to_close_s") or {}).get("median")
    if _num(per) and per > 0:
        tail = f" and closes them a median {_fmt(span)} s later" if _num(span) else ""
        items.append(f"Does this beat open or close a loop? The creator opens a median {_fmt(per)} loop(s) per "
                     f"video{tail}; is a loop open now, and is this the beat that pays it off?")
    else:
        items.append("Does this beat open or close a loop? The creator rarely uses open loops: open one only if "
                     "this content needs it.")
    order = st.get("order") or []
    if order:
        items.append(f"Which story part is this beat in ({' → '.join(order)} is the creator's usual order), and "
                     "does it do that part's job?")
    else:
        items.append("Which story part is this beat in (hook, setup, turn, payoff, cta, ...) and does it do that "
                     "part's job?")
    pr = [p["principle"] for p in (mind.get("principles") or [])[:3]]
    items.append("Which problem does this beat solve (attention dip, clarity, proof, emotion, rhythm), and which "
                 "principle justifies it" + (f" (the creator's top: {', '.join(pr)})?" if pr else "?"))
    if mind.get("never_do"):
        items.append("Does this beat do something the creator avoids (see never_do in the mind)? If so, change it "
                     "or say why this content needs it.")
    items.append("Is this the creator's habit (the style) or only what this content needs (a one-off)? Keep the "
                 "habit, and solve a one-off on its own terms.")
    items.append("Is this new? No idea, object or move already used in this video or for this client.")
    return items


# ---------------------------------------------------------------- ideation brief

def _tokens(s):
    return {w for w in norm_text(s).split() if len(w) >= 4}


def _wild_card(ide, mt, exclude, used_methods, line):
    """A method that fits the message type (vocab fit table), that the creator rarely uses (macro share under
    RARE_SHARE), and that is not a top method. Prefer one not used yet in this plan; ties rotate by the line."""
    dist = ide.get("methods") or {}
    fit = [m for m in FIT.get(mt, ()) if m not in exclude and m not in NOT_WILD]
    if not fit:
        return None, "no plausible method left for this message type"
    rare = [m for m in fit if (dist.get(m) or {}).get("share", 0) < RARE_SHARE]
    note = None
    if not rare:
        rare = fit
        note = "the creator uses every fitting method often; the wild card is their least used one"
    key = lambda m: (used_methods.get(m, 0), (dist.get(m) or {}).get("count", 0))  # noqa: E731
    low = min(key(m) for m in rare)
    tied = [m for m in rare if key(m) == low]
    h = int(hashlib.sha256((line or "").encode("utf-8")).hexdigest(), 16)
    return tied[h % len(tied)], note


def _used(used, message_type):
    """(method counts of this message type's beats, example ids used anywhere in the plan). Entries without a
    message_type still block their example, but do not count towards any type's method shares."""
    methods, ids = Counter(), set()
    for u in used or []:
        if isinstance(u, dict):
            if u.get("method") in TRANSLATIONS and u.get("message_type") == message_type:
                methods[u["method"]] += 1
            if isinstance(u.get("example_id"), str):
                ids.add(u["example_id"])
    return methods, ids


def _never_do(mind):
    """never_do items seen in at least 2 videos, or in at least 30 % of them."""
    n = len(mind.get("trained_on") or [])
    return [x for x in (mind.get("never_do") or []) if isinstance(x, dict)
            and (x.get("seen_in", 0) >= NEVER_DO_MIN_SEEN or (n and x.get("seen_in", 0) / n >= RULE_SHARE - 1e-9))]


def ideate(mind, line, message_type, used=None):
    """The brief Claude uses to invent visuals for `line` the creator's way.

    used: [{message_type, method, example_id}] already chosen in this plan. Methods rotate by deficit against
    the creator's share: with k beats of this message type so far, this beat is beat k + 1, a method's target is
    share x (k + 1), and the method furthest below its target comes first. A creator who uses one method 83 %
    of the time still gets it on about 5 beats of 6. Used examples are never offered again."""
    if message_type not in MESSAGE_TYPES:
        raise ValueError(f"message type {message_type!r} is not one of {'|'.join(MESSAGE_TYPES)}")
    mind = mind or {}
    ide = mind.get("ideation") or {}
    used_methods, used_ids = _used(used, message_type)
    play = (ide.get("playbook") or {}).get(message_type)
    fallback = None
    if play and play.get("methods"):
        methods = play["methods"]
    else:
        fallback = "overall"
        allex = {}
        for p in (ide.get("playbook") or {}).values():
            for m in p.get("methods", []):
                allex.setdefault(m["method"], []).extend(m.get("examples", []))
        methods = [{"method": m, **d, "principles": (ide.get("principles_by_method") or {}).get(m, {}),
                    "examples": allex.get(m, [])} for m, d in (ide.get("methods") or {}).items()]
    notes = []
    beat = sum(used_methods.values()) + 1
    deficit = {m["method"]: (m.get("share") or 0.0) * beat - used_methods.get(m["method"], 0) for m in methods}
    ranked = sorted(range(len(methods)), key=lambda i: (-deficit[methods[i]["method"]], i))
    top = [methods[i] for i in ranked][:2]
    skipped = [m["method"] for m in methods[:2] if m not in top]
    if skipped:
        notes.append(f"ahead of the creator's share in this plan, moved back: {', '.join(skipped)}")
    if methods and all(used_methods.get(m["method"], 0) for m in methods):
        notes.append(f"ran out: every method the creator uses for a {message_type} line is already used in this "
                     "plan; methods now repeat in the creator's proportions, so make each idea clearly new")
    top_out = []
    for m in top:
        exs = [e for e in m.get("examples", []) if e.get("example_id") not in used_ids]
        if m.get("examples") and not exs:
            notes.append(f"ran out of reference examples for {m['method']}: invent from the method and the "
                         "why-patterns alone")
        top_out.append({"method": m["method"], "share": m.get("share"), "pooled_share": m.get("pooled_share"),
                        "seen_in": m.get("seen_in"), "count": m.get("count"),
                        "examples": [{"example_id": e.get("example_id"), "line": e.get("line", ""),
                                      "idea": e.get("idea", ""), "why_this": e.get("why_this", "")}
                                     for e in exs[:3]]})
    top_names = [m["method"] for m in top]
    wild, wild_note = _wild_card(ide, message_type, top_names, used_methods, line)
    if wild_note:
        notes.append(wild_note)

    why, seen = [], set()
    for m in top:
        for e in m.get("examples", []):
            w = e.get("why_this", "")
            if _str(w) and norm_text(w) not in seen:
                seen.add(norm_text(w))
                why.append(w)
    pc = Counter()
    for m in top:
        pc.update(m.get("principles") or {})
    vocab_p = VOCAB.get("principles", {})
    principles = [{"principle": p, "means": vocab_p.get(p, ""), "count": c} for p, c in pc.most_common()]
    rel = _tokens(" ".join([message_type, line or ""] + [m.replace("_", " ") for m in top_names]))
    rules = [r for r in (mind.get("rules_of_thumb") or []) if isinstance(r, dict)]
    rules = sorted(rules, key=lambda r: (not (_tokens(r.get("text", "")) & rel), r.get("status") != "rule-candidate",
                                         -r.get("seen_in", 0)))
    rules_out = [{"rule": r.get("text"), "seen_in": r.get("seen_in"), "status": r.get("status"),
                  "relevant": bool(_tokens(r.get("text", "")) & rel)} for r in rules[:6]]
    pats = [p for p in (ide.get("patterns") or [])
            if message_type in (p.get("message_types") or {}) or set(p.get("methods") or {}) & set(top_names)]
    craft_all = ide.get("craft") or {}
    craft_notes = {m: craft_all.get(m, {k: "" for k in CRAFT_KEYS}) for m in top_names}
    wild_share = ((ide.get("methods") or {}).get(wild) or {}).get("share", 0.0) if wild else None
    if top_names:
        uses = " and ".join(top_names) if len(top_names) > 1 else f"{top_names[0]} (twice, two different ideas)"
        rule = (f"Generate 3 ideas: 2 using the creator's top methods for a {message_type} line ({uses}), "
                f"1 wild card using {wild}, a method the creator rarely uses that fits this message type. Take "
                "the method, never the reference idea itself; avoid the never_do items; each idea must answer "
                "the five design questions of craft-rules §1.")
    else:
        rule = (f"Generate 3 ideas: no references to lean on, so try 3 different methods that fit a "
                f"{message_type} line, one of them {wild}. Each idea must answer the five design questions of "
                "craft-rules §1.")
    return {"line": line, "message_type": message_type, "fallback": fallback, "top_methods": top_out,
            "wild_card": {"method": wild, "share_overall": wild_share,
                          "note": "fits this message type and is rarely used by the creator: a fresh angle, "
                                  "still built on their craft"},
            "principles": principles, "why_patterns": why, "rules_of_thumb": rules_out,
            "emotions": (play or {}).get("emotions", {}),
            "patterns": [{"pattern": p["pattern"], "seen_in": p["seen_in"], "methods": p.get("methods", {})}
                         for p in pats[:5]],
            "never_do": [{"avoid": x["avoid"], "seen_in": x["seen_in"]} for x in _never_do(mind)[:5]],
            "craft_notes": craft_notes, "rule": rule,
            "rotation": {"beat_of_type": beat, "used_methods": dict(used_methods),
                         "deficit": {m: round(d, 4) for m, d in deficit.items()}, "used_examples": len(used_ids),
                         "notes": notes},
            "record_when_chosen": "add {message_type, method, example_id} of the chosen idea to the plan's used list",
            "for_each_idea": ["idea (concrete: object, action, framing)", "translation method", "principle",
                              "why it serves the message better than a literal shot",
                              "craft: framing, motion, text_role, sound_role"]}


# ---------------------------------------------------------------- retention candidates

def candidates(measure, sound=None):
    """Candidate device times for Claude to classify: cuts and jump cuts (with the shot length before them
    against the median shot), sfx, and big text (h_pct >= 8, consecutive samples of one box merged)."""
    m = measure if isinstance(measure, dict) else {}
    shots = [s.get("dur_s") for s in m.get("shots") or [] if isinstance(s, dict) and _num(s.get("dur_s"))]
    med = stat(shots)["median"]
    out = []
    cuts = sorted((c for c in m.get("cuts") or [] if isinstance(c, dict) and _num(c.get("t"))), key=lambda c: c["t"])
    prev = 0.0
    for c in cuts:
        before = c["t"] - prev
        ratio = before / med if med else None
        lo, hi = DEVICE_SHOT_RATIO
        d = {"t": round(float(c["t"]), 3), "source": "jump" if c.get("kind") == "jump" else "cut",
             "kind": c.get("kind"), "shot_before_s": round(before, 3),
             "vs_median_shot": round(ratio, 2) if ratio is not None else None,
             "passes_shot_ratio": bool(ratio is not None and (ratio < lo or ratio > hi))}
        out.append(d)
        if c.get("kind") != "jump":
            prev = c["t"]
    so = sound if isinstance(sound, dict) else {}
    for e in so.get("sfx") or []:
        if isinstance(e, dict) and _num(e.get("t")):
            out.append({"t": round(float(e["t"]), 3), "source": "sfx", "class_guess": e.get("class_guess")})
    txt = m.get("text") if isinstance(m.get("text"), dict) else {}
    last = {}
    for b in sorted((b for b in txt.get("boxes") or [] if isinstance(b, dict) and _num(b.get("t"))),
                    key=lambda b: b["t"]):
        if not (_num(b.get("h_pct")) and b["h_pct"] >= BIG_TEXT_PCT):
            continue
        key = json.dumps(b.get("box"), sort_keys=True)
        new = key not in last or b["t"] - last[key] > 1.0
        last[key] = b["t"]
        if new:
            out.append({"t": round(float(b["t"]), 3), "source": "big_text", "h_pct": b["h_pct"]})
    out.sort(key=lambda d: (d["t"], d["source"]))
    v = m.get("video") if isinstance(m.get("video"), dict) else {}
    return {"duration_s": v.get("duration_s"), "median_shot_s": med, "candidates": out,
            "device_shot_ratio": list(DEVICE_SHOT_RATIO),
            "rule": "a device resets attention beyond the style's normal cutting rhythm. A cut is a device only when "
                    f"its shot is under {DEVICE_SHOT_RATIO[0]}x or over {DEVICE_SHOT_RATIO[1]}x the median shot "
                    "(passes_shot_ratio) or when it brings a new visual idea. Classify each candidate, and add "
                    "devices the measurements cannot see (an open loop, a reveal in the words)."}


# ---------------------------------------------------------------- merging free-text wordings

MERGE_KINDS = ("rules", "never", "pattern", "promise", "device")


def _kind_slots(doc, kind):
    """[(container, key)] holding the free-text values of `kind` in one creator.json."""
    if not isinstance(doc, dict):
        return []
    story = _story_of(doc)
    if kind == "rules":
        rules = doc.get("rules_of_thumb")
        return [(rules, i) for i in range(len(rules))] if isinstance(rules, list) else []
    if kind == "never":
        return [(d, "alternative_rejected") for d in _lst(doc, "decisions")]
    if kind == "pattern":
        return [(v, "pattern") for v in _lst(doc, "visual_ideas")]
    if kind == "promise":
        return [(story, "audience_promise")] if story else []
    if kind == "device":
        h = story.get("hook")
        return [(h, "visual_device")] if isinstance(h, dict) else []
    raise ValueError(f"unknown kind {kind!r}: one of {'|'.join(MERGE_KINDS)}|all")


def _kind_texts(doc, kind):
    out = []
    for c, k in _kind_slots(doc, kind):
        v = c[k] if isinstance(c, list) else c.get(k)
        if isinstance(v, str) and norm_text(v):
            out.append(v)
    return out


def similar_texts(docs, kind="rules", threshold=MERGE_JACCARD):
    """Clusters of near-duplicate wordings of one kind (token Jaccard >= threshold) over {id: doc}. Each cluster
    lists the wordings with their videos and suggests the most used wording."""
    items = {}
    for vid, doc in docs.items():
        for t in _kind_texts(doc, kind):
            k = norm_text(t)
            it = items.setdefault(k, {"text": t.strip(), "key": k, "videos": []})
            if vid not in it["videos"]:
                it["videos"].append(vid)
    keys = sorted(items)
    toks = {k: set(k.split()) for k in keys}
    parent = {k: k for k in keys}

    def find(k):
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            if _jaccard(toks[a], toks[b]) >= threshold - 1e-9:
                parent[find(a)] = find(b)
    groups = {}
    for k in keys:
        groups.setdefault(find(k), []).append(items[k])
    clusters = []
    for g in groups.values():
        if len(g) < 2:
            continue
        g.sort(key=lambda r: (-len(r["videos"]), r["key"]))
        clusters.append({"suggested": g[0]["text"], "rules": g})
    clusters.sort(key=lambda c: c["suggested"])
    return clusters


def similar_rules(docs, threshold=MERGE_JACCARD):
    return similar_texts(docs, "rules", threshold)


def _jaccard(a, b):
    return len(a & b) / len(a | b) if a | b else 0.0


def apply_map(doc, mapping, kind="rules"):
    """Rewrite the wordings of `kind` from {old wording: new wording} (matched after normalising). Rules of
    thumb are de-duplicated inside the video. Returns the number of values changed."""
    nm = {norm_text(k): v.strip() for k, v in mapping.items()}
    changed = 0
    for c, k in _kind_slots(doc, kind):
        v = c[k] if isinstance(c, list) else c.get(k)
        if isinstance(v, str) and norm_text(v) in nm and nm[norm_text(v)] != v:
            c[k] = nm[norm_text(v)]
            changed += 1
    if kind == "rules" and isinstance(doc.get("rules_of_thumb"), list):
        out, seen = [], set()
        for r in doc["rules_of_thumb"]:
            if isinstance(r, str) and norm_text(r) and norm_text(r) not in seen:
                seen.add(norm_text(r))
                out.append(r)
        doc["rules_of_thumb"] = out
    return changed


def apply_rule_map(doc, mapping):
    return apply_map(doc, mapping, "rules")


# ---------------------------------------------------------------- hold-out scoring (called by prove.py)

def _boundaries(structure):
    segs = [s for s in (structure or []) if isinstance(s, dict) and _num(s.get("t0")) and s.get("part")]
    return sorted(((float(s["t0"]), str(s["part"])) for s in segs), key=lambda b: b[0])


def score_structure(pred, actual, tol_s=STRUCTURE_TOL_S):
    """Score predicted story parts against the actual ones. Each part's start is a boundary labelled with the
    part; a predicted boundary matches an actual one of the same part within tol_s (nearest first,
    one-to-one). Boundaries within STRUCTURE_START_S (0.5 s) of t=0 are the opening and are not scored: they
    would be a free match. Returns {precision, recall, f1, overall_f1, tp, n_pred, n_actual, timing_mae_s,
    abs_err_sum, order_match, first_part_match} (keys compatible with prove.pool_scores)."""
    p_all, a_all = _boundaries(pred), _boundaries(actual)
    p = [b for b in p_all if b[0] >= STRUCTURE_START_S]
    a = [b for b in a_all if b[0] >= STRUCTURE_START_S]
    pairs = sorted((abs(pt - at), i, j) for i, (pt, pp) in enumerate(p) for j, (at, ap) in enumerate(a)
                   if pp == ap and abs(pt - at) <= tol_s + 1e-9)
    used_p, used_a, errs = set(), set(), []
    for d, i, j in pairs:
        if i in used_p or j in used_a:
            continue
        used_p.add(i)
        used_a.add(j)
        errs.append(d)
    tp = len(errs)
    prec = tp / len(p) if p else 0.0
    rec = tp / len(a) if a else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0

    def seq(bs):
        out = []
        for _, part in bs:
            if not out or out[-1] != part:
                out.append(part)
        return out
    return {"precision": prec, "recall": rec, "f1": f1, "overall_f1": f1, "tp": tp, "n_pred": len(p),
            "n_actual": len(a), "timing_mae_s": (sum(errs) / tp) if tp else None, "abs_err_sum": sum(errs),
            "order_match": bool(p_all) and seq(p_all) == seq(a_all),
            "first_part_match": bool(p_all and a_all) and p_all[0][1] == a_all[0][1], "tol_s": tol_s}


def retention_events(retention):
    """creator.json retention -> [{t, type}] events, so prove.score_placement scores devices like events."""
    return [{"t": float(r["t"]), "type": str(r["device"])} for r in (retention or [])
            if isinstance(r, dict) and _num(r.get("t")) and r.get("device")]


# ---------------------------------------------------------------- pack io

def _read(p):
    try:
        return store.read_json(p, None)
    except (OSError, ValueError):
        return None


def _pack(pack):
    if not Path(pack).is_dir():
        raise BadInput(f"pack folder not found: {Path(pack).name}")
    root = Path(pack) / "analysis"
    if not root.is_dir():
        raise BadInput(f"no analysis folder in {Path(pack).name}")
    return root


def creator_files(pack):
    root = _pack(pack)
    files = {d.name: d / "creator.json" for d in sorted(root.iterdir()) if (d / "creator.json").is_file()}
    if not files:
        raise BadInput("no creator.json in analysis/<id>/ (write them first, see references/creator-mind.md)")
    return files


def _load_json(f):
    return json.loads(Path(f).read_bytes().decode("utf-8-sig"))


def load_checked(files):
    """({id: doc}, [error lines]) for the given {id: path}."""
    docs, errors = {}, []
    for vid, f in files.items():
        try:
            doc = _load_json(f)
        except (OSError, ValueError) as e:
            errors.append(f"{vid}/creator.json: (file): not valid JSON ({type(e).__name__})")
            continue
        errs = validate(doc)
        errors += [f"{vid}/creator.json: {e}" for e in errs]
        if not errs:
            docs[vid] = doc
    return docs, errors


def _hamming_hex(a, b):
    try:
        return bin(int(a, 16) ^ int(b, 16)).count("1")
    except (TypeError, ValueError):
        return 64


def _measure(pack, vid):
    m = _read(Path(pack) / "analysis" / vid / "measure.json")
    return m if isinstance(m, dict) else None


def _ident(pack, vid):
    m = _measure(pack, vid) or {}
    v = m.get("video") if isinstance(m.get("video"), dict) else {}
    sha = v.get("sha256") if isinstance(v.get("sha256"), str) else None
    th = _read(Path(pack) / "analysis" / vid / "thumbs.json")
    th = th if isinstance(th, list) and th and all(isinstance(h, str) for h in th) else None
    return sha, th


def measured_durations(pack, ids):
    out = {}
    for i in ids:
        v = (_measure(pack, i) or {}).get("video")
        if isinstance(v, dict) and _num(v.get("duration_s")) and v["duration_s"] > 0:
            out[i] = float(v["duration_s"])
    return out


def dedupe(pack, ids):
    """(kept, [(kept, dropped)]): the first of two videos with the same source sha256 or near-identical thumbs."""
    idents = {i: _ident(pack, i) for i in ids}
    kept, dupes = [], []
    for i in ids:
        sa, ta = idents[i]
        hit = None
        for k in kept:
            sb, tb = idents[k]
            if sa and sa == sb:
                hit = k
            elif ta and tb:
                n = min(len(ta), len(tb))
                if sum(_hamming_hex(x, y) for x, y in zip(ta[:n], tb[:n])) / n <= DUP_HAMMING:
                    hit = k
            if hit:
                break
        if hit:
            dupes.append((hit, i))
        else:
            kept.append(i)
    return kept, dupes


def _ids(s):
    return [x.strip() for x in (s or "").split(",") if x.strip()]


# ---------------------------------------------------------------- cli

def _dump(obj):
    print(json.dumps(obj, ensure_ascii=False, indent=2))


def cmd_check(a):
    files = creator_files(a.pack)
    docs, errors = load_checked(files)
    for e in errors:
        print(e, file=sys.stderr)
    if errors:
        print(f"creator.json: {len(files) - len(docs)} of {len(files)} file(s) have schema errors", file=sys.stderr)
        return 1
    print(f"creator.json: {len(files)} file(s) valid")
    return 0


def cmd_candidates(a):
    root = _pack(a.pack)
    d = root / a.id
    m = _read(d / "measure.json")
    if not isinstance(m, dict):
        raise BadInput(f"{a.id}: measure.json missing or not an object (run measure_video.py first)")
    _dump({"id": a.id, **candidates(m, _read(d / "sound.json"))})
    return 0


def _kind_maps(mapping, kinds):
    """{kind: {old: new}} from a flat map (applied to every selected kind) or a map nested by kind."""
    if not isinstance(mapping, dict) or not mapping:
        raise BadInput('--apply: the map must be {"old wording": "new wording"} or {kind: {...}}')
    if all(isinstance(v, dict) for v in mapping.values()):
        unknown = [k for k in mapping if k not in MERGE_KINDS]
        if unknown:
            raise BadInput(f"--apply: unknown kind {', '.join(unknown)} (one of {'|'.join(MERGE_KINDS)})")
        out = {k: v for k, v in mapping.items() if k in kinds}
    else:
        out = {k: mapping for k in kinds}
    for m in out.values():
        if not all(isinstance(k, str) and _str(v) for k, v in m.items()):
            raise BadInput('--apply: every entry must be "old wording": "new wording"')
    return out


def cmd_merge_rules(a):
    files = creator_files(a.pack)
    kinds = list(MERGE_KINDS) if a.kind == "all" else [a.kind]
    if a.apply:
        try:
            mapping = _load_json(a.apply)
        except (OSError, ValueError) as e:
            raise BadInput(f"--apply: cannot read the map ({type(e).__name__})")
        maps = _kind_maps(mapping, kinds)
        changed, bad = Counter(), []
        for vid, f in files.items():
            try:
                doc = _load_json(f)
            except (OSError, ValueError):
                bad.append(vid)
                continue
            if not isinstance(doc, dict):
                bad.append(vid)
                continue
            n = 0
            for kind, m in maps.items():
                c = apply_map(doc, m, kind)
                changed[kind] += c
                n += c
            if n:
                store.write_json(f, doc)
        done = ", ".join(f"{k} {changed[k]}" for k in maps) or "nothing"
        print(f"rewritten: {done}" + (f"; unreadable: {', '.join(bad)}" if bad else ""))
        return 1 if bad else 0
    docs = {}
    for vid, f in files.items():
        try:
            docs[vid] = _load_json(f)
        except (OSError, ValueError):
            print(f"{vid}/creator.json: (file): not valid JSON", file=sys.stderr)
    found = {k: similar_texts(docs, k) for k in kinds}
    nxt = ("write {\"old wording\": \"new wording\"} for each cluster to a map.json (nested by kind with --kind all) "
           "and run creator.py merge-rules <pack> --kind <kind> --apply map.json"
           if any(found.values()) else "nothing to merge")
    if a.kind == "all":
        _dump({"threshold": MERGE_JACCARD, "kinds": found, "next": nxt})
    else:
        _dump({"kind": a.kind, "threshold": MERGE_JACCARD, "clusters": found[a.kind], "next": nxt})
    return 0


def cmd_synth(a):
    files = creator_files(a.pack)
    train, exclude = _ids(a.train), _ids(a.exclude)
    for label, lst in (("--train", train), ("--exclude", exclude)):
        unknown = [i for i in lst if i not in files]
        if unknown:
            raise BadInput(f"{label}: no creator.json for {', '.join(unknown)}")
    held = [i for i in files if i in exclude]
    ids = [i for i in (train or list(files)) if i not in held]
    if not ids:
        raise BadInput("no training videos left after --exclude")
    docs, errors = load_checked({i: files[i] for i in ids})
    if errors:
        for e in errors:
            print(e, file=sys.stderr)
        print("creator-mind.json not written: fix the files above (creator.py check)", file=sys.stderr)
        return 1
    kept, dupes = dedupe(a.pack, ids)
    mind = synth({i: docs[i] for i in kept}, held_out=held, dupes=dupes,
                 durations=measured_durations(a.pack, kept))
    store.write_json(Path(a.pack) / "creator-mind.json", mind)
    ide = mind["ideation"]
    print(f"creator-mind.json: {len(kept)} videos, {ide['n_ideas']} visual ideas, "
          f"{len(ide['playbook'])} message types in the playbook, max gap p90 "
          f"{mind['retention']['never_more_than_s']} s" + (f"; {len(dupes)} duplicate(s) counted once" if dupes else ""))
    return 0


def cmd_ideate(a):
    if not Path(a.pack).is_dir():
        raise BadInput(f"pack folder not found: {Path(a.pack).name}")
    mind = _read(Path(a.pack) / "creator-mind.json")
    if not isinstance(mind, dict) or mind.get("schema") != MIND_SCHEMA:
        raise BadInput("no valid creator-mind.json in the pack (run creator.py synth first)")
    used = None
    if a.used:
        try:
            used = _load_json(a.used)
        except (OSError, ValueError) as e:
            raise BadInput(f"--used: cannot read the list ({type(e).__name__})")
        if not isinstance(used, list):
            raise BadInput("--used: must be a JSON list of {message_type, method, example_id}")
    _dump(ideate(mind, a.line, a.type, used))
    return 0


def main(argv=None):
    store.print_utf8()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check", help="validate every analysis/<id>/creator.json")
    c.add_argument("pack")
    cd = sub.add_parser("candidates", help="candidate device times for one video (JSON)")
    cd.add_argument("pack")
    cd.add_argument("id")
    mr = sub.add_parser("merge-rules", help="list near-duplicate wordings, or --apply a rewrite map")
    mr.add_argument("pack")
    mr.add_argument("--kind", default="rules", choices=list(MERGE_KINDS) + ["all"],
                    help="rules (rules_of_thumb), never (alternative_rejected), pattern, promise "
                         "(audience_promise), device (hook.visual_device) or all; default rules")
    mr.add_argument("--apply", help='JSON map {"old wording": "new wording"} (or {kind: {...}} with --kind all)')
    s = sub.add_parser("synth", help="write <pack>/creator-mind.json")
    s.add_argument("pack")
    g = s.add_mutually_exclusive_group()
    g.add_argument("--train", help="comma list of video ids to train on")
    g.add_argument("--all", action="store_true", help="train on every video with a creator.json (default)")
    s.add_argument("--exclude", help="comma list of video ids held out of training")
    i = sub.add_parser("ideate", help="print the ideation brief for one line (JSON)")
    i.add_argument("pack")
    i.add_argument("--line", required=True, help="the spoken line under the visual")
    i.add_argument("--type", required=True, choices=MESSAGE_TYPES or None, help="message type of the line")
    i.add_argument("--used", help="JSON list of {message_type, method, example_id} already chosen in this plan")
    a = ap.parse_args(argv)
    try:
        if VOCAB_ERROR:
            raise BadInput(VOCAB_ERROR)
        return {"check": cmd_check, "candidates": cmd_candidates, "merge-rules": cmd_merge_rules,
                "synth": cmd_synth, "ideate": cmd_ideate}[a.cmd](a)
    except (BadInput, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except OSError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(store.exit_on_lock_timeout(main))

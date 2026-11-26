#!/usr/bin/env python3
"""Think like the creator: why each visual, cut and beat was made, synthesised into a creator mind.

The measurements (measure, motion, sound, look) say WHAT the creator did. This layer records WHY, so a new
video is made with the creator's judgement and not by copying their moves. The centre is the visual ideas:
for every visual beat, what line it sits on, what the viewer should get, how the creator translated that
message into a picture (literal, metaphor, scale shift, ...), and why that beats the literal shot. Story
structure, decisions, retention devices and the toolchain come second.

Claude writes one <pack>/analysis/<id>/creator.json per training video, following
references/creator-mind.md, after reading the sheets, the transcript, measure, motion, sound and look:

  {schema "creator/1",
   story: {structure: [{part, t0, t1, words}], hook: {type, first_words_s, visual_device},
           open_loops: [{opened_t, closed_t|null, text}], audience_promise},
   visual_ideas: [{t0, t1, line, message, message_type, emotion, idea, translation, why_this,
                   alternatives: [2+ str], craft: {framing, motion, text_role, sound_role}, pattern?}],
   decisions: [{t, what, why, problem_solved, principle, alternative_rejected?, evidence}],
   retention: [{t, device, gap_s_since_last|null}],
   toolchain: [{element, likely_tool, how, confidence 0..1, evidence}],
   rules_of_thumb: [str]}

  part           hook setup context turn escalation payoff recap cta other
  hook.type      paradox question claim number visual story other
  translation    literal metaphor symbol contrast exaggeration visual_pun data_viz pov before_after
                 juxtaposition personification scale_shift reveal other
  message_type   fact number claim emotion contrast process reveal cta
  problem_solved attention_dip clarity proof emotion rhythm other
  device         pattern_interrupt reveal open_loop payoff tempo_change sound_hit text_slam other
  evidence       measure motion sound look transcript sheet (one, or a list)

Usage:
  creator.py check <pack>                                   validate every analysis/<id>/creator.json
  creator.py synth <pack> [--train V01,V02 | --all] [--exclude V03]   write <pack>/creator-mind.json
  creator.py ideate <pack> --line "<spoken line>" --type <message_type>   print the ideation brief (JSON)

synth aggregates the training videos into creator-mind.json {schema "creator-mind/1", trained_on, held_out,
dupes, ideation, story_template, hooks, retention, principles, problems, toolchain, rules_of_thumb,
checklist}. Numbers are {median, p10, p90, n, seen_in}. Free text that is counted (rules of thumb,
principles, idea patterns, tools) is normalised first: lowercase, punctuation stripped, whitespace
collapsed. Two videos with the same source sha256 (measure.json video.sha256) or near-identical thumbs
(thumbs.json, mean Hamming <= 6 of 64 bits) count once.

Python: validate(doc) -> [errors], synth({id: doc}) -> mind, checklist(mind) -> [questions],
ideate(mind, line, message_type) -> brief, score_structure(pred, actual, tol_s=1.0) -> score (the hold-out
scorer prove.py calls), retention_events(retention) -> [{t, type}] for prove.score_placement.

Exit: 0 ok, 1 schema errors (each printed as "<id>/creator.json: <field>: <problem>"), 2 bad input.
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

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import store  # noqa: E402

SCHEMA = "creator/1"
MIND_SCHEMA = "creator-mind/1"
RULE_SHARE = 0.30
DUP_HAMMING = 6
STRUCTURE_TOL_S = 1.0
EXAMPLES_PER_METHOD = 3

PARTS = ("hook", "setup", "context", "turn", "escalation", "payoff", "recap", "cta", "other")
HOOK_TYPES = ("paradox", "question", "claim", "number", "visual", "story", "other")
TRANSLATIONS = ("literal", "metaphor", "symbol", "contrast", "exaggeration", "visual_pun", "data_viz", "pov",
                "before_after", "juxtaposition", "personification", "scale_shift", "reveal", "other")
MESSAGE_TYPES = ("fact", "number", "claim", "emotion", "contrast", "process", "reveal", "cta")
PROBLEMS = ("attention_dip", "clarity", "proof", "emotion", "rhythm", "other")
DEVICES = ("pattern_interrupt", "reveal", "open_loop", "payoff", "tempo_change", "sound_hit", "text_slam", "other")
EVIDENCE = ("measure", "motion", "sound", "look", "transcript", "sheet")
CRAFT_KEYS = ("framing", "motion", "text_role", "sound_role")
NOT_WILD = ("literal", "other")


class BadInput(ValueError):
    """Bad pack or id list: the CLI prints one line and exits 2."""


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


def norm_tool(s):
    """Tool names keep '+' (ai_image+i2v) and '_' but drop spaces and other punctuation."""
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFKC", s).lower()
    s = re.sub(r"[^\w+]+", " ", s)
    s = re.sub(r"\s*\+\s*", "+", s)
    return re.sub(r"\s+", " ", s).strip()


def stat(values):
    vals = [float(v) for v in (values or []) if _num(v)]
    if not vals:
        return {"median": None, "p10": None, "p90": None, "n": 0, "seen_in": 0}
    a = np.asarray(vals, float)
    return {"median": round(float(np.median(a)), 4), "p10": round(float(np.percentile(a, 10)), 4),
            "p90": round(float(np.percentile(a, 90)), 4), "n": len(vals), "seen_in": len(vals)}


def _status(seen_in, n):
    return "rule-candidate" if n and seen_in / n >= RULE_SHARE - 1e-9 else "option"


def _share(k, n):
    return round(k / n, 4) if n else None


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


def validate(doc):
    """Schema errors of one creator.json as 'field: problem' strings (empty list = valid)."""
    errs = []
    if not isinstance(doc, dict):
        return ["(root): must be a JSON object"]
    if doc.get("schema") != SCHEMA:
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
        for k in ("line", "message", "emotion", "idea", "why_this"):
            _req_str(v, k, p, errs)
        _req_enum(v, "translation", TRANSLATIONS, p, errs)
        _req_enum(v, "message_type", MESSAGE_TYPES, p, errs)
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
        for k in ("what", "why", "principle"):
            _req_str(d, k, p, errs)
        _req_enum(d, "problem_solved", PROBLEMS, p, errs)
        if "alternative_rejected" in d and d["alternative_rejected"] is not None \
                and not isinstance(d["alternative_rejected"], str):
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
        _req_num(r, "gap_s_since_last", p, errs, nullable=True)

    for p, tc in _list_of_dicts(doc, "toolchain", "", errs):
        for k in ("element", "likely_tool", "how"):
            _req_str(tc, k, p, errs)
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


# ---------------------------------------------------------------- synthesis

def _as_dict(analyses):
    if isinstance(analyses, dict):
        return dict(analyses)
    return {f"V{i + 1:02d}": a for i, a in enumerate(analyses or [])}


def _lst(doc, key):
    v = doc.get(key) if isinstance(doc, dict) else None
    return [e for e in v if isinstance(e, dict)] if isinstance(v, list) else []


def _duration(doc):
    ends = []
    story = doc.get("story") if isinstance(doc.get("story"), dict) else {}
    ends += [s.get("t1") for s in _lst(story, "structure")]
    ends += [v.get("t1") for v in _lst(doc, "visual_ideas")]
    ends += [r.get("t") for r in _lst(doc, "retention")]
    ends += [d.get("t") for d in _lst(doc, "decisions")]
    if _num(doc.get("duration_s")):
        ends.append(doc["duration_s"])
    ends = [e for e in ends if _num(e)]
    return max(ends) if ends else None


def _order(doc):
    story = doc.get("story") if isinstance(doc.get("story"), dict) else {}
    segs = sorted((s for s in _lst(story, "structure") if _num(s.get("t0")) and s.get("part") in PARTS),
                  key=lambda s: s["t0"])
    out = []
    for s in segs:
        if not out or out[-1] != s["part"]:
            out.append(s["part"])
    return out, segs


def _story(docs):
    n = len(docs)
    orders = Counter()
    part_durs, part_seen = {}, Counter()
    for doc in docs.values():
        order, segs = _order(doc)
        if order:
            orders[tuple(order)] += 1
        per = {}
        for s in segs:
            if _num(s.get("t1")):
                per[s["part"]] = per.get(s["part"], 0.0) + (s["t1"] - s["t0"])
        for p, d in per.items():
            part_durs.setdefault(p, []).append(d)
            part_seen[p] += 1
    ranked = sorted(orders.items(), key=lambda kv: (-kv[1], len(kv[0]), kv[0]))
    top, cnt = (list(ranked[0][0]), ranked[0][1]) if ranked else ([], 0)
    parts = {p: {"duration_s": stat(part_durs[p]), "share": _share(part_seen[p], n),
                 "in_template": p in top} for p in PARTS if p in part_durs}
    return {"order": top, "seen_in": cnt, "share": _share(cnt, n),
            "status": _status(cnt, n) if top else "option",
            "orders": [{"order": list(o), "count": c} for o, c in ranked], "parts": parts}


def _hooks(docs):
    types, first = Counter(), []
    for doc in docs.values():
        h = (doc.get("story") or {}).get("hook") if isinstance(doc.get("story"), dict) else None
        if isinstance(h, dict) and h.get("type") in HOOK_TYPES:
            types[h["type"]] += 1
            first.append(h.get("first_words_s"))
    return {"types": dict(types.most_common()), "first_words_s": stat(first)}


def _gaps(rets):
    rets = sorted((r for r in rets if _num(r.get("t"))), key=lambda r: r["t"])
    gaps, prev = [], 0.0
    for r in rets:
        g = r.get("gap_s_since_last")
        gaps.append(float(g) if _num(g) else r["t"] - prev)
        prev = r["t"]
    return gaps


def _retention(docs):
    rates, maxes, all_gaps, devices = [], [], [], Counter()
    for doc in docs.values():
        rets = [r for r in _lst(doc, "retention") if _num(r.get("t"))]
        for r in rets:
            if r.get("device") in DEVICES:
                devices[r["device"]] += 1
        dur = _duration(doc)
        if dur and dur > 0:
            rates.append(len(rets) / dur * 10.0)
        gaps = _gaps(rets)
        if gaps:
            maxes.append(max(gaps))
            all_gaps += gaps
    mg = stat(maxes)
    return {"devices_per_10s": stat(rates), "gap_s_since_last": stat(all_gaps), "max_gap_s": mg,
            "never_more_than_s": mg["p90"], "devices": dict(devices.most_common())}


class _Counted:
    """Counts normalised free text: occurrences, the videos it appears in, and the first original wording."""

    def __init__(self, norm=norm_text):
        self.norm = norm
        self.items = {}

    def add(self, text, vid, extra=None):
        key = self.norm(text)
        if not key:
            return None
        it = self.items.setdefault(key, {"key": key, "text": text.strip(), "count": 0, "videos": []})
        it["count"] += 1
        if vid not in it["videos"]:
            it["videos"].append(vid)
        if extra:
            for k, v in extra.items():
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


def _decisions(docs):
    n = len(docs)
    pr, problems = _Counted(), Counter()
    for vid, doc in docs.items():
        for d in _lst(doc, "decisions"):
            if d.get("problem_solved") in PROBLEMS:
                problems[d["problem_solved"]] += 1
            if _str(d.get("principle")):
                pr.add(d["principle"], vid, {"problems": d.get("problem_solved", "other")})
    principles = [{"principle": p["text"], "key": p["key"], "count": p["count"], "seen_in": p["seen_in"],
                   "videos": p["videos"], "problems": p.get("problems", {}), "status": p["status"]}
                  for p in pr.ranked(n)]
    return principles, dict(problems.most_common())


def _toolchain(docs):
    n = len(docs)
    by_el = {}
    for vid, doc in docs.items():
        seen = {}
        for tc in _lst(doc, "toolchain"):
            el, tool = norm_text(tc.get("element")), norm_tool(tc.get("likely_tool"))
            if not el or not tool:
                continue
            prev = seen.get(el)
            c = tc.get("confidence") if _num(tc.get("confidence")) else 0.0
            if prev is None or c > prev[1]:
                seen[el] = (tool, c, tc.get("how") if isinstance(tc.get("how"), str) else "")
        for el, (tool, c, how) in seen.items():
            e = by_el.setdefault(el, {"videos": [], "votes": {}})
            e["videos"].append(vid)
            v = e["votes"].setdefault(tool, {"tool": tool, "videos": 0, "stated_confidence": [], "how": []})
            v["videos"] += 1
            v["stated_confidence"].append(c)
            if how and how not in v["how"] and len(v["how"]) < 3:
                v["how"].append(how)
    out = {}
    for el, e in sorted(by_el.items()):
        k = len(e["videos"])
        votes = sorted(e["votes"].values(), key=lambda v: (-v["videos"], v["tool"]))
        for v in votes:
            v["stated_confidence"] = stat(v["stated_confidence"])["median"]
        top = votes[0]
        out[el] = {"consensus": top["tool"], "confidence": round(top["videos"] / k, 4), "seen_in": k,
                   "share": _share(k, n), "status": _status(k, n), "how": top["how"], "votes": votes}
    return out


def _rules(docs):
    rc = _Counted()
    for vid, doc in docs.items():
        rules = doc.get("rules_of_thumb") if isinstance(doc.get("rules_of_thumb"), list) else []
        for r in rules:
            if isinstance(r, str):
                rc.add(r, vid)
    return rc.ranked(len(docs))


def _top_text(counter):
    return counter.most_common(1)[0][0] if counter else ""


def _ideation(docs):
    n = len(docs)
    methods, method_vids = Counter(), {}
    play, examples, craft, emotions = {}, {}, {}, Counter()
    patterns = _Counted()
    total = literal = 0
    for vid, doc in docs.items():
        for v in _lst(doc, "visual_ideas"):
            m, mt = v.get("translation"), v.get("message_type")
            if m not in TRANSLATIONS:
                continue
            total += 1
            literal += m == "literal"
            methods[m] += 1
            method_vids.setdefault(m, set()).add(vid)
            if _str(v.get("emotion")):
                emotions[norm_text(v["emotion"])] += 1
            if mt in MESSAGE_TYPES:
                play.setdefault(mt, Counter())[m] += 1
                ex = examples.setdefault((mt, m), [])
                if len(ex) < EXAMPLES_PER_METHOD and not any(e["video"] == vid for e in ex):
                    ex.append({"video": vid, "line": v.get("line", ""), "message": v.get("message", ""),
                               "idea": v.get("idea", ""), "why_this": v.get("why_this", ""),
                               "craft": v.get("craft") if isinstance(v.get("craft"), dict) else {}})
            c = v.get("craft") if isinstance(v.get("craft"), dict) else {}
            cm = craft.setdefault(m, {k: Counter() for k in CRAFT_KEYS})
            for k in CRAFT_KEYS:
                if _str(c.get(k)):
                    cm[k][c[k].strip()] += 1
            pat = v.get("pattern") if _str(v.get("pattern")) else v.get("idea")
            if _str(pat):
                patterns.add(pat, vid, {"methods": m})
    dist = {m: {"count": c, "share": _share(c, total), "seen_in": len(method_vids[m]),
                "status": _status(len(method_vids[m]), n)} for m, c in methods.most_common()}
    playbook = {}
    for mt in MESSAGE_TYPES:
        if mt not in play:
            continue
        cnt = play[mt]
        k = sum(cnt.values())
        ms = [{"method": m, "count": c, "share": _share(c, k), "examples": examples.get((mt, m), [])}
              for m, c in sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0]))]
        summary = f"{mt} → " + ", ".join(f"{e['method']} {round(e['share'] * 100)} %" for e in ms)
        playbook[mt] = {"n": k, "methods": ms, "top": ms[0]["method"], "summary": summary}
    craft_notes = {m: {k: _top_text(cm[k]) for k in CRAFT_KEYS} for m, cm in craft.items()}
    pats = [{"pattern": p["text"], "key": p["key"], "count": p["count"], "seen_in": p["seen_in"],
             "videos": p["videos"], "methods": p.get("methods", {}), "status": p["status"]}
            for p in patterns.ranked(n)]
    return {"n_ideas": total, "methods": dist, "playbook": playbook, "patterns": pats,
            "literal_share": _share(literal, total), "figurative_share": _share(total - literal, total),
            "emotions": dict(emotions.most_common()), "craft": craft_notes}


def synth(analyses, held_out=None, dupes=None):
    """Aggregate {video id: creator.json doc} into the creator mind (see the module docstring)."""
    docs = {k: v for k, v in _as_dict(analyses).items() if isinstance(v, dict)}
    principles, problems = _decisions(docs)
    mind = {"schema": MIND_SCHEMA, "trained_on": list(docs), "held_out": list(held_out or []),
            "dupes": [list(d) for d in (dupes or [])],
            "ideation": _ideation(docs), "story_template": _story(docs), "hooks": _hooks(docs),
            "retention": _retention(docs), "principles": principles, "problems": problems,
            "toolchain": _toolchain(docs), "rules_of_thumb": _rules(docs)}
    mind["checklist"] = checklist(mind)
    return mind


# ---------------------------------------------------------------- planner questions

def _fmt(x):
    return str(round(float(x), 2))


def checklist(mind):
    """The "what would this creator do here?" questions Plan 5's beat planner asks for every beat. Generic
    wording; only numbers and the fixed vocabularies (methods, parts, devices) come from the mind."""
    mind = mind if isinstance(mind, dict) else {}
    ide = mind.get("ideation") or {}
    ret = mind.get("retention") or {}
    st = mind.get("story_template") or {}
    items = [
        "What should the viewer understand or feel on this line (the message, not the noun), and which "
        "message type is it (fact, number, claim, emotion, contrast, process, reveal, cta)?",
    ]
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
    items.append("Does this beat open or close a loop? If it opens one, where is it closed; if it closes one, "
                 "was it opened clearly enough?")
    order = st.get("order") or []
    if order:
        items.append(f"Which story part is this beat in ({' → '.join(order)} is the creator's usual order), and "
                     "does it do that part's job?")
    else:
        items.append("Which story part is this beat in (hook, setup, turn, payoff, cta, ...) and does it do that "
                     "part's job?")
    items.append("Which problem does this beat solve (attention dip, clarity, proof, emotion, rhythm), and which "
                 "of the creator's ranked principles justifies it?")
    items.append("Is this the creator's habit (the style) or only what this content needs (a one-off)? Keep the "
                 "habit, and solve a one-off on its own terms.")
    items.append("Is this new? No idea, object or move already used in this video or for this client.")
    return items


# ---------------------------------------------------------------- ideation brief

def _wild_card(ide, exclude, line):
    dist = ide.get("methods") or {}
    cands = [m for m in TRANSLATIONS if m not in exclude and m not in NOT_WILD]
    if not cands:
        return None
    low = min(dist.get(m, {}).get("count", 0) for m in cands)
    tied = [m for m in cands if dist.get(m, {}).get("count", 0) == low]
    h = int(hashlib.sha256((line or "").encode("utf-8")).hexdigest(), 16)
    return tied[h % len(tied)]


def ideate(mind, line, message_type):
    """The brief Claude uses to invent visuals for `line` the creator's way: top methods for the message type
    with reference examples (line -> idea), the creator's why_this patterns, the 3-ideas rule with a wild card
    from a method the creator rarely uses, and the craft notes per method."""
    if message_type not in MESSAGE_TYPES:
        raise ValueError(f"message type {message_type!r} is not one of {'|'.join(MESSAGE_TYPES)}")
    ide = (mind or {}).get("ideation") or {}
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
        methods = [{"method": m, "count": d["count"], "share": d["share"],
                    "examples": allex.get(m, [])[:EXAMPLES_PER_METHOD]}
                   for m, d in (ide.get("methods") or {}).items()]
    top = methods[:2]
    top_names = [m["method"] for m in top]
    wild = _wild_card(ide, top_names, line)
    why, seen = [], set()
    for m in top:
        for e in m.get("examples", []):
            w = e.get("why_this", "")
            if _str(w) and norm_text(w) not in seen:
                seen.add(norm_text(w))
                why.append(w)
    craft_all = ide.get("craft") or {}
    craft_notes = {m: craft_all.get(m, {k: "" for k in CRAFT_KEYS}) for m in top_names}
    wild_share = ((ide.get("methods") or {}).get(wild) or {}).get("share", 0.0) if wild else None
    if top_names:
        uses = " and ".join(top_names) if len(top_names) > 1 else f"{top_names[0]} (twice, two different ideas)"
        rule = (f"Generate 3 ideas: 2 using the creator's top methods for a {message_type} line ({uses}), "
                f"1 wild card using {wild}, a method the creator rarely uses. Take the method, never the "
                "reference idea itself; each idea must answer the five design questions of craft-rules §1.")
    else:
        rule = (f"Generate 3 ideas: no references to lean on, so try 3 different methods, one of them {wild}. "
                "Each idea must answer the five design questions of craft-rules §1.")
    return {"line": line, "message_type": message_type, "fallback": fallback,
            "top_methods": [{"method": m["method"], "share": m.get("share"), "count": m.get("count"),
                             "examples": [{"line": e.get("line", ""), "idea": e.get("idea", ""),
                                           "why_this": e.get("why_this", "")} for e in m.get("examples", [])]}
                            for m in top],
            "wild_card": {"method": wild, "share_overall": wild_share,
                          "note": "rarely or never used by the creator: a fresh angle, still built on their craft"},
            "why_patterns": why, "craft_notes": craft_notes, "rule": rule,
            "for_each_idea": ["idea (concrete: object, action, framing)", "translation method",
                              "why it serves the message better than a literal shot",
                              "craft: framing, motion, text_role, sound_role"]}


# ---------------------------------------------------------------- hold-out scoring (called by prove.py)

def _boundaries(structure):
    segs = [s for s in (structure or []) if isinstance(s, dict) and _num(s.get("t0")) and s.get("part")]
    return sorted(((float(s["t0"]), str(s["part"])) for s in segs), key=lambda b: b[0])


def score_structure(pred, actual, tol_s=STRUCTURE_TOL_S):
    """Score predicted story parts against the actual ones. Each part's start is a boundary labelled with the
    part; a predicted boundary matches an actual one of the same part within tol_s (nearest first,
    one-to-one). Returns {precision, recall, overall_f1, f1, tp, n_pred, n_actual, timing_mae_s, abs_err_sum,
    order_match} (keys compatible with prove.pool_scores)."""
    p, a = _boundaries(pred), _boundaries(actual)
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
            "order_match": bool(p) and seq(p) == seq(a), "tol_s": tol_s}


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


def creator_files(pack):
    root = Path(pack) / "analysis"
    if not Path(pack).is_dir():
        raise BadInput(f"pack folder not found: {Path(pack).name}")
    if not root.is_dir():
        raise BadInput(f"no analysis folder in {Path(pack).name}")
    files = {d.name: d / "creator.json" for d in sorted(root.iterdir()) if (d / "creator.json").is_file()}
    if not files:
        raise BadInput("no creator.json in analysis/<id>/ (write them first, see references/creator-mind.md)")
    return files


def load_checked(files):
    """({id: doc}, [error lines]) for the given {id: path}."""
    docs, errors = {}, []
    for vid, f in files.items():
        try:
            doc = json.loads(Path(f).read_bytes().decode("utf-8-sig"))
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


def _ident(pack, vid):
    d = Path(pack) / "analysis" / vid
    m = _read(d / "measure.json")
    sha = None
    if isinstance(m, dict) and isinstance(m.get("video"), dict) and isinstance(m["video"].get("sha256"), str):
        sha = m["video"]["sha256"]
    th = _read(d / "thumbs.json")
    th = th if isinstance(th, list) and th and all(isinstance(h, str) for h in th) else None
    return sha, th


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

def cmd_check(a):
    files = creator_files(a.pack)
    docs, errors = load_checked(files)
    for e in errors:
        print(e, file=sys.stderr)
    bad = len(files) - len(docs)
    if errors:
        print(f"creator.json: {bad} of {len(files)} file(s) have schema errors", file=sys.stderr)
        return 1
    print(f"creator.json: {len(files)} file(s) valid")
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
    mind = synth({i: docs[i] for i in kept}, held_out=held, dupes=dupes)
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
    brief = ideate(mind, a.line, a.type)
    print(json.dumps(brief, ensure_ascii=False, indent=2))
    return 0


def main(argv=None):
    store.print_utf8()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check", help="validate every analysis/<id>/creator.json")
    c.add_argument("pack")
    s = sub.add_parser("synth", help="write <pack>/creator-mind.json")
    s.add_argument("pack")
    g = s.add_mutually_exclusive_group()
    g.add_argument("--train", help="comma list of video ids to train on")
    g.add_argument("--all", action="store_true", help="train on every video with a creator.json (default)")
    s.add_argument("--exclude", help="comma list of video ids held out of training")
    i = sub.add_parser("ideate", help="print the ideation brief for one line (JSON)")
    i.add_argument("pack")
    i.add_argument("--line", required=True, help="the spoken line under the visual")
    i.add_argument("--type", required=True, choices=MESSAGE_TYPES, help="message type of the line")
    a = ap.parse_args(argv)
    try:
        return {"check": cmd_check, "synth": cmd_synth, "ideate": cmd_ideate}[a.cmd](a)
    except (BadInput, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except OSError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(store.exit_on_lock_timeout(main))

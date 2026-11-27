#!/usr/bin/env python3
"""B2: the edit plan. Claude writes the beats and 3 concepts per beat into plan.json; this script feeds
Claude the candidates (style laws that fire on the beat, banked ideas, the creator's way of turning a line
into a picture), checks every set, writes the signatures, scores and recommends, and records what was shown.

Every set has exactly 3 options and exactly one wild card, which is a new invention. A set uses a law
when one fires, an idea when one is available, and always an invention. Every option answers the five
design questions (point, real_thing, invisible, reference, new). Nothing already shown for the beat,
rejected by the client, used for the client in another project, or picked for another beat repeats.

Commands:
    plan.py candidates <project> [--beat N] [--json]
    plan.py check <project>                 problems, one per line (exit 1 when any)
    plan.py recommend <project>             score the latest sets and write the recommendation
    plan.py finalize <project>              check, recommend, record what was shown (exit 1 on problems)
    plan.py more <project> --beat N [--owner-ok-more --owner-words "..."]   open the next set of one beat

Option fields Claude writes: id (B03-p<k>), set, key, title, category, scene_type, desc, source
(law|idea|invention), law ("<when>-><then>" or null), idea_id, recipe, wild, concept_key, tags, fit (0..1),
design_q {point, real_thing, invisible, reference, new}, translation, message, why_this.
The script writes: sig, score; per beat: beat_types, lang, sets, recommended, pick, taste_decided, taste_note.

Exit codes: 0 ok, 1 a check failed or the budget is used up, 2 bad input.
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import store  # noqa: E402

STEP = "plan"
DESIGN_Q = ("point", "real_thing", "invisible", "reference", "new")
DESIGN_Q_MIN = 8
GAP_TOL = 0.05
W_LAW, W_IDEA, W_PREFER, W_AVOID = 0.15, 0.10, 0.10, 0.20
SOURCES = ("law", "idea", "invention")


class BadInput(Exception):
    pass


# ---------------------------------------------------------------- loading

def _plan_path(d):
    return Path(d) / "plan.json"


def load_plan(d):
    p = _plan_path(d)
    if not p.is_file():
        raise BadInput(f"no plan.json in {Path(d).name}")
    data = store.read_json(p, None)
    if not isinstance(data, dict) or not isinstance(data.get("beats"), list):
        raise BadInput("plan.json has no beats list")
    return data


def _ctx(d):
    """{project entry, style entry, pack path or None, client slug or None, aspect, lang}."""
    import project
    import styles
    try:
        entry = project.load(d)
    except KeyError as e:
        raise BadInput(str(e.args[0] if e.args else e))
    st = styles.find(styles.load_registry(), entry.get("style") or "") or {}
    pack = Path(st["pack"]) if str(st.get("pack") or "").strip() and Path(st["pack"]).is_dir() else None
    return {"entry": entry, "style": st, "pack": pack, "client": entry.get("client"),
            "aspect": st.get("aspect"), "slug": st.get("slug") or entry.get("style")}


def _norm_type(t):
    t = str(t or "").strip().lower()
    return t[5:] if t.startswith("word:") else t


def load_laws(pack):
    if pack is None:
        return []
    data = store.read_json(Path(pack) / "laws.json", None)
    rows = data.get("laws", []) if isinstance(data, dict) else data if isinstance(data, list) else []
    return [r for r in rows if isinstance(r, dict) and r.get("when") and r.get("then")]


def load_recipes(pack):
    if pack is None:
        return set()
    data = store.read_json(Path(pack) / "recipes.json", None)
    rows = data.get("recipes", []) if isinstance(data, dict) else data if isinstance(data, list) else []
    return {r.get("kind") for r in rows if isinstance(r, dict) and r.get("kind")}


def _words_in(d, beat):
    """The beat's words: edit_ready.words.json inside [start, end), else the beat's script split."""
    doc = store.read_json(Path(d) / "edit_ready.words.json", None)
    ws = doc.get("words", []) if isinstance(doc, dict) else doc if isinstance(doc, list) else []
    t0, t1 = float(beat.get("start", 0)), float(beat.get("end", 0))
    out = []
    for w in ws:
        if not isinstance(w, dict):
            continue
        s = w.get("start", w.get("start_ms", 0) / 1000 if "start_ms" in w else None)
        if s is None:
            continue
        if t0 - 1e-6 <= float(s) < t1:
            out.append({"word": w.get("word") or w.get("text") or ""})
    if not out:
        out = [{"word": x} for x in str(beat.get("script") or "").split()]
    return out


def _lang_of(d, plan):
    doc = store.read_json(Path(d) / "edit_ready.words.json", None)
    if isinstance(doc, dict) and doc.get("language"):
        return str(doc["language"])
    return str(plan.get("lang") or "he")


# ---------------------------------------------------------------- triggers and candidates

def beat_triggers(words, lang, beat_types=()):
    """{"number", "date", "question"} found in the words (profile of `lang`), plus the beat's types."""
    import lang as langmod
    texts = [str(w.get("word") or w.get("text") or "") if isinstance(w, dict) else str(w) for w in words]
    out = {_norm_type(t) for t in beat_types if t}
    try:
        prof = langmod.load_profile(lang)
    except KeyError:
        prof = {"triggers": {}}
    for hit in langmod.find_triggers(texts, prof):
        out.add({"month": "date"}.get(hit["kind"], hit["kind"]))
    if any(re.search(r"\d", t) for t in texts):
        out.add("number")
    return out


def _law_id(law):
    return f"{_norm_type(law['when'])}->{law['then']}"


def fired_laws(laws, triggers):
    hits = [l for l in laws if _norm_type(l["when"]) in triggers]
    return sorted(hits, key=lambda l: (l.get("status") != "rule-candidate", -float(l.get("confidence") or 0)))


def _beat_key(b):
    return str(int(b["n"]))


def _latest(beat):
    opts = [o for o in beat.get("options") or [] if isinstance(o, dict)]
    if not opts:
        return 0, []
    s = max(int(o.get("set") or 1) for o in opts)
    return s, [o for o in opts if int(o.get("set") or 1) == s]


def _mind(pack):
    if pack is None:
        return None
    m = store.read_json(Path(pack) / "creator-mind.json", None)
    return m if isinstance(m, dict) else None


def candidates(d, beat=None):
    import ideas
    import project
    import taste
    plan = load_plan(d)
    cx = _ctx(d)
    laws = load_laws(cx["pack"])
    lang = _lang_of(d, plan)
    try:
        ts = taste.summary(cx["client"], cx["slug"])
    except Exception:
        ts = {"prefers": [], "avoids": []}
    mind = _mind(cx["pack"])
    out = {}
    for b in plan["beats"]:
        if beat is not None and int(b["n"]) != int(beat):
            continue
        bt = list(b.get("beat_types") or [])
        trig = beat_triggers(_words_in(d, b), lang, bt)
        blocked = project.blocked(d, STEP, int(b["n"]))
        try:  # a beat with no trigger and no type does not pull every banked idea in
            found = [] if not trig else ideas.search(beat_types=sorted(trig), aspect=cx["aspect"], style=cx["slug"],
                                 exclude_sigs=frozenset(blocked), avoid_client=cx["client"], limit=5)
        except ValueError:
            found = []
        row = {"triggers": sorted(trig), "laws": fired_laws(laws, trig),
               "ideas": [{"id": i["id"], "title": i["title"], "kind": i["kind"], "tags": i.get("tags", []),
                          "rate": round((i.get("picked", 0) + 1) / (i.get("offered", 0) + 2), 3)} for i in found],
               "blocked": sorted(blocked), "prefers": ts.get("prefers", []), "avoids": ts.get("avoids", [])}
        if mind:
            import creator
            try:
                if b.get("message_type"):
                    row["creator"] = creator.ideate(mind, str(b.get("script") or ""), b["message_type"])
                row["checklist"] = creator.checklist(mind)
            except Exception as e:  # a hand-edited or old mind must not stop the plan
                row["creator_error"] = f"{type(e).__name__}: {e}"
        out[_beat_key(b)] = row
    return out


# ---------------------------------------------------------------- signatures and checks

def option_sig(o):
    import project
    return project.sig(STEP, o.get("category") or "", o.get("concept_key") or o.get("title") or "")


def check(d):
    plan = load_plan(d)
    cx = _ctx(d)
    cands = candidates(d)
    recipes = load_recipes(cx["pack"])
    problems = []
    beats = sorted(plan["beats"], key=lambda b: float(b.get("start", 0)))
    length = float((plan.get("clip") or {}).get("length_s") or (beats[-1].get("end") if beats else 0) or 0)
    t = 0.0
    for b in beats:
        s, e = float(b.get("start", 0)), float(b.get("end", 0))
        if abs(s - t) > GAP_TOL:
            problems.append(f"B{b['n']}: {'gap' if s > t else 'overlap'} of {abs(s - t):.2f} s before it")
        if e <= s:
            problems.append(f"B{b['n']}: end {e:.2f} is not after start {s:.2f}")
        t = e
    if beats and abs(t - length) > GAP_TOL:
        problems.append(f"the beats end at {t:.2f} s, the clip is {length:.2f} s")
    shown_items = store.read_json(Path(d) / "shown.json", {}).get("items", []) or []
    for b in plan["beats"]:
        n = b["n"]
        set_no, opts = _latest(b)
        cd = dict(cands.get(_beat_key(b), {}))
        # the set being checked may already be recorded as shown (finalize run twice): not a repeat of itself
        own = {it.get("sig") for it in shown_items
               if it.get("step") == STEP and int(it.get("beat", -1)) == int(n) and int(it.get("set", 0)) == set_no}
        own |= {f"idea:{it['idea_id']}" for it in shown_items if it.get("idea_id") and it.get("step") == STEP
                and int(it.get("beat", -1)) == int(n) and int(it.get("set", 0)) == set_no}
        cd["blocked"] = [x for x in cd.get("blocked", []) if x not in own]
        if len(opts) != 3:
            problems.append(f"B{n}: the latest set has {len(opts)} options, needs exactly 3")
        wild = [o for o in opts if o.get("wild")]
        if len(wild) != 1:
            problems.append(f"B{n}: {len(wild)} wild cards in the set, needs exactly 1")
        for o in wild:
            if o.get("source") != "invention":
                problems.append(f"B{n}: wild card {o.get('id')} must be an invention, not {o.get('source')}")
        srcs = {o.get("source") for o in opts}
        bad_src = srcs - set(SOURCES)
        if bad_src:
            problems.append(f"B{n}: unknown source {sorted(bad_src)}")
        if any(l.get("status") == "rule-candidate" for l in cd.get("laws", [])) and "law" not in srcs:
            problems.append(f"B{n}: a style law fires here ({_law_id(cd['laws'][0])}) but no option uses a law")
        if cd.get("ideas") and "idea" not in srcs:
            problems.append(f"B{n}: a banked idea fits ({cd['ideas'][0]['id']}) but no option uses one")
        if "invention" not in srcs:
            problems.append(f"B{n}: no new invention in the set")
        sigs = []
        for o in opts:
            dq = o.get("design_q") or {}
            for q in DESIGN_Q:
                if len(str(dq.get(q) or "").strip()) < DESIGN_Q_MIN:
                    problems.append(f"B{n} {o.get('id')}: design question '{q}' is missing or too short")
            sg = option_sig(o)
            if sg in sigs:
                problems.append(f"B{n} {o.get('id')}: same concept as another option in the set")
            sigs.append(sg)
            if sg in set(cd.get("blocked", [])):
                problems.append(f"B{n} {o.get('id')}: repeats a concept already shown, rejected or used ({sg})")
            if o.get("idea_id") and f"idea:{o['idea_id']}" in set(cd.get("blocked", [])):
                problems.append(f"B{n} {o.get('id')}: idea {o['idea_id']} is blocked for this beat or client")
            if o.get("recipe") and o["recipe"] not in recipes:
                problems.append(f"B{n} {o.get('id')}: recipe '{o['recipe']}' is not in the style's recipes.json")
            if o.get("source") == "law" and not o.get("law"):
                problems.append(f"B{n} {o.get('id')}: source law but no law named")
            if o.get("source") == "idea" and not o.get("idea_id"):
                problems.append(f"B{n} {o.get('id')}: source idea but no idea_id")
    return problems


# ---------------------------------------------------------------- scoring

def _score(o, laws_by_id, idea_rate, prefers, avoids, taste=True):
    s = float(o.get("fit") or 0)
    if o.get("source") == "law" and o.get("law") in laws_by_id:
        s += W_LAW * float(laws_by_id[o["law"]].get("confidence") or 0)
    if o.get("source") == "idea" and o.get("idea_id"):
        s += W_IDEA * idea_rate.get(o["idea_id"], 0.5)
    if taste:
        tags = {str(t).lower() for t in o.get("tags") or []}
        s += W_PREFER * len(tags & prefers) - W_AVOID * len(tags & avoids)
    return round(s, 4)


def _argmax(opts, key):
    best = None
    for o in opts:  # a tie goes to the earlier option
        if best is None or key(o) > key(best) + 1e-12:
            best = o
    return best


def recommend(d):
    import ideas
    import taste
    cx = _ctx(d)
    laws_by_id = {_law_id(l): l for l in load_laws(cx["pack"])}
    try:
        ts = taste.summary(cx["client"], cx["slug"])
    except Exception:
        ts = {"prefers": [], "avoids": []}
    prefers = {t.lower() for t in ts.get("prefers", [])}
    avoids = {t.lower() for t in ts.get("avoids", [])}
    rates = {}
    for i in ideas.load_index()["ideas"]:
        rates[i["id"]] = (i.get("picked", 0) + 1) / (i.get("offered", 0) + 2)
    rows = {}

    def go(plan):
        ranked = {}
        for b in plan["beats"]:
            _, opts = _latest(b)
            for o in opts:
                o["sig"] = option_sig(o)
                o["score"] = _score(o, laws_by_id, rates, prefers, avoids)
            if not opts:
                continue
            order = sorted(range(len(opts)), key=lambda k: (-opts[k]["score"], k))
            plain = _argmax(opts, lambda o: _score(o, laws_by_id, rates, prefers, avoids, taste=False))
            ranked[_beat_key(b)] = {"beat": b, "opts": [opts[k] for k in order], "plain": plain, "pos": 0}
        # different beats never share a recommended concept: the lower-scoring beat moves on
        notes = {k: "" for k in ranked}
        changed = True
        while changed:
            changed = False
            seen = {}
            for k, r in sorted(ranked.items(), key=lambda kv: -kv[1]["opts"][kv[1]["pos"]]["score"]):
                sg = r["opts"][r["pos"]]["sig"]
                if sg in seen and r["pos"] + 1 < len(r["opts"]):
                    r["pos"] += 1
                    notes[k] = f"הרעיון הכי חזק כבר מומלץ בביט {seen[sg]}; כאן ההמלצה היא הבא בתור"
                    changed = True
                    break
                seen.setdefault(sg, ranked[k]["beat"]["n"])
        for k, r in ranked.items():
            b, rec = r["beat"], r["opts"][r["pos"]]
            decided = r["pos"] == 0 and rec is not r["plain"]
            b["recommended"] = rec.get("id")
            b["pick"] = rec.get("key") or rec.get("id")
            b["taste_decided"] = bool(decided)
            b["taste_note"] = notes[k] or ("הטעם של הלקוח הכריע" if decided else "")
            b["sets"] = max(int(b.get("sets") or 0), _latest(b)[0])
            rows[k] = {"beat": b["n"], "recommended": rec.get("id"), "taste_decided": bool(decided),
                       "note": b["taste_note"]}

    store.update_json(_plan_path(d), {"beats": []}, go)
    return [rows[k] for k in sorted(rows, key=int)]


# ---------------------------------------------------------------- finalize and more

def finalize(d):
    import project
    probs = check(d)
    if probs:
        for p in probs:
            print(p, file=sys.stderr)
        return 1
    recommend(d)
    plan = load_plan(d)
    shown = store.read_json(Path(d) / "shown.json", {}).get("sets", {}) or {}
    for b in plan["beats"]:
        s, opts = _latest(b)
        if int(shown.get(f"{STEP}:{int(b['n'])}", 0)) >= s:
            continue  # this set was already recorded
        if s == 1 and int(b.get("sets") or 0) <= 1:
            project.next_set(d, STEP, int(b["n"]))  # the first set counts toward the budget
        project.record_shown(d, STEP, int(b["n"]), s, opts)
    return 0


def more(d, beat, owner_ok_more=False, owner_words=None):
    import project
    n = project.next_set(d, STEP, int(beat), owner_ok_more, owner_words)

    def go(plan):
        for b in plan["beats"]:
            if int(b["n"]) == int(beat):
                b["sets"] = n
                return
        raise BadInput(f"no beat {beat} in plan.json")
    store.update_json(_plan_path(d), {"beats": []}, go)
    return n, candidates(d, beat)


# ---------------------------------------------------------------- CLI

def _main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("candidates"); a.add_argument("project"); a.add_argument("--beat", type=int)
    a.add_argument("--json", action="store_true")
    for name in ("check", "recommend", "finalize"):
        sub.add_parser(name).add_argument("project")
    a = sub.add_parser("more"); a.add_argument("project"); a.add_argument("--beat", type=int, required=True)
    a.add_argument("--owner-ok-more", action="store_true"); a.add_argument("--owner-words")
    args = ap.parse_args(argv)
    store.print_utf8()
    import project
    try:
        if args.cmd == "candidates":
            c = candidates(args.project, args.beat)
            if args.json:
                print(json.dumps(c, ensure_ascii=False, indent=2))
            else:
                for k, r in c.items():
                    laws = ", ".join(_law_id(l) for l in r["laws"][:3]) or "-"
                    ids = ", ".join(i["id"] for i in r["ideas"]) or "-"
                    print(f"B{k}: טריגרים {', '.join(r['triggers']) or '-'} | חוקים {laws} | רעיונות {ids} | "
                          f"חסומים {len(r['blocked'])}")
            return 0
        if args.cmd == "check":
            probs = check(args.project)
            for p in probs:
                print(p)
            return 1 if probs else 0
        if args.cmd == "recommend":
            for r in recommend(args.project):
                extra = f" ({r['note']})" if r["note"] else ""
                print(f"B{r['beat']}: {r['recommended']}{extra}")
            return 0
        if args.cmd == "finalize":
            rc = finalize(args.project)
            if rc == 0:
                print("התוכנית נבדקה; ההמלצות והסטים שהוצגו נשמרו")
            return rc
        n, c = more(args.project, args.beat, args.owner_ok_more, args.owner_words)
        print(json.dumps({"beat": args.beat, "set": n, "candidates": c.get(str(args.beat), {})},
                         ensure_ascii=False, indent=2))
        return 0
    except project.BudgetExceeded as e:
        print(str(e), file=sys.stderr)
        return 1
    except (BadInput, project._Bad) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


def main(argv=None):
    return store.exit_on_lock_timeout(_main, argv)


if __name__ == "__main__":
    sys.exit(main())

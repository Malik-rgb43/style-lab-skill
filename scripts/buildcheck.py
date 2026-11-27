#!/usr/bin/env python3
"""B5: the build gates, the review checklist and one draft per round.

Order is enforced: captions first; beats one at a time in plan order; the first 2 non-a-roll beats are
compared with their picked frame stills (SSIM and palette) before any later beat starts. The checklist holds
every effective rule (craft, style, client), the fixed craft checks and automatic checks the script ticks
itself (AI beats placed only from look_pass items, a complete generation log, spend within the approved
batches). A draft needs every "pre" item ticked; presenting it needs every "post" item.

Commands:
    buildcheck.py captions-done <project>
    buildcheck.py start-beat <project> --beat N [--override "<reason>"]
    buildcheck.py beat-done <project> --beat N [--snippet <file> --params <json file>]
    buildcheck.py compare <project> --beat N --snapshot <png>
    buildcheck.py checklist <project>
    buildcheck.py tick <project> --phase pre|post --n K [--note "..."]
    buildcheck.py draft <project> --file <mp4>
    buildcheck.py present <project>
    buildcheck.py next-round <project>
    buildcheck.py status <project> [--json]
Exit codes: 0 ok, 1 a gate refused or a check failed, 2 bad input, 3 missing tool.
"""
import argparse
import datetime as dt
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import store  # noqa: E402

COMPARE_SSIM = 0.60
COMPARE_PALETTE_DE = 10.0
FIRST_N = 2
FIXED_PRE = ("אין פריימים ריקים", "אין שאריות (אלמנטים שנשארו מסצנה קודמת)", "אין טקסט חתוך",
             "אין אלמנטים על הפנים", "אין כתוביות על הסנטר", "0 שגיאות ב־hyperframes check")
POST = ("בדיקת תנועה: prove.py qa <pack> --build <draft> נקי, או שכל דגל תוקן או הוסבר",
        "צפיתי בטיוטה מתחילה ועד סוף")
AUTO = {"ai_look_pass": "כל ביט AI הוצב רק מפריטים שעברו את בדיקת המראה (look_pass)",
        "gen_log_complete": "בכל שורה ביומן היצירה יש prompt, model, seed, cost, file",
        "spend_ok": "ההוצאה בתוך המנות שאושרו (אף מנה לא over)"}


class Refused(Exception):
    pass


class BadInput(ValueError):
    pass


def _now():
    return dt.date.today().isoformat()


def _path(d):
    return Path(d) / "build" / "state.json"


def _empty():
    return {"version": 1, "round": 1, "captions": {"done": False, "date": None}, "beats": {}, "override": [],
            "checklist": {"pre": [], "post": []}, "drafts": []}


def load(d):
    s = store.read_json(_path(d), None)
    base = _empty()
    if isinstance(s, dict):
        base.update(s)
    return base


def _update(d, fn):
    def go(s):
        for k, v in _empty().items():
            s.setdefault(k, v)
        return fn(s)
    box = {}
    store.update_json(_path(d), _empty(), lambda s: box.update(r=go(s)))
    return box.get("r")


# ---------------------------------------------------------------- plan helpers

def _plan(d):
    p = store.read_json(Path(d) / "plan.json", None)
    if not isinstance(p, dict) or not p.get("beats"):
        raise BadInput("no plan.json with beats")
    return p


def _chosen(d, b):
    import picks
    e = picks.load(d)["plan"].get(str(b["n"])) or {}
    pid = e.get("picked") or b.get("recommended")
    for o in b.get("options") or []:
        if o.get("id") == pid:
            return o
    return None


def _order(d):
    return [int(b["n"]) for b in sorted(_plan(d)["beats"], key=lambda b: float(b.get("start", 0)))]


def first_n(d):
    plan = _plan(d)
    order = sorted(plan["beats"], key=lambda b: float(b.get("start", 0)))
    out = [int(b["n"]) for b in order if ((_chosen(d, b) or {}).get("scene_type") or "") != "a-roll"]
    return out[:FIRST_N]


# ---------------------------------------------------------------- gates

def captions_done(d):
    _update(d, lambda s: s.update(captions={"done": True, "date": _now()}))


def start_beat(d, beat, override=None):
    beat = int(beat)
    s = load(d)
    if not s["captions"].get("done"):
        raise Refused("קודם הכתוביות: buildcheck.py captions-done אחרי שהן מוכנות")
    order = _order(d)
    if beat not in order:
        raise BadInput(f"no beat {beat} in plan.json")
    built = {int(k) for k, v in s["beats"].items() if v.get("built")}
    nxt = next((b for b in order if b not in built), None)
    if beat != nxt:
        raise Refused(f"הביטים נבנים לפי הסדר: הבא בתור הוא {nxt}, לא {beat}")
    fn = first_n(d)
    if beat not in fn:
        bad = [b for b in fn if not ((s["beats"].get(str(b)) or {}).get("compared") or {}).get("pass")]
        if bad and not str(override or "").strip():
            raise Refused(f"קודם להשוות את הביטים {', '.join(map(str, bad))} לפריימים שנבחרו (compare), "
                          "או --override \"<סיבה>\"")

    def go(st):
        e = st["beats"].setdefault(str(beat), {"started": False, "built": False, "compared": None,
                                               "snippet": None, "params": None})
        e["started"] = True
        if beat not in fn and str(override or "").strip():
            st["override"].append({"beat": beat, "reason": override.strip(), "date": _now()})
    _update(d, go)


def beat_done(d, beat, snippet=None, params=None):
    beat = int(beat)
    s = load(d)
    if not (s["beats"].get(str(beat)) or {}).get("started"):
        raise Refused(f"ביט {beat} לא התחיל (start-beat)")
    plan = _plan(d)
    b = next(x for x in plan["beats"] if int(x["n"]) == beat)
    o = _chosen(d, b) or {}
    rel = None
    if snippet:
        sp = Path(snippet)
        sp = sp if sp.is_absolute() else Path(d) / sp
        if not sp.is_file():
            raise BadInput(f"snippet not found: {snippet}")
        try:
            rel = sp.resolve().relative_to(Path(d).resolve()).as_posix()
        except ValueError:
            raise BadInput("the snippet must be inside the project")

    def go(st):
        e = st["beats"][str(beat)]
        e.update(built=True, snippet=rel, params=params if rel else None, title=o.get("title") or b.get("script"),
                 kind="b-roll" if o.get("scene_type") in ("cutaway", "overlay") else "text-animation",
                 beat_types=list(b.get("beat_types") or []))
    _update(d, go)


def _picked_still(d, beat):
    import picks
    e = picks.load(d)["frames"].get(str(int(beat))) or {}
    for _, v in picks.options(d, "frames", beat):
        if v.get("id") == e.get("picked"):
            return Path(d) / v["still"], v.get("at_s")
    raise Refused(f"לביט {beat} אין פריים שנבחר להשוואה")


def compare(d, beat, snapshot_png):
    import imgmetrics as im
    still, at_s = _picked_still(d, beat)
    a, b = im.decode_image(still), im.decode_image(snapshot_png)
    ss = im.ssim(a, b)
    sa, sb = im.look_stats(a), im.look_stats(b)
    de = im.palette_distance(sb["palette"], [p["hex"] for p in sa["palette"]]) or 0.0
    res = {"ssim": ss, "palette_de": de, "pass": bool(ss >= COMPARE_SSIM and de <= COMPARE_PALETTE_DE), "at_s": at_s}
    _update(d, lambda st: st["beats"].setdefault(str(int(beat)), {"started": True, "built": False}).update(compared=res))
    return res


# ---------------------------------------------------------------- checklist

def _auto(d):
    q = (store.read_json(Path(d) / "gen" / "queue.json", {"items": []}) or {}).get("items", [])
    log = []
    lp = Path(d) / "gen" / "log.jsonl"
    if lp.is_file():
        for line in lp.read_text(encoding="utf-8").splitlines():
            try:
                log.append(json.loads(line))
            except ValueError:
                pass
    placed = [i for i in q if i.get("stage") == "build" or i.get("status") == "placed"]
    ai_ok = all(i.get("status") in ("look_pass", "placed") for i in placed if i.get("kind") in ("image", "video", "3d"))
    gens = [x for x in log if x.get("type") == "gen"]
    log_ok = all(all(x.get(k) not in (None, "") for k in ("prompt", "model", "seed", "cost", "file"))
                 for x in gens if x.get("status") == "generated")
    sp = store.read_json(Path(d) / "spend.json", {"batches": []}) or {}
    spend_ok = not any(b.get("status") == "over" for b in sp.get("batches", []))
    return {"ai_look_pass": ai_ok, "gen_log_complete": log_ok, "spend_ok": spend_ok}


def _rule_items(d):
    import lessons
    import project
    e = project.load(d)
    out = []
    for r in lessons.effective_rules(e.get("style"), e.get("client")):
        if r.get("overridden_by"):
            continue
        src = {"craft": "craft", "style": "style", "client": "client"}[r["layer"]]
        out.append({"text": f"[{r['layer']} {r.get('rule_no')}] {r['measurable']} ({r['id']})", "source": src,
                    "lesson": r["id"]})
    return out


def checklist(d):
    rules = _rule_items(d)
    auto = _auto(d)
    s = load(d)
    old = {(i.get("source"), i.get("text")): i for i in s["checklist"].get("pre", []) + s["checklist"].get("post", [])}

    def keep(text, source):
        o = old.get((source, text)) or {}
        return {"ticked": bool(o.get("ticked")), "note": o.get("note", "")}
    pre = []
    for r in rules:
        pre.append({"text": r["text"], "source": r["source"], **keep(r["text"], r["source"])})
    for t in FIXED_PRE:
        pre.append({"text": t, "source": "craft", **keep(t, "craft")})
    for k, t in AUTO.items():
        pre.append({"text": t, "source": "auto", "auto": k, "ticked": bool(auto[k]), "note": ""})
    post = [{"text": t, "source": "craft", **keep(t, "craft")} for t in POST]
    for n, it in enumerate(pre, 1):
        it["n"] = n
    for n, it in enumerate(post, 1):
        it["n"] = n
    _update(d, lambda st: st.update(checklist={"pre": pre, "post": post}))
    L = [f"# רשימת בדיקה · סבב {s['round']}", "", "## לפני הטיוטה"]
    L += [f"{it['n']}. [{'x' if it['ticked'] else ' '}] {it['text']}" + (" (אוטומטי)" if it["source"] == "auto" else "")
          for it in pre]
    L += ["", "## אחרי הטיוטה"] + [f"{it['n']}. [{'x' if it['ticked'] else ' '}] {it['text']}" for it in post]
    out = Path(d) / "review-checklist.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    return out


def tick(d, phase, n, note=""):
    if phase not in ("pre", "post"):
        raise BadInput("phase must be pre or post")

    def go(st):
        items = st["checklist"].get(phase) or []
        it = next((x for x in items if int(x.get("n", -1)) == int(n)), None)
        if it is None:
            raise BadInput(f"no {phase} item {n} (run checklist first)")
        if it.get("source") == "auto":
            raise Refused("פריט אוטומטי: הסקריפט מסמן אותו לבד כשהוא נכון")
        it["ticked"] = True
        it["note"] = note
    _update(d, go)


def draft(d, file):
    checklist(d)  # recompute the automatic items first
    s = load(d)
    if not Path(file).is_file():
        raise BadInput(f"draft not found: {file}")
    open_items = [i for i in s["checklist"]["pre"] if not i.get("ticked")]
    if open_items:
        raise Refused("לפני טיוטה כל פריטי 'לפני' מסומנים. פתוחים: " + ", ".join(str(i["n"]) for i in open_items))
    if any(x.get("round") == s["round"] for x in s["drafts"]):
        raise Refused(f"כבר יש טיוטה בסבב {s['round']}: טיוטה אחת לסבב (next-round אחרי ההערות)")
    try:
        rel = Path(file).resolve().relative_to(Path(d).resolve()).as_posix()
    except ValueError:
        rel = str(file)
    entry = {"round": s["round"], "file": rel, "date": _now(), "presented": False}
    _update(d, lambda st: st["drafts"].append(entry))
    return entry


def present(d):
    s = load(d)
    mine = [x for x in s["drafts"] if x.get("round") == s["round"]]
    if not mine:
        raise Refused("אין טיוטה בסבב הזה")
    open_items = [i for i in s["checklist"]["post"] if not i.get("ticked")]
    if open_items:
        raise Refused("לפני הצגה כל פריטי 'אחרי' מסומנים. פתוחים: " + ", ".join(str(i["n"]) for i in open_items))

    def go(st):
        for x in st["drafts"]:
            if x.get("round") == st["round"]:
                x["presented"] = True
    _update(d, go)
    return dict(mine[-1], presented=True)


def next_round(d):
    def go(st):
        st["round"] = int(st.get("round") or 1) + 1
        for ph in ("pre", "post"):
            for it in st["checklist"].get(ph) or []:
                it["ticked"], it["note"] = False, ""
    _update(d, go)
    checklist(d)
    return load(d)["round"]


# ---------------------------------------------------------------- CLI

def _main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("captions-done", "checklist", "present", "next-round"):
        sub.add_parser(name).add_argument("project")
    a = sub.add_parser("start-beat"); a.add_argument("project"); a.add_argument("--beat", type=int, required=True)
    a.add_argument("--override")
    a = sub.add_parser("beat-done"); a.add_argument("project"); a.add_argument("--beat", type=int, required=True)
    a.add_argument("--snippet"); a.add_argument("--params")
    a = sub.add_parser("compare"); a.add_argument("project"); a.add_argument("--beat", type=int, required=True)
    a.add_argument("--snapshot", required=True)
    a = sub.add_parser("tick"); a.add_argument("project"); a.add_argument("--phase", required=True, choices=("pre", "post"))
    a.add_argument("--n", type=int, required=True); a.add_argument("--note", default="")
    a = sub.add_parser("draft"); a.add_argument("project"); a.add_argument("--file", required=True)
    a = sub.add_parser("status"); a.add_argument("project"); a.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    store.print_utf8()
    import imgmetrics
    try:
        d = args.project
        if not Path(d).is_dir():
            raise BadInput(f"project folder not found: {d}")
        if args.cmd == "captions-done":
            captions_done(d)
            print("הכתוביות סומנו כמוכנות")
        elif args.cmd == "start-beat":
            start_beat(d, args.beat, args.override)
            print(f"ביט {args.beat}: התחלנו")
        elif args.cmd == "beat-done":
            params = None
            if args.params:
                try:
                    params = json.loads(Path(args.params).read_text(encoding="utf-8"))
                except (OSError, ValueError) as e:
                    raise BadInput(f"cannot read --params: {e}")
            beat_done(d, args.beat, args.snippet, params)
            print(f"ביט {args.beat}: נבנה")
        elif args.cmd == "compare":
            r = compare(d, args.beat, args.snapshot)
            print(f"ביט {args.beat}: SSIM {r['ssim']:.2f} (צריך {COMPARE_SSIM}), פלטה ΔE {r['palette_de']:.1f} "
                  f"(עד {COMPARE_PALETTE_DE:g}): {'עבר' if r['pass'] else 'נכשל'}")
            return 0 if r["pass"] else 1
        elif args.cmd == "checklist":
            print(checklist(d))
        elif args.cmd == "tick":
            tick(d, args.phase, args.n, args.note)
            print(f"סומן: {args.phase} {args.n}")
        elif args.cmd == "draft":
            e = draft(d, args.file)
            print(f"טיוטה לסבב {e['round']}: {e['file']}")
        elif args.cmd == "present":
            present(d)
            print("הטיוטה מוכנה להצגה")
        elif args.cmd == "next-round":
            print(f"סבב {next_round(d)}")
        else:
            s = load(d)
            if args.json:
                print(json.dumps(s, ensure_ascii=False, indent=2))
            else:
                print(f"סבב {s['round']}, כתוביות {'מוכנות' if s['captions'].get('done') else 'לא'}; "
                      f"ביטים שנבנו: {sorted(int(k) for k, v in s['beats'].items() if v.get('built'))}")
        return 0
    except Refused as e:
        print(str(e), file=sys.stderr)
        return 1
    except imgmetrics.MissingTool as e:
        print(str(e), file=sys.stderr)
        return 3
    except (BadInput, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


def main(argv=None):
    return store.exit_on_lock_timeout(_main, argv)


if __name__ == "__main__":
    sys.exit(main())

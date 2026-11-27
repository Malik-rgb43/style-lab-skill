#!/usr/bin/env python3
"""The owner's picks for one edit: plan concepts, frame variants and motion variants, in picks.json.

picks.json is the Studio contract (the Studio, a pick page and the chat all write this one shape):
  {"version": 1, "project": "<id>", "updated": "ISO", "source": "studio|page|chat",
   "plan":   {"3": {"picked": "B03-p2", "own": "", "more": false, "decided_by": "owner", "sig": "plan:..."}},
   "frames": {"3": {"picked": "B03-f1-b", "more": false, "decided_by": "owner"}},
   "motion": {"mode": "three", "overrides": {"5": "one"},
              "beats": {"3": {"picked": "B03-m1-a", "more": false, "decided_by": "owner"}}},
   "notes":  {"3": "free text"}}
picks.logged.json = {"logged": ["plan:3:B03-p2", ...]} records which owner picks are already in the
taste file, so `sync` logs each pick once. Only the owner's picks are logged (decided_by "owner").

Commands:
    picks.py pick <project> --step plan|frames|motion --beat N (--id ID | --more | --own "...")
                  [--note "..."] [--by owner|claude] [--source chat|page|studio]
    picks.py sync <project>          log new owner picks to taste; print "more" requests and own designs
    picks.py validate <project>      problems, one per line (exit 1 when any)
    picks.py show <project> [--json]
Exit codes: 0 ok, 1 a check failed, 2 bad input.
"""
import argparse
import datetime as dt
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import store  # noqa: E402

STEPS = ("plan", "frames", "motion")
BY = ("owner", "claude")
SOURCES = ("chat", "page", "studio")
MODES = ("three", "one")


class BadInput(ValueError):
    pass


def _path(d):
    return Path(d) / "picks.json"


def _logged_path(d):
    return Path(d) / "picks.logged.json"


def _empty():
    return {"version": 1, "project": None, "updated": None, "source": None, "plan": {}, "frames": {},
            "motion": {"mode": None, "overrides": {}, "beats": {}}, "notes": {}}


def _now():
    return dt.datetime.now().isoformat(timespec="seconds")


def load(d):
    data = store.read_json(_path(d), None)
    base = _empty()
    if isinstance(data, dict):
        base.update({k: v for k, v in data.items() if v is not None})
        if not isinstance(base.get("motion"), dict):
            base["motion"] = _empty()["motion"]
        base["motion"].setdefault("beats", {})
        base["motion"].setdefault("overrides", {})
    return base


def _section(data, step):
    return data["motion"]["beats"] if step == "motion" else data[step]


# ---------------------------------------------------------------- options of a step

def options(d, step, beat):
    """[(set_no, option dict)] of one beat at one step, from plan.json, frames.json or motion-variants.json."""
    d = Path(d)
    beat = int(beat)
    if step == "plan":
        plan = store.read_json(d / "plan.json", {}) or {}
        for b in plan.get("beats", []) or []:
            if isinstance(b, dict) and int(b.get("n", -1)) == beat:
                return [(int(o.get("set") or 1), o) for o in b.get("options") or [] if isinstance(o, dict)]
        return []
    name = "frames.json" if step == "frames" else "motion-variants.json"
    doc = store.read_json(d / name, {}) or {}
    b = (doc.get("beats") or {}).get(str(beat)) or {}
    out = []
    for s in b.get("sets") or []:
        for v in s.get("variants") or []:
            if isinstance(v, dict):
                out.append((int(s.get("set") or 1), v))
    return out


def _find(d, step, beat, pid):
    for set_no, o in options(d, step, beat):
        if o.get("id") == pid:
            return set_no, o
    return None, None


# ---------------------------------------------------------------- writing

def pick(d, step, beat, picked=None, *, own="", more=False, note="", by="owner", source="chat"):
    if step not in STEPS:
        raise BadInput(f"step must be one of {', '.join(STEPS)}")
    if by not in BY:
        raise BadInput(f"--by must be owner or claude")
    if source not in SOURCES:
        raise BadInput(f"source must be one of {', '.join(SOURCES)}")
    beat = int(beat)
    own = (own or "").strip()
    if own and step != "plan":
        raise BadInput("an own design is only for the plan step")
    if not picked and not own and not more and not note:
        raise BadInput("give an id, --more, --own or --note")
    entry = None
    if picked or own or more:
        entry = {"picked": picked or None, "own": own, "more": bool(more), "decided_by": by}
        if step == "plan" and picked:
            _, o = _find(d, "plan", beat, picked)
            if o is not None:
                import plan as planmod
                entry["sig"] = o.get("sig") or planmod.option_sig(o)
    try:
        import project
        pid = project.load(d).get("id")
    except (ImportError, KeyError):
        pid = None
    out = {}

    def go(data):
        data.setdefault("version", 1)
        for k in ("plan", "frames", "notes"):
            data.setdefault(k, {})
        mo = data.setdefault("motion", {})
        mo.setdefault("mode", None)
        mo.setdefault("overrides", {})
        mo.setdefault("beats", {})
        if pid and not data.get("project"):
            data["project"] = pid
        if entry is not None:
            sec = mo["beats"] if step == "motion" else data[step]
            old = sec.get(str(beat)) or {}
            if more and not picked and old.get("picked"):
                entry["picked"] = old["picked"]  # asking for more keeps the earlier pick visible
            sec[str(beat)] = entry
            out.update(entry)
        if note:
            data["notes"][str(beat)] = note.strip()
        data["updated"] = _now()
        data["source"] = source
    store.update_json(_path(d), _empty(), go)
    return out


def set_motion_mode(d, mode, overrides=None):
    if mode is not None and mode not in MODES:
        raise BadInput("mode must be three or one")
    for k, v in (overrides or {}).items():
        if v not in MODES:
            raise BadInput(f"override for beat {k} must be three or one")

    def go(data):
        mo = data.setdefault("motion", {"mode": None, "overrides": {}, "beats": {}})
        if mode is not None:
            mo["mode"] = mode
        mo.setdefault("overrides", {}).update({str(k): v for k, v in (overrides or {}).items()})
        mo.setdefault("beats", {})
        data["updated"] = _now()
    store.update_json(_path(d), _empty(), go)


# ---------------------------------------------------------------- checks

def validate(d):
    data = load(d)
    probs = []
    for step in STEPS:
        for b, e in sorted(_section(data, step).items(), key=lambda kv: int(kv[0])):
            if not isinstance(e, dict):
                probs.append(f"{step} B{b}: not an object")
                continue
            if e.get("picked"):
                set_no, o = _find(d, step, b, e["picked"])
                if o is None:
                    probs.append(f"{step} B{b}: {e['picked']} is not an option of this beat")
            if step == "frames" and (e.get("picked")) and not _has_plan(data, b):
                probs.append(f"frames B{b}: picked before the plan pick of this beat")
            if step == "motion" and e.get("picked") and not (data["frames"].get(str(b)) or {}).get("picked"):
                probs.append(f"motion B{b}: picked before the frames pick of this beat")
    seen = {}
    for b, e in sorted(data["plan"].items(), key=lambda kv: int(kv[0])):
        sg = (e or {}).get("sig")
        if not sg and (e or {}).get("picked"):
            _, o = _find(d, "plan", b, e["picked"])
            if o is not None:
                import plan as planmod
                sg = o.get("sig") or planmod.option_sig(o)
        if sg:
            if sg in seen:
                probs.append(f"plan B{b}: same concept as beat {seen[sg]} (never twice in one video)")
            seen.setdefault(sg, b)
    return probs


def _has_plan(data, b):
    e = data["plan"].get(str(b)) or {}
    return bool(e.get("picked") or e.get("own"))


# ---------------------------------------------------------------- taste sync

def sync(d):
    import taste
    data = load(d)
    try:
        import project
        entry = project.load(d)
    except (ImportError, KeyError):
        entry = {}
    logged = set((store.read_json(_logged_path(d), {}) or {}).get("logged", []))
    new, more, own = [], [], []
    for step in STEPS:
        for b, e in sorted(_section(data, step).items(), key=lambda kv: int(kv[0])):
            if not isinstance(e, dict):
                continue
            if e.get("more"):
                more.append({"step": step, "beat": int(b)})
            if step == "plan" and e.get("own"):
                own.append({"beat": int(b), "text": e["own"]})
            if e.get("decided_by") != "owner" or not e.get("picked") or e.get("more"):
                continue
            key = f"{step}:{b}:{e['picked']}"
            if key in logged:
                continue
            set_no, o = _find(d, step, b, e["picked"])
            if o is None:
                continue
            others = [x for s, x in options(d, step, b) if s == set_no and x.get("id") != e["picked"]]
            sig_of = (lambda x: x.get("sig")) if step != "plan" else \
                (lambda x: x.get("sig") or __import__("plan").option_sig(x))
            taste.log(entry.get("client"), entry.get("style"), step, e["picked"], [x["id"] for x in others],
                      entry.get("id"), note=(data["notes"].get(str(b)) or ""),
                      tags={"picked": list(o.get("tags") or []),
                            "rejected": sorted({t for x in others for t in x.get("tags") or []})},
                      sigs={"picked": sig_of(o), "rejected": [s for s in (sig_of(x) for x in others) if s]})
            if o.get("idea_id"):
                import ideas
                try:
                    ideas.picked(o["idea_id"], entry.get("client"))
                except (KeyError, ValueError):
                    pass
            logged.add(key)
            new.append(key)
    if new:
        def go(doc):
            doc["logged"] = sorted(set(doc.get("logged", [])) | set(new))
        store.update_json(_logged_path(d), {"logged": []}, go)
    return {"logged": new, "more": more, "own": own}


# ---------------------------------------------------------------- CLI

def _main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("pick"); a.add_argument("project"); a.add_argument("--step", required=True, choices=STEPS)
    a.add_argument("--beat", type=int, required=True)
    g = a.add_mutually_exclusive_group()
    g.add_argument("--id"); g.add_argument("--more", action="store_true"); g.add_argument("--own")
    a.add_argument("--note", default=""); a.add_argument("--by", default="owner", choices=BY)
    a.add_argument("--source", default="chat", choices=SOURCES)
    for name in ("sync", "validate"):
        sub.add_parser(name).add_argument("project")
    a = sub.add_parser("show"); a.add_argument("project"); a.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    store.print_utf8()
    try:
        if not Path(args.project).is_dir():
            raise BadInput(f"project folder not found: {args.project}")
        if args.cmd == "pick":
            pick(args.project, args.step, args.beat, args.id, own=args.own or "", more=args.more,
                 note=args.note, by=args.by, source=args.source)
            print(f"נשמר: {args.step} ביט {args.beat}")
            return 0
        if args.cmd == "sync":
            r = sync(args.project)
            print(f"נרשמו לטעם: {len(r['logged'])}")
            for m in r["more"]:
                print(f"  עוד רעיונות: {m['step']} ביט {m['beat']}")
            for o in r["own"]:
                print(f"  עיצוב משלך לביט {o['beat']}: {o['text']}")
            return 0
        if args.cmd == "validate":
            probs = validate(args.project)
            for p in probs:
                print(p)
            return 1 if probs else 0
        data = load(args.project)
        if args.json:
            print(json.dumps(data, ensure_ascii=False, indent=2))
        else:
            for step in STEPS:
                for b, e in sorted(_section(data, step).items(), key=lambda kv: int(kv[0])):
                    print(f"{step} B{b}: {e.get('picked') or ('own' if e.get('own') else '-')}"
                          f"{' + עוד רעיונות' if e.get('more') else ''} ({e.get('decided_by')})")
        return 0
    except BadInput as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


def main(argv=None):
    return store.exit_on_lock_timeout(_main, argv)


if __name__ == "__main__":
    sys.exit(main())

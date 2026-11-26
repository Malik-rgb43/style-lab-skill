#!/usr/bin/env python3
"""Project state for stage B: registry entry, option signatures, the no-repeat set and the more-ideas budget.

Files: <store home>/projects.json = {"version": 1, "projects": [{id, title, path, style, client, stage,
status, created, updated, last_draft}]}; per project folder: shown.json (options already shown).

Commands:
    project.py init <dir> --title T --clip <mp4> --style S [--client C]
    project.py show <dir> [--json]
    project.py stage <dir> B3
    project.py close <dir> --final <mp4>
    project.py list [--json]
    project.py blocked <dir> --step plan|frames|motion --beat N [--json]
    project.py next-set <dir> --step plan|frames|motion --beat N [--owner-ok-more --owner-words "<owner words>"]

Signatures: sig = "<step>:" + sha1(norm(parts joined by "|"))[:12]. blocked = signatures a new option set for
that beat and step must not contain: already shown in this project, rejected by the client at that step, and (plan
step only) picked for this client in another project or for another beat of this video.
More ideas: next_set gives set 1, then at most 3 extra sets per beat and step; after that it exits 1 unless
--owner-ok-more is given.
Exit codes: 0 ok, 1 refused (budget), 2 bad input.
"""
import argparse
import datetime as dt
import hashlib
import json
import os
import re
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import store  # noqa: E402
import lessons  # noqa: E402
import taste  # noqa: E402

STAGES = tuple(f"B{i}" for i in range(8))
STEPS = ("plan", "frames", "motion")
MAX_SETS = 4  # 1 + 3 extra


class BudgetExceeded(Exception):
    pass


class _Bad(Exception):
    pass


_FOLD = re.compile(r"[-_\u2010-\u2015\u05be\u05f4\"]+")


def _part(p):
    if isinstance(p, bool):
        return "1" if p else "0"
    if isinstance(p, float):
        return f"{p:.2f}"
    t = unicodedata.normalize("NFC", str(p))
    t = _FOLD.sub(" ", t)
    return lessons.norm(t).replace("|", "\\|")


def sig(step, *parts):
    text = "|".join(_part(p) for p in parts)
    return f"{step}:" + hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def _posix(p):
    return str(p).replace("\\", "/")


def _abs(d):
    return _posix(Path(d).resolve())


def _key(p):
    return os.path.normcase(_abs(p))


def _now():
    return dt.datetime.now().isoformat(timespec="seconds")


def _registry_path():
    return store.path("projects.json")


def _registry():
    return store.read_json(_registry_path(), {"version": 1, "projects": []})


def load(d):
    k = _key(d)
    for e in _registry().get("projects", []):
        if os.path.normcase(e.get("path", "")) == k:
            return e
    raise KeyError(f"project not initialised: {_abs(d)}")


def _update(d, fn):
    k = _key(d)

    def go(reg):
        reg.setdefault("version", 1)
        for e in reg.setdefault("projects", []):
            if os.path.normcase(e.get("path", "")) == k:
                fn(e)
                e["updated"] = _now()
                return
        raise KeyError(f"project not initialised: {_abs(d)}")
    store.update_json(_registry_path(), {"version": 1, "projects": []}, go)


def init(d, title, clip, style, client=None):
    import styles
    st = styles.find(styles.load_registry(), style)
    if not st:
        raise _Bad(f"unknown style: {style}")
    cslug = None
    if client:
        import clients
        c = clients.load_client(client)
        if not c:
            raise _Bad(f"unknown client: {client}")
        cslug = c["slug"]
    dp = Path(d)
    dp.mkdir(parents=True, exist_ok=True)
    key = _key(dp)
    out = {}

    def go(reg):
        reg.setdefault("version", 1)
        projs = reg.setdefault("projects", [])
        for e in projs:
            if os.path.normcase(e.get("path", "")) == key:
                e.update({"title": title, "style": st["slug"], "client": cslug or e.get("client"),
                          "updated": _now()})
                out.update(e)
                return
        pid = store.slugify(title, {e["id"] for e in projs})
        e = {"id": pid, "title": title, "path": _abs(dp), "style": st["slug"], "client": cslug,
             "stage": "B0", "status": "active", "created": _now(), "updated": _now(), "last_draft": None,
             "clip": _posix(clip)}
        projs.append(e)
        out.update(e)
    store.update_json(_registry_path(), {"version": 1, "projects": []}, go)

    cslug = out.get("client")

    def fill(plan):
        if not plan.get("client"):
            plan["client"] = cslug
        if not plan.get("style"):
            plan["style"] = st["slug"]
    store.update_json(dp / "plan.json", {}, fill)
    return out


def set_stage(d, stage):
    if stage not in STAGES:
        raise _Bad(f"bad stage {stage!r}: one of {', '.join(STAGES)}")
    _update(d, lambda e: e.update(stage=stage))


def close(d, final):
    def go(e):
        e["status"] = "final"
        e["final"] = _posix(final)
    _update(d, go)


# ---- shown.json, no-repeat set, budget

def _shown_path(d):
    return Path(d) / "shown.json"


_SHOWN_DEFAULT = {"version": 1, "items": [], "sets": {}}


def record_shown(d, step, beat, set_no, options):
    today = dt.date.today().isoformat()

    def go(data):
        data.setdefault("version", 1)
        items = data.setdefault("items", [])
        sets = data.setdefault("sets", {})
        for o in options:
            it = {"beat": int(beat), "step": step, "set": set_no, "id": o["id"], "sig": o["sig"], "date": today}
            if o.get("idea_id"):
                it["idea_id"] = o["idea_id"]
            items.append(it)
        k = f"{step}:{int(beat)}"
        sets[k] = max(int(sets.get(k, 0)), int(set_no))
    store.update_json(_shown_path(d), _SHOWN_DEFAULT, go)
    try:
        import ideas
    except ImportError:
        return
    for o in options:
        if o.get("idea_id"):
            try:
                ideas.offered(o["idea_id"])
            except (KeyError, ValueError):
                pass  # an idea that is not in the bank must not lose the shown record


def _bad_words(w):
    # Same rule as spend.py `_blank` (duplicated to avoid editing a file other work owns): blank when
    # only whitespace and invisible Cf marks (RTL/LRM, zero-width) remain. Pasted Hebrew keeps its marks.
    return (not isinstance(w, str)
            or all(c.isspace() or unicodedata.category(c) == "Cf" for c in w))


def next_set(d, step, beat, owner_ok_more=False, owner_words=None):
    """Next set number for (step, beat). Does not burn the budget: only record_shown counts a set."""
    if owner_ok_more and _bad_words(owner_words):
        raise _Bad("--owner-ok-more needs --owner-words with the owner's own message (non-blank, "
                   "no invisible marks)")
    beat = int(beat)
    n = int(store.read_json(_shown_path(d), _SHOWN_DEFAULT).get("sets", {}).get(f"{step}:{beat}", 0)) + 1
    if n > MAX_SETS:
        if not owner_ok_more:
            raise BudgetExceeded(f"הגענו ל־3 סטים נוספים לביט {beat}; לשאול את הבעלים")

        def go(data):
            data.setdefault("owner_ok_more", []).append(
                {"beat": beat, "step": step, "words": owner_words.strip(), "at": _now()})
        store.update_json(_shown_path(d), _SHOWN_DEFAULT, go)
    return n


def _opt_sigs(plan, picked_id):
    """Signatures (and idea:<id>) of plan option `picked_id` in a plan.json dict."""
    out = set()
    for b in plan.get("beats") or []:
        for o in b.get("options") or []:
            if o.get("id") == picked_id:
                if o.get("sig"):
                    out.add(o["sig"])
                if o.get("idea_id"):
                    out.add(f"idea:{o['idea_id']}")
    return out


def _plan_picks(d):
    """{beat string: (option id, stored sig or None)} of the plan picks in <d>/picks.json (any decided_by)."""
    picks = store.read_json(Path(d) / "picks.json", {})
    plan = picks.get("plan") if isinstance(picks, dict) else None
    out = {}
    for b, v in (plan or {}).items():
        if isinstance(v, dict) and v.get("picked"):
            out[str(b)] = (v["picked"], v.get("sig"))
    return out


def _pick_sigs(plan, pick):
    """The pick's own stored sig first; plan.json's option only when the pick has none."""
    pid, sg = pick
    return {sg} if sg else _opt_sigs(plan, pid)


def blocked(d, step, beat):
    entry = load(d)
    beat = int(beat)
    out = set()
    for it in store.read_json(_shown_path(d), _SHOWN_DEFAULT).get("items", []):
        if it.get("step") == step and int(it.get("beat", -1)) == beat:
            if it.get("sig"):
                out.add(it["sig"])
            if it.get("idea_id"):
                out.add(f"idea:{it['idea_id']}")
    client = entry.get("client")
    out |= taste.used_sigs(client, step)["rejected"]
    if step == "plan":
        me = store.read_json(Path(d) / "plan.json", {})
        for b, pk in _plan_picks(d).items():
            if b != str(beat):
                out |= _pick_sigs(me, pk)
        if client:
            for e in _registry().get("projects", []):
                if e.get("client") != client or os.path.normcase(e.get("path", "")) == _key(d):
                    continue
                other = store.read_json(Path(e["path"]) / "plan.json", {})
                for pk in _plan_picks(e["path"]).values():
                    out |= _pick_sigs(other, pk)
    return out


# ---- CLI

def main(argv=None):
    return store.exit_on_lock_timeout(_main, argv)


def _main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("init"); a.add_argument("dir"); a.add_argument("--title", required=True)
    a.add_argument("--clip", required=True); a.add_argument("--style", required=True); a.add_argument("--client")
    a = sub.add_parser("show"); a.add_argument("dir"); a.add_argument("--json", action="store_true")
    a = sub.add_parser("stage"); a.add_argument("dir"); a.add_argument("stage")
    a = sub.add_parser("close"); a.add_argument("dir"); a.add_argument("--final", required=True)
    a = sub.add_parser("list"); a.add_argument("--json", action="store_true")
    for name in ("blocked", "next-set"):
        a = sub.add_parser(name); a.add_argument("dir"); a.add_argument("--step", required=True, choices=STEPS)
        a.add_argument("--beat", type=int, required=True)
        if name == "blocked":
            a.add_argument("--json", action="store_true")
        else:
            a.add_argument("--owner-ok-more", action="store_true")
            a.add_argument("--owner-words")
    args = p.parse_args(argv)
    store.print_utf8()
    if args.cmd not in ("init", "list"):
        try:
            load(args.dir)
        except KeyError as e:
            print(e.args[0], file=sys.stderr)
            return 2
    try:
        if args.cmd == "init":
            e = init(args.dir, args.title, args.clip, args.style, args.client)
            print(json.dumps(e, ensure_ascii=False))
        elif args.cmd == "show":
            e = load(args.dir)
            if args.json:
                print(json.dumps(e, ensure_ascii=False, indent=2))
            else:
                for k, v in e.items():
                    print(f"{k}: {v}")
        elif args.cmd == "stage":
            set_stage(args.dir, args.stage)
            print(f"stage {args.stage}")
        elif args.cmd == "close":
            close(args.dir, args.final)
            print("closed (final)")
        elif args.cmd == "list":
            projs = _registry().get("projects", [])
            if args.json:
                print(json.dumps(projs, ensure_ascii=False, indent=2))
            else:
                for e in projs:
                    print(f"{e['id']}\t{e['stage']}\t{e['status']}\t{e['path']}")
        elif args.cmd == "blocked":
            b = sorted(blocked(args.dir, args.step, args.beat))
            if args.json:
                print(json.dumps(b, ensure_ascii=False))
            else:
                print("\n".join(b))
        else:
            print(next_set(args.dir, args.step, args.beat, args.owner_ok_more, args.owner_words))
    except BudgetExceeded as e:
        print(str(e), file=sys.stderr)
        return 1
    except (_Bad, taste._Bad) as e:
        print(str(e), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())

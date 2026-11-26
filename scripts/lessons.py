"""Lessons registry: the owner's review notes, interpreted and layered (craft / style / client / video).

A lesson may later become a rule. Registry: <store home>/lessons/lessons.json
    {"version": 1, "next_id": 1, "lessons": [...]}   ids L0001, L0002, ...

Commands:
    lessons.py add --text "<owner words>" --measurable "<measurable reading>" --layer craft|style|client|video
                   [--target <style|client>] [--kind quality|taste] [--scope ..]
                   --project P --video V [--t S] [--round N] [--before ..] [--after ..]
    lessons.py list [--status S] [--layer L] [--target T] [--json]
    lessons.py show <id>
    lessons.py edit <id> field=value [...]    (text, measurable, scope, layer, target, kind)
    lessons.py reject <id> [--reason ..]

craft needs --kind and no target; style needs a style target; client needs a client target;
video has neither (status one-off). Other layers start as candidate.
"""
import argparse
import datetime as dt
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import store  # noqa: E402

LAYERS = ("craft", "style", "client", "video")
KINDS = ("quality", "taste")
STATUSES = ("candidate", "proposed", "rule", "rejected", "merged", "one-off")
EDITABLE = ("text", "measurable", "scope", "layer", "target", "kind")


class _Abort(Exception):
    def __init__(self, rc):
        self.rc = rc


def _fail(msg):
    print(msg, file=sys.stderr)
    return 1


def registry_path():
    return store.path("lessons", "lessons.json")


def _empty():
    return {"version": 1, "next_id": 1, "lessons": []}


def today():
    return dt.date.today().isoformat()


def load():
    return store.read_json(registry_path(), _empty())


def get(lesson_id):
    return next((x for x in load()["lessons"] if x["id"] == lesson_id), None)


def norm(text):
    """Lowercase, collapse whitespace, strip punctuation/space at both ends (Hebrew kept)."""
    t = re.sub(r"\s+", " ", (text or "").lower())
    return re.sub(r"^[\W_]+|[\W_]+$", "", t)


def _resolve(layer, target, kind):
    """Validate the layer/target/kind combination. Returns (target_slug, kind) or raises ValueError."""
    if layer not in LAYERS:
        raise ValueError(f"layer must be one of {', '.join(LAYERS)}")
    if kind not in (None, *KINDS):
        raise ValueError(f"kind must be one of {', '.join(KINDS)}")
    if layer == "craft":
        if kind is None:
            raise ValueError("craft lessons need --kind quality|taste")
        if target:
            raise ValueError("craft lessons have no target")
        return None, kind
    if kind is not None:
        raise ValueError(f"kind applies to craft lessons only, not {layer}")
    if layer == "style":
        import styles
        s = styles.find(styles.load_registry(), target or "")
        if not s:
            raise ValueError(f"no style {target!r}" if target else "no style given: style lessons need --target <style>")
        return s["slug"], None
    if layer == "client":
        import clients
        c = clients.load_client(target or "")
        if not c:
            raise ValueError(f"no client {target!r}" if target else "no client given: client lessons need --target <client>")
        return c["slug"], None
    if target:
        raise ValueError("video lessons have no target")
    return None, None


def _rejected_match(data, text):
    n = norm(text)
    return [x["id"] for x in data["lessons"] if x["status"] == "rejected" and norm(x["text"]) == n]


def add(text, measurable, layer, target=None, kind=None, scope="", evidence=None):
    """Add a lesson (locked read-modify-write). Returns the lesson plus "warnings". Raises ValueError."""
    ev = dict(evidence or {})
    out = {}

    def fn(data):
        slug, k = _resolve(layer, target, kind)
        warnings = [f"same text as rejected lesson {i}" for i in _rejected_match(data, text)]
        e = {"project": ev.get("project", ""), "video": ev.get("video", ""), "t": ev.get("t"),
             "round": ev.get("round"), "note": ev.get("note", ""), "before": ev.get("before"),
             "after": ev.get("after"), "added": today()}
        lesson = {"id": f"L{data['next_id']:04d}", "text": text, "measurable": measurable, "layer": layer,
                  "target": slug, "kind": k, "scope": scope, "evidence": [e],
                  "status": "one-off" if layer == "video" else "candidate", "merged_into": None,
                  "rule_no": None, "exception_of": None, "reason": "", "created": today(),
                  "updated": today(), "promoted": None}
        data["next_id"] += 1
        data["lessons"].append(lesson)
        out.update(lesson, warnings=warnings)

    store.update_json(registry_path(), _empty(), fn)
    return out


def _edit(data, args):
    x = next((i for i in data["lessons"] if i["id"] == args.id), None)
    if not x:
        return _fail(f"no lesson {args.id!r}")
    new = {}
    for pair in args.pairs:
        if "=" not in pair:
            return _fail(f"expected field=value, got {pair!r}")
        k, v = pair.split("=", 1)
        if k not in EDITABLE:
            return _fail(f"{k!r} is not editable; use one of {list(EDITABLE)}")
        new[k] = v
    for k in ("text", "measurable"):
        if k in new and not new[k].strip():
            return _fail(f"{k} must not be empty")
    layer = new.get("layer", x["layer"])
    target = (new["target"] or None) if "target" in new else x["target"]
    kind = (new["kind"] or None) if "kind" in new else x["kind"]
    if "layer" in new and "target" not in new:
        target = None  # a target belongs to the old layer
    if "layer" in new and "kind" not in new and layer != "craft":
        kind = None
    try:
        x["target"], x["kind"] = _resolve(layer, target, kind)
    except ValueError as e:
        return _fail(str(e))
    x["layer"] = layer
    for k in ("text", "measurable", "scope"):
        if k in new:
            x[k] = new[k]
    if "layer" in new:
        if layer == "video" and x["status"] == "candidate":
            x["status"] = "one-off"
        elif layer != "video" and x["status"] == "one-off":
            x["status"] = "candidate"
    x["updated"] = today()
    print(f"updated {x['id']}")


def _reject(data, args):
    x = next((i for i in data["lessons"] if i["id"] == args.id), None)
    if not x:
        return _fail(f"no lesson {args.id!r}")
    x["status"] = "rejected"
    x["reason"] = args.reason or ""
    x["updated"] = today()
    print(f"rejected {x['id']}")


def _locked(fn, args):
    def inner(data):
        rc = fn(data, args)
        if rc is not None:
            raise _Abort(rc)
    try:
        store.update_json(registry_path(), _empty(), inner)
    except _Abort as e:
        return e.rc
    return 0


def cmd_add(args):
    ev = {"project": args.project, "video": args.video, "t": args.t, "round": args.round,
          "before": args.before, "after": args.after}
    try:
        lesson = add(args.text, args.measurable, args.layer, args.target, args.kind, args.scope, ev)
    except ValueError as e:
        return _fail(str(e))
    for w in lesson["warnings"]:
        print(f"warning: {w}", file=sys.stderr)
    print(f"added {lesson['id']} ({lesson['layer']}, {lesson['status']})")
    return 0


def cmd_list(args):
    rows = [x for x in load()["lessons"]
            if (not args.status or x["status"] == args.status)
            and (not args.layer or x["layer"] == args.layer)
            and (not args.target or x["target"] == args.target)]
    if args.json:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
        return 0
    if not rows:
        print("no lessons")
    for x in rows:
        print(f"{x['id']}\t{x['status']}\t{x['layer']}\t{x['target'] or '-'}\t{x['text']}")
    return 0


def cmd_show(args):
    x = get(args.id)
    if not x:
        return _fail(f"no lesson {args.id!r}")
    print(json.dumps(x, ensure_ascii=False, indent=2))
    return 0


def main(argv=None):
    return store.exit_on_lock_timeout(_main, argv)


def _main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("add")
    a.add_argument("--text", required=True); a.add_argument("--measurable", required=True)
    a.add_argument("--layer", required=True, choices=LAYERS); a.add_argument("--target")
    a.add_argument("--kind", choices=KINDS); a.add_argument("--scope", default="")
    a.add_argument("--project", required=True); a.add_argument("--video", required=True)
    a.add_argument("--t"); a.add_argument("--round"); a.add_argument("--before"); a.add_argument("--after")
    a = sub.add_parser("list")
    a.add_argument("--status", choices=STATUSES); a.add_argument("--layer", choices=LAYERS)
    a.add_argument("--target"); a.add_argument("--json", action="store_true")
    a = sub.add_parser("show"); a.add_argument("id")
    a = sub.add_parser("edit"); a.add_argument("id"); a.add_argument("pairs", nargs="+")
    a = sub.add_parser("reject"); a.add_argument("id"); a.add_argument("--reason")
    args = p.parse_args(argv)

    store.print_utf8()
    if args.cmd == "add":
        return cmd_add(args)
    if args.cmd == "list":
        return cmd_list(args)
    if args.cmd == "show":
        return cmd_show(args)
    return _locked(_edit if args.cmd == "edit" else _reject, args)


if __name__ == "__main__":
    sys.exit(main())

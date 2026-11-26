"""Registry of trained editing styles, so several styles can live side by side and the user
can pick one by name when editing.

The registry is one JSON file outside the skill folder (survives skill updates):
    <store home>/styles.json   (~/.claude/style-lab, or STYLE_LAB_HOME; override with --registry or STYLE_LAB_REGISTRY)

Commands:
    styles.py list [--all] [--json]
    styles.py migrate                (v1 -> v2, writes styles.json.bak-YYYYMMDD once)
    styles.py add [<slug>] --name "<display name>" --pack <style pack folder> [--description ...]
                  [--skill <path to style SKILL.md>] [--creator ...] [--fonts brand|reference]
                  [--colors brand|reference] [--theme dark|light|both] [--status training|ready]
                  [--kind trained|described|derived] [--aspect 9:16|16:9|1:1|4:5] [--name-source user|claude]
                  (slug optional: made from the name)
    styles.py show <slug or name>
    styles.py set <slug> key=value [key=value ...]
    styles.py remove <slug>          (removes the registry entry only, never files)
    styles.py check                  (every pack and skill path still exists)
"""
import argparse
import datetime as dt
import json
import os
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import store  # noqa: E402

STATUSES = ("training", "ready", "archived")
KINDS = ("trained", "described", "derived")
ASPECTS = ("9:16", "16:9", "1:1", "4:5")
NAME_SOURCES = ("user", "claude")
EDITABLE = {"name", "description", "pack", "skill", "creator", "fonts", "colors", "status",
            "board_url", "aliases", "notes", "theme",
            "kind", "aspect", "name_source", "version", "rules_count"}
V2_DEFAULTS = {"kind": "trained", "aspect": "9:16", "name_source": "user", "version": 1, "rules_count": 0}
_REG_OVERRIDE = None  # set from --registry


class _Abort(Exception):
    def __init__(self, rc):
        self.rc = rc


def registry_path():
    arg = _REG_OVERRIDE or os.environ.get("STYLE_LAB_REGISTRY")
    return Path(arg) if arg else store.path("styles.json")


def migrate(data):
    """v1 -> v2 in place. Returns (data, changed)."""
    changed = False
    if data.get("version", 1) < 2:
        data["version"] = 2
        changed = True
    for s in data.setdefault("styles", []):
        for k, v in V2_DEFAULTS.items():
            if k not in s:
                s[k] = v
                changed = True
    return data, changed


def load_registry():
    """Read-only load; v1 files are upgraded in memory only (use `migrate` to write)."""
    data = store.read_json(registry_path(), {"version": 2, "styles": []})
    return migrate(data)[0]


def save_registry(data):
    store.write_json(registry_path(), data)


def today():
    return dt.date.today().isoformat()


def find(data, key):
    """Match by slug, display name or alias, case-insensitive."""
    k = key.strip().lower()
    for s in data["styles"]:
        names = [s["slug"], s.get("name", "")] + s.get("aliases", [])
        if k in (n.lower() for n in names if n):
            return s
    return None


def clients_for(slug):
    try:
        import clients
    except ModuleNotFoundError:
        return []
    return clients.linked_clients(slug)


def _client_names(slug):
    return [c.get("name") or c.get("slug", "") for c in clients_for(slug)]


def cmd_list(data, args):
    rows = [s for s in data["styles"] if args.all or s.get("status") != "archived"]
    if args.json:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
        return 0
    if not rows:
        print("no styles yet")
        return 0
    for s in rows:
        print(f"{s['slug']}\t{s.get('name','')}\t{s.get('status','')}\t{s.get('description','')}")
    return 0


def cmd_add(data, args):
    if not args.slug:
        args.slug = store.slugify(args.name, {x["slug"] for x in data["styles"]})
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", args.slug):
        print("slug must be lowercase latin letters, digits and dashes", file=sys.stderr)
        return 1
    if find(data, args.slug) or find(data, args.name):
        print(f"a style named {args.slug!r} or {args.name!r} already exists", file=sys.stderr)
        return 1
    skill = args.skill or str(Path.home() / ".claude" / "skills" / f"style-{args.slug}" / "SKILL.md")
    data["styles"].append({
        "slug": args.slug,
        "name": args.name,
        "description": args.description or "",
        "creator": args.creator or "",
        "pack": str(Path(args.pack).resolve()),
        "skill": skill,
        "fonts": args.fonts,
        "colors": args.colors,
        "theme": args.theme,
        "status": args.status,
        "aliases": [],
        "board_url": "",
        "kind": args.kind,
        "aspect": args.aspect,
        "name_source": args.name_source,
        "version": 1,
        "rules_count": 0,
        "created": today(),
        "updated": today(),
    })
    print(f"added {args.slug} ({args.name}), status {args.status}")
    return None


def cmd_show(data, args):
    s = find(data, args.key)
    if not s:
        print(f"no style {args.key!r}", file=sys.stderr)
        return 1
    print(json.dumps({**s, "clients": clients_for(s["slug"])}, ensure_ascii=False, indent=2))
    return 0


def cmd_set(data, args):
    s = find(data, args.slug)
    if not s:
        print(f"no style {args.slug!r}", file=sys.stderr)
        return 1
    for pair in args.pairs:
        if "=" not in pair:
            print(f"expected key=value, got {pair!r}", file=sys.stderr)
            return 1
        k, v = pair.split("=", 1)
        if k not in EDITABLE:
            print(f"{k!r} is not editable; use one of {sorted(EDITABLE)}", file=sys.stderr)
            return 1
        if k == "status" and v not in STATUSES:
            print(f"status must be one of {STATUSES}", file=sys.stderr)
            return 1
        if k in ("fonts", "colors") and v not in ("brand", "reference"):
            print(f"{k} must be brand or reference", file=sys.stderr)
            return 1
        if k == "theme" and v not in ("dark", "light", "both"):
            print("theme must be dark, light or both", file=sys.stderr)
            return 1
        if k == "kind" and v not in KINDS:
            print(f"kind must be one of {KINDS}", file=sys.stderr)
            return 1
        if k == "aspect" and v not in ASPECTS:
            print(f"aspect must be one of {ASPECTS}", file=sys.stderr)
            return 1
        if k == "name_source" and v not in NAME_SOURCES:
            print(f"name_source must be one of {NAME_SOURCES}", file=sys.stderr)
            return 1
        if k in ("version", "rules_count"):
            if not v.isdigit():
                print(f"{k} must be a whole number", file=sys.stderr)
                return 1
            v = int(v)
        if k == "name":
            other = find(data, v)
            if other and other is not s:
                print(f"another style is already called {v!r}", file=sys.stderr)
                return 1
        s[k] = [a.strip() for a in v.split(",") if a.strip()] if k == "aliases" else v
    s["updated"] = today()
    print(f"updated {s['slug']}")
    return None


def cmd_remove(data, args):
    s = find(data, args.slug)
    if not s:
        print(f"no style {args.slug!r}", file=sys.stderr)
        return 1
    data["styles"].remove(s)
    print(f"removed {s['slug']} from the registry (files untouched: {s['pack']})")
    return None


def cmd_check(data, args):
    bad = 0
    for s in data["styles"]:
        for key in ("pack", "skill"):
            if s.get(key) and not Path(s[key]).exists():
                level = "missing" if s.get("status") == "ready" else "not yet"
                print(f"{s['slug']}: {key} {level}: {s[key]}")
                bad += level == "missing"
    print("ok" if not bad else f"{bad} missing path(s) on ready styles")
    return 1 if bad else 0


def cmd_migrate(args):
    path = registry_path()
    if not path.exists():
        print("no registry yet")
        return 0
    state = {}

    def fn(data):
        _, changed = migrate(data)
        state["changed"] = changed
        if not changed:
            raise _Abort(0)
        bak = path.with_name(f"{path.name}.bak-{dt.date.today():%Y%m%d}")
        if not bak.exists():
            shutil.copy2(path, bak)
            state["bak"] = bak

    try:
        store.update_json(path, {"version": 2, "styles": []}, fn)
    except _Abort:
        print("already v2")
        return 0
    print(f"migrated to v2 ({path})" + (f", backup {state['bak'].name}" if "bak" in state else ""))
    return 0


WRITERS = {"add": cmd_add, "set": cmd_set, "remove": cmd_remove}


def main():
    global _REG_OVERRIDE
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--registry")
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("list"); a.add_argument("--all", action="store_true"); a.add_argument("--json", action="store_true")
    a = sub.add_parser("add")
    a.add_argument("slug", nargs="?"); a.add_argument("--name", required=True); a.add_argument("--pack", required=True)
    a.add_argument("--description"); a.add_argument("--skill"); a.add_argument("--creator")
    a.add_argument("--fonts", choices=("brand", "reference"), default="brand")
    a.add_argument("--colors", choices=("brand", "reference"), default="brand")
    a.add_argument("--status", choices=STATUSES, default="training")
    a.add_argument("--theme", choices=("dark", "light", "both"), default="dark")
    a.add_argument("--kind", choices=KINDS, default="trained")
    a.add_argument("--aspect", choices=ASPECTS, default="9:16")
    a.add_argument("--name-source", dest="name_source", choices=NAME_SOURCES, default="user")
    a = sub.add_parser("show"); a.add_argument("key")
    a = sub.add_parser("set"); a.add_argument("slug"); a.add_argument("pairs", nargs="+")
    a = sub.add_parser("remove"); a.add_argument("slug")
    sub.add_parser("check")
    sub.add_parser("migrate")
    args = p.parse_args()

    store.print_utf8()
    _REG_OVERRIDE = args.registry
    if args.cmd == "migrate":
        return cmd_migrate(args)
    if args.cmd in WRITERS:

        def fn(data):
            migrate(data)  # any write leaves a v2 registry
            rc = WRITERS[args.cmd](data, args)
            if rc is not None:  # failed: abort before anything is written
                raise _Abort(rc)

        try:
            store.update_json(registry_path(), {"version": 2, "styles": []}, fn)
        except _Abort as e:
            return e.rc
        return 0
    data = load_registry()
    return {"list": cmd_list, "show": cmd_show, "check": cmd_check}[args.cmd](data, args)


if __name__ == "__main__":
    sys.exit(main())

"""Client profiles: a client holds a brand (palette, fonts, logo, taste); any style can be linked to any client.

Registry: <store home>/clients.json  = {"version": 1, "clients": [{"slug","name","dir"}]}
Profile:  <store home>/clients/<slug>/client.json

Commands:
    clients.py add [--slug S] --name "<name>" [--industry ..] [--tone ..] [--color <hex> ...] [--theme dark|light|both]
    clients.py list [--json]
    clients.py show <slug, name or alias>
    clients.py set <key> field=value [field=value ...]   (lists split on ","; fonts.heading=Heebo)
    clients.py link <client> <style> [--default]         (style must exist; --default clears other defaults)
    clients.py unlink <client> <style>
    clients.py remove <key>                              (registry entry only; the folder is kept)
    clients.py restore <slug>                            (re-register a kept folder)
    clients.py palette <client> [--pick monochrome|analogous|accent]
                                                         (3 proposals from the first brand colour; --pick saves one)
"""
import argparse
import datetime as dt
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import store  # noqa: E402
import palette  # noqa: E402
from palette import parse_hex  # noqa: E402

THEMES = ("dark", "light", "both")
DEFAULT_FONTS = {"heading": "Karantina 700", "body": "Heebo", "mono": "Consolas"}
LIST_FIELDS = {"aliases", "languages", "logo", "brand_colors", "dos", "donts", "assets"}
TEXT_FIELDS = {"name", "industry", "audience", "tone", "notes"}
FONT_KEYS = ("heading", "body", "mono")
SLUG_RE = r"[a-z0-9][a-z0-9-]*"
SLUG_MSG = "slug must be lowercase latin letters, digits and dashes"


class _Abort(Exception):
    def __init__(self, rc):
        self.rc = rc


def _fail(msg):
    print(msg, file=sys.stderr)
    return 1


def registry_path():
    return store.path("clients.json")


def profile_path(slug):
    return store.path("clients", slug, "client.json")


def _empty():
    return {"version": 1, "clients": []}


def today():
    return dt.date.today().isoformat()


def load_registry():
    return store.read_json(registry_path(), _empty())


def _find_entry(reg, key):
    """Registry entry by slug, then name, then alias (each across all clients), case-insensitive."""
    k = key.strip().lower()
    if not k:
        return None
    for e in reg["clients"]:
        if k == e["slug"].lower():
            return e
    for e in reg["clients"]:
        if k == (e.get("name") or "").strip().lower():
            return e
    for e in reg["clients"]:
        c = store.read_json(profile_path(e["slug"]), None)
        if c and k in (a.strip().lower() for a in c.get("aliases", [])):
            return e
    return None


def load_client(key):
    e = _find_entry(load_registry(), key)
    return store.read_json(profile_path(e["slug"]), None) if e else None


def save_client(c):
    c["updated"] = today()
    store.write_json(profile_path(c["slug"]), c)


def all_links():
    """(client name, style slug) for every style link of every registered client."""
    out = []
    for e in load_registry()["clients"]:
        c = store.read_json(profile_path(e["slug"]), None) or {}
        for s in c.get("styles", []):
            out.append((c.get("name") or e["slug"], s["slug"]))
    return out


def linked_clients(style_slug):
    out = []
    for e in load_registry()["clients"]:
        c = store.read_json(profile_path(e["slug"]), None)
        for s in (c or {}).get("styles", []):
            if s["slug"] == style_slug:
                out.append({"slug": c["slug"], "name": c.get("name", ""), "default": bool(s.get("default"))})
    return out


def _styles():
    import styles
    return styles


# ---- writers run inside the registry lock; return None on success or an exit code (nothing is written then).

def cmd_add(reg, args):
    args.name = args.name.strip()
    if not args.name:
        return _fail("name must not be empty")
    try:
        colors = [parse_hex(c) for c in args.color]
    except ValueError as e:
        return _fail(str(e))
    taken = {e["slug"] for e in reg["clients"]}
    folders = store.path("clients")
    kept = {d.name for d in folders.iterdir() if d.is_dir()} if folders.is_dir() else set()
    slug = args.slug or store.slugify(args.name, taken | kept)
    if not re.fullmatch(SLUG_RE, slug):
        return _fail(SLUG_MSG)
    if _find_entry(reg, slug) or _find_entry(reg, args.name):
        return _fail(f"a client named {slug!r} or {args.name!r} already exists")
    if profile_path(slug).exists():
        return _fail(f"a client folder already exists at {profile_path(slug).parent}; "
                     f"restore it with `clients.py restore {slug}` or pick another --slug")
    c = {"slug": slug, "name": args.name, "aliases": [], "industry": args.industry or "", "audience": "",
         "tone": args.tone or "", "languages": [], "logo": [], "fonts": dict(DEFAULT_FONTS),
         "brand_colors": colors, "palette": None, "theme": args.theme, "styles": [], "dos": [],
         "donts": [], "assets": [], "notes": "", "created": today(), "updated": today()}
    save_client(c)
    reg["clients"].append({"slug": slug, "name": args.name, "dir": f"clients/{slug}"})
    print(f"added {slug} ({args.name})")
    base = store.slugify(args.name, set())
    if not args.slug and slug != base and base in kept and not _find_entry(reg, base):
        print(f"note: a kept folder clients/{base} exists; `clients.py restore {base}` brings it back")


def _get(reg, key):
    e = _find_entry(reg, key)
    c = store.read_json(profile_path(e["slug"]), None) if e else None
    if not c:
        print(f"no client {key!r}", file=sys.stderr)
    return e, c


def cmd_set(reg, args):
    e, c = _get(reg, args.key)
    if not c:
        return 1
    old_first = (c.get("brand_colors") or [None])[0]
    for pair in args.pairs:
        if "=" not in pair:
            return _fail(f"expected field=value, got {pair!r}")
        k, v = pair.split("=", 1)
        items = [a.strip() for a in v.split(",") if a.strip()]
        if k.startswith("fonts.") and k[6:] in FONT_KEYS:
            c["fonts"][k[6:]] = v
        elif k == "theme":
            if v not in THEMES:
                return _fail("theme must be dark, light or both")
            c["theme"] = v
        elif k == "brand_colors":
            try:
                c[k] = [parse_hex(x) for x in items]
            except ValueError as ex:
                return _fail(str(ex))
        elif k == "aliases":
            for a in items:
                o = _find_entry(reg, a)
                if o and o["slug"] != c["slug"]:
                    return _fail(f"alias {a!r} is already the slug, name or alias of client {o['slug']!r}")
            c[k] = items
        elif k in LIST_FIELDS:
            c[k] = items
        elif k in TEXT_FIELDS:
            if k == "name":
                v = v.strip()
                if not v:
                    return _fail("name must not be empty")
                o = _find_entry(reg, v)
                if o and o["slug"] != c["slug"]:
                    return _fail(f"another client is already called {v!r}")
            c[k] = v
        else:
            ok = sorted(TEXT_FIELDS | LIST_FIELDS | {"theme"} | {f"fonts.{f}" for f in FONT_KEYS})
            return _fail(f"{k!r} is not editable; use one of {ok}")
    save_client(c)
    e["name"] = c["name"]
    print(f"updated {c['slug']}")
    pal = c.get("palette") or {}
    first = (c.get("brand_colors") or [None])[0]
    if pal and first != old_first and first != pal.get("source_hex"):
        kind = pal.get("kind", "<kind>")
        print(f"note: the saved {kind} palette was made from {pal.get('source_hex')}; rerun "
              f"`clients.py palette {c['slug']} --pick {kind}` to rebuild it from the new first brand colour")


def cmd_link(reg, args):
    e, c = _get(reg, args.client)
    if not c:
        return 1
    s = _styles().find(_styles().load_registry(), args.style)
    if not s:
        return _fail(f"no style {args.style!r}")
    if args.default:
        for x in c["styles"]:
            x["default"] = False
    link = next((x for x in c["styles"] if x["slug"] == s["slug"]), None)
    if link is None:
        link = {"slug": s["slug"], "default": False}
        c["styles"].append(link)
    if args.default:
        link["default"] = True
    save_client(c)
    print(f"linked {c['slug']} -> {s['slug']}" + (" (default)" if link["default"] else ""))


def cmd_unlink(reg, args):
    e, c = _get(reg, args.client)
    if not c:
        return 1
    s = _styles().find(_styles().load_registry(), args.style)
    slug = s["slug"] if s else args.style  # a removed style can still be unlinked by slug
    keep = [x for x in c["styles"] if x["slug"] != slug]
    if len(keep) == len(c["styles"]):
        return _fail(f"{c['slug']} is not linked to {args.style!r}")
    c["styles"] = keep
    save_client(c)
    print(f"unlinked {c['slug']} from {slug}")


def cmd_remove(reg, args):
    e = _find_entry(reg, args.key)
    if not e:
        return _fail(f"no client {args.key!r}")
    reg["clients"].remove(e)
    print(f"removed {e['slug']} from the registry (folder kept: {store.path(e['dir'])})")


def cmd_restore(reg, args):
    slug = args.slug
    if not re.fullmatch(SLUG_RE, slug):
        return _fail(SLUG_MSG)
    if _find_entry(reg, slug):
        return _fail(f"{slug!r} is already registered")
    c = store.read_json(profile_path(slug), None)
    if not c:
        return _fail(f"no client folder for {slug!r} at {profile_path(slug).parent}")
    if c.get("slug") != slug:  # a copied or renamed folder: the folder name is the truth
        print(f"note: the profile said slug {c.get('slug')!r}; rewritten to the folder name {slug!r}")
        c["slug"] = slug
        save_client(c)
    name = c.get("name") or slug
    reg["clients"].append({"slug": slug, "name": name, "dir": f"clients/{slug}"})
    print(f"restored {slug} ({name})")
    twin = next((x for x in reg["clients"] if x["slug"] != slug
                 and (x.get("name") or "").strip().lower() == name.strip().lower()), None)
    if twin:
        print(f"note: client {twin['slug']!r} has the same name; rename one with "
              f"`clients.py set {slug} name=<new name>`")


def cmd_list(args):
    rows = load_registry()["clients"]
    if args.json:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
        return 0
    if not rows:
        print("no clients yet")
    for e in rows:
        c = store.read_json(profile_path(e["slug"]), {}) or {}
        linked = ", ".join(s["slug"] for s in c.get("styles", []))
        print(f"{e['slug']}\t{e['name']}\t{c.get('industry', '')}\tstyles:{linked}")
    return 0


def cmd_show(args):
    c = load_client(args.key)
    if not c:
        return _fail(f"no client {args.key!r}")
    known = _styles().load_registry()
    c = dict(c)
    c["styles"] = [
        s if _styles().find(known, s["slug"]) else {**s, "note": "(missing style)"} for s in c["styles"]]
    print(json.dumps(c, ensure_ascii=False, indent=2))
    return 0


def cmd_palette(args):
    c = load_client(args.client)
    if not c:
        return _fail(f"no client {args.client!r}")
    if not c.get("brand_colors"):
        return _fail(f"{c['slug']} has no brand colour; set one with "
                     f"`clients.py set {c['slug']} brand_colors=#RRGGBB`")
    src = c["brand_colors"][0]
    if args.pick:
        return palette.save(c["slug"], src, args.pick)
    print(f"{c['slug']}: proposals from {src}; save one with `clients.py palette {c['slug']} --pick <kind>`")
    print()
    print(palette.format_proposals(palette.proposals(src)))
    return 0


WRITERS = {"add": cmd_add, "set": cmd_set, "link": cmd_link, "unlink": cmd_unlink, "remove": cmd_remove,
           "restore": cmd_restore}


def main(argv=None):
    return store.exit_on_lock_timeout(_main, argv)


def _main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("add")
    a.add_argument("--slug"); a.add_argument("--name", required=True); a.add_argument("--industry")
    a.add_argument("--tone"); a.add_argument("--color", action="append", default=[])
    a.add_argument("--theme", choices=THEMES, default="dark")
    a = sub.add_parser("list"); a.add_argument("--json", action="store_true")
    a = sub.add_parser("show"); a.add_argument("key")
    a = sub.add_parser("set"); a.add_argument("key"); a.add_argument("pairs", nargs="+")
    a = sub.add_parser("link"); a.add_argument("client"); a.add_argument("style"); a.add_argument("--default", action="store_true")
    a = sub.add_parser("unlink"); a.add_argument("client"); a.add_argument("style")
    a = sub.add_parser("remove"); a.add_argument("key")
    a = sub.add_parser("restore"); a.add_argument("slug")
    a = sub.add_parser("palette"); a.add_argument("client")
    a.add_argument("--pick", choices=("monochrome", "analogous", "accent"))
    args = p.parse_args(argv)

    store.print_utf8()
    if args.cmd == "list":
        return cmd_list(args)
    if args.cmd == "show":
        return cmd_show(args)
    if args.cmd == "palette":
        return cmd_palette(args)

    def fn(reg):
        rc = WRITERS[args.cmd](reg, args)
        if rc is not None:
            raise _Abort(rc)

    try:
        store.update_json(registry_path(), _empty(), fn)
    except _Abort as e:
        return e.rc
    return 0


if __name__ == "__main__":
    sys.exit(main())

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
    styles.py set-status <slug> <training|ready|archived> [--force --reason "<text>"]
                  [--waive-recipes --reason "<text>"]
                  (ready needs <pack>/proof.json with recreation.passed and holdout.passed, and no
                   recipe in <pack>/recipes.json whose `how` starts with ai_ and whose test.status is
                   not passed; --force logs ready_forced, --waive-recipes logs recipes_waived)
    styles.py set-cover <slug> <image>
                  (the owner's pick from cover.py candidates: writes <pack>/cover.jpg, 1080 px long side,
                   JPEG q88, and stores `cover` relative to the pack; a style without a pick has no cover)
    styles.py remove <slug>          (removes the registry entry only, never files)
    styles.py check                  (every pack and skill path still exists)
"""
import argparse
import datetime as dt
import hashlib
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
V2_DEFAULTS = {"kind": "trained", "aspect": "9:16", "name_source": "user", "version": 1, "rules_count": 0,
               "requires": []}
LEVELS = ("required", "recommended")
_REG_OVERRIDE = None  # set from --registry


class _Abort(Exception):
    def __init__(self, rc):
        self.rc = rc


def registry_path():
    arg = _REG_OVERRIDE or os.environ.get("STYLE_LAB_REGISTRY")
    return Path(arg) if arg else store.path("styles.json")


def _backup_once(path):
    """Copy path to <name>.bak-YYYYMMDD unless today's backup exists. Returns the backup path or None."""
    bak = path.with_name(f"{path.name}.bak-{dt.date.today():%Y%m%d}")
    if bak.exists():
        return None
    shutil.copy2(path, bak)
    return bak


def migrate(data):
    """v1 -> v2 in place. Returns (data, changed)."""
    changed = False
    if data.get("version", 1) < 2:
        data["version"] = 2
        changed = True
    for s in data.setdefault("styles", []):
        for k, v in V2_DEFAULTS.items():
            if k not in s:
                s[k] = json.loads(json.dumps(v))  # fresh list per style
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
    """Match by slug, then display name, then alias (each across all styles), case-insensitive."""
    k = key.strip().lower()
    if not k:
        return None
    for names in (lambda s: [s["slug"]], lambda s: [s.get("name") or ""], lambda s: s.get("aliases", [])):
        for s in data["styles"]:
            if k in (n.strip().lower() for n in names(s) if n):
                return s
    return None


def _alias_clash(data, s, aliases):
    """(alias, other slug) for the first alias that is another style's slug, name or alias, else None."""
    for a in aliases:
        other = find(data, a)
        if other is not None and other is not s:
            return a, other["slug"]
    return None


def dangling_links(data):
    """(client name, style slug) for every client link to a style missing from the registry."""
    try:
        import clients
    except ModuleNotFoundError as e:
        if e.name != "clients":
            raise
        return []
    known = {s["slug"] for s in data["styles"]}
    try:
        links = clients.all_links()
    except Exception as e:  # e.g. corrupt clients.json: report nothing, do not crash
        print(f"warning: could not read client links: {e}", file=sys.stderr)
        return []
    return [(name, slug) for name, slug in links if slug not in known]


def _dangling_lines(data):
    return [f"dangling: client {name} -> style {slug} (missing style)" for name, slug in dangling_links(data)]


def clients_for(slug):
    try:
        import clients
    except ModuleNotFoundError as e:
        if e.name != "clients":
            raise
        return []
    try:
        return clients.linked_clients(slug)
    except Exception as e:  # e.g. corrupt clients.json: show no clients, do not crash
        print(f"warning: could not read clients for {slug}: {e}", file=sys.stderr)
        return []


def _client_names(slug):
    return [c.get("name") or c.get("slug", "") for c in clients_for(slug)]


def cmd_list(data, args):
    rows = [s for s in data["styles"] if args.all or s.get("status") != "archived"]
    if args.json:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
    elif not rows:
        print("no styles yet")
    else:
        for s in rows:
            print(f"{s['slug']}\t{s.get('name','')}\t{s.get('status','')}\t{s.get('description','')}"
                  f"\tclients:{', '.join(_client_names(s['slug']))}\tcover:{cover_path(s)}")
    for line in _dangling_lines(data):  # stderr keeps the rows (and --json) clean for parsing
        print(line, file=sys.stderr)
    return 0


def cmd_add(data, args):
    args.name = args.name.strip()
    if not args.name:
        print("name must not be empty", file=sys.stderr)
        return 1
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
        "requires": [],
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
    print(json.dumps({**s, "cover_path": cover_path(s), "clients": clients_for(s["slug"])},
                     ensure_ascii=False, indent=2))
    return 0


def cmd_require(data, args):
    s = find(data, args.style)
    if not s:
        print(f"no style {args.style!r}", file=sys.stderr)
        return 1
    name = args.name.strip()
    if not name or not args.why.strip():
        print("--name and --why must not be empty", file=sys.stderr)
        return 1
    entry = {"name": name, "why": args.why.strip(), "level": args.level, "fallback": (args.fallback or "").strip()}
    req = s.setdefault("requires", [])
    req[:] = [r for r in req if r.get("name") != name] + [entry]  # same name: replaced
    s["updated"] = today()
    print(f"{s['slug']} {args.level}: {name}")
    return None


def cmd_unrequire(data, args):
    s = find(data, args.style)
    if not s:
        print(f"no style {args.style!r}", file=sys.stderr)
        return 1
    before = len(s.get("requires", []))
    s["requires"] = [r for r in s.get("requires", []) if r.get("name") != args.name.strip()]
    if len(s["requires"]) == before:
        print(f"{s['slug']} does not require {args.name!r}", file=sys.stderr)
        return 1
    s["updated"] = today()
    print(f"{s['slug']}: removed {args.name}")
    return None


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
        if k == "status" and v == "ready":
            rc, _ = ready_gate(s)  # no override through `set`: use set-status --force
            if rc is not None:
                return rc
            s.pop("ready_forced", None)
            s.pop("recipes_waived", None)
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
            v = v.strip()
            if not v:
                print("name must not be empty", file=sys.stderr)
                return 1
            other = find(data, v)
            if other and other is not s:
                print(f"another style is already called {v!r}", file=sys.stderr)
                return 1
        if k == "aliases":
            v = [a.strip() for a in v.split(",") if a.strip()]
            clash = _alias_clash(data, s, v)
            if clash:
                print(f"alias {clash[0]!r} is already the slug, name or alias of style {clash[1]!r}",
                      file=sys.stderr)
                return 1
        s[k] = v
    s["updated"] = today()
    print(f"updated {s['slug']}")
    return None


def _read_pack_json(pack, name):
    """(data, error). data is None when the file is absent; error is a message when unreadable."""
    p = Path(pack) / name
    if not p.exists():
        return None, None
    try:
        return json.loads(p.read_text(encoding="utf-8")), None
    except (OSError, ValueError) as e:
        return None, f"{name} unreadable: {e}"


def trained_hash(trained_on):
    """sha256 of the sorted training ids; stored in proof.holdout so a changed fingerprint stales the proof."""
    return hashlib.sha256("\n".join(sorted(str(x) for x in trained_on or [])).encode("utf-8")).hexdigest()


def proof_problems(pack):
    """What is missing for `ready`: a list of short strings (empty = the proof passed)."""
    if not str(pack or "").strip():
        return ["the style has no pack folder"]
    proof, err = _read_pack_json(pack, "proof.json")
    if err:
        return [err]
    if not isinstance(proof, dict):
        return ["no proof.json (run prove.py holdout and prove.py recreation)"]
    out = []
    for key in ("recreation", "holdout"):
        part = proof.get(key)
        if not (isinstance(part, dict) and part.get("passed") is True):
            out.append(f"{key} test not passed")
    hold = proof.get("holdout")
    if isinstance(hold, dict) and hold.get("passed") is True:
        fp, ferr = _read_pack_json(pack, "fingerprint.json")
        trained = fp.get("trained_on") if isinstance(fp, dict) else None
        if ferr:
            out.append(ferr)
        elif hold.get("trained_unknown"):
            out.append("hold-out was picked without a fingerprint (training ids unknown)")
        else:
            ov = sorted(set(hold.get("ids") or []) & set(trained or []))
            if ov:
                out.append("hold-out overlaps training: " + ", ".join(ov))
            elif hold.get("trained_hash") != trained_hash(trained or []):
                out.append("proof is stale: re-run prove.py holdout")
    return out


def untested_ai_recipes(pack):
    """(names of ai_ recipes whose test.status is not passed, error message or None)."""
    data, err = _read_pack_json(pack, "recipes.json")
    if err:
        return [], err
    items = data.get("recipes", []) if isinstance(data, dict) else (data or [])
    bad = []
    for r in items if isinstance(items, list) else []:
        if not isinstance(r, dict) or not str(r.get("how", "")).startswith("ai_"):
            continue
        test = r.get("test")
        if not (isinstance(test, dict) and test.get("status") == "passed"):
            bad.append(str(r.get("kind") or r.get("how")))
    return bad, None


def ready_gate(s, force=False, waive=False, reason=""):
    """Check a style may become `ready`. Returns (rc or None, log) where log is merged into the entry."""
    reason = (reason or "").strip()
    if (force or waive) and not reason:
        print("--force and --waive-recipes need --reason \"<text>\"", file=sys.stderr)
        return 1, {}
    log = {}
    problems = proof_problems(s.get("pack", ""))
    if problems:
        if not force:
            print(f"{s['slug']} cannot be ready: {'; '.join(problems)}. "
                  f"Pass the tests, or --force --reason \"<text>\".", file=sys.stderr)
            return 1, {}
        log["ready_forced"] = {"reason": reason, "date": today(), "missing": problems}
    bad, err = untested_ai_recipes(s.get("pack", "")) if str(s.get("pack") or "").strip() else ([], None)
    if err:
        bad = [err]
    if bad:
        if not waive:
            print(f"{s['slug']} cannot be ready: untested AI recipes: {', '.join(bad)}. "
                  f"Test them (recipes.json test.status passed), or --waive-recipes --reason \"<text>\".",
                  file=sys.stderr)
            return 1, {}
        log["recipes_waived"] = {"reason": reason, "date": today(), "recipes": bad}
    return None, log


def cmd_set_status(data, args):
    s = find(data, args.slug)
    if not s:
        print(f"no style {args.slug!r}", file=sys.stderr)
        return 1
    if args.status == "ready":
        rc, log = ready_gate(s, args.force, args.waive_recipes, args.reason)
        if rc is not None:
            return rc
        s.pop("ready_forced", None)  # a clean ready clears old overrides
        s.pop("recipes_waived", None)
        s.update(log)
    s["status"] = args.status
    s["updated"] = today()
    print(f"{s['slug']}: status {args.status}" + (" (forced)" if "ready_forced" in s else ""))
    return None


COVER_LONG_SIDE = 1080
COVER_QUALITY = 88


def _write_cover(src, dst):
    """src image -> dst JPEG, long side COVER_LONG_SIDE, quality COVER_QUALITY. Returns an error text or None."""
    tmp = dst.with_name(dst.name + ".tmp.jpg")
    try:
        try:
            from PIL import Image
        except ImportError:
            Image = None
        if Image is not None:
            with Image.open(src) as im:
                im = im.convert("RGB")
                k = COVER_LONG_SIDE / max(im.size)
                im = im.resize((max(1, round(im.width * k)), max(1, round(im.height * k))), Image.LANCZOS)
                im.save(tmp, "JPEG", quality=COVER_QUALITY)
        else:
            import subprocess
            vf = (f"scale='if(gt(iw,ih),{COVER_LONG_SIDE},-2)':'if(gt(iw,ih),-2,{COVER_LONG_SIDE})'")
            r = subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-vf", vf, "-q:v", "2", str(tmp)],
                               capture_output=True, text=True)
            if r.returncode != 0:
                return "could not convert the image (install Pillow: pip install pillow)"
        os.replace(tmp, dst)
    except OSError as e:
        return f"could not write the cover: {e}"
    except Exception as e:  # PIL: unreadable or unsupported image
        return f"could not read the image: {e}"
    finally:
        if tmp.exists():
            tmp.unlink()
    return None


def cover_path(s):
    return str(Path(s["pack"]) / s["cover"]) if s.get("cover") and s.get("pack") else ""


def cmd_set_cover(data, args):
    s = find(data, args.slug)
    if not s:
        print(f"no style {args.slug!r}", file=sys.stderr)
        return 1
    src = Path(args.image)
    if not src.is_file():
        print(f"image not found: {src}", file=sys.stderr)
        return 1
    pack = Path(s.get("pack") or "")
    if not str(s.get("pack") or "").strip() or not pack.is_dir():
        print(f"the pack folder does not exist: {s.get('pack')}", file=sys.stderr)
        return 1
    err = _write_cover(src, pack / "cover.jpg")
    if err:
        print(err, file=sys.stderr)
        return 1
    s["cover"] = "cover.jpg"
    s["updated"] = today()
    print(f"{s['slug']}: cover {pack / 'cover.jpg'}")
    return None


def cmd_remove(data, args):
    s = find(data, args.slug)
    if not s:
        print(f"no style {args.slug!r}", file=sys.stderr)
        return 1
    data["styles"].remove(s)
    print(f"removed {s['slug']} from the registry (files untouched: {s['pack']})")
    n = len(clients_for(s["slug"]))
    if n:
        print(f"{n} client(s) still link to it (shown as missing style; "
              f"`clients.py unlink <client> {s['slug']}` removes a link)")
    return None


def cmd_check(data, args):
    bad = 0
    for s in data["styles"]:
        for key in ("pack", "skill"):
            if s.get(key) and not Path(s[key]).exists():
                level = "missing" if s.get("status") == "ready" else "not yet"
                print(f"{s['slug']}: {key} {level}: {s[key]}")
                bad += level == "missing"
    dangling = _dangling_lines(data)
    for line in dangling:
        print(line)
    if not bad and not dangling:
        print("ok")
    if bad:
        print(f"{bad} missing path(s) on ready styles")
    if dangling:
        print(f"{len(dangling)} dangling client link(s)")
    return 1 if bad else 0  # a dangling link is reported, never fatal


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
        bak = _backup_once(path)
        if bak:
            state["bak"] = bak

    try:
        store.update_json(path, {"version": 2, "styles": []}, fn)
    except _Abort:
        print("already v2")
        return 0
    print(f"migrated to v2 ({path})" + (f", backup {state['bak'].name}" if "bak" in state else ""))
    return 0


WRITERS = {"add": cmd_add, "set": cmd_set, "remove": cmd_remove, "require": cmd_require,
           "unrequire": cmd_unrequire, "set-cover": cmd_set_cover,
           "set-status": cmd_set_status}


def main(argv=None):
    return store.exit_on_lock_timeout(_main, argv)


def _main(argv=None):
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
    a = sub.add_parser("set-status", help="change status; ready is gated by proof.json and recipes.json")
    a.add_argument("slug"); a.add_argument("status", choices=STATUSES)
    a.add_argument("--force", action="store_true"); a.add_argument("--waive-recipes", action="store_true")
    a.add_argument("--reason", default="")
    a = sub.add_parser("set-cover", help="store the owner's chosen cover frame for a style")
    a.add_argument("slug"); a.add_argument("image")
    a = sub.add_parser("remove"); a.add_argument("slug")
    a = sub.add_parser("require", help="record a connection the style depends on")
    a.add_argument("style"); a.add_argument("--name", required=True); a.add_argument("--why", required=True)
    a.add_argument("--level", choices=LEVELS, required=True); a.add_argument("--fallback")
    a = sub.add_parser("unrequire"); a.add_argument("style"); a.add_argument("--name", required=True)
    sub.add_parser("check")
    sub.add_parser("migrate")
    args = p.parse_args(argv)

    store.print_utf8()
    _REG_OVERRIDE = args.registry
    if args.cmd == "migrate":
        return cmd_migrate(args)
    if args.cmd in WRITERS:

        def fn(data):
            if migrate(data)[1] and registry_path().exists():  # any write leaves a v2 registry
                _backup_once(registry_path())
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

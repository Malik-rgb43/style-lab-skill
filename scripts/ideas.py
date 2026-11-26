#!/usr/bin/env python3
"""Ideas bank: B-roll and text-animation beats kept as reusable snippets with parameters.

Store: <store home>/ideas/index.json = {"version": 1, "next_id": n, "ideas": [Idea]}
       <store home>/ideas/<id>/  meta.json, still.png, clip.mp4, snippet.html
       <store home>/ideas/_removed/<id>/   (removed ideas are archived, never deleted)
A "style-link" idea points at files inside a style pack (nothing is copied).

A snippet never carries a client's name, alias, logo file name or brand/palette hex: those are
parameters ({{name}}) that `render` fills from the client's tokens (or assets/brand-defaults.json).

Commands:
    ideas.py add <title> --kind K --beat-type T... --aspect A --snippet FILE [--params FILE.json]
                 [--still FILE] [--clip FILE] [--client C] [--style S] [--project P]
                 [--source final|studio] [--tag T]...
    ideas.py list [--json]
    ideas.py search [--kind K] [--beat-type T]... [--tag T]... [--client C] [--style S] [--aspect A]
                    [--avoid-client C] [--limit N] [--json]
    ideas.py show <id>
    ideas.py render <id> [--client C] [--param k=v]... --out FILE
    ideas.py remove <id>
Exit codes: 0 ok, 1 refused, 2 bad input, 3 missing tool.
"""
import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import store  # noqa: E402

KINDS = ("b-roll", "text-animation", "transition", "caption", "cta", "camera")
SOURCES = ("final", "style-link", "studio")
PARAM_TYPES = ("text", "color", "image", "number", "font")
FONT_ROLES = ("heading", "body", "mono")
CLIP_SECONDS = 4.0
CLIP_SHORT_SIDE = 720
DEFAULTS_FILE = Path(__file__).resolve().parent.parent / "assets" / "brand-defaults.json"
_PLACEHOLDER = re.compile(r"\{\{\s*([A-Za-z0-9_\-]+)\s*\}\}")
_HEX = re.compile(r"#(?:[0-9a-fA-F]{6}|[0-9a-fA-F]{3})(?![0-9a-fA-F])")


class MissingTool(Exception):
    pass


class Refused(ValueError):
    """The snippet would carry a client's identity; exit code 1."""


def _index_path():
    return store.path("ideas", "index.json")


def _empty():
    return {"version": 1, "next_id": 1, "ideas": []}


def _rel(p):
    return Path(p).relative_to(store.home()).as_posix()


def _abs(rel):
    return store.home() / rel


def _norm_hex(h):
    h = h.lower()
    return "#" + "".join(c * 2 for c in h[1:]) if len(h) == 4 else h


def load_index():
    return store.read_json(_index_path(), _empty())


def get(idea_id):
    for i in load_index()["ideas"]:
        if i["id"] == idea_id:
            return i
    raise KeyError(idea_id)


# ---- client-safety check

def _client_literals(client):
    """(names, logo file names, hexes) that must not appear in a snippet made for this client."""
    names = [n for n in [client.get("name"), client.get("slug"), *client.get("aliases", [])] if n]
    logos = [Path(str(x)).name for x in client.get("logo", []) if x]
    hexes = {_norm_hex(h) for h in client.get("brand_colors", []) if h}
    pal = client.get("palette") or {}
    for theme in (pal.get("roles") or {}).values():
        hexes.update(_norm_hex(v) for v in theme.values() if isinstance(v, str) and v.startswith("#"))
    for k in ("brand", "accent", "source_hex"):
        if isinstance(pal.get(k), str) and pal[k].startswith("#"):
            hexes.add(_norm_hex(pal[k]))
    return [n.strip() for n in names if len(n.strip()) >= 2], [x for x in logos if x], hexes


def check_snippet(text, client):
    """Raise Refused when the snippet holds the client's name, alias, logo file name or a brand hex."""
    names, logos, hexes = _client_literals(client)
    low = text.lower()
    for n in names:
        if n.lower() in low:
            raise Refused(f"the snippet contains the client name {n!r}: make it a parameter")
    for f in logos:
        if f.lower() in low:
            raise Refused(f"the snippet contains the logo file name {f!r}: make it a parameter")
    for m in _HEX.findall(text):
        if _norm_hex(m) in hexes:
            raise Refused(f"the snippet contains the brand colour {m}: make it a parameter "
                          f"(a {{{{brand}}}} placeholder with role brand)")


# ---- params

def _check_params(params):
    if not isinstance(params, dict):
        raise ValueError("params must be an object {name: {type, default, client_bound, role?}}")
    import palette
    out = {}
    for name, p in params.items():
        if not re.fullmatch(r"[A-Za-z0-9_\-]+", name or ""):
            raise ValueError(f"bad param name {name!r}")
        if not isinstance(p, dict) or p.get("type") not in PARAM_TYPES:
            raise ValueError(f"param {name!r}: type must be one of {', '.join(PARAM_TYPES)}")
        role = p.get("role")
        if role is not None:
            ok = palette.ROLE_KEYS if p["type"] == "color" else FONT_ROLES if p["type"] == "font" else ()
            if role not in ok:
                raise ValueError(f"param {name!r}: role {role!r} is not valid for type {p['type']}")
        q = {"type": p["type"], "default": p.get("default"), "client_bound": bool(p.get("client_bound"))}
        if role is not None:
            q["role"] = role
        out[name] = q
    return out


# ---- clip

def _make_clip(src, dst):
    ff = shutil.which("ffmpeg")
    if not ff:
        raise MissingTool("ffmpeg is missing: install it (winget install Gyan.FFmpeg, or brew install ffmpeg)")
    vf = (f"scale=w='if(gt(iw,ih),-2,min({CLIP_SHORT_SIDE},iw))':"
          f"h='if(gt(iw,ih),min({CLIP_SHORT_SIDE},ih),-2)'")
    r = subprocess.run([ff, "-y", "-v", "error", "-i", str(src), "-t", str(CLIP_SECONDS), "-vf", vf,
                        "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(dst)],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0 or not Path(dst).exists():
        raise ValueError(f"ffmpeg could not read the clip {src}: {r.stderr.strip()[:200]}")


# ---- core

def _folder(idea_id, removed=False):
    return store.path("ideas", "_removed", idea_id) if removed else store.path("ideas", idea_id)


def _write_meta(idea):
    d = _folder(idea["id"], idea.get("removed"))
    if d.is_dir():
        store.write_json(d / "meta.json", idea)


def add(title, kind, beat_types, aspect, snippet_file, params, *, still=None, clip=None, made_for=None,
        source="studio", tags=(), link=None):
    if not str(title or "").strip():
        raise ValueError("title must not be empty")
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {', '.join(KINDS)}")
    if link:
        source = "style-link"
    if source not in SOURCES:
        raise ValueError(f"source must be one of {', '.join(SOURCES)}")
    if source == "style-link" and not link:
        raise ValueError("a style-link idea needs link={style, entry, path}")
    made = {"style": None, "client": None, "project": None, **(made_for or {})}
    params = _check_params(params or {})

    tmp = None
    if source == "style-link":
        paths = {"still": still, "clip": clip, "snippet": str(snippet_file) if snippet_file else None}
    else:
        sp = Path(snippet_file) if snippet_file else None
        if not sp or not sp.is_file():
            raise ValueError(f"snippet file not found: {snippet_file}")
        text = sp.read_text(encoding="utf-8")
        if made["client"]:
            import clients
            c = clients.load_client(made["client"])
            if c:
                made["client"] = c["slug"]
                check_snippet(text, c)
        ideas_dir = store.path("ideas")
        ideas_dir.mkdir(parents=True, exist_ok=True)
        tmp = Path(tempfile.mkdtemp(prefix="_new-", dir=str(ideas_dir)))
        try:
            shutil.copyfile(sp, tmp / "snippet.html")
            if still:
                shutil.copyfile(still, tmp / ("still" + (Path(still).suffix.lower() or ".png")))
            if clip:
                _make_clip(clip, tmp / "clip.mp4")
        except BaseException:
            shutil.rmtree(tmp, ignore_errors=True)
            raise

    box = {}

    def fn(idx):
        n = idx["next_id"]
        idx["next_id"] = n + 1
        iid = f"I{n:04d}"
        dest = _folder(iid)
        if tmp is not None:
            store._retry_perm(lambda: shutil.move(str(tmp), str(dest)))
            files = {f.stem: f for f in dest.iterdir() if f.is_file()}
            p = {k: _rel(files[k]) if k in files else None for k in ("still", "clip", "snippet")}
        else:
            dest.mkdir(parents=True, exist_ok=True)
            p = paths
        idea = {"id": iid, "title": title.strip(), "kind": kind, "beat_types": list(beat_types),
                "aspect": aspect, **p, "params": params, "made_for": made, "source": source, "link": link,
                "approved": False, "offered": 0, "picked": 0, "tags": list(tags), "removed": False}
        store.write_json(dest / "meta.json", idea)
        idx["ideas"].append(idea)
        box["idea"] = idea

    try:
        store.update_json(_index_path(), _empty(), fn)
    except BaseException:
        if tmp is not None:
            shutil.rmtree(tmp, ignore_errors=True)
        raise
    return box["idea"]


def _matches(i, kind, beat_types, tags, client, style, aspect):
    """Number of matched tags and beat types, or None when a filter rules the idea out."""
    if i.get("removed"):
        return None
    if kind and i["kind"] != kind:
        return None
    if aspect and i.get("aspect") != aspect:
        return None
    mf = i.get("made_for") or {}
    if client and (mf.get("client") or "").lower() != client.lower():
        return None
    if style and (mf.get("style") or "").lower() != style.lower():
        return None
    have_t = {t.lower() for t in i.get("tags", [])}
    want_t = {t.lower() for t in tags}
    if not want_t <= have_t:
        return None
    have_b = {b.lower() for b in i.get("beat_types", [])}
    want_b = {b.lower() for b in beat_types}
    if want_b and not (want_b & have_b):
        return None
    return len(want_t & have_t) + len(want_b & have_b)


def search(kind=None, beat_types=(), tags=(), client=None, style=None, aspect=None,
           exclude_sigs=frozenset(), avoid_client=None, limit=10):
    """Filters are ANDed (all tags, any beat type); ranked by pick rate (picked+1)/(offered+2), then matches."""
    if client:
        import clients
        c = clients.load_client(client)
        client = c["slug"] if c else client
    rejected = set()
    if avoid_client:
        import taste
        try:
            rejected = set(taste.summary(avoid_client)["rejected_twice"])
        except taste._Bad as e:
            raise ValueError(str(e))
    scored = []
    for i in load_index()["ideas"]:
        m = _matches(i, kind, beat_types, tags, client, style, aspect)
        key = f"idea:{i['id']}"
        if m is None or key in exclude_sigs or key in rejected:
            continue
        rate = (i.get("picked", 0) + 1) / (i.get("offered", 0) + 2)
        scored.append((-rate, -m, i["id"], i))
    scored.sort(key=lambda t: t[:3])
    return [t[3] for t in scored[:limit]]


def _bump(idea_id, field):
    box = {}

    def fn(idx):
        for i in idx["ideas"]:
            if i["id"] == idea_id:
                i[field] = i.get(field, 0) + 1
                box["n"] = i[field]
                _write_meta(i)
                return
        raise KeyError(idea_id)

    store.update_json(_index_path(), _empty(), fn)
    return box["n"]


def offered(idea_id):
    return _bump(idea_id, "offered")


def picked(idea_id):
    return _bump(idea_id, "picked")


def remove(idea_id):
    """Mark removed and archive the folder under ideas/_removed/ (never a permanent delete)."""
    def fn(idx):
        for i in idx["ideas"]:
            if i["id"] != idea_id:
                continue
            src = _folder(idea_id)
            if src.is_dir():
                dest = _folder(idea_id, True)
                n = 1
                while dest.exists():
                    n += 1
                    dest = store.path("ideas", "_removed", f"{idea_id}.{n}")
                dest.parent.mkdir(parents=True, exist_ok=True)
                store._retry_perm(lambda: shutil.move(str(src), str(dest)))
                if i["source"] != "style-link":
                    for k in ("still", "clip", "snippet"):
                        if i.get(k):
                            i[k] = _rel(dest / Path(i[k]).name)
                i["removed"] = True
                store.write_json(dest / "meta.json", i)
            else:
                i["removed"] = True
            return
        raise KeyError(idea_id)

    store.update_json(_index_path(), _empty(), fn)


def brand_defaults():
    return json.loads(DEFAULTS_FILE.read_text(encoding="utf-8"))


def tokens(client, theme):
    """(roles, fonts) from the client's palette and fonts; assets/brand-defaults.json when there is none."""
    d = brand_defaults()
    fonts = dict(d["fonts"])
    roles = None
    if client:
        fonts.update({k: v for k, v in (client.get("fonts") or {}).items() if v})
        pal = client.get("palette")
        if pal and pal.get("roles"):
            roles = pal["roles"].get(theme) or next(iter(pal["roles"].values()))
    return roles or d["roles"][theme], fonts


def render(idea_id, client=None, overrides=None, theme=None):
    """The snippet with {{name}} replaced by override, client token or default (client: slug/name or dict)."""
    idea = get(idea_id)
    snip = Path(idea["snippet"]) if idea["source"] == "style-link" else _abs(idea["snippet"])
    text = snip.read_text(encoding="utf-8")
    c = None
    if client:
        if isinstance(client, str):
            import clients
            c = clients.load_client(client)
            if not c:
                raise ValueError(f"unknown client: {client}")
        else:
            c = client
    if theme is None:
        t = (c or {}).get("theme")
        theme = t if t in ("dark", "light") else "dark"
    roles, fonts = tokens(c, theme)
    ov = dict(overrides or {})
    values = {}
    for name, p in idea["params"].items():
        v = ov.get(name)
        if v is None or v == "":
            role = p.get("role")
            if p["type"] == "color" and role:
                v = roles[role]
            elif p["type"] == "font" and role:
                v = fonts[role]
            else:
                v = p.get("default")
        if (v is None or v == "") and p["type"] == "image" and p.get("client_bound"):
            raise KeyError(name)
        values[name] = "" if v is None else str(v)
    values.update({k: str(v) for k, v in ov.items() if k not in values})
    return _PLACEHOLDER.sub(lambda m: values.get(m.group(1), m.group(0)), text)


# ---- CLI

def _short(i):
    mf = i.get("made_for") or {}
    return (f"{i['id']}  {i['kind']:<14} {i['aspect'] or '-':<5} "
            f"picked {i.get('picked', 0)}/{i.get('offered', 0)}  {i['title']}"
            + (f"  [{mf['style']}]" if mf.get("style") else ""))


def _print(ideas, as_json):
    if as_json:
        print(json.dumps(ideas, ensure_ascii=False, indent=2))
    else:
        for i in ideas:
            print(_short(i))
        if not ideas:
            print("no ideas")


def _bad(msg):
    print(msg, file=sys.stderr)
    return 2


def main(argv=None):
    return store.exit_on_lock_timeout(_main, argv)


def _main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("add")
    a.add_argument("title"); a.add_argument("--kind", required=True)
    a.add_argument("--beat-type", action="append", default=[]); a.add_argument("--aspect", required=True)
    a.add_argument("--snippet", required=True); a.add_argument("--params")
    a.add_argument("--still"); a.add_argument("--clip")
    a.add_argument("--client"); a.add_argument("--style"); a.add_argument("--project")
    a.add_argument("--source", default="studio", choices=("final", "studio"))
    a.add_argument("--tag", action="append", default=[])
    a = sub.add_parser("list"); a.add_argument("--json", action="store_true")
    a = sub.add_parser("search")
    a.add_argument("--kind"); a.add_argument("--beat-type", action="append", default=[])
    a.add_argument("--tag", action="append", default=[]); a.add_argument("--client"); a.add_argument("--style")
    a.add_argument("--aspect"); a.add_argument("--avoid-client"); a.add_argument("--limit", type=int, default=10)
    a.add_argument("--json", action="store_true")
    a = sub.add_parser("show"); a.add_argument("id")
    a = sub.add_parser("render")
    a.add_argument("id"); a.add_argument("--client"); a.add_argument("--param", action="append", default=[])
    a.add_argument("--out", required=True)
    a = sub.add_parser("remove"); a.add_argument("id")
    args = p.parse_args(argv)
    store.print_utf8()

    try:
        if args.cmd == "add":
            params = {}
            if args.params:
                try:
                    params = json.loads(Path(args.params).read_text(encoding="utf-8"))
                except (OSError, ValueError) as e:
                    return _bad(f"cannot read --params {args.params}: {e}")
            try:
                idea = add(args.title, args.kind, args.beat_type, args.aspect, args.snippet, params,
                           still=args.still, clip=args.clip,
                           made_for={"style": args.style, "client": args.client, "project": args.project},
                           source=args.source, tags=args.tag)
            except Refused as e:
                print(e, file=sys.stderr)
                return 1
            print(f"added {idea['id']} ({idea['title']})")
            return 0
        if args.cmd == "list":
            _print([i for i in load_index()["ideas"] if not i.get("removed")], args.json)
            return 0
        if args.cmd == "search":
            _print(search(args.kind, args.beat_type, args.tag, args.client, args.style, args.aspect,
                          avoid_client=args.avoid_client, limit=args.limit), args.json)
            return 0
        if args.cmd == "show":
            print(json.dumps(get(args.id), ensure_ascii=False, indent=2))
            return 0
        if args.cmd == "render":
            ov = {}
            for kv in args.param:
                k, sep, v = kv.partition("=")
                if not sep:
                    return _bad(f"bad --param {kv!r}: use name=value")
                ov[k] = v
            out = render(args.id, args.client, ov)
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(out, encoding="utf-8")
            print(f"wrote {args.out}")
            return 0
        remove(args.id)
        print(f"removed {args.id} (archived in ideas/_removed)")
        return 0
    except KeyError as e:
        what = e.args[0] if e.args else ""
        if args.cmd == "render" and isinstance(what, str) and not re.fullmatch(r"I\d+", what):
            return _bad(f"missing value for client-bound parameter {what!r}: pass --param {what}=<value>")
        return _bad(f"no idea {what!r}")
    except ValueError as e:
        return _bad(str(e))
    except MissingTool as e:
        print(str(e), file=sys.stderr)
        return 3


if __name__ == "__main__":
    sys.exit(main())

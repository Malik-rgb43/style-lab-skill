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
    ideas.py seed-style <style> [--categories a,b]     link a style's reusable library beats (no copies)
    ideas.py from-final <project>                     the final's beats that can become ideas (pre-ticked)
    ideas.py commit-final <project> --final <mp4> [--untick 2,5]   bank them and close the project
Exit codes: 0 ok, 1 refused, 2 bad input, 3 missing tool.
"""
import argparse
import colorsys
import json
import time
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
_HEX = re.compile(r"#([0-9a-fA-F]{8}|[0-9a-fA-F]{6}|[0-9a-fA-F]{4}|[0-9a-fA-F]{3})(?![0-9a-fA-F])")
_FUNC = re.compile(r"\b(rgba?|hsla?)\(\s*([^()]*?)\s*\)", re.I)
NEUTRAL_HEX = {"#ffffff", "#000000"}
BRAND_ROLES = ("brand", "brand_deep", "accent", "highlight", "tint", "cta")
STALE_NEW_SECONDS = 3600


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


def load_index():
    return store.read_json(_index_path(), _empty())


def get(idea_id):
    for i in load_index()["ideas"]:
        if i["id"] == idea_id:
            return i
    raise KeyError(idea_id)


# ---- client-safety check

def _hex6(h):
    h = h.lower()
    if len(h) in (3, 4):
        h = "".join(c * 2 for c in h)
    return "#" + h[:6]


def _func_hex(kind, body):
    parts = re.split(r"[\s,/]+", body.strip())
    nums = []
    for p in parts:
        m = re.fullmatch(r"(-?[\d.]+)(%|deg)?", p)
        if not m:
            return None
        nums.append((float(m.group(1)), m.group(2)))
    if len(nums) < 3:
        return None
    if kind.startswith("rgb"):
        r, g, b = [max(0, min(255, round(v * 2.55) if u == "%" else round(v))) for v, u in nums[:3]]
    else:
        r, g, b = [round(x * 255) for x in colorsys.hls_to_rgb(
            (nums[0][0] % 360) / 360, max(0, min(1, nums[2][0] / 100)), max(0, min(1, nums[1][0] / 100)))]
    return f"#{r:02x}{g:02x}{b:02x}"


def colors_in(text):
    """Every colour literal in the text as lowercase #rrggbb: hex (3/4/6/8 digits), rgb(a), hsl(a)."""
    found = {_hex6(m) for m in _HEX.findall(text)}
    for kind, body in _FUNC.findall(text):
        h = _func_hex(kind.lower(), body)
        if h:
            found.add(h)
    return found


def _near(a, b, tol=2):
    """Same colour within tol per channel (hsl() round trips are not exact)."""
    return all(abs(int(a[i:i + 2], 16) - int(b[i:i + 2], 16)) <= tol for i in (1, 3, 5))


def _brand_hexes(client):
    """Brand/palette colours of a client; neutrals (background, surface, text, muted) and pure white/black are never in it."""
    vals = list(client.get("brand_colors", []))
    pal = client.get("palette") or {}
    for theme in (pal.get("roles") or {}).values():
        vals += [theme.get(r) for r in BRAND_ROLES]
    vals += [pal.get("brand"), pal.get("accent"), pal.get("source_hex")]
    out = set()
    for v in vals:
        if isinstance(v, str):
            out |= colors_in(v)
    return out - NEUTRAL_HEX


def _name_patterns(client):
    pats = []
    for n in [client.get("name"), client.get("slug"), *client.get("aliases", [])]:
        k = re.sub(r"[\s_\-]", "", n or "").lower()
        if len(k) < 3:
            continue
        body = r"[\s_\-]*".join(re.escape(c) for c in k)
        pats.append((n, re.compile(r"(?<!\w)" + body + r"(?!\w)", re.I)))
    return pats


def check_snippet(texts, client):
    """Raise Refused when any text holds the client's name, alias, logo file name or a brand colour."""
    if isinstance(texts, str):
        texts = [texts]
    hexes = _brand_hexes(client)
    logos = [Path(str(x)).name for x in client.get("logo", []) if x]
    for text in texts:
        low = text.lower()
        for n, pat in _name_patterns(client):
            if pat.search(text):
                raise Refused(f"the snippet contains the client name {n!r}: make it a parameter")
        for f in logos:
            if f.lower() in low:
                raise Refused(f"the snippet contains the logo file name {f!r}: make it a parameter")
        hit = {c for c in colors_in(text) if any(_near(c, h) for h in hexes)}
        if hit:
            raise Refused(f"the snippet contains the brand colour {sorted(hit)[0]}: make it a parameter "
                          f"(a {{{{brand}}}} placeholder with role brand)")


def check_all(texts, made_client=None):
    """Check against the made-for client (unknown = ValueError) and every other client in the store."""
    import clients
    if made_client and not clients.load_client(made_client):
        raise ValueError(f"unknown client {made_client!r}: add it with clients.py add, or leave --client out")
    for e in clients.load_registry()["clients"]:
        c = clients.load_client(e["slug"])
        if c:
            check_snippet(texts, c)


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
    vf = (f"scale=w='if(gt(iw,ih),-2,trunc(min({CLIP_SHORT_SIDE},iw)/2)*2)':"
          f"h='if(gt(iw,ih),trunc(min({CLIP_SHORT_SIDE},ih)/2)*2,-2)'")
    r = subprocess.run([ff, "-y", "-v", "error", "-i", str(src), "-t", str(CLIP_SECONDS), "-vf", vf,
                        "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(dst)],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0 or not Path(dst).exists():
        raise ValueError(f"ffmpeg could not read the clip {src}: {r.stderr.strip()[:200]}")


# ---- core

def _folder(idea_id, removed=False):
    return store.path("ideas", "_removed", idea_id) if removed else store.path("ideas", idea_id)


def _write_meta(idea):
    d = _abs(idea["archive"]) if idea.get("removed") and idea.get("archive") else _folder(idea["id"])
    if d.is_dir():
        store.write_json(d / "meta.json", idea)


def _clean_stale():
    d = store.path("ideas")
    if not d.is_dir():
        return
    for old in d.glob("_new-*"):
        try:
            if time.time() - old.stat().st_mtime > STALE_NEW_SECONDS:
                shutil.rmtree(old, ignore_errors=True)
        except OSError:
            pass


def _link_key(i):
    lk = i.get("link") or {}
    return (lk.get("style"), lk.get("entry"))


def find_by_link(style, entry):
    for i in load_index()["ideas"]:
        if not i.get("removed") and i.get("link") and _link_key(i) == (style, entry):
            return i
    return None


def find(made_for_style=None, client=None, source=None, kind=None):
    """Live ideas whose made_for style/client (and source, kind) equal the given values."""
    if client:
        import clients
        c = clients.load_client(client)
        client = c["slug"] if c else client
    out = []
    for i in load_index()["ideas"]:
        mf = i.get("made_for") or {}
        if i.get("removed"):
            continue
        if made_for_style and (mf.get("style") or "").lower() != made_for_style.lower():
            continue
        if client and (mf.get("client") or "").lower() != client.lower():
            continue
        if (source and i["source"] != source) or (kind and i["kind"] != kind):
            continue
        out.append(i)
    return out


def set_approved(idea_id, flag=True):
    def fn(idx):
        for i in idx["ideas"]:
            if i["id"] == idea_id:
                i["approved"] = bool(flag)
                _write_meta(i)
                return
        raise KeyError(idea_id)

    store.update_json(_index_path(), _empty(), fn)


def add(title, kind, beat_types, aspect, snippet_file, params, *, still=None, clip=None, made_for=None,
        source="studio", tags=(), link=None, approved=None):
    """Add an idea. A link idea (style-link) is updated in place when its (style, entry) is already banked."""
    if not str(title or "").strip():
        raise ValueError("title must not be empty")
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {', '.join(KINDS)}")
    if link:
        source = "style-link"
        if not (link.get("style") and link.get("entry")):
            raise ValueError("link needs {style, entry, path}")
    if source not in SOURCES:
        raise ValueError(f"source must be one of {', '.join(SOURCES)}")
    if source == "style-link" and not link:
        raise ValueError("a style-link idea needs link={style, entry, path}")
    made = {"style": None, "client": None, "project": None, **(made_for or {})}
    params = _check_params(params or {})
    for label, f in (("still", still), ("clip", clip)):
        if f and not Path(f).is_file():
            raise ValueError(f"{label} file not found: {f}")
    texts = [str(p["default"]) for p in params.values() if p.get("default") is not None]

    tmp = None
    sp = Path(snippet_file) if snippet_file else None
    if source == "style-link":
        if not sp and not still:
            raise ValueError("a style-link idea needs a snippet or a still")
        if sp and not sp.is_file():
            raise ValueError(f"snippet file not found: {snippet_file}")
        if sp:
            texts.append(sp.read_text(encoding="utf-8"))
        paths = {"still": str(still) if still else None, "clip": str(clip) if clip else None,
                 "snippet": str(sp) if sp else None}
    else:
        if not sp or not sp.is_file():
            raise ValueError(f"snippet file not found: {snippet_file}")
        texts.append(sp.read_text(encoding="utf-8"))
    check_all(texts, made["client"])
    if made["client"]:
        import clients
        made["client"] = clients.load_client(made["client"])["slug"]

    if source != "style-link":
        ideas_dir = store.path("ideas")
        ideas_dir.mkdir(parents=True, exist_ok=True)
        _clean_stale()
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
        fields = {"title": title.strip(), "kind": kind, "beat_types": list(beat_types), "aspect": aspect,
                  "params": params, "made_for": made, "link": link, "tags": list(tags)}
        if link:
            old = next((i for i in idx["ideas"] if not i.get("removed") and i.get("link")
                        and _link_key(i) == (link["style"], link["entry"])), None)
            if old:
                old.update(fields)
                old.update(paths)
                if approved is not None:
                    old["approved"] = bool(approved)
                _folder(old["id"]).mkdir(parents=True, exist_ok=True)
                _write_meta(old)
                box["idea"] = old
                return
        n = idx["next_id"]
        idx["next_id"] = n + 1
        iid = f"I{n:04d}"
        dest = _folder(iid)
        box["dest"] = dest
        try:
            if tmp is not None:
                store._retry_perm(lambda: shutil.move(str(tmp), str(dest)))
                files = {f.stem: f for f in dest.iterdir() if f.is_file()}
                p = {k: _rel(files[k]) if k in files else None for k in ("still", "clip", "snippet")}
            else:
                dest.mkdir(parents=True, exist_ok=True)
                p = paths
            idea = {"id": iid, **{k: fields[k] for k in ("title", "kind", "beat_types", "aspect")}, **p,
                    "params": params, "made_for": made, "source": source, "link": link,
                    "approved": bool(approved), "offered": 0, "picked": 0, "by_client": {},
                    "tags": list(tags), "removed": False}
            store.write_json(dest / "meta.json", idea)
        except BaseException:
            shutil.rmtree(dest, ignore_errors=True)
            raise
        idx["ideas"].append(idea)
        box["idea"] = idea

    try:
        store.update_json(_index_path(), _empty(), fn)
    except BaseException:
        if tmp is not None:
            shutil.rmtree(tmp, ignore_errors=True)
        if box.get("dest") is not None and not any(i["id"] == box["dest"].name for i in load_index()["ideas"]):
            shutil.rmtree(box["dest"], ignore_errors=True)
        raise
    return box["idea"]


def _matches(i, kind, beat_types, tags, client, style, aspect, include_generic=False):
    """Number of matched tags and beat types, or None when a filter rules the idea out."""
    if i.get("removed"):
        return None
    if kind and i["kind"] != kind:
        return None
    if aspect and i.get("aspect") != aspect:
        return None
    mf = i.get("made_for") or {}
    mc = (mf.get("client") or "").lower()
    if client and mc != client.lower() and not (include_generic and not mc):
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


def _rate(i, client):
    """(picked+1)/(offered+2) from the client's own counters when it has any, else the global ones."""
    src = (i.get("by_client") or {}).get(client) if client else None
    src = src or i
    return (src.get("picked", 0) + 1) / (src.get("offered", 0) + 2)


def search(kind=None, beat_types=(), tags=(), client=None, style=None, aspect=None,
           exclude_sigs=frozenset(), avoid_client=None, limit=10, include_generic=False):
    """Filters are ANDed (all tags, any beat type); ranked by pick rate, then matches.
    include_generic: with `client`, also return ideas made for no client."""
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
        m = _matches(i, kind, beat_types, tags, client, style, aspect, include_generic)
        key = f"idea:{i['id']}"
        if m is None or key in exclude_sigs or key in rejected:
            continue
        scored.append((-_rate(i, client), -m, i["id"], i))
    scored.sort(key=lambda t: t[:3])
    return [t[3] for t in scored[:limit]]


def _bump(idea_id, field, client=None):
    slug = None
    if client:
        import clients
        c = clients.load_client(client)
        slug = c["slug"] if c else client
    box = {}

    def fn(idx):
        for i in idx["ideas"]:
            if i["id"] == idea_id:
                i[field] = i.get(field, 0) + 1
                if slug:
                    bc = i.setdefault("by_client", {}).setdefault(slug, {"offered": 0, "picked": 0})
                    bc[field] = bc.get(field, 0) + 1
                box["n"] = i[field]
                _write_meta(i)
                return
        raise KeyError(idea_id)

    store.update_json(_index_path(), _empty(), fn)
    return box["n"]


def offered(idea_id, client=None):
    return _bump(idea_id, "offered", client)


def picked(idea_id, client=None):
    return _bump(idea_id, "picked", client)


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
                i["archive"] = _rel(dest)
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
    return (roles if roles is not None else d["roles"][theme]), fonts


def render(idea_id, client=None, overrides=None, theme=None):
    """The snippet with {{name}} replaced by override, client token or default (client: slug/name or dict).
    A link idea with no snippet renders as an <img> of its still."""
    idea = get(idea_id)
    if not idea.get("snippet"):
        if not idea.get("still"):
            raise ValueError(f"idea {idea_id} has no snippet and no still to render")
        still = Path(idea["still"]) if idea["source"] == "style-link" else _abs(idea["still"])
        return f'<img src="{still.as_posix()}" alt="">\n'
    snip = Path(idea["snippet"]) if idea["source"] == "style-link" else _abs(idea["snippet"])
    if not snip.is_file():
        raise ValueError(f"idea {idea_id}: snippet file not found: {snip}")
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
                if role not in roles:
                    raise ValueError(f"client palette has no role {role!r} (param {name!r})")
                v = roles[role]
            elif p["type"] == "font" and role:
                if role not in fonts:
                    raise ValueError(f"client fonts have no role {role!r} (param {name!r})")
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



# ---- seeding: a style's library (links) and an approved final

def _kind_of(category, scene_type=None):
    c = f"{category or ''} {scene_type or ''}".lower()
    for word, kind in (("caption", "caption"), ("transition", "transition"), ("cta", "cta"),
                       ("call to action", "cta"), ("camera", "camera"), ("zoom", "camera"),
                       ("text", "text-animation"), ("title", "text-animation"), ("type", "text-animation")):
        if word in c:
            return kind
    return "b-roll"


def _style_entry(style):
    import styles
    s = styles.find(styles.load_registry(), style)
    if not s:
        raise KeyError(style)
    if not str(s.get("pack") or "").strip() or not Path(s["pack"]).is_dir():
        raise ValueError(f"style {s['slug']} has no pack folder")
    return s


def seed_style(style, categories=()):
    """Link a style's reusable library entries into the bank. Idempotent per (style, entry)."""
    s = _style_entry(style)
    pack = Path(s["pack"])
    lib = pack / "library"
    index = store.read_json(lib / "index.json", None)
    if not isinstance(index, dict):
        raise ValueError(f"style {s['slug']} has no library/index.json (build_reference_library.py)")
    flags = {}
    anim = store.read_json(pack / "animations.json", None)
    rows = anim.get("animations", []) if isinstance(anim, dict) else anim if isinstance(anim, list) else []
    for a in rows:
        if isinstance(a, dict) and a.get("id") is not None:
            flags[str(a["id"])] = bool(a.get("reusable"))
    cats = {c.strip().lower() for c in categories if c and c.strip()}
    added, skipped = [], []
    for e in index.get("animations", []):
        if not isinstance(e, dict) or e.get("id") is None:
            continue
        eid = str(e["id"])
        reusable = bool(e.get("reusable")) or flags.get(eid, False) or \
            (str(e.get("category") or "").strip().lower() in cats)
        if not reusable or find_by_link(s["slug"], eid):
            skipped.append(eid)
            continue
        still = lib / e["still"] if e.get("still") else None
        clip = lib / e["clip"] if e.get("clip") else None
        if still is None or not still.is_file():
            skipped.append(eid)
            continue
        idea = add(e.get("title") or eid, _kind_of(e.get("category"), e.get("scene_type")),
                   [e["scene_type"]] if e.get("scene_type") else [], s.get("aspect") or "9:16", None, {},
                   still=still, clip=clip if clip and clip.is_file() else None,
                   made_for={"style": s["slug"], "client": None, "project": None},
                   tags=[t for t in (e.get("category"),) if t],
                   link={"style": s["slug"], "entry": eid, "path": str(e.get("folder") or "")},
                   approved=True)
        added.append(idea["id"])
    return {"added": added, "skipped": skipped}


def _beat_times(project_dir):
    plan = store.read_json(Path(project_dir) / "plan.json", {}) or {}
    out = {}
    for b in plan.get("beats", []) if isinstance(plan, dict) else []:
        if isinstance(b, dict) and b.get("n") is not None:
            try:
                out[str(b["n"])] = (float(b["start"]), float(b["end"]))
            except (KeyError, TypeError, ValueError):
                pass
    return out


def final_candidates(project_dir):
    """Built beats with a snippet file and non-empty params, pre-ticked; also written to closing.json."""
    d = Path(project_dir)
    state = store.read_json(d / "build" / "state.json", {}) or {}
    times = _beat_times(d)
    out = []
    beats = state.get("beats", {}) if isinstance(state, dict) else {}
    for key in sorted(beats, key=lambda k: int(k) if str(k).isdigit() else 10 ** 9):
        b = beats[key]
        if not isinstance(b, dict) or not b.get("snippet") or not b.get("params"):
            continue
        if not (d / b["snippet"]).is_file() or str(key) not in times:
            continue
        t0, t1 = times[str(key)]
        end = min(t1, t0 + CLIP_SECONDS)
        out.append({"beat": int(key) if str(key).isdigit() else key, "title": b.get("title") or f"beat {key}",
                    "kind": b.get("kind") or "b-roll", "beat_types": list(b.get("beat_types") or []),
                    "snippet": b["snippet"], "params": b["params"], "start_s": round(t0, 3),
                    "end_s": round(end, 3), "still_t": round((t0 + end) / 2, 3), "ticked": True})
    store.write_json(d / "closing.json", {"version": 1, "candidates": out})
    return out


def _cut(final, t0, dur, dst, still=False):
    ff = shutil.which("ffmpeg")
    if not ff:
        raise MissingTool("ffmpeg is missing: install it (winget install Gyan.FFmpeg, or brew install ffmpeg)")
    cmd = [ff, "-y", "-v", "error", "-ss", f"{t0:.3f}", "-i", str(final)]
    if still:
        cmd += ["-frames:v", "1", str(dst)]
    else:
        cmd += ["-t", f"{dur:.3f}", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(dst)]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0 or not Path(dst).is_file():
        raise ValueError(f"ffmpeg could not cut the final at {t0:.2f} s: {r.stderr.strip()[:200]}")


def _aspect(video):
    ffp = shutil.which("ffprobe")
    try:
        r = subprocess.run([ffp, "-v", "error", "-select_streams", "v:0", "-show_entries",
                            "stream=width,height", "-of", "csv=p=0", str(video)],
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
        w, h = (int(x) for x in r.stdout.strip().split(",")[:2])
    except (OSError, TypeError, ValueError):
        return "9:16"
    return min((("9:16", 9 / 16), ("16:9", 16 / 9), ("1:1", 1.0), ("4:5", 0.8)),
               key=lambda a: abs(a[1] - w / h))[0]


def _set_field(idea_id, key, value):
    def fn(idx):
        for i in idx["ideas"]:
            if i["id"] == idea_id:
                i[key] = value
                _write_meta(i)
                return
    store.update_json(_index_path(), _empty(), fn)


def commit_final(project_dir, final_mp4, untick=()):
    """Bank every ticked candidate (still + clip cut from the final), then close the project.
    A candidate refused by the client-literal check is reported, not banked. Returns (added ids, refused)."""
    import project
    d = Path(project_dir)
    final = Path(final_mp4)
    if not final.is_file():
        raise ValueError(f"final not found: {final_mp4}")
    entry = project.load(d)
    skip = {str(u) for u in untick}
    cands = final_candidates(d)
    added, refused = [], []
    today = time.strftime("%Y-%m-%d")
    aspect = _aspect(final)
    with tempfile.TemporaryDirectory() as tmp:
        for c in cands:
            if str(c["beat"]) in skip:
                c["ticked"] = False
                continue
            still, clip = Path(tmp) / f"b{c['beat']}.png", Path(tmp) / f"b{c['beat']}.mp4"
            _cut(final, c["still_t"], 0, still, still=True)
            _cut(final, c["start_s"], c["end_s"] - c["start_s"], clip)
            try:
                idea = add(c["title"], c["kind"], c["beat_types"], aspect, d / c["snippet"], c["params"],
                           still=still, clip=clip, source="final", approved=True,
                           made_for={"style": entry.get("style"), "client": entry.get("client"),
                                     "project": entry.get("id")})
            except Refused as e:
                refused.append({"beat": c["beat"], "why": str(e)})
                continue
            _set_field(idea["id"], "approved_date", today)
            added.append(idea["id"])
    store.write_json(d / "closing.json", {"version": 1, "candidates": cands, "added": added, "refused": refused})
    project.close(d, str(final))
    return added, refused

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
    a.add_argument("--aspect"); a.add_argument("--avoid-client"); a.add_argument("--include-generic", action="store_true"); a.add_argument("--limit", type=int, default=10)
    a.add_argument("--json", action="store_true")
    a = sub.add_parser("show"); a.add_argument("id")
    a = sub.add_parser("render")
    a.add_argument("id"); a.add_argument("--client"); a.add_argument("--param", action="append", default=[])
    a.add_argument("--out", required=True)
    a = sub.add_parser("remove"); a.add_argument("id")
    a = sub.add_parser("seed-style"); a.add_argument("style"); a.add_argument("--categories", default="")
    a = sub.add_parser("from-final"); a.add_argument("project")
    a = sub.add_parser("commit-final"); a.add_argument("project"); a.add_argument("--final", required=True)
    a.add_argument("--untick", default="")
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
                          avoid_client=args.avoid_client, limit=args.limit,
                          include_generic=args.include_generic), args.json)
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
        if args.cmd == "seed-style":
            r = seed_style(args.style, [c for c in args.categories.split(",") if c.strip()])
            print(f"נוספו {len(r['added'])} רעיונות (קישורים לסגנון), דולגו {len(r['skipped'])}")
            return 0
        if args.cmd == "from-final":
            cands = final_candidates(args.project)
            if not cands:
                print("אין בסרטון קטעים עם סניפט ופרמטרים")
                return 0
            print("הקטעים האלה ייכנסו לבנק הרעיונות (כולם מסומנים; --untick כדי להוציא):")
            for c in cands:
                print(f"  [x] {c['beat']}: {c['title']} ({c['kind']}, {c['start_s']:.1f}-{c['end_s']:.1f} שניות)")
            return 0
        if args.cmd == "commit-final":
            untick = [u.strip() for u in args.untick.split(",") if u.strip()]
            added, refused = commit_final(args.project, args.final, untick)
            print(f"נוספו לבנק: {', '.join(added) or 'אף אחד'}; הפרויקט נסגר")
            for r in refused:
                print(f"  לא נוסף קטע {r['beat']}: {r['why']}", file=sys.stderr)
            return 0
        remove(args.id)
        print(f"removed {args.id} (archived in ideas/_removed)")
        return 0
    except KeyError as e:
        what = e.args[0] if e.args else ""
        if args.cmd == "render" and isinstance(what, str) and not re.fullmatch(r"I\d+", what):
            return _bad(f"missing value for client-bound parameter {what!r}: pass --param {what}=<value>")
        if args.cmd == "seed-style":
            return _bad(f"no style {what!r}")
        if args.cmd in ("from-final", "commit-final"):
            return _bad(str(what))
        return _bad(f"no idea {what!r}")
    except ValueError as e:
        return _bad(str(e))
    except MissingTool as e:
        print(str(e), file=sys.stderr)
        return 3


if __name__ == "__main__":
    sys.exit(main())

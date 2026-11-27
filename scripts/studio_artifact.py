#!/usr/bin/env python3
"""The Studio as a claude.ai Artifact: the same screens as studio.py, published as a private artifact whose
owner actions (picks, motion mode, lesson approve/reject, idea remove, style rename) are saved as rows in the
artifact's own database. Claude reads the rows and applies them here, through the same code as the local
Studio (studio.dispatch), so picks.json stays the one source of truth.

    studio_artifact.py build --out DIR [--project ID] [--no-video]
        writes the page, the data snapshot and the media it shows into DIR (a folder in Claude's scratchpad or
        the working folder: the Artifact tool publishes only from there; DIR is emptied first)
        and prints a JSON manifest: file_path, files (published path -> source), capabilities, url (saved
        from an earlier publish, or null), counts and anything skipped. Claude publishes it with the Artifact
        tool: file_path, files, capabilities {"db": {}}, and url when the manifest has one.
    studio_artifact.py set-url URL     remember the published artifact (so the next build republishes it)
    studio_artifact.py apply DIR       apply the rows Claude saved from the artifact's "actions" collection
                                       (ArtifactData list with out_dir=DIR); rows already applied are skipped
    studio_artifact.py status          the saved url, the last build and how many rows were applied

Reference videos (any refs/ folder) are never copied: studio.resolve_file refuses them. Everything published
stays private to the owner until the owner shares it. Exit codes: 0 ok, 1 a row failed, 2 bad input.
"""
import argparse
import datetime as dt
import json
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import store  # noqa: E402
import studio  # noqa: E402

APP = studio.APP
MAX_FILES = 250          # the Artifact tool takes 255 files per call; keep room for the page itself
MAX_FILE = 15 << 20      # per binary file
MAX_TOTAL = 60 << 20     # per call (64 MB), with room to spare
VIDEO = (".mp4", ".webm", ".mov", ".m4v")


def _home():
    return store.path("studio-artifact")


def _state():
    return store.read_json(_home() / "state.json", {}) or {}


def _save_state(upd):
    store.update_json(_home() / "state.json", {}, lambda cur: cur.update(upd))


def _safe(s):
    return re.sub(r"[^A-Za-z0-9._-]", "_", str(s))[:120] or "_"


class Bundle:
    def __init__(self, out, no_video=False):
        self.out = Path(out)
        self.no_video = no_video
        self.files = {}      # published path -> absolute source
        self.total = 0
        self.skipped = []

    def add(self, url):
        """'/files/<kind>/<key>/<rest>' -> a relative published path, or None when it cannot ship."""
        if not url:
            return None
        m = re.fullmatch(r"/files/(project|pack|idea|client)/([^/]+)/(.+)", url)
        if not m:
            return None
        kind, key, rest = m.groups()
        try:
            src = studio.resolve_file(kind, key, rest)
        except studio.Refused as e:
            self.skipped.append({"url": url, "why": str(e)})
            return None
        rel = "/".join(["files", kind, _safe(key)] + [_safe(x) for x in Path(rest).parts if x not in ("", ".", "..")])
        if src.suffix.lower() in VIDEO or url.endswith("/clip"):
            if self.no_video:
                self.skipped.append({"url": url, "why": "videos left out (--no-video)"})
                return None
            if not Path(rel).suffix:
                rel += src.suffix.lower()
        elif not Path(rel).suffix:
            rel += src.suffix.lower()
        if rel in self.files:
            return rel
        size = src.stat().st_size
        if size > MAX_FILE:
            self.skipped.append({"url": url, "why": f"{size >> 20} MB is over the 15 MB per-file limit"})
            return None
        if len(self.files) >= MAX_FILES or self.total + size > MAX_TOTAL:
            self.skipped.append({"url": url, "why": "over the artifact's size or file-count limit"})
            return None
        dst = self.out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        self.files[rel] = str(dst)
        self.total += size
        return rel

    def write_json(self, rel, obj):
        dst = self.out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        data = json.dumps(obj, ensure_ascii=False, indent=1).encode("utf-8")
        dst.write_bytes(data)
        self.files[rel] = str(dst)
        self.total += len(data)


def _room_media(room, bundle, pid):
    """Copy the stills and clips the room shows; point room.media at the copied folder."""
    prefix = f"/files/project/{pid}/"
    seen = {}

    def walk(doc, field):
        for bt in ((doc or {}).get("beats") or {}).values():
            for st in bt.get("sets") or []:
                for v in st.get("variants") or []:
                    p = v.get(field)
                    if p:
                        rel = bundle.add(prefix + p)
                        if rel:
                            seen[p] = rel
                        else:
                            v[field] = None
    walk(room.get("frames"), "still")
    walk(room.get("motion"), "clip")
    # the page joins room.media + path: give it the copied tree as the base and keep the original names
    base = f"files/project/{_safe(pid)}/"
    for doc, field in ((room.get("frames"), "still"), (room.get("motion"), "clip")):
        for bt in ((doc or {}).get("beats") or {}).values():
            for st in bt.get("sets") or []:
                for v in st.get("variants") or []:
                    if v.get(field) in seen:
                        v[field] = seen[v[field]][len(base):]
    room["media"] = base
    return room


def page_html(built_at):
    css = (APP / "app.css").read_text(encoding="utf-8")
    js = (APP / "app.js").read_text(encoding="utf-8")
    body = (APP / "index.html").read_text(encoding="utf-8")
    body = body[body.index("<body>") + len("<body>"):body.index("</body>")]
    body = re.sub(r'<script src="/static/app.js"></script>\s*', "", body)
    conf = json.dumps({"built_at": built_at}).replace("<", "\\u003c")
    return ("<title>סטודיו סגנון</title>\n"
            f"<style>\n{css}\nhtml,body{{direction:rtl}}\n</style>\n"
            f'<div dir="rtl" lang="he">{body}</div>\n'
            f"<script>window.STUDIO_ARTIFACT={conf};</script>\n"
            f"<script>\n{js}\n</script>\n")


def build(out, project=None, no_video=False):
    out = Path(out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    built_at = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
    b = Bundle(out, no_video)
    ov = studio.overview()
    for s in ov["styles"]:
        s["cover"] = b.add(s.get("cover"))
    for c in ov["clients"]:
        c["logo"] = b.add(c.get("logo"))
    for i in ov["ideas"]:
        i["still"] = b.add(i.get("still"))
        i["clip"] = b.add(i.get("clip"))
    for p in ov["projects"]:
        if project and p["id"] != project:
            p["room_file"] = None
            continue
        rel = f"data/project-{_safe(p['id'])}.json"
        try:
            room = _room_media(studio.project_room(p["id"]), b, p["id"])
        except Exception as e:  # a broken project never stops the whole Studio
            b.skipped.append({"project": p["id"], "why": f"{type(e).__name__}: {e}"})
            p["room_file"] = None
            continue
        room["entry"].pop("path", None)
        b.write_json(rel, room)
        p["room_file"] = rel
    ov["built_at"] = built_at
    b.write_json("data/overview.json", ov)
    for font in sorted((APP / "fonts").glob("*.ttf")):
        dst = out / "fonts" / font.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(font, dst)
        b.files["fonts/" + font.name] = str(dst)
        b.total += font.stat().st_size
    page = out / "index.html"
    page.write_text(page_html(built_at), encoding="utf-8")
    st = _state()
    _save_state({"built_at": built_at, "out": str(out)})
    return {"file_path": str(page), "files": b.files, "capabilities": {"db": {}}, "url": st.get("url"),
            "title": "סטודיו סגנון", "icon": "palette", "built_at": built_at, "count": len(b.files),
            "bytes": b.total, "skipped": b.skipped}


def _rows(src):
    """Rows saved by ArtifactData out_dir: <dir>/actions/<id>.json (or the folder itself)."""
    src = Path(src)
    folder = src / "actions" if (src / "actions").is_dir() else src
    out = []
    for f in sorted(folder.glob("*.json")):
        try:
            doc = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            out.append((f.stem, None))
            continue
        if isinstance(doc, dict) and "route" not in doc and isinstance(doc.get("data"), dict):
            doc = doc["data"]
        out.append((f.stem, doc))
    return out


def apply(src):
    done = (_state().get("applied") or {})
    results, failed = [], 0
    rows = sorted(_rows(src), key=lambda r: (r[1] or {}).get("at", "") if isinstance(r[1], dict) else "")
    for rid, row in rows:
        if rid in done:
            continue
        if not isinstance(row, dict) or not isinstance(row.get("route"), str):
            results.append({"id": rid, "ok": False, "error": "not an action row"})
            failed += 1
            continue
        try:
            res = studio.dispatch(row["route"], row.get("body") or {})
            ok = bool(res.get("ok", True))
            results.append({"id": rid, "route": row["route"], "ok": ok,
                            **({} if ok else {"error": (res.get("err") or res.get("out") or "")[-600:]})})
        except studio.Refused as e:
            ok = False
            results.append({"id": rid, "route": row["route"], "ok": False, "error": str(e)})
        except Exception as e:
            ok = False
            results.append({"id": rid, "route": row["route"], "ok": False, "error": f"{type(e).__name__}: {e}"})
        if ok:
            done[rid] = row.get("at") or True
        else:
            failed += 1
    _save_state({"applied": done})
    return {"applied": sum(1 for r in results if r["ok"]), "failed": failed, "results": results}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--out", required=True)
    b.add_argument("--project")
    b.add_argument("--no-video", action="store_true")
    u = sub.add_parser("set-url")
    u.add_argument("url")
    a = sub.add_parser("apply")
    a.add_argument("dir")
    sub.add_parser("status")
    args = ap.parse_args(argv)
    if args.cmd == "build":
        print(json.dumps(build(args.out, args.project, args.no_video), ensure_ascii=False, indent=1))
        return 0
    if args.cmd == "set-url":
        if not re.fullmatch(r"https://claude\.ai/(code/)?artifact/[A-Za-z0-9-]+/?", args.url):
            print("not a claude.ai artifact link", file=sys.stderr)
            return 2
        _save_state({"url": args.url.rstrip("/")})
        print(json.dumps({"url": args.url.rstrip("/")}))
        return 0
    if args.cmd == "apply":
        if not Path(args.dir).is_dir():
            print(f"no folder {args.dir}", file=sys.stderr)
            return 2
        r = apply(args.dir)
        print(json.dumps(r, ensure_ascii=False, indent=1))
        return 1 if r["failed"] else 0
    st = _state()
    print(json.dumps({"url": st.get("url"), "built_at": st.get("built_at"), "applied": len(st.get("applied") or {})},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())

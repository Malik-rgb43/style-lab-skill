#!/usr/bin/env python3
"""Style Studio: a local web app over the global store (styles, clients, ideas, lessons, projects) and the
project room where the owner picks plan concepts, frames and motion variants.

    studio.py [--port 0] [--no-browser]            start; prints the URL (with the per-run token)
    studio.py --print-url                          start and only print the URL (for Claude to share)

Python standard library only; binds 127.0.0.1. Every request needs the per-run token (in the first URL,
then a SameSite=Strict cookie); every write also needs an Origin equal to the Studio's own address.
All logic stays in the scripts: the Studio reads the store and project files and writes picks through
picks.py (the same picks.json shape as the pick pages and the chat). Reference videos (any refs/ folder)
are never served. Exit codes: 0 stopped, 2 bad input.
"""
import argparse
import http.server
import json
import mimetypes
import os
import re
import secrets
import socketserver
import subprocess
import sys
import threading
import urllib.parse
import webbrowser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import store  # noqa: E402

SCRIPTS = Path(__file__).resolve().parent
APP = SCRIPTS.parent / "assets" / "studio"
COOKIE = "sl_studio"
MAX_BODY = 1 << 20


class Refused(Exception):
    def __init__(self, code, msg):
        super().__init__(msg)
        self.code = code


# ---------------------------------------------------------------- data

def _projects():
    import project
    return project._registry().get("projects", [])


def _project(pid):
    for e in _projects():
        if e.get("id") == pid:
            return e
    raise Refused(404, f"no project {pid}")


def overview():
    import clients
    import ideas
    import styles
    reg = styles.load_registry()
    all_clients = []
    for c in clients.load_registry().get("clients", []):
        try:
            full = clients.load_client(c.get("slug")) or c
        except Exception:
            full = c
        if not full.get("removed"):
            all_clients.append(full)

    def slugs(c):
        return [x.get("slug") if isinstance(x, dict) else x for x in (c or {}).get("styles") or []]
    links = {}
    for c in all_clients:
        for s in slugs(c):
            links.setdefault(s, []).append(c.get("slug"))
    out_styles = []
    for s in reg.get("styles", []):
        out_styles.append({k: s.get(k) for k in ("slug", "name", "description", "status", "kind", "aspect",
                                                    "rules_count", "theme", "version", "requires")}
                          | {"cover": f"/files/pack/{s['slug']}/{s['cover']}" if s.get("cover") else None,
                             "clients": links.get(s.get("slug"), [])})
    out_clients = []
    for c in all_clients:
        if not c:
            continue
        pal = (c.get("palette") or {}).get("roles") or {}
        roles = pal.get("dark") or next(iter(pal.values()), {}) if pal else {}
        logo = c.get("logo")
        logo = logo[0] if isinstance(logo, list) and logo else logo if isinstance(logo, str) else None
        out_clients.append({"slug": c.get("slug"), "name": c.get("name"), "styles": slugs(c),
                            "palette": {k: roles.get(k) for k in ("brand", "brand_deep", "accent", "highlight",
                                                                  "background", "text") if roles.get(k)},
                            "fonts": c.get("fonts") or {}, "dos": c.get("dos") or [], "donts": c.get("donts") or [],
                            "logo": f"/files/client/{c['slug']}/{Path(logo).name}" if logo else None})
    idx = ideas.load_index()
    out_ideas = []
    for i in idx.get("ideas", []):
        if i.get("removed"):
            continue
        out_ideas.append({k: i.get(k) for k in ("id", "title", "kind", "beat_types", "aspect", "tags", "made_for",
                                                 "source", "offered", "picked", "approved")}
                         | {"still": f"/files/idea/{i['id']}/still" if i.get("still") else None,
                            "clip": f"/files/idea/{i['id']}/clip" if i.get("clip") else None})
    return {"styles": out_styles, "clients": out_clients, "ideas": out_ideas, "lessons": _lessons(),
            "projects": [{k: e.get(k) for k in ("id", "title", "style", "client", "stage", "status", "updated",
                                                 "last_draft")} for e in _projects()],
            "caps": _caps()}


def _lessons():
    try:
        import lessons
        return [{k: x.get(k) for k in ("id", "text", "measurable", "layer", "target", "status", "kind", "scope",
                                        "rule_no", "evidence")} for x in lessons.load().get("lessons", [])]
    except Exception:
        return []


def _caps():
    c = store.read_json(store.path("capabilities.json"), {}) or {}
    st = c.get("stages") or {}
    return {"checked": c.get("checked"), "missing": {k: v.get("missing", []) for k, v in st.items() if v.get("missing")}}


def project_room(pid):
    e = _project(pid)
    d = Path(e["path"])

    def rd(name, default):
        v = store.read_json(d / name, default)
        return v if v is not None else default
    import picks
    return {"entry": e, "plan": rd("plan.json", {}), "frames": rd("frames.json", {"beats": {}}),
            "motion": rd("motion-variants.json", {"beats": {}}), "picks": picks.load(d),
            "shown": rd("shown.json", {}).get("sets", {}), "build": rd("build/state.json", {}),
            "spend": [{k: b.get(k) for k in ("id", "status", "total", "spent", "currency")}
                      for b in rd("spend.json", {}).get("batches", [])],
            "problems": picks.validate(d), "media": f"/files/project/{pid}/"}


# ---------------------------------------------------------------- writes

def write_pick(pid, body):
    import picks
    d = Path(_project(pid)["path"])
    step = body.get("step")
    if step not in picks.STEPS:
        raise Refused(400, "step must be plan, frames or motion")
    try:
        beat = int(body.get("beat"))
    except (TypeError, ValueError):
        raise Refused(400, "beat must be a number")
    try:
        picks.pick(d, step, beat, body.get("id") or None, own=str(body.get("own") or ""),
                   more=bool(body.get("more")), note=str(body.get("note") or ""), by="owner", source="studio")
    except ValueError as e:
        raise Refused(400, str(e))
    return {"ok": True, "picks": picks.load(d), "problems": picks.validate(d)}


def write_motion_mode(pid, body):
    import picks
    d = Path(_project(pid)["path"])
    mode = body.get("mode")
    if mode not in ("three", "one"):
        raise Refused(400, "mode must be three or one")
    if body.get("beat") is not None:
        picks.set_motion_mode(d, None, {str(int(body["beat"])): mode})
    else:
        picks.set_motion_mode(d, mode)
    return {"ok": True, "picks": picks.load(d)}


def _cli(script, *args):
    r = subprocess.run([sys.executable, str(SCRIPTS / script), *args], capture_output=True, text=True,
                       encoding="utf-8", errors="replace", stdin=subprocess.DEVNULL, timeout=120)
    return {"ok": r.returncode == 0, "code": r.returncode, "out": r.stdout.strip()[-2000:],
            "err": r.stderr.strip()[-2000:]}


def lesson_action(lid, action, body):
    if not re.fullmatch(r"L\d+", lid or ""):
        raise Refused(400, "bad lesson id")
    if action == "approve":
        args = ["approve", lid]
        if body.get("one_note"):
            args.append("--one-note")
        if body.get("yes_no_conflict"):
            args.append("--yes-no-conflict")
        return _cli("lessons.py", *args)
    if action == "reject":
        return _cli("lessons.py", "reject", lid, *(["--reason", str(body["reason"])] if body.get("reason") else []))
    raise Refused(404, "unknown lesson action")


def idea_remove(iid):
    if not re.fullmatch(r"I\d+", iid or ""):
        raise Refused(400, "bad idea id")
    return _cli("ideas.py", "remove", iid)


def style_rename(slug, body):
    name = str(body.get("name") or "").strip()
    if not name or len(name) > 80:
        raise Refused(400, "a name of 1-80 characters")
    return _cli("styles.py", "set", slug, f"name={name}")


ROUTES = [
    (r"/api/project/([^/]+)/pick", lambda m, b: write_pick(m.group(1), b)),
    (r"/api/project/([^/]+)/motion-mode", lambda m, b: write_motion_mode(m.group(1), b)),
    (r"/api/lesson/([^/]+)/(approve|reject)", lambda m, b: lesson_action(m.group(1), m.group(2), b)),
    (r"/api/idea/([^/]+)/remove", lambda m, b: idea_remove(m.group(1))),
    (r"/api/style/([^/]+)/rename", lambda m, b: style_rename(m.group(1), b)),
]


def dispatch(path, body):
    """Run one owner action (the local server's POST, or a row the Studio artifact saved). Only ROUTES run."""
    if not isinstance(body, dict):
        raise Refused(400, "a JSON object")
    for pat, fn in ROUTES:
        m = re.fullmatch(pat, path or "")
        if m:
            return fn(m, body)
    raise Refused(404, "not found")


# ---------------------------------------------------------------- files

def _inside(p, root):
    try:
        Path(os.path.realpath(p)).relative_to(Path(os.path.realpath(root)))
        return True
    except ValueError:
        return False


def resolve_file(kind, key, rest):
    import ideas
    import styles
    if kind == "project":
        base = Path(_project(key)["path"])
        p = base / rest
    elif kind == "pack":
        s = styles.find(styles.load_registry(), key)
        if not s or not s.get("pack"):
            raise Refused(404, "no such style")
        base = Path(s["pack"])
        p = base / rest
    elif kind == "idea":
        i = ideas.get(key)
        field = rest.split("/")[0]
        if field not in ("still", "clip", "snippet") or not i.get(field):
            raise Refused(404, "no such file")
        p = ideas._abs(i[field])
        base = p.parent
    elif kind == "client":
        base = store.path("clients", key)
        p = base / rest
    else:
        raise Refused(404, "unknown file kind")
    if not _inside(p, base):
        raise Refused(403, "outside its folder")
    rel_parts = [x.lower() for x in Path(os.path.realpath(p)).parts]
    if "refs" in rel_parts:
        raise Refused(403, "reference videos are never served")
    if not Path(p).is_file():
        raise Refused(404, "not found")
    return Path(p)


# ---------------------------------------------------------------- HTTP

class Handler(http.server.BaseHTTPRequestHandler):
    server_version = "style-lab-studio"
    token = ""
    origin = ""

    def log_message(self, fmt, *args):  # quiet
        pass

    def _send(self, code, body=b"", ctype="application/json; charset=utf-8", headers=None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, code, obj):
        self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"))

    def _authed(self, query):
        if query.get("t", [""])[0] == self.token:
            return True
        cookies = self.headers.get("Cookie") or ""
        return any(c.strip() == f"{COOKIE}={self.token}" for c in cookies.split(";"))

    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        u = urllib.parse.urlsplit(self.path)
        q = urllib.parse.parse_qs(u.query)
        if not self._authed(q):
            return self._send(403, "צריך את הקישור המלא שהסטודיו הדפיס (עם הטוקן)".encode("utf-8"),
                              "text/plain; charset=utf-8")
        set_cookie = {"Set-Cookie": f"{COOKIE}={self.token}; Path=/; HttpOnly; SameSite=Strict"}
        try:
            path = urllib.parse.unquote(u.path)
            if path in ("/", "/index.html"):
                return self._send(200, (APP / "index.html").read_bytes(), "text/html; charset=utf-8", set_cookie)
            if path.startswith("/static/"):
                f = (APP / path[len("/static/"):]).resolve()
                if not _inside(f, APP) or not f.is_file():
                    raise Refused(404, "not found")
                return self._file(f)
            if path == "/api/overview":
                return self._json(200, overview())
            m = re.fullmatch(r"/api/project/([^/]+)", path)
            if m:
                return self._json(200, project_room(m.group(1)))
            m = re.fullmatch(r"/files/(project|pack|idea|client)/([^/]+)/(.+)", path)
            if m:
                return self._file(resolve_file(m.group(1), m.group(2), m.group(3)))
            raise Refused(404, "not found")
        except Refused as e:
            self._json(e.code, {"error": str(e)})
        except Exception as e:  # never a traceback page
            self._json(500, {"error": f"{type(e).__name__}: {e}"})

    def do_POST(self):
        u = urllib.parse.urlsplit(self.path)
        if not self._authed(urllib.parse.parse_qs(u.query)):
            return self._json(403, {"error": "token"})
        if (self.headers.get("Origin") or "") != self.origin:
            return self._json(403, {"error": "origin"})
        try:
            n = int(self.headers.get("Content-Length") or 0)
            if n > MAX_BODY:
                raise Refused(413, "too large")
            body = json.loads(self.rfile.read(n) or b"{}") if n else {}
            if not isinstance(body, dict):
                raise Refused(400, "a JSON object")
            return self._json(200, dispatch(urllib.parse.unquote(u.path), body))
        except Refused as e:
            self._json(e.code, {"error": str(e)})
        except ValueError as e:
            self._json(400, {"error": str(e)})
        except Exception as e:
            self._json(500, {"error": f"{type(e).__name__}: {e}"})

    def _file(self, f):
        ctype = mimetypes.guess_type(str(f))[0] or "application/octet-stream"
        size = f.stat().st_size
        rng = self.headers.get("Range")
        m = re.fullmatch(r"bytes=(\d*)-(\d*)", rng or "")
        if m and (m.group(1) or m.group(2)):
            a = int(m.group(1)) if m.group(1) else max(0, size - int(m.group(2)))
            b = int(m.group(2)) if m.group(1) and m.group(2) else size - 1
            b = min(b, size - 1)
            if a > b:
                return self._send(416, b"", ctype, {"Content-Range": f"bytes */{size}"})
            with open(f, "rb") as fh:
                fh.seek(a)
                data = fh.read(b - a + 1)
            return self._send(206, data, ctype, {"Content-Range": f"bytes {a}-{b}/{size}", "Accept-Ranges": "bytes"})
        return self._send(200, f.read_bytes(), ctype, {"Accept-Ranges": "bytes"})


class Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = False


def make_server(port=0, token=None):
    token = token or secrets.token_urlsafe(24)
    handler = type("H", (Handler,), {})
    srv = Server(("127.0.0.1", int(port)), handler)
    host, p = srv.server_address[:2]
    handler.token = token
    handler.origin = f"http://127.0.0.1:{p}"
    return srv, f"http://127.0.0.1:{p}/?t={token}"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", type=int, default=0)
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--print-url", action="store_true", help="print the URL and keep serving (no browser)")
    ap.add_argument("--token", help="a fixed token instead of a random one (for a preview pane)")
    a = ap.parse_args(argv)
    store.print_utf8()
    try:
        srv, url = make_server(a.port, a.token)
    except OSError as e:
        print(f"error: cannot start the Studio: {e}", file=sys.stderr)
        return 2
    print(url, flush=True)
    if not (a.no_browser or a.print_url):
        threading.Timer(0.3, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

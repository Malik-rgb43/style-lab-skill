"""Criteria for behaviour evals: one function per type, each returning (ok: bool, detail: str).

Paths may use {work}, {home}, {fakehome} and {pack} (the first registered style's pack).
"""
import glob as _glob
import hashlib
import json
import re
from pathlib import Path


class Ctx:
    def __init__(self, work, real_home=None):
        self.work = Path(work)
        self.home = self.work / "home"
        self.fakehome = self.work / "fakehome"
        self.real_home = Path(real_home) if real_home else None
        self.state = _read(self.work / "state.json", {}) or {}

    @property
    def pack(self):
        reg = _read(self.home / "styles.json", {}) or {}
        for s in reg.get("styles", []):
            if s.get("pack"):
                return s["pack"]
        return str(self.work / "_no_pack_")

    def path(self, p):
        p = str(p)
        for k in ("work", "home", "fakehome"):
            p = p.replace("{" + k + "}", str(getattr(self, k)))
        if "{pack}" in p:
            p = p.replace("{pack}", str(self.pack))
        return Path(p)

    def transcript(self):
        f = self.work / "transcript.md"
        return f.read_text(encoding="utf-8", errors="replace") if f.is_file() else ""


def _read(p, default):
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def tree_hashes(p):
    """{relative path: sha256} for a file or every file under a folder."""
    p = Path(p)
    if p.is_file():
        return {".": sha256(p)}
    if not p.is_dir():
        return {}
    return {str(f.relative_to(p)).replace("\\", "/"): sha256(f) for f in sorted(p.rglob("*")) if f.is_file()}


def json_get(doc, path):
    """Dotted path with [*] (fan out over a list). Returns a list of matches."""
    cur = [doc]
    for part in re.findall(r"[^.\[\]]+|\[\*\]", path or ""):
        nxt = []
        for c in cur:
            if part == "[*]":
                if isinstance(c, list):
                    nxt.extend(c)
                elif isinstance(c, dict):
                    nxt.extend(c.values())
            elif isinstance(c, dict) and part in c:
                nxt.append(c[part])
            elif isinstance(c, list) and part.isdigit() and int(part) < len(c):
                nxt.append(c[int(part)])
        cur = nxt
    return cur


# ---------------------------------------------------------------- criteria

def file_exists(ctx, path):
    p = ctx.path(path)
    return p.exists(), str(p)


def file_count(ctx, glob, min=0, max=None):
    n = len(_glob.glob(str(ctx.path(glob)), recursive=True))
    ok = n >= min and (max is None or n <= max)
    return ok, f"{n} files (min {min}, max {max})"


def json_path(ctx, file, path, op, value=None):
    doc = _read(ctx.path(file), None)
    if doc is None:
        return False, f"cannot read {ctx.path(file)}"
    vals = json_get(doc, path)
    one = vals[0] if len(vals) == 1 else vals
    if op == "eq":
        ok = one == value
    elif op == "ne":
        ok = bool(vals) and one != value
    elif op == "gte":
        ok = bool(vals) and all(isinstance(v, (int, float)) and v >= value for v in vals)
    elif op == "lte":
        ok = bool(vals) and all(isinstance(v, (int, float)) and v <= value for v in vals)
    elif op == "in":
        ok = bool(vals) and all(v in value for v in vals)
    elif op == "contains":
        ok = any((value in v) if isinstance(v, (list, str, dict)) else v == value for v in vals)
    elif op == "len_eq":
        ok = isinstance(one, (list, dict, str)) and len(one) == value if len(vals) == 1 else len(vals) == value
    elif op == "len_gte":
        ok = isinstance(one, (list, dict, str)) and len(one) >= value if len(vals) == 1 else len(vals) >= value
    elif op == "all_eq":
        ok = bool(vals) and all(v == value for v in vals)
    elif op == "sorted_desc":
        seq = one if len(vals) == 1 and isinstance(one, list) else vals
        ok = bool(seq) and all(a >= b for a, b in zip(seq, seq[1:]))
    elif op == "disjoint_with":
        other = json_get(doc, value)
        a = set(map(str, vals[0] if len(vals) == 1 and isinstance(vals[0], list) else vals))
        b = set(map(str, other[0] if len(other) == 1 and isinstance(other[0], list) else other))
        ok = bool(a) and not (a & b)
    else:
        return False, f"unknown op {op}"
    return bool(ok), f"{path} {op} {value!r}: got {str(one)[:200]}"


def _lines(ctx, regex):
    rx = re.compile(regex, re.M)
    return [m for m in rx.finditer(ctx.transcript())]


def transcript_has(ctx, regex, min=1):
    n = len(_lines(ctx, regex))
    return n >= min, f"{n} matches of {regex!r} (min {min})"


def transcript_lacks(ctx, regex):
    m = _lines(ctx, regex)
    return not m, ("none" if not m else f"found: {m[0].group(0)[:120]}")


def transcript_order(ctx, before, after):
    a, b = _lines(ctx, before), _lines(ctx, after)
    if not a or not b:
        return False, f"before found {len(a)}, after found {len(b)}"
    return a[0].start() < b[0].start(), f"before at {a[0].start()}, after at {b[0].start()}"


def lessons(ctx, where=None, min=0, max=None, evidence_min=None):
    doc = _read(ctx.home / "lessons" / "lessons.json", {}) or {}
    rows = [x for x in doc.get("lessons", []) if all(x.get(k) == v for k, v in (where or {}).items())]
    ok = len(rows) >= min and (max is None or len(rows) <= max)
    if ok and evidence_min is not None:
        ok = all(len(x.get("evidence") or []) >= evidence_min for x in rows)
    return ok, f"{len(rows)} lessons where {where} (min {min}, max {max}, evidence_min {evidence_min})"


def unchanged(ctx, path):
    want = (ctx.state.get("hashes") or {}).get(path)
    if want is None:
        return False, f"{path} was not hashed at prepare"
    have = tree_hashes(ctx.path(path))
    return have == want, "unchanged" if have == want else "changed"


def _ytdlp_log(ctx):
    f = ctx.work / "ytdlp.log.jsonl"
    out = []
    if f.is_file():
        for line in f.read_text(encoding="utf-8").splitlines():
            try:
                out.append(json.loads(line))
            except ValueError:
                pass
    return out


def _is_download(call):
    return "-J" not in (call.get("argv") or []) and "--dump-json" not in (call.get("argv") or []) \
        and "--flat-playlist" not in (call.get("argv") or [])


def ytdlp_calls(ctx, kind, min=0, max=None):
    calls = [c for c in _ytdlp_log(ctx) if _is_download(c) == (kind == "download")]
    ok = len(calls) >= min and (max is None or len(calls) <= max)
    return ok, f"{len(calls)} {kind} calls (min {min}, max {max})"


def ytdlp_after_consent(ctx, proposal="{pack}/refs/proposal.json"):
    import datetime as dt
    prop = _read(ctx.path(proposal), {}) or {}
    at = (prop.get("consent") or {}).get("at")
    downs = [c for c in _ytdlp_log(ctx) if _is_download(c)]
    if not downs:
        return True, "no downloads"
    if not at:
        return False, f"{len(downs)} downloads and no consent"
    t0 = dt.datetime.fromisoformat(at.replace("Z", "+00:00")).timestamp()
    early = [c for c in downs if c.get("t", 0) < t0 - 1]
    return not early, f"{len(early)} downloads before consent at {at}"


def no_leak(ctx):
    snap = ctx.state.get("real_snapshot")
    if snap is None or not ctx.real_home:
        return True, "no real home given"
    now = real_snapshot(ctx.real_home)
    if now == snap:
        return True, "real store and skills unchanged"
    a, b = set(map(tuple, snap)), set(map(tuple, now))
    diff = sorted(b - a)[:3] or sorted(a - b)[:3]
    return False, f"real files changed: {diff}"


def real_snapshot(real_home):
    """[rel, size, mtime_ns] under <real>/.claude/style-lab, and [name, mtime_ns] under <real>/.claude/skills."""
    base = Path(real_home) / ".claude"
    out = []
    st = base / "style-lab"
    if st.is_dir():
        for f in sorted(st.rglob("*")):
            if f.is_file() and not f.name.endswith((".lock", ".tmp")):
                s = f.stat()
                out.append(["style-lab/" + str(f.relative_to(st)).replace("\\", "/"), s.st_size, s.st_mtime_ns])
    sk = base / "skills"
    if sk.is_dir():
        for d in sorted(sk.iterdir()):
            out.append(["skills/" + d.name, d.stat().st_mtime_ns])
    return out


CHECKS = {f.__name__: f for f in (file_exists, file_count, json_path, transcript_has, transcript_lacks,
                                  transcript_order, lessons, unchanged, ytdlp_calls, ytdlp_after_consent, no_leak)}
AUTO = [{"type": "no_leak", "args": {}},
        {"type": "transcript_lacks", "args": {"regex": "^STUCK:"}},
        {"type": "transcript_has", "args": {"regex": "^(ASK|SAY):", "min": 1}}]


def run(ctx, crit):
    fn = CHECKS.get(crit.get("type"))
    if not fn:
        return False, f"unknown criteria type {crit.get('type')}"
    try:
        return fn(ctx, **(crit.get("args") or {}))
    except TypeError as e:
        return False, f"bad args: {e}"

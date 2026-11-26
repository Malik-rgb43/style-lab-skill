"""Lessons registry: the owner's review notes, interpreted and layered (craft / style / client / video).

A lesson may later become a rule. Registry: <store home>/lessons/lessons.json
    {"version": 1, "next_id": 1, "lessons": [...]}   ids L0001, L0002, ...

Commands:
    lessons.py add --text "<owner words>" --measurable "<measurable reading>" --layer craft|style|client|video
                   [--target <style|client>] [--kind quality|taste] [--scope ..]
                   --project P --video V [--t S] [--round N] [--before ..] [--after ..]
    lessons.py list [--status S] [--layer L] [--target T] [--json]
    lessons.py show <id>
    lessons.py edit <id> field=value [...]    (text, measurable, scope, layer, target, kind;
                                               on a rule only text/measurable/scope, re-rendered)
    lessons.py reject <id> [--reason ..]          (not for rules or merged lessons)
    lessons.py match <id> --project P --video V [--t S] [--round N] [--note ..] [--before ..] [--after ..]
    lessons.py approve <id> [--one-note]          (owner yes: proposed -> rule, then render;
                                                   a candidate with one note needs --one-note)
    lessons.py merge <src> <into>                 (moves evidence; src becomes merged)
    lessons.py render --layer craft|style|client [--target T]
    lessons.py settings [auto_promote=true|false]

craft needs --kind and no target; style needs a style target; client needs a client target;
video has neither (status one-off). Other layers start as candidate.

A lesson becomes proposed when its evidence spans 2 distinct (project, video) pairs or 2 distinct
rounds. The same (project, video, t, round) is never counted twice. Only `approve` (or
settings auto_promote=true) makes a rule. Rules are rendered to:
    craft  -> <store home>/craft-rules.learned.md            (whole file generated)
    client -> <store home>/clients/<slug>/lessons.md         (whole file generated)
    style  -> the pack's STYLE.md and the style skill file, only between the markers
              <!-- style-lab:learned-rules:start --> and <!-- style-lab:learned-rules:end -->;
              hand text outside them is never changed. Each file is backed up once per day
              (<file>.bak-YYYYMMDD) before its first write. The style's styles.json entry
              gets rules_count, and version + 1 whenever its rule files change.
"""
import argparse
import contextlib
import datetime as dt
import json
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import store  # noqa: E402

LAYERS = ("craft", "style", "client", "video")
KINDS = ("quality", "taste")
STATUSES = ("candidate", "proposed", "rule", "rejected", "merged", "one-off")
EDITABLE = ("text", "measurable", "scope", "layer", "target", "kind")
START = "<!-- style-lab:learned-rules:start -->"
END = "<!-- style-lab:learned-rules:end -->"
LEARNED_HEADING = "## Review-round rules (learned)"
SETTINGS_DEFAULT = {"auto_promote": False}


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


def _check_kind(layer, kind):
    """Kind rules without touching the target (used when only kind is edited)."""
    if kind not in (None, *KINDS):
        raise ValueError(f"kind must be one of {', '.join(KINDS)}")
    if layer == "craft" and kind is None:
        raise ValueError("craft lessons need kind quality|taste")
    if layer != "craft" and kind is not None:
        raise ValueError(f"kind applies to craft lessons only, not {layer}")


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
        e = _make_evidence(ev)
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
    if "layer" in new and x["status"] not in ("candidate", "one-off"):
        return _fail(f"{x['id']} is {x['status']}: layer can change only on candidate or one-off lessons")
    for k in ("target", "kind"):
        if k in new and x["status"] == "rule":
            return _fail(f"{x['id']} is a rule: its {k} cannot change")
    layer = new.get("layer", x["layer"])
    target = (new["target"] or None) if "target" in new else x["target"]
    kind = (new["kind"] or None) if "kind" in new else x["kind"]
    if "layer" in new and "target" not in new:
        target = None  # a target belongs to the old layer
    if "layer" in new and "kind" not in new and layer != "craft":
        kind = None
    try:
        if "layer" in new or "target" in new:
            x["target"], x["kind"] = _resolve(layer, target, kind)
        elif "kind" in new:
            _check_kind(layer, kind)
            x["kind"] = kind
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
    if x["status"] == "rule" and ("measurable" in new or "scope" in new):
        warnings = []
        try:
            _render(x["layer"], x["target"], data, warnings)  # before the registry write: all or nothing
        except store.LockTimeout:
            raise
        except RENDER_ERRORS as e:
            _flush(warnings)
            return _fail(f"{e}; the edit was not saved")
        _flush(warnings)
        args.rendered = True
    print(f"updated {x['id']}")


def _reject(data, args):
    x = next((i for i in data["lessons"] if i["id"] == args.id), None)
    if not x:
        return _fail(f"no lesson {args.id!r}")
    if x["status"] == "rule":
        return _fail(f"{x['id']} is a rule: use a later retire command for rules")
    if x["status"] == "merged":
        return _fail(f"{x['id']} is merged into {x['merged_into']}; reject that lesson instead")
    x["status"] = "rejected"
    x["reason"] = args.reason or ""
    x["updated"] = today()
    print(f"rejected {x['id']}")


# ---------- evidence, proposal, approval, merge ----------

def _find(data, lesson_id):
    return next((i for i in data["lessons"] if i["id"] == lesson_id), None)


def _make_evidence(ev):
    rnd = ev.get("round")
    t = ev.get("t")
    return {"project": ev.get("project") or "", "video": ev.get("video") or "",
            "t": None if t in (None, "") else str(t),
            "round": None if rnd in (None, "") else int(rnd),
            "note": ev.get("note") or "", "before": ev.get("before"), "after": ev.get("after"),
            "added": today()}


def _ev_key(e):
    """One note = one (project, video, t, round); recording it twice never counts twice."""
    return (e.get("project"), e.get("video"), e.get("t"), e.get("round"))


def _ready(x):
    """2+ distinct (project, video) pairs or 2+ distinct rounds."""
    pairs = {(e.get("project"), e.get("video")) for e in x["evidence"]}
    rounds = {e.get("round") for e in x["evidence"] if e.get("round") is not None}
    return len(pairs) >= 2 or len(rounds) >= 2


def _not_live(x):
    if x["status"] == "one-off":
        return f"{x['id']} is a one-off (video layer): it is kept for history and never becomes a rule"
    if x["status"] == "merged":
        return f"{x['id']} is merged into {x['merged_into']}"
    if x["status"] == "rejected":
        return f"{x['id']} is rejected"
    return None


def settings():
    return store.read_json(store.path("settings.json"), dict(SETTINGS_DEFAULT))


def _record_match(lesson_id, evidence):
    out = {}

    def fn(data):
        x = _find(data, lesson_id)
        if not x:
            raise ValueError(f"no lesson {lesson_id!r}")
        msg = _not_live(x)
        if msg:
            raise ValueError(msg)
        e = _make_evidence(evidence)
        added = _ev_key(e) not in {_ev_key(i) for i in x["evidence"]}
        if added:
            x["evidence"].append(e)
            x["updated"] = today()
        if x["status"] == "candidate" and _ready(x):
            x["status"] = "proposed"
            x["updated"] = today()
        out.update(x, added=added)

    store.update_json(registry_path(), _empty(), fn)
    return out


def _auto_promote(lesson, warnings=None):
    if lesson["status"] == "proposed" and settings().get("auto_promote"):
        return approve(lesson["id"], warnings)
    return None


def match(lesson_id, evidence, warnings=None):
    """Add evidence (deduplicated); candidate -> proposed at 2 projects/videos or 2 rounds.
    With settings auto_promote the lesson is approved at once. Raises ValueError (or the render
    error of an auto-promotion; the evidence is saved either way)."""
    out = _record_match(lesson_id, evidence)
    promoted = _auto_promote(out, warnings)
    if promoted:
        out.update(promoted)
    return out


RESYNC = "rule files were updated; re-run approve or `lessons.py render` to resync"


def _save_after_render(fn, state):
    """update_json on the lessons registry; if it fails after rule files were written, say how to resync."""
    try:
        store.update_json(registry_path(), _empty(), fn)
    except store.LockTimeout:
        raise
    except _Abort:
        raise
    except Exception as e:
        if state.get("rendered"):
            raise OSError(f"saving the lessons registry failed ({e}); {RESYNC}") from e
        raise


def approve(lesson_id, warnings=None, one_note=False):
    """Owner yes: proposed -> rule with the next rule_no of its (layer, target), then render.
    A candidate (one note so far) needs one_note=True, the owner's explicit yes.
    If rendering fails nothing is saved (the status stays) and the error is raised."""
    out = {}
    state = {}

    def fn(data):
        x = _find(data, lesson_id)
        if not x:
            raise ValueError(f"no lesson {lesson_id!r}")
        msg = _not_live(x)
        if msg:
            raise ValueError(msg)
        if x["status"] not in ("candidate", "proposed"):
            raise ValueError(f"{x['id']} is already a {x['status']}")
        if x["status"] == "candidate" and not one_note:
            raise ValueError(f"{x['id']} is a candidate: only one note so far; "
                             f"the owner can still approve it with --one-note")
        nos = [i["rule_no"] for i in data["lessons"]
               if i["layer"] == x["layer"] and i["target"] == x["target"] and i.get("rule_no")]
        x["rule_no"] = max(nos, default=0) + 1
        x["status"] = "rule"
        x["promoted"] = x["updated"] = today()
        paths = _render(x["layer"], x["target"], data, warnings)  # raises before the registry is written
        state["rendered"] = True
        out.update(x, paths=[str(q) for q in paths])

    _save_after_render(fn, state)
    return out


def merge(src_id, into_id, warnings=None):
    """Move src's evidence into into_id (deduplicated); src becomes merged. Raises ValueError."""
    out = {}

    def fn(data):
        if src_id == into_id:
            raise ValueError("a lesson cannot be merged into itself")
        a, b = _find(data, src_id), _find(data, into_id)
        for lid, x in ((src_id, a), (into_id, b)):
            if not x:
                raise ValueError(f"no lesson {lid!r}")
        if a["status"] in ("one-off", "rule", "merged"):
            raise ValueError(f"{src_id} is {a['status']}: it cannot be merged")
        if b["status"] in ("one-off", "rejected", "merged"):
            raise ValueError(f"{into_id} is {b['status']}: nothing can be merged into it")
        keys = {_ev_key(e) for e in b["evidence"]}
        for e in a["evidence"]:
            if _ev_key(e) not in keys:
                b["evidence"].append(e)
                keys.add(_ev_key(e))
        a["evidence"] = []
        a["status"], a["merged_into"] = "merged", into_id
        if b["status"] == "candidate" and _ready(b):
            b["status"] = "proposed"
        a["updated"] = b["updated"] = today()
        out.update(b)

    store.update_json(registry_path(), _empty(), fn)
    promoted = _auto_promote(out, warnings)
    if promoted:
        out.update(promoted)
    return out


# ---------- rendering ----------

def _clean(text):
    """One line, and never a marker look-alike."""
    return " ".join(str(text or "").split()).replace("<!--", "&lt;!--")


def rule_line(x):
    scope = _clean(x.get("scope")).rstrip(". ") or "everywhere"
    line = (f"{x['rule_no']}. {_clean(x['measurable'])} — {scope}. "
            f"(evidence: {len(x['evidence'])}, since {x['promoted']}, {x['id']})")
    if x["layer"] == "craft" and x.get("kind"):
        line += f" [{x['kind']}]"
    if x.get("exception_of"):
        line += f" (exception to {x['exception_of']}: {_clean(x.get('reason'))})"
    return line


def _rule_lines(data, layer, target):
    rules = [x for x in data["lessons"]
             if x["status"] == "rule" and x["layer"] == layer and x["target"] == target]
    return [rule_line(x) for x in sorted(rules, key=lambda x: x["rule_no"])]


def _block(lines, nl):
    return nl.join([START.encode(), *(ln.encode("utf-8") for ln in lines), END.encode()])


_HEADING = re.compile(rb"(?mi)^(?:\xef\xbb\xbf)?#{1,6}[ \t][^\n]*review-round rules[^\n]*$")


def splice(data: bytes, lines, where="file") -> bytes:
    """Put the rule block into data; every byte outside the block stays as it is.
    Existing markers: replace between them. Else insert right after the first heading containing
    "Review-round rules", else append a learned-rules section. Damaged markers raise ValueError."""
    nl = b"\r\n" if b"\r\n" in data else b"\n"
    block = _block(lines, nl)
    s_b, e_b = START.encode(), END.encode()
    ns, ne = data.count(s_b), data.count(e_b)
    if ns or ne:
        if ns != 1 or ne != 1 or data.index(e_b) < data.index(s_b):
            raise ValueError(f"learned-rules markers in {where} are damaged (need one start marker, then one "
                             f"end marker); fix them by hand, nothing was written")
        i, j = data.index(s_b), data.index(e_b) + len(e_b)
        return data[:i] + block + data[j:]
    m = _HEADING.search(data)
    if m:
        pos = m.end()
        if data[pos:pos + 1] == b"\n":
            return data[:pos + 1] + block + nl + data[pos + 1:]
        return data[:pos] + nl + block + nl + data[pos:]  # heading is the last line, no newline
    lead = b"" if not data or data.endswith(b"\n") else nl
    sep = nl if data else b""
    return data + lead + sep + LEARNED_HEADING.encode() + nl + nl + block + nl


def _backup_once(p):
    bak = p.with_name(f"{p.name}.bak-{dt.date.today():%Y%m%d}")
    if p.exists() and not bak.exists():
        shutil.copy2(p, bak)


def _write_bytes(p, data):
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(p.parent), prefix=p.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        store._retry_perm(lambda: os.replace(tmp, p))
    finally:
        if os.path.exists(tmp):
            with contextlib.suppress(OSError):
                os.unlink(tmp)


def _commit(plan, after=None):
    """plan = [(path, old bytes or None, new bytes)]. Back up and write each changed file, then run
    after(changed); if a later write or after() fails, put the earlier files back so nothing is
    half-written."""
    done = []
    try:
        for p, old, new in plan:
            if old == new:
                continue
            _backup_once(p)
            _write_bytes(p, new)
            done.append((p, old))
        if after:
            after(bool(done))
    except BaseException:
        for p, old in reversed(done):
            with contextlib.suppress(OSError):
                if old is None:
                    os.unlink(p)
                else:
                    _write_bytes(p, old)
        raise


def _generated(title, lines):
    body = [f"# {title}", "", "Generated by lessons.py from the lessons registry; edits here are overwritten.",
            "", START, *lines, END, ""]
    return "\n".join(body).encode("utf-8")


def _read(p):
    return p.read_bytes() if p.exists() else None


def _render(layer, target, data, warnings=None):
    """Plan every file first, then write. Returns the paths."""
    lines = _rule_lines(data, layer, target)
    if layer == "craft":
        p = store.path("craft-rules.learned.md")
        _commit([(p, _read(p), _generated("Learned craft rules", lines))])
        return [p]
    if layer == "client":
        if not target:
            raise ValueError("client rules need a client target")
        p = store.path("clients", target, "lessons.md")
        _commit([(p, _read(p), _generated(f"Learned rules for client {target}", lines))])
        return [p]
    if layer != "style":
        raise ValueError(f"{layer} lessons are never rendered")
    import styles
    s = styles.find(styles.load_registry(), target or "")
    if not s:
        raise ValueError(f"style {target!r} is not in the styles registry")
    style_md = Path(s["pack"]) / "STYLE.md"
    if not style_md.is_file():
        raise FileNotFoundError(f"STYLE.md not found: {style_md} (pack missing or drive not connected); "
                                f"nothing was written")
    old = style_md.read_bytes()
    plan = [(style_md, old, splice(old, lines, style_md))]
    skill = Path(s["skill"]) if s.get("skill") else None
    if skill and skill.is_file():
        sk = skill.read_bytes()
        plan.append((skill, sk, splice(sk, lines, skill)))
    else:
        msg = f"warning: style skill file not found, not updated: {skill or '(none registered)'}"
        if warnings is None:
            print(msg, file=sys.stderr)
        else:
            warnings.append(msg)
    n = sum(1 for x in data["lessons"] if x["status"] == "rule" and x["layer"] == "style" and x["target"] == s["slug"])
    _commit(plan, lambda changed: _sync_style_entry(s["slug"], n, changed))
    return [p for p, _, _ in plan]


def _sync_style_entry(slug, rules_count, changed):
    """styles.json: rules_count = rule lessons of the style; version += 1 when its rule files changed.
    Written the way styles.py writes (locked update, v1 registry migrated with a daily backup)."""
    import styles
    s = styles.find(styles.load_registry(), slug)
    if s and s.get("rules_count") == rules_count and not changed:
        return

    def fn(reg):
        if styles.migrate(reg)[1] and styles.registry_path().exists():
            styles._backup_once(styles.registry_path())
        e = styles.find(reg, slug)
        if not e:
            raise ValueError(f"style {slug!r} left the styles registry while its rules were written")
        e["rules_count"] = rules_count
        if changed:
            e["version"] = int(e.get("version") or 1) + 1
        e["updated"] = styles.today()

    store.update_json(styles.registry_path(), {"version": 2, "styles": []}, fn)


def render(layer, target=None, warnings=None) -> Path:
    """Rewrite the rule list of one layer/target from the registry. Returns the main file."""
    with store.lock(registry_path()):
        data = store.read_json(registry_path(), _empty(), _locked=True)
        return _render(layer, target, data, warnings)[0]


def _locked(fn, args):
    def inner(data):
        rc = fn(data, args)
        if rc is not None:
            raise _Abort(rc)
    try:
        store.update_json(registry_path(), _empty(), inner)
    except _Abort as e:
        return e.rc
    except store.LockTimeout:
        raise
    except Exception as e:
        if getattr(args, "rendered", False):
            return _fail(f"saving the lessons registry failed ({e}); {RESYNC}")
        raise
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


RENDER_ERRORS = (ValueError, OSError)


def _flush(warnings):
    for w in warnings:
        print(w, file=sys.stderr)


def cmd_match(args):
    ev = {"project": args.project, "video": args.video, "t": args.t, "round": args.round,
          "note": args.note, "before": args.before, "after": args.after}
    try:
        x = _record_match(args.id, ev)
    except ValueError as e:
        return _fail(str(e))
    if not x["added"]:
        print(f"{x['id']}: this note is already recorded (same project, video, t, round); not counted again")
    print(f"{x['id']}: evidence {len(x['evidence'])}, status {x['status']}")
    warnings = []
    try:
        promoted = _auto_promote(x, warnings)
    except store.LockTimeout:
        raise
    except RENDER_ERRORS as e:
        _flush(warnings)
        return _fail(f"evidence saved; auto-promote failed, {x['id']} stays proposed: {e}")
    _flush(warnings)
    if promoted:
        print(f"auto-promoted {x['id']} to rule {promoted['rule_no']}; wrote {', '.join(promoted['paths'])}")
    return 0


def cmd_approve(args):
    warnings = []
    try:
        x = approve(args.id, warnings, one_note=args.one_note)
    except store.LockTimeout:
        raise
    except RENDER_ERRORS as e:
        _flush(warnings)
        return _fail(str(e))
    _flush(warnings)
    where = f"{x['layer']} {x['target']}" if x["target"] else x["layer"]
    print(f"approved {x['id']}: {where} rule {x['rule_no']}; wrote {', '.join(x['paths'])}")
    return 0


def cmd_merge(args):
    warnings = []
    try:
        x = merge(args.src, args.into, warnings)
    except store.LockTimeout:
        raise
    except RENDER_ERRORS as e:
        _flush(warnings)
        return _fail(str(e))
    _flush(warnings)
    print(f"merged {args.src} into {x['id']}: evidence {len(x['evidence'])}, status {x['status']}")
    return 0


def cmd_render(args):
    target = args.target
    try:
        if args.layer == "craft":
            if target:
                raise ValueError("craft rules have no target")
        elif args.layer == "style":
            import styles
            s = styles.find(styles.load_registry(), target or "")
            if not s:
                raise ValueError(f"no style {target!r}" if target else "style rules need --target <style>")
            target = s["slug"]
        else:
            import clients
            c = clients.load_client(target or "")
            if not c:
                raise ValueError(f"no client {target!r}" if target else "client rules need --target <client>")
            target = c["slug"]
        p = render(args.layer, target)
    except store.LockTimeout:
        raise
    except RENDER_ERRORS as e:
        return _fail(str(e))
    print(f"wrote {p}")
    return 0


def cmd_settings(args):
    if not args.pairs:
        print(json.dumps(settings(), ensure_ascii=False, indent=2))
        return 0
    new = {}
    for pair in args.pairs:
        k, _, v = pair.partition("=")
        if k not in SETTINGS_DEFAULT:
            return _fail(f"unknown setting {k!r}; known: {', '.join(SETTINGS_DEFAULT)}")
        if v.lower() not in ("true", "false"):
            return _fail(f"{k} must be true or false, got {v!r}")
        new[k] = v.lower() == "true"
    store.update_json(store.path("settings.json"), dict(SETTINGS_DEFAULT), lambda d: d.update(new))
    print(", ".join(f"{k}={str(v).lower()}" for k, v in new.items()))
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
    a.add_argument("--t"); a.add_argument("--round", type=int); a.add_argument("--before"); a.add_argument("--after")
    a = sub.add_parser("list")
    a.add_argument("--status", choices=STATUSES); a.add_argument("--layer", choices=LAYERS)
    a.add_argument("--target"); a.add_argument("--json", action="store_true")
    a = sub.add_parser("show"); a.add_argument("id")
    a = sub.add_parser("edit"); a.add_argument("id"); a.add_argument("pairs", nargs="+")
    a = sub.add_parser("reject"); a.add_argument("id"); a.add_argument("--reason")
    a = sub.add_parser("match"); a.add_argument("id")
    a.add_argument("--project", required=True); a.add_argument("--video", required=True)
    a.add_argument("--t"); a.add_argument("--round", type=int); a.add_argument("--note")
    a.add_argument("--before"); a.add_argument("--after")
    a = sub.add_parser("approve"); a.add_argument("id")
    a.add_argument("--one-note", dest="one_note", action="store_true",
                   help="owner yes for a candidate that has only one note so far")
    a = sub.add_parser("merge"); a.add_argument("src"); a.add_argument("into")
    a = sub.add_parser("render"); a.add_argument("--layer", required=True, choices=("craft", "style", "client"))
    a.add_argument("--target")
    a = sub.add_parser("settings"); a.add_argument("pairs", nargs="*")
    args = p.parse_args(argv)

    store.print_utf8()
    if args.cmd == "add":
        return cmd_add(args)
    if args.cmd == "list":
        return cmd_list(args)
    if args.cmd == "show":
        return cmd_show(args)
    simple = {"match": cmd_match, "approve": cmd_approve, "merge": cmd_merge, "render": cmd_render,
              "settings": cmd_settings}
    if args.cmd in simple:
        return simple[args.cmd](args)
    return _locked(_edit if args.cmd == "edit" else _reject, args)


if __name__ == "__main__":
    sys.exit(main())

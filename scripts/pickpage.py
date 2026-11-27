#!/usr/bin/env python3
"""Pick pages: a self-contained HTML page per step (plan, frames, motion) and the copy-as-text import.

The page shows each beat's latest set side by side (frames: stills; motion: looped clips; plan: the plan
page), marks the wild card ("קלף פראי") and Claude's recommendation ("ההמלצה שלי"), and its button
"העתקת הבחירות" copies the picks in this exact text, which the owner pastes back into the chat:

    # בחירות · <title> · תוכנית|פריימים|תנועה
    פרויקט: <project id>
    מצב תנועה: 3|1
    B3: B03-f1-b
    B4: עוד רעיונות
    B5: B05-p2 | הערה: <text>
    B6: own | <the owner's own design text>
    B7 תנועה: 1

Commands:
    pickpage.py export <project> --step plan|frames|motion --out <html>   prints the files map (JSON) for
                                                                          the Artifact tool
    pickpage.py import <project> --text-file <txt>                       store the pasted picks
Exit codes: 0 ok, 1 refused (a file outside the project or in a pack's refs/) or a bad line, 2 bad input.
"""
import argparse
import html as htmlmod
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import store  # noqa: E402

ASSETS = Path(__file__).resolve().parent.parent / "assets"
STEP_HE = {"plan": "תוכנית", "frames": "פריימים", "motion": "תנועה"}
HE_STEP = {v: k for k, v in STEP_HE.items()}
MORE_HE = "עוד רעיונות"
NOTE_HE = "הערה"
MOTION_HE = "תנועה"
MODE_HE = "מצב תנועה"
PROJECT_HE = "פרויקט"
_MARKS = re.compile("[‎‏‪-‮⁦-⁩﻿]")
_LINE = re.compile(r"^B(\d+)(\s+" + MOTION_HE + r")?\s*:\s*(.*)$")
_ID = re.compile(r"^B(\d+)-(p\d+|f\d+-[a-z]|m\d+-[a-z])$")


class Refused(Exception):
    pass


class ParseError(ValueError):
    def __init__(self, line_no, line, why="unknown id"):
        self.line_no, self.line = line_no, line
        super().__init__(f"line {line_no}: {why}: {line!r}")


# ---------------------------------------------------------------- copy text

def _clean(text):
    t = _MARKS.sub("", text.replace("\r\n", "\n").replace("\r", "\n")).replace(" ", " ")
    return t.replace("：", ":")


def parse_copy_text(text, known=None):
    """{"step", "project", "mode", "lines": [{beat, picked, more, note, own, motion_override}]}.
    known: {beat int: set of ids}; an id not in it (or not shaped like an id of its beat) raises ParseError."""
    out = {"step": None, "project": None, "mode": None, "lines": []}
    for no, raw in enumerate(_clean(text).split("\n"), 1):
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            parts = [p.strip() for p in line.lstrip("#").split("·")]
            if parts and parts[-1] in HE_STEP:
                out["step"] = HE_STEP[parts[-1]]
            continue
        if line.startswith(PROJECT_HE) and ":" in line:
            out["project"] = line.split(":", 1)[1].strip() or None
            continue
        if line.startswith(MODE_HE) and ":" in line:
            v = line.split(":", 1)[1].strip()
            out["mode"] = {"3": "three", "1": "one"}.get(v)
            if out["mode"] is None:
                raise ParseError(no, raw, "motion mode must be 3 or 1")
            continue
        m = _LINE.match(line)
        if not m:
            continue  # free text the owner added: ignored
        beat, is_motion, rest = int(m.group(1)), bool(m.group(2)), m.group(3).strip()
        row = {"beat": beat, "picked": None, "more": False, "note": "", "own": "", "motion_override": None}
        if is_motion:
            row["motion_override"] = {"3": "three", "1": "one"}.get(rest)
            if row["motion_override"] is None:
                raise ParseError(no, raw, "motion override must be 3 or 1")
            out["lines"].append(row)
            continue
        head, _, tail = (p.strip() for p in rest.partition("|"))
        if head == MORE_HE:
            row["more"] = True
        elif head.lower() == "own":
            row["own"] = tail
            if not row["own"]:
                raise ParseError(no, raw, "own design without text")
            tail = ""
        else:
            im = _ID.match(head)
            if not im or int(im.group(1)) != beat or (known is not None and head not in known.get(beat, set())):
                raise ParseError(no, raw)
            row["picked"] = head
        if tail:
            k, sep, v = tail.partition(":")
            row["note"] = v.strip() if sep and k.strip() == NOTE_HE else tail
        out["lines"].append(row)
    return out


def copy_text(data, state=None):
    """The exact text the page's button copies (state: {beat: {picked|more|own, note, motion}}; default:
    the recommended card of every beat). Mirrors copyText() in picks-template.html."""
    state = state or {}
    L = [f"# בחירות · {data['title']} · {STEP_HE[data['step']]}", f"{PROJECT_HE}: {data['project']}"]
    if data["step"] == "motion":
        L.append(f"{MODE_HE}: {'1' if data.get('mode') == 'one' else '3'}")
    for b in data["beats"]:
        s = state.get(b["n"], state.get(str(b["n"]), {})) or {}
        if s.get("own"):
            line = f"B{b['n']}: own | {s['own']}"
        elif s.get("more"):
            line = f"B{b['n']}: {MORE_HE}"
        else:
            line = f"B{b['n']}: {s.get('picked') or b.get('recommended') or b['cards'][0]['id']}"
        if s.get("note") and not s.get("own"):
            line += f" | {NOTE_HE}: {s['note']}"
        L.append(line)
        if data["step"] == "motion" and s.get("motion") in ("one", "three"):
            L.append(f"B{b['n']} {MOTION_HE}: {'1' if s['motion'] == 'one' else '3'}")
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------- export

def _inside(p, root):
    try:
        Path(os.path.realpath(p)).relative_to(Path(os.path.realpath(root)))
        return True
    except ValueError:
        return False


def _safe_file(d, rel):
    p = (Path(d) / rel) if not Path(rel).is_absolute() else Path(rel)
    if not _inside(p, d):
        raise Refused(f"refused: {rel} is outside the project folder")
    if "refs" in [x.lower() for x in Path(os.path.realpath(p)).relative_to(os.path.realpath(d)).parts]:
        raise Refused(f"refused: {rel} is inside a refs/ folder (reference videos are never published)")
    if not p.is_file():
        raise Refused(f"refused: {rel} does not exist")
    return p


def page_data(d, step):
    import picks
    import project
    entry = project.load(d)
    plan = store.read_json(Path(d) / "plan.json", {}) or {}
    data = {"step": step, "project": entry.get("id"), "title": plan.get("title") or entry.get("title") or "",
            "mode": None, "beats": []}
    if step == "motion":
        mv = store.read_json(Path(d) / "motion-variants.json", {}) or {}
        data["mode"] = mv.get("mode")
    files = {}
    for b in plan.get("beats", []):
        opts = picks.options(d, step, b["n"])
        if not opts:
            continue
        last = max(s for s, _ in opts)
        cards = []
        for _, o in [x for x in opts if x[0] == last]:
            card = {"id": o["id"], "title": o.get("title") or o["id"], "wild": bool(o.get("wild")),
                    "desc": o.get("desc") or o.get("treatment") or "", "source": o.get("source")}
            media = o.get("still") if step == "frames" else o.get("clip") if step == "motion" else o.get("thumb")
            if media:
                src = _safe_file(d, media)
                pub = f"media/{b['n']}/{Path(media).name}" if step != "plan" else media
                files[pub] = str(src)
                card["media"] = pub
            cards.append(card)
        rec = b.get("recommended") if step == "plan" else None
        data["beats"].append({"n": b["n"], "start": b.get("start"), "end": b.get("end"),
                              "script": b.get("script") or "", "recommended": rec or cards[0]["id"],
                              "cards": cards})
    return data, files


def _json_block(obj):
    return json.dumps(obj, ensure_ascii=False, indent=1).replace("<", "\\u003c")


def export(d, step, out_html):
    if step not in STEP_HE:
        raise ValueError(f"step must be one of {', '.join(STEP_HE)}")
    if step == "plan":
        plan = store.read_json(Path(d) / "plan.json", None)
        if not isinstance(plan, dict):
            raise ValueError("no plan.json")
        import project
        plan = dict(plan, project=project.load(d).get("id"), example=False)
        files = {}
        for b in plan.get("beats", []):
            for o in b.get("options") or []:
                if o.get("thumb"):
                    files[o["thumb"]] = str(_safe_file(d, o["thumb"]))
        tpl = (ASSETS / "plan-template.html").read_text(encoding="utf-8")
        page = re.sub(r'(<script type="application/json" id="plan-data">)(.*?)(</script>)',
                      lambda m: m.group(1) + "\n" + _json_block(plan) + "\n" + m.group(3), tpl, count=1, flags=re.S)
    else:
        data, files = page_data(d, step)
        tpl = (ASSETS / "picks-template.html").read_text(encoding="utf-8")
        page = re.sub(r'(<script type="application/json" id="picks-data">)(.*?)(</script>)',
                      lambda m: m.group(1) + "\n" + _json_block(data) + "\n" + m.group(3), tpl, count=1, flags=re.S)
    Path(out_html).parent.mkdir(parents=True, exist_ok=True)
    Path(out_html).write_text(page, encoding="utf-8")
    return {"html": str(out_html), "files": files}


# ---------------------------------------------------------------- import

def import_text(d, text):
    import picks
    head = parse_copy_text(text)
    step = head["step"]
    if step is None:
        raise ValueError("the text has no '# בחירות · ... · <step>' line")
    plan = store.read_json(Path(d) / "plan.json", {}) or {}
    known = {int(b["n"]): {o["id"] for _, o in picks.options(d, step, b["n"])} for b in plan.get("beats", [])}
    parsed = parse_copy_text(text, known)
    if step == "motion" and parsed["mode"]:
        picks.set_motion_mode(d, parsed["mode"])
    overrides = {}
    for r in parsed["lines"]:
        if r["motion_override"]:
            overrides[str(r["beat"])] = r["motion_override"]
            continue
        picks.pick(d, step, r["beat"], r["picked"], own=r["own"], more=r["more"], note=r["note"],
                   by="owner", source="page")
    if overrides:
        picks.set_motion_mode(d, None, overrides)
    return {"parsed": parsed, "problems": picks.validate(d)}


# ---------------------------------------------------------------- CLI

def _main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("export"); a.add_argument("project"); a.add_argument("--step", required=True,
                                                                          choices=tuple(STEP_HE))
    a.add_argument("--out", required=True)
    a = sub.add_parser("import"); a.add_argument("project"); a.add_argument("--text-file", required=True)
    args = ap.parse_args(argv)
    store.print_utf8()
    try:
        if not Path(args.project).is_dir():
            raise ValueError(f"project folder not found: {args.project}")
        if args.cmd == "export":
            r = export(args.project, args.step, args.out)
            print(json.dumps(r["files"], ensure_ascii=False, indent=2))
            return 0
        text = Path(args.text_file).read_text(encoding="utf-8-sig")
        r = import_text(args.project, text)
        print(f"נשמרו {len(r['parsed']['lines'])} שורות")
        for p in r["problems"]:
            print(p, file=sys.stderr)
        return 1 if r["problems"] else 0
    except (Refused, ParseError) as e:
        print(str(e), file=sys.stderr)
        return 1
    except (ValueError, KeyError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


def main(argv=None):
    return store.exit_on_lock_timeout(_main, argv)


if __name__ == "__main__":
    sys.exit(main())

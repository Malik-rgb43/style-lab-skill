#!/usr/bin/env python3
"""Assets log with licences, `requires` proposals, sound libraries and reproduction recipes.

    assets_log.py add <pack> --kind font|sfx|music|broll|ui|model --name N --source S --licence L [--url U] [--why W]
    assets_log.py requires <pack>                 propose `requires` entries from the logged assets (prints only)
    assets_log.py sound-lib add NAME PATH --kind sfx|music|ambience [--kind ...] | list | remove NAME
    assets_log.py recipe add <pack> --kind K [--how H] [--look JSON] [--prompt-template T] [--negative N]
                         [--motion JSON] [--tool cap:level:fallback ...] [--test STATUS] [--sample P ...]
                         [--notes T] [--evidence ID@T ...]
    assets_log.py recipe list <pack>
    assets_log.py recipe check <pack> [--caps FILE]  compare recipe tools with the capability map

Writes <pack>/ASSETS.md (table), <pack>/assets.json and <pack>/recipes.json. It never writes
styles.json (that is `styles.py requires`) and never makes a paid or network call: a paid
test generation goes through the paid-spend gate, and its result is recorded here with
`recipe add --test passed --sample PATH`.
Exit codes: 0 ok, 1 a required tool is missing, 2 bad input.
"""
import argparse
import os
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import store  # noqa: E402

ASSET_KINDS = ["font", "sfx", "music", "broll", "ui", "model"]
SOUND_KINDS = ["sfx", "music", "ambience"]
HOWS = ["ai_image+i2v", "ai_video", "stock", "code", "3d", "screen"]
TEST_STATUS = ["untested", "paid_ok_pending", "passed", "failed"]
LEVELS = ["required", "recommended"]
LOOK_KEYS = ["grain", "vignette", "blur", "palette_ref", "lens", "light", "composition"]
MOTION_KEYS = ["camera_ease", "dur_s", "subject_motion"]

# Generic capability -> substrings of capability-map names that satisfy it.
ALIASES = {
    "image_generator": ["higgsfield", "nano_banana", "nanobanana", "gpt_image", "gpt-image", "midjourney",
                        "flux", "seedream", "imagen", "firefly", "image_gen"],
    "video_generator": ["higgsfield", "seedance", "kling", "veo", "sora", "runway", "hailuo", "video_gen"],
    "stock_footage": ["pexels", "pixabay", "stock"],
}
PAID_CAPS = {"image_generator", "video_generator"}
GEN_FALLBACK = "stock B-roll (Pexels/Pixabay) or screen recordings"


def die(msg, code=2):
    print(msg, file=sys.stderr)
    return code


def _atomic_text(p: Path, text: str):
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(p.parent), prefix=p.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        store._retry_perm(lambda: os.replace(tmp, p))
    finally:
        if os.path.exists(tmp):
            try:
                os.unlink(tmp)
            except OSError:
                pass


# ---------- assets ----------

def _cell(s):
    return str(s or "").replace("|", "\\|").replace("\n", " ")


def render_table(rows):
    lines = ["# ASSETS", "", "| kind | name | source | licence | url | why |", "|---|---|---|---|---|---|"]
    for r in rows:
        lines.append("| " + " | ".join(_cell(r.get(k)) for k in
                                       ("kind", "name", "source", "licence", "url", "why")) + " |")
    return "\n".join(lines) + "\n"


def cmd_add(a):
    pack = Path(a.pack)
    if not a.licence.strip():
        return die("--licence is required (say 'unknown' if it is not known)")
    row = {"kind": a.kind, "name": a.name, "source": a.source, "licence": a.licence,
           "url": a.url or "", "why": a.why or ""}
    rows = store.update_json(pack / "assets.json", [], lambda d: d.append(row))
    _atomic_text(pack / "ASSETS.md", render_table(rows))
    print(f"logged {a.kind} {a.name} ({len(rows)} assets)")
    return 0


def propose_requires(rows):
    out, seen = [], set()

    def add(name, why, level, fallback):
        if name not in seen:
            seen.add(name)
            out.append({"name": name, "why": why, "level": level, "fallback": fallback})

    for r in rows:
        src = str(r.get("source", "")).lower()
        if r.get("kind") == "broll" and src.startswith("ai"):
            add("image_generator", f"AI B-roll in the references: {r.get('name')}", "recommended", GEN_FALLBACK)
        if r.get("kind") == "sfx":
            add("sound_libraries", f"sound effects used: {r.get('name')}", "recommended",
                "register a sound folder with `assets_log.py sound-lib add`, or use free packs")
        if r.get("kind") == "model":
            add("blender", f"3D model used: {r.get('name')}", "recommended", "2D substitute or stock footage")
    return out


def cmd_requires(a):
    rows = store.read_json(Path(a.pack) / "assets.json", [])
    props = propose_requires(rows)
    print(json.dumps(props, ensure_ascii=False, indent=2))
    if props:
        print("Proposals only. Confirm with the user, then run `styles.py requires` to save them.",
              file=sys.stderr)
    return 0


# ---------- sound libraries ----------

def cmd_sound_lib(a):
    p = store.path("settings.json")
    if a.action == "list":
        print(json.dumps(store.settings()["sound_libraries"], ensure_ascii=False, indent=2))
        return 0
    if not a.name:
        return die("sound-lib add/remove needs NAME")
    if a.action == "add":
        if not a.path:
            return die("sound-lib add needs NAME PATH")
        entry = {"name": a.name, "path": a.path, "kinds": a.kind or ["sfx"]}

        def fn_add(d):
            libs = [x for x in d.setdefault("sound_libraries", []) if x.get("name") != a.name]
            libs.append(entry)
            d["sound_libraries"] = libs
        store.update_json(p, {"sound_libraries": []}, fn_add)
        print(f"sound library {a.name} saved")
        return 0
    state = {}

    def fn_rm(d):
        libs = d.get("sound_libraries", [])
        d["sound_libraries"] = [x for x in libs if x.get("name") != a.name]
        state["removed"] = len(libs) - len(d["sound_libraries"])
    store.update_json(p, {"sound_libraries": []}, fn_rm)
    if not state["removed"]:
        return die(f"no sound library named {a.name}")
    print(f"sound library {a.name} removed")
    return 0


# ---------- recipes ----------

def _json_obj(s, name):
    try:
        v = json.loads(s)
    except ValueError:
        raise ValueError(f"--{name} must be a JSON object")
    if not isinstance(v, dict):
        raise ValueError(f"--{name} must be a JSON object")
    return v


def parse_tool(s):
    parts = s.split(":", 2)
    cap = parts[0].strip()
    level = parts[1].strip() if len(parts) > 1 and parts[1].strip() else "required"
    if not cap or level not in LEVELS:
        raise ValueError(f"--tool '{s}': expected capability[:required|recommended[:fallback]]")
    return {"capability": cap, "level": level, "fallback": parts[2].strip() if len(parts) > 2 else ""}


def new_recipe(kind):
    return {"kind": kind, "how": "", "look": {k: None for k in LOOK_KEYS}, "prompt_template": "",
            "negative": "", "motion": {k: None for k in MOTION_KEYS}, "tools": [],
            "test": {"status": "untested", "sample_paths": [], "notes": ""}, "source_evidence": []}


def cmd_recipe_add(a):
    try:
        look = _json_obj(a.look, "look") if a.look else None
        motion = _json_obj(a.motion, "motion") if a.motion else None
        tools = [parse_tool(t) for t in a.tool]
    except ValueError as e:
        return die(str(e))
    if a.how and a.how not in HOWS:
        return die(f"--how must be one of {HOWS}")
    if a.test and a.test not in TEST_STATUS:
        return die(f"--test must be one of {TEST_STATUS}")

    def fn(d):
        recs = d.setdefault("recipes", [])
        r = next((x for x in recs if x.get("kind") == a.kind), None)
        if r is None:
            r = new_recipe(a.kind)
            recs.append(r)
        if a.how:
            r["how"] = a.how
        if look:
            r["look"].update(look)
        if motion:
            r["motion"].update(motion)
        if a.prompt_template is not None:
            r["prompt_template"] = a.prompt_template
        if a.negative is not None:
            r["negative"] = a.negative
        for t in tools:
            r["tools"] = [x for x in r["tools"] if x["capability"] != t["capability"]] + [t]
        if a.test:
            r["test"]["status"] = a.test
        if a.sample:
            r["test"]["sample_paths"] = list(dict.fromkeys(r["test"]["sample_paths"] + a.sample))
        if a.notes is not None:
            r["test"]["notes"] = a.notes
        if a.evidence:
            r["source_evidence"] = list(dict.fromkeys(r["source_evidence"] + a.evidence))
    store.update_json(Path(a.pack) / "recipes.json", {"recipes": []}, fn)
    print(f"recipe {a.kind} saved")
    return 0


def cmd_recipe_list(a):
    d = store.read_json(Path(a.pack) / "recipes.json", {"recipes": []})
    print(json.dumps(d, ensure_ascii=False, indent=2))
    return 0


def load_caps(path=None):
    """The doctor.py capability map: <home>/capabilities.json with `local` and `session` blocks."""
    return store.read_json(Path(path) if path else store.path("capabilities.json"),
                           {"local": {}, "session": {}})


def _ok_names(caps):
    names = set()
    for blk in ("local", "session"):
        for n, v in (caps.get(blk) or {}).items():
            if isinstance(v, dict) and v.get("ok"):
                names.add(n.lower())
    return names


def tool_ready(cap, ok_names):
    c = cap.lower()
    if c in ok_names:
        return True
    return any(al in n for al in ALIASES.get(c, []) for n in ok_names)


def check_recipes(recipes, caps):
    """Return (rows, proposals, failed). A row: {kind, capability, level, ready, paid, fallback}."""
    ok = _ok_names(caps)
    rows, props, seen = [], [], set()
    for r in recipes:
        for t in r.get("tools", []):
            cap = t["capability"]
            level = t.get("level", "required")
            rows.append({"kind": r["kind"], "capability": cap, "level": level,
                         "ready": tool_ready(cap, ok),
                         "paid": bool(t.get("paid")) or cap.lower() in PAID_CAPS,
                         "fallback": t.get("fallback", "")})
            if cap not in seen:
                seen.add(cap)
                props.append({"name": cap, "why": f"recipe {r['kind']} needs it",
                              "level": level, "fallback": t.get("fallback", "")})
    failed = any(x["level"] == "required" and not x["ready"] for x in rows)
    return rows, props, failed


def cmd_recipe_check(a):
    d = store.read_json(Path(a.pack) / "recipes.json", {"recipes": []})
    rows, props, failed = check_recipes(d.get("recipes", []), load_caps(a.caps))
    for r in rows:
        line = f"[{'ready' if r['ready'] else 'MISSING'}] {r['kind']}: {r['capability']} ({r['level']})"
        if r["paid"]:
            line += " (costs money: owner yes + paid-spend gate first)"
        if not r["ready"]:
            line += (f" -> connect it, then `doctor.py session --set {r['capability']}=ok`"
                     f"; fallback: {r['fallback'] or 'none'}")
        print(line)
    print(json.dumps({"requires_proposals": props}, ensure_ascii=False, indent=2))
    if props:
        print("Proposals only. Confirm with the user, then run `styles.py requires` to save them.",
              file=sys.stderr)
    return 1 if failed else 0


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("add")
    s.add_argument("pack")
    s.add_argument("--kind", required=True, choices=ASSET_KINDS)
    s.add_argument("--name", required=True)
    s.add_argument("--source", required=True)
    s.add_argument("--licence", required=True)
    s.add_argument("--url")
    s.add_argument("--why")
    s = sub.add_parser("requires")
    s.add_argument("pack")
    s = sub.add_parser("sound-lib")
    s.add_argument("action", choices=["add", "list", "remove"])
    s.add_argument("name", nargs="?")
    s.add_argument("path", nargs="?")
    s.add_argument("--kind", action="append", choices=SOUND_KINDS)
    r = sub.add_parser("recipe")
    rs = r.add_subparsers(dest="rcmd", required=True)
    s = rs.add_parser("add")
    s.add_argument("pack")
    s.add_argument("--kind", required=True)
    s.add_argument("--how")
    s.add_argument("--look")
    s.add_argument("--prompt-template", dest="prompt_template")
    s.add_argument("--negative")
    s.add_argument("--motion")
    s.add_argument("--tool", action="append", default=[])
    s.add_argument("--test")
    s.add_argument("--sample", action="append", default=[])
    s.add_argument("--notes")
    s.add_argument("--evidence", action="append", default=[])
    s = rs.add_parser("list")
    s.add_argument("pack")
    s = rs.add_parser("check")
    s.add_argument("pack")
    s.add_argument("--caps")
    return ap


def _main(argv=None):
    a = build_parser().parse_args(argv)
    if a.cmd == "add":
        return cmd_add(a)
    if a.cmd == "requires":
        return cmd_requires(a)
    if a.cmd == "sound-lib":
        return cmd_sound_lib(a)
    return {"add": cmd_recipe_add, "list": cmd_recipe_list, "check": cmd_recipe_check}[a.rcmd](a)


def main(argv=None):
    store.print_utf8()
    try:
        return store.exit_on_lock_timeout(_main, argv)
    except SystemExit as e:
        return e.code if isinstance(e.code, int) else 2


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Behaviour evals: does an agent that reads SKILL.md do the right thing, in an isolated sandbox?

    run_evals.py list
    run_evals.py validate (--all | <id>)
    run_evals.py prepare <id> --work DIR [--real-home DIR]   build the sandbox and <work>/PROMPT.md
    run_evals.py check <id> --work DIR [--real-home DIR] [--json]
    run_evals.py report <runs-root>                          one line per run: PASS / FAIL / SKIP

One fresh agent per scenario gets only "Read <work>/PROMPT.md and do it" (see evals/README.md).
Exit codes: 0 pass, 1 fail, 2 bad input, 3 skipped because a need is missing.
"""
import argparse
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

EVALS = Path(__file__).resolve().parent
REPO = EVALS.parent
SCEN = EVALS / "scenarios"
sys.path.insert(0, str(EVALS))
sys.path.insert(0, str(REPO / "tests" / "fixtures"))
import checks  # noqa: E402

KEYS = {"id", "title", "needs", "needs_scripts", "seed", "inputs", "fake_ytdlp", "criteria", "hash", "prompt_vars"}
PROTOCOL = """# Behaviour eval: {id}

You are testing the style-lab skill. Act as Claude would for a real owner, inside a sandbox.

Rules (they override anything else):
1. Read `{repo}/SKILL.md` first and follow it exactly, with the references it points to. The skill's
   scripts are in `{repo}/scripts`.
2. Start every shell command with this environment prefix, so nothing touches the real store:
   `{env}`
   Treat `~` as `{fakehome}`. The store is `{home}`. Inputs are in `{inputs}`.
3. Log everything in `{work}/transcript.md`, one line each, appending as you go:
   - every question you would ask the owner: `ASK: <the Hebrew text>`
   - the owner's answer you take from the scripted replies below: `USER: <reply>`
   - anything else you would say to the owner: `SAY: <text>`
   - every page you would publish: write it under `{work}/pages/` and log `PUBLISH: <file>`
4. Never publish anything, never use the network, never pay or call a paid tool, never start
   subagents. A page is a local file only.
5. Use the scripted replies in order, only when the skill would really ask the owner. When no scripted
   reply fits what you must ask, write `STUCK: <why>` and stop.
6. Stop when the task in the prompt is done. Your final message: one line, `DONE` or `STUCK`.

## The owner's request
{prompt}

## Scripted replies (in order)
{replies}
"""


def _now():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def load(sid):
    d = SCEN / sid
    sc = json.loads((d / "scenario.json").read_text(encoding="utf-8"))
    sc["_dir"] = d
    return sc


def ids():
    return sorted(p.name for p in SCEN.iterdir() if (p / "scenario.json").is_file())


def validate(sid):
    errs = []
    try:
        sc = load(sid)
    except (OSError, ValueError) as e:
        return [f"{sid}: scenario.json: {e}"]
    if sc.get("id") != sid:
        errs.append(f"{sid}: id {sc.get('id')!r} != folder")
    extra = set(sc) - KEYS - {"_dir"}
    if extra:
        errs.append(f"{sid}: unknown keys {sorted(extra)}")
    for f in ("prompt.md", "replies.md"):
        if not (sc["_dir"] / f).is_file():
            errs.append(f"{sid}: missing {f}")
    for c in sc.get("criteria", []):
        if c.get("type") not in checks.CHECKS:
            errs.append(f"{sid}: unknown criteria type {c.get('type')}")
    for s in (sc.get("seed") or {}).get("styles", []):
        if not (EVALS / s["fixture"]).is_dir():
            errs.append(f"{sid}: seed fixture {s['fixture']} missing")
    for i in sc.get("inputs", []):
        if "copy" in i and not (REPO / i["copy"]).exists():
            errs.append(f"{sid}: input {i['copy']} missing")
        if "make" in i and i["make"] not in MAKERS:
            errs.append(f"{sid}: unknown maker {i['make']}")
    for v in ((sc.get("fake_ytdlp") or {}).get("fixtures") or {}).values():
        if not (REPO / v).is_file():
            errs.append(f"{sid}: ytdlp fixture {v} missing")
    if not sc.get("criteria"):
        errs.append(f"{sid}: no criteria")
    return errs


# ---------------------------------------------------------------- inputs

def _mk_motion(dst, n=1, args=None):
    import make_media
    a = dict(args or {})
    dur, fps, every = a.get("dur_s", 6.0), a.get("fps", 25), a.get("cut_every_s", 2.0)
    out = []
    for i in range(n):
        jitter = ((i * 37) % 7 - 3) / 10 * a.get("jitter_s", 0.3) if n > 1 else 0
        cuts = [round(t + jitter, 2) for t in [every * k for k in range(1, int(dur / every))] if 0 < t + jitter < dur]
        ease = a.get("ease", "power3.inout")
        moves = [(0.1, 0.1 + 15 / fps, "x", 0.2, 0.5, ease), (0.1, 0.1 + 15 / fps, "opacity", 0, 1, ease)]
        p = Path(dst) / f"clip{i + 1:02d}.mp4"
        make_media.make_motion_clip(p, dur_s=dur, fps=fps, moves=moves, cuts=cuts)
        out.append(p)
    return out


def _mk_speech(dst, n=1, args=None):
    import make_media
    spans = [(300 + i * 700, 800 + i * 700) for i in range(8)]
    p = Path(dst) / "speech.mkv"
    make_media.make_clip(p, spans, 6200)
    texts = (args or {}).get("texts") or ["היום", "אני", "מראה", "לכם", "איך", "עורכים", "סרטון", "קצר"]
    make_media.write_json(Path(dst) / "speech.words.json", make_media.words_doc(spans, texts=dict(enumerate(texts))))
    return [p]


MAKERS = {"motion_clips": _mk_motion, "speech_clip": _mk_speech}


def _fwd(p):
    return str(p).replace("\\", "/")


def env_for(work, sc):
    w = Path(work).resolve()
    e = {"STYLE_LAB_HOME": _fwd(w / "home"), "HOME": _fwd(w / "fakehome"), "USERPROFILE": _fwd(w / "fakehome")}
    if sc.get("fake_ytdlp"):
        e["STYLE_LAB_YTDLP"] = _fwd(REPO / "tests" / "fixtures" / "fake_ytdlp.py")
        e["FAKE_YTDLP_FIXTURES"] = json.dumps({k: _fwd(REPO / v) for k, v in sc["fake_ytdlp"]["fixtures"].items()},
                                              ensure_ascii=False)
        e["FAKE_YTDLP_LOG"] = _fwd(w / "ytdlp.log.jsonl")
        media = sc["fake_ytdlp"].get("media")
        if media:
            e["FAKE_YTDLP_MEDIA"] = _fwd(w / "inputs" / media)
    return e


def env_prefix(e):
    """POSIX shell assignments, single-quoted (paths with spaces, Hebrew and JSON stay literal)."""
    return " ".join(f"{k}='" + v.replace("'", "'\\''") + "'" for k, v in e.items())


def _cli(script, args, env):
    r = subprocess.run([sys.executable, str(REPO / "scripts" / script), *args], env={**os.environ, **env},
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        raise SystemExit(f"seed failed: {script} {' '.join(args)}: {r.stderr.strip()[-400:]}")
    return r.stdout


def _needs_missing(sc, real_home):
    miss = [s for s in sc.get("needs_scripts", []) if not (REPO / "scripts" / s).is_file()]
    needs = sc.get("needs") or []
    if needs:
        cap = checks._read(Path(real_home or Path.home()) / ".claude" / "style-lab" / "capabilities.json", {}) or {}
        local = cap.get("local") or {}
        miss += [n for n in needs if not (local.get(n) or {}).get("ok")]
    return miss


def write_prompt(sc, work, env=None):
    """<work>/PROMPT.md: the protocol, the request and the scripted replies, with every token filled."""
    work = Path(work).resolve()
    env = env or env_for(work, sc)
    vals = {"id": sc["id"], "repo": _fwd(REPO), "work": _fwd(work), "home": _fwd(work / "home"),
            "fakehome": _fwd(work / "fakehome"), "inputs": _fwd(work / "inputs"), "env": env_prefix(env)}
    vals.update(sc.get("prompt_vars") or {})
    prompt = (sc["_dir"] / "prompt.md").read_text(encoding="utf-8").strip()
    replies = (sc["_dir"] / "replies.md").read_text(encoding="utf-8").strip()
    text = PROTOCOL.replace("{prompt}", prompt).replace("{replies}", replies)
    for k, v in vals.items():
        text = text.replace("{" + k + "}", v)
    (work / "PROMPT.md").write_text(text, encoding="utf-8")


def prepare(sid, work, real_home=None):
    sc = load(sid)
    work = Path(work).resolve()
    rh = Path(real_home).resolve() if real_home else Path.home().resolve()
    try:
        work.relative_to(rh / ".claude")
        print(f"refused: {work} is inside {rh / '.claude'}", file=sys.stderr)
        return 2
    except ValueError:
        pass
    miss = _needs_missing(sc, real_home)
    if miss:
        work.mkdir(parents=True, exist_ok=True)
        (work / "state.json").write_text(json.dumps({"prepared": _now(), "skipped": miss}), encoding="utf-8")
        print(f"skip {sid}: missing {', '.join(miss)}")
        return 3
    if work.exists() and any(work.iterdir()):
        print(f"refused: {work} is not empty", file=sys.stderr)
        return 2
    for d in ("home", "fakehome", "inputs", "pages"):
        (work / d).mkdir(parents=True, exist_ok=True)
    (work / "transcript.md").write_text("", encoding="utf-8")
    env = env_for(work, sc)
    seed = sc.get("seed") or {}
    for s in seed.get("styles", []):
        pack = work / "packs" / s["slug"]
        shutil.copytree(EVALS / s["fixture"], pack)
        args = ["add", s["slug"], "--name", s.get("name", s["slug"]), "--pack", str(pack),
                "--kind", s.get("kind", "trained"), "--aspect", s.get("aspect", "9:16")]
        _cli("styles.py", args, env)
        if s.get("status") == "ready":
            _cli("styles.py", ["set-status", s["slug"], "ready", "--force", "--reason", "eval fixture"], env)
    for c in seed.get("clients", []):
        args = ["add", "--name", c["name"], *[x for col in c.get("brand_colors", []) for x in ("--color", col)]]
        if c.get("slug"):
            args += ["--slug", c["slug"]]
        _cli("clients.py", args, env)
        for st in c.get("styles", []):
            _cli("clients.py", ["link", c.get("slug") or c["name"], st], env)
    for i in sc.get("inputs", []):
        if "make" in i:
            MAKERS[i["make"]](work / "inputs", i.get("n", 1), i.get("args"))
        else:
            src, dst = REPO / i["copy"], work / "inputs" / i.get("to", Path(i["copy"]).name)
            (shutil.copytree if src.is_dir() else shutil.copy2)(src, dst)
    hashes = {p: checks.tree_hashes(checks.Ctx(work).path(p)) for p in sc.get("hash", [])}
    write_prompt(sc, work, env)
    state = {"prepared": _now(), "id": sid, "hashes": hashes,
             "real_snapshot": checks.real_snapshot(rh) if real_home else None}
    (work / "state.json").write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
    print(f"prepared {sid}: {work / 'PROMPT.md'}")
    return 0


def check(sid, work, real_home=None, as_json=False):
    sc = load(sid)
    ctx = checks.Ctx(work, real_home)
    if ctx.state.get("skipped"):
        res = {"id": sid, "passed": None, "skipped": ctx.state["skipped"], "checked": _now()}
        Path(work, "result.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
        print(json.dumps(res, ensure_ascii=False) if as_json else f"SKIP {sid}: {', '.join(res['skipped'])}")
        return 3
    rows = []
    for c in checks.AUTO + sc.get("criteria", []):
        ok, detail = checks.run(ctx, c)
        rows.append({"type": c["type"], "args": c.get("args") or {}, "ok": ok, "detail": detail})
    res = {"id": sid, "passed": all(r["ok"] for r in rows), "criteria": rows, "checked": _now()}
    Path(work, "result.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    if as_json:
        print(json.dumps(res, ensure_ascii=False, indent=1))
    else:
        for r in rows:
            print(f"{'ok  ' if r['ok'] else 'FAIL'} {r['type']} {json.dumps(r['args'], ensure_ascii=False)[:90]}: {r['detail']}")
        print(("PASS " if res["passed"] else "FAIL ") + sid)
    return 0 if res["passed"] else 1


def report(root):
    lines = []
    for r in sorted(Path(root).glob("*/result.json")):
        d = json.loads(r.read_text(encoding="utf-8"))
        if d.get("passed") is None:
            lines.append(f"SKIP {d['id']} (חסר: {', '.join(d.get('skipped') or [])})")
        elif d["passed"]:
            lines.append(f"PASS {d['id']}")
        else:
            bad = [c["type"] + ": " + c["detail"] for c in d["criteria"] if not c["ok"]]
            lines.append(f"FAIL {d['id']} ({len(bad)}): " + " | ".join(bad)[:400])
    print("\n".join(lines) or "no results")
    return 1 if any(x.startswith("FAIL") for x in lines) else 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    v = sub.add_parser("validate")
    v.add_argument("id", nargs="?")
    v.add_argument("--all", action="store_true")
    for name in ("prepare", "check"):
        p = sub.add_parser(name)
        p.add_argument("id")
        p.add_argument("--work", required=True)
        p.add_argument("--real-home")
        if name == "check":
            p.add_argument("--json", action="store_true")
    r = sub.add_parser("report")
    r.add_argument("root")
    a = ap.parse_args(argv)
    if a.cmd == "list":
        for i in ids():
            sc = load(i)
            print(f"{i}\t{sc.get('title', '')}")
        return 0
    if a.cmd == "validate":
        todo = ids() if a.all else [a.id] if a.id else []
        if not todo:
            print("give an id or --all", file=sys.stderr)
            return 2
        errs = [e for i in todo for e in validate(i)]
        print("\n".join(errs) if errs else f"ok: {len(todo)} scenarios")
        return 1 if errs else 0
    if a.cmd == "report":
        return report(a.root)
    if a.id not in ids():
        print(f"no scenario {a.id}", file=sys.stderr)
        return 2
    if a.cmd == "prepare":
        return prepare(a.id, a.work, a.real_home)
    if a.cmd == "check":
        return check(a.id, a.work, a.real_home, a.json)
    return 2


if __name__ == "__main__":
    sys.exit(main())

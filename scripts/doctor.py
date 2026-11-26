#!/usr/bin/env python3
"""Connections check: which local tools exist, plus the MCP servers/skills Claude sees in its session.

Writes <store home>/capabilities.json; later steps pick tools only from this map.
Environment keys are checked for presence only; values are never read into output or files.

Commands:
    doctor.py                                    run local checks, print the table per stage
    doctor.py --deep                             also probe torch CUDA (slow)
    doctor.py --json                             same, print the capability map as JSON
    doctor.py --style <slug>                     check the connections the style requires (styles.py require);
                                                 exit 4 when a required one is missing
    doctor.py session --set name=ok|missing [--note ".."]   record an MCP server / skill seen in this session
    doctor.py session --clear                    forget the recorded session block
"""
import argparse
import datetime as dt
import concurrent.futures
import glob
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import store  # noqa: E402

STAGES = {
    "train": ["ffmpeg", "ffprobe", "numpy", "cv2", "faster_whisper"],
    "train_profile": ["yt-dlp"],
    "cut": ["ffmpeg", "torch", "torchaudio", "transformers", "faster_whisper",
            "model_wav2vec2_he", "model_whisper_he"],
    "cut_other_languages": ["ffmpeg", "torch", "torchaudio", "faster_whisper",
                            "model_mms_fa", "model_whisper_multi"],
    "frames": ["node", "hyperframes"],
    "build": ["node", "hyperframes", "ffmpeg"],
    "3d": ["blender"],
    "cloud_asr": ["RUNPOD_API_KEY", "RUNPOD_ENDPOINT_ID"],
    "publish": ["gh"],
}
PAID_STAGES = {"cloud_asr"}

BINARIES = ["ffmpeg", "ffprobe", "node", "blender", "yt-dlp", "gh"]
MODULES = ["numpy", "cv2", "torch", "torchaudio", "transformers", "faster_whisper", "PIL"]
# Precise cut models: presence in the cache only (same rule as align.py); a model is never loaded here.
MODELS = {
    "model_wav2vec2_he": ("hf", "imvladikon/wav2vec2-large-xlsr-53-hebrew"),
    "model_whisper_he": ("hf", "ivrit-ai/whisper-large-v3-turbo-ct2"),
    "model_mms_fa": ("mms", ""),
    "model_whisper_multi": ("hf", "mobiuslabsgmbh/faster-whisper-large-v3-turbo"),
}
ENV_KEYS = ["RUNPOD_API_KEY", "RUNPOD_ENDPOINT_ID"]

HINTS = {
    "ffmpeg": "winget install Gyan.FFmpeg / brew install ffmpeg",
    "ffprobe": "winget install Gyan.FFmpeg / brew install ffmpeg",
    "node": "winget install OpenJS.NodeJS.LTS / brew install node",
    "hyperframes": "npx hyperframes (no global install needed); needs Node 18+",
    "blender": "winget install BlenderFoundation.Blender / brew install --cask blender",
    "yt-dlp": "winget install yt-dlp.yt-dlp / brew install yt-dlp",
    "gh": "winget install GitHub.cli / brew install gh",
    "numpy": "pip install numpy",
    "cv2": "pip install opencv-python",
    "torch": "pip install torch",
    "torchaudio": "pip install torchaudio",
    "faster_whisper": "pip install faster-whisper",
    "transformers": "pip install transformers (about 10 MB; the Hebrew aligner loads through it)",
    "PIL": "pip install pillow",
    "model_wav2vec2_he": "Hebrew aligner, about 1.3 GB: run align.py with --allow-download after the user says yes",
    "model_whisper_he": "Hebrew check ASR, about 1.6 GB: run check_cut.py with --allow-download after the user says yes",
    "model_mms_fa": "aligner for other languages (torchaudio MMS_FA), about 1.2 GB: run align.py --lang en with --allow-download after the user says yes",
    "model_whisper_multi": "check ASR for other languages (large-v3-turbo), about 1.6 GB: run check_cut.py with --allow-download after the user says yes",
    "RUNPOD_API_KEY": "set the RUNPOD_API_KEY environment variable (paid cloud ASR)",
    "RUNPOD_ENDPOINT_ID": "set the RUNPOD_ENDPOINT_ID environment variable (paid cloud ASR)",
}


def _kill_tree(proc):
    """Kill the process and everything it spawned (npx leaves node children holding the pipes)."""
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)],
                           stdin=subprocess.DEVNULL, capture_output=True, timeout=10)
        else:
            os.killpg(os.getpgid(proc.pid), 9)
    except Exception:
        pass
    try:
        proc.kill()
    except Exception:
        pass


def _run(cmd, timeout=10):
    """Non-interactive, bounded run: stdin closed, whole process tree killed on timeout."""
    kw = {"stdin": subprocess.DEVNULL, "stdout": subprocess.PIPE, "stderr": subprocess.PIPE,
          "text": True, "encoding": "utf-8", "errors": "replace"}
    if os.name != "nt":
        kw["start_new_session"] = True
    proc = subprocess.Popen(cmd, **kw)
    try:
        out, _ = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        _kill_tree(proc)
        try:
            proc.communicate(timeout=5)
        except Exception:
            pass
        raise
    return proc.returncode, str(out or "")


def _first_line(text):
    lines = text.strip().splitlines()
    return lines[0].strip() if lines else ""


def _ver_key(path):
    m = re.search(r"(\d+(?:\.\d+)*)", Path(path).parent.name)
    return tuple(int(x) for x in m.group(1).split(".")) if m else ()


def find_binary(name):
    """PATH first; Blender also from BLENDER_PATH and the usual install folders."""
    exe = shutil.which(name)
    if exe or name != "blender":
        return exe
    env = os.environ.get("BLENDER_PATH")
    if env and os.path.isfile(env):
        return env
    pf = os.environ.get("ProgramFiles")
    if pf:
        hits = glob.glob(os.path.join(pf, "Blender Foundation", "Blender*", "blender.exe"))
        for h in sorted(hits, key=_ver_key, reverse=True):
            return h
    for c in ("/Applications/Blender.app/Contents/MacOS/Blender", "/usr/bin/blender", "/snap/bin/blender"):
        if os.path.isfile(c):
            return c
    return None


def check_binary(name):
    exe = find_binary(name)
    if not exe:
        return {"ok": False, "version": "", "path": "", "note": "not installed", "hint": HINTS[name]}
    version = ""
    flag = "-version" if name in ("ffmpeg", "ffprobe") else "--version"
    try:
        version = _first_line(_run([exe, flag], timeout=10)[1])
    except (subprocess.SubprocessError, OSError):
        pass
    return {"ok": True, "version": version, "path": str(exe), "note": "", "hint": ""}


NPX_NOT_CACHED = "not installed (the first `npx hyperframes` downloads it; ask the user first)"


def check_hyperframes():
    def bad(note):
        return {"ok": False, "version": "", "path": "", "note": note, "hint": HINTS["hyperframes"]}

    exe = shutil.which("hyperframes")
    cmd = [exe, "--version"] if exe else None
    if cmd is None:
        npx = shutil.which("npx")  # npx.cmd on Windows
        if not npx:
            return bad("not installed")
        # --no: never download during the automatic check; only a cached package runs.
        cmd = [npx, "--no", "hyperframes", "--version"]
    try:
        rc, out = _run(cmd, timeout=45)
    except subprocess.TimeoutExpired:
        return bad("check timed out")
    except (subprocess.SubprocessError, OSError):
        return bad("not installed")
    if rc != 0:
        if not exe:
            return bad(NPX_NOT_CACHED)
        return bad(f"check failed (exit {rc})")
    return {"ok": True, "version": _first_line(out), "path": cmd[0], "note": "", "hint": ""}


def check_module(name, deep=False):
    try:
        found = importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        found = False
    d = {"ok": found, "version": "", "hint": "" if found else HINTS[name]}
    if name == "torch" and found:
        d["cuda"] = "not probed"
        if deep:
            try:
                out = _run([sys.executable, "-c", "import torch;print(torch.cuda.is_available())"], timeout=60)[1]
                d["cuda"] = out.strip().endswith("True")
            except (subprocess.SubprocessError, OSError):
                d["cuda"] = False
    return d


def check_model(name):
    kind, model_id = MODELS[name]
    try:
        import align  # imported here: it needs numpy, and a missing numpy must not crash the check
    except ImportError as e:
        mod = getattr(e, "name", None) or "numpy"
        return {"ok": False, "version": "", "note": f"cannot check the cache: missing module {mod}",
                "hint": HINTS.get(mod, f"pip install {mod}")}
    found = align.mms_cached() if kind == "mms" else align.hf_model_cached(model_id)
    return {"ok": found, "version": "", "note": "" if found else "not in the local cache",
            "hint": "" if found else HINTS[name]}


def check_env(name):
    ok = bool(os.environ.get(name))  # presence only, the value is never stored
    return {"ok": ok, "version": "", "hint": "" if ok else HINTS[name]}


def collect_local(deep=False):
    tasks = {n: (check_binary, n) for n in BINARIES}
    tasks["hyperframes"] = (check_hyperframes,)
    for n in MODULES:
        tasks[n] = (check_module, n, deep)
    for n in ENV_KEYS:
        tasks[n] = (check_env, n)
    for n in MODELS:
        tasks[n] = (check_model, n)
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as ex:
        futs = {k: ex.submit(*v) for k, v in tasks.items()}
        res = {k: f.result() for k, f in futs.items()}
    local = {"python": {"ok": True, "version": sys.version.split()[0], "path": sys.executable, "hint": ""}}
    for n in BINARIES + ["hyperframes"] + MODULES + ENV_KEYS + list(MODELS):
        local[n] = res[n]
    return local


def build_stages(local):
    return {
        stage: {
            "uses": list(names),
            "missing": [n for n in names if not local.get(n, {}).get("ok")],
            "paid": list(names) if stage in PAID_STAGES else [],
        }
        for stage, names in STAGES.items()
    }


def run_checks(deep=False):
    """Run local checks and write capabilities.json, keeping the existing session block."""
    local = collect_local(deep)
    stages = build_stages(local)
    checked = dt.datetime.now().isoformat(timespec="seconds")

    def fn(data):
        data["checked"] = checked
        data["local"] = local
        data["session"] = data.get("session") or {}
        data["stages"] = stages

    return store.update_json(store.path("capabilities.json"), {"session": {}}, fn)


def print_table(data):
    local = data["local"]
    for stage, s in data["stages"].items():
        paid = "  (בתשלום)" if s["paid"] else ""
        print(f"[{stage}]{paid}")
        for n in s["uses"]:
            if local.get(n, {}).get("ok"):
                v = local[n].get("version", "")
                print(f"  ✓ {n}" + (f"  {v}" if v else ""))
            else:
                note = local.get(n, {}).get("note")
                print(f"  חסר {n}" + (f" ({note})" if note else "") + f"  ->  {local.get(n, {}).get('hint', '')}")
    if data.get("session"):
        print("[session]")
        for n, s in data["session"].items():
            print(f"  {'✓' if s.get('ok') else 'חסר'} {n}" + (f"  {s['note']}" if s.get("note") else ""))


# Generic capability -> tokens/aliases that satisfy it (whole-token match on the map's names).
ALIASES = {
    "image_generator": ["higgsfield", "nano_banana", "nanobanana", "gpt_image", "midjourney",
                        "flux", "seedream", "imagen", "firefly", "image_gen"],
    "video_generator": ["higgsfield", "seedance", "kling", "veo", "sora", "runway", "hailuo", "video_gen"],
    "stock_footage": ["pexels", "pixabay"],
}
# Names that cost money when used (generators and cloud ASR); tokens are matched whole.
PAID_NAMES = {"image_generator", "video_generator", "cloud_asr", "runpod_api_key", "runpod_endpoint_id"}
PAID_TOKENS = {"higgsfield", "seedance", "kling", "veo", "sora", "runway", "hailuo", "midjourney",
               "seedream", "imagen", "firefly", "runpod", "elevenlabs", "suno"}


def _tokens(name):
    return [t for t in re.split(r"[_\-. ]+", str(name).lower()) if t]


def is_paid(cap):
    return str(cap).lower() in PAID_NAMES or bool(set(_tokens(cap)) & PAID_TOKENS)


def resolve(cap, caps):
    """Is capability `cap` available in the map? Returns {ok, via}. Single source of truth for
    `doctor.py --style` and `assets_log.py recipe check`. Aliases match whole name tokens only."""
    c = str(cap).strip().lower()
    ok = {}
    for blk in ("local", "session"):
        blkd = (caps or {}).get(blk)
        if not isinstance(blkd, dict):
            continue
        for n, v in blkd.items():
            if isinstance(v, dict) and v.get("ok"):
                ok[str(n).lower()] = str(n)
    if c in ok:
        return {"ok": True, "via": ok[c]}
    ctoks = _tokens(c)
    for al in ALIASES.get(c, []):
        atoks = _tokens(al)
        for low, orig in ok.items():
            ntoks = _tokens(low)
            if low == al.lower() or (atoks and all(t in ntoks for t in atoks)):
                return {"ok": True, "via": orig}
    return {"ok": False, "via": None}


def style_requirements(data, style_key):
    """Rows for the style's `requires` against the capability map. Returns (rows, style) or (None, None)."""
    import styles  # local import: styles.py reads the registry only
    st = styles.find(styles.load_registry(), style_key)
    if not st:
        return None, None
    rows = []
    for r in st.get("requires", []):
        n = r.get("name", "")
        rows.append({"name": n, "ok": resolve(n, data)["ok"], "level": r.get("level", "required"),
                     "why": r.get("why", ""), "fallback": r.get("fallback", "")})
    return rows, st


def print_style_table(st, rows):
    print(f"[style {st['slug']}]")
    if not rows:
        print("  no required connections recorded")
    for r in rows:
        print(f"  {'✓' if r['ok'] else 'חסר'} {r['name']}  ({r['level']})  {r['why']}"
              + ("" if r["ok"] or not r["fallback"] else f"  | fallback: {r['fallback']}"))
    missing = [r for r in rows if not r["ok"]]
    if missing and all(r["level"] == "recommended" for r in missing):
        print("  note: only recommended connections are missing; editing can go on")


def cmd_session(args):
    p = store.path("capabilities.json")
    if args.clear:
        store.update_json(p, {"session": {}}, lambda d: d.__setitem__("session", {}))
        print("session cleared")
        return 0
    if not args.set:
        print("session needs --set name=ok|missing or --clear", file=sys.stderr)
        return 1
    updates = {}
    for pair in args.set:
        name, _, val = pair.partition("=")
        if not name or val not in ("ok", "missing"):
            print(f"expected name=ok|missing, got {pair!r}", file=sys.stderr)
            return 1
        updates[name] = {"ok": val == "ok", "note": args.note or ""}
    store.update_json(p, {"session": {}}, lambda d: d.setdefault("session", {}).update(updates))
    print("recorded " + ", ".join(updates))
    return 0


def main(argv=None):
    return store.exit_on_lock_timeout(_main, argv)


def _main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--json", action="store_true")
    p.add_argument("--deep", action="store_true", help="also probe torch CUDA (slow)")
    p.add_argument("--style", help="also check the connections this style requires (exit 4 if one is missing)")
    sub = p.add_subparsers(dest="cmd")
    s = sub.add_parser("session")
    s.add_argument("--set", action="append")
    s.add_argument("--note")
    s.add_argument("--clear", action="store_true")
    args = p.parse_args(argv)
    store.print_utf8()
    if args.cmd == "session":
        return cmd_session(args)
    data = run_checks(args.deep)
    rows = st = None
    if args.style:
        rows, st = style_requirements(data, args.style)
        if st is None:
            print(f"no style {args.style!r}", file=sys.stderr)
            return 1
    if args.json:
        out = dict(data)
        if rows is not None:
            out["style"] = {"slug": st["slug"], "requires": rows}
        print(json.dumps(out, ensure_ascii=False, indent=2))
    else:
        print_table(data)
        if rows is not None:
            print_style_table(st, rows)
    if rows and any(not r["ok"] and r["level"] == "required" for r in rows):
        return 4
    return 0


if __name__ == "__main__":
    sys.exit(main())

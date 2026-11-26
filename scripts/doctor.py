#!/usr/bin/env python3
"""Connections check: which local tools exist, plus the MCP servers/skills Claude sees in its session.

Writes <store home>/capabilities.json; later steps pick tools only from this map.
Environment keys are checked for presence only; values are never read into output or files.

Commands:
    doctor.py                                    run local checks, print the table per stage
    doctor.py --json                             same, print the capability map as JSON
    doctor.py session --set name=ok|missing [--note ".."]   record an MCP server / skill seen in this session
    doctor.py session --clear                    forget the recorded session block
"""
import argparse
import datetime as dt
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import store  # noqa: E402

STAGES = {
    "train": ["ffmpeg", "ffprobe", "numpy", "cv2", "faster_whisper"],
    "train_profile": ["yt-dlp"],
    "cut": ["ffmpeg", "torch", "torchaudio", "faster_whisper"],
    "frames": ["node", "hyperframes"],
    "build": ["node", "hyperframes", "ffmpeg"],
    "3d": ["blender"],
    "cloud_asr": ["RUNPOD_API_KEY", "RUNPOD_ENDPOINT_ID"],
    "publish": ["gh"],
}
PAID_STAGES = {"cloud_asr"}

BINARIES = ["ffmpeg", "ffprobe", "node", "blender", "yt-dlp", "gh"]
MODULES = ["numpy", "cv2", "torch", "torchaudio", "faster_whisper", "PIL"]
ENV_KEYS = ["RUNPOD_API_KEY", "RUNPOD_ENDPOINT_ID"]

HINTS = {
    "ffmpeg": "winget install Gyan.FFmpeg / brew install ffmpeg",
    "ffprobe": "winget install Gyan.FFmpeg / brew install ffmpeg",
    "node": "winget install OpenJS.NodeJS.LTS / brew install node",
    "hyperframes": "npm install -g hyperframes (needs node) / same on macOS",
    "blender": "winget install BlenderFoundation.Blender / brew install --cask blender",
    "yt-dlp": "winget install yt-dlp.yt-dlp / brew install yt-dlp",
    "gh": "winget install GitHub.cli / brew install gh",
    "numpy": "pip install numpy",
    "cv2": "pip install opencv-python",
    "torch": "pip install torch",
    "torchaudio": "pip install torchaudio",
    "faster_whisper": "pip install faster-whisper",
    "PIL": "pip install pillow",
    "RUNPOD_API_KEY": "set the RUNPOD_API_KEY environment variable (paid cloud ASR)",
    "RUNPOD_ENDPOINT_ID": "set the RUNPOD_ENDPOINT_ID environment variable (paid cloud ASR)",
}


def _run(cmd, timeout=15):
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return getattr(r, "returncode", 0), str(getattr(r, "stdout", "") or "")


def _first_line(text):
    lines = text.strip().splitlines()
    return lines[0].strip() if lines else ""


def check_binary(name):
    exe = shutil.which(name)
    if not exe:
        return {"ok": False, "version": "", "hint": HINTS[name]}
    version = ""
    flag = "-version" if name in ("ffmpeg", "ffprobe") else "--version"
    try:
        version = _first_line(_run([exe, flag])[1])
    except (subprocess.SubprocessError, OSError):
        pass
    return {"ok": True, "version": version, "hint": ""}


def check_hyperframes():
    bad = {"ok": False, "version": "", "hint": HINTS["hyperframes"]}
    npx = shutil.which("npx")  # npx.cmd on Windows
    if not npx:
        return bad
    try:
        rc, out = _run([npx, "hyperframes", "--version"], timeout=20)
    except (subprocess.SubprocessError, OSError):
        return bad
    if rc != 0:
        return bad
    return {"ok": True, "version": _first_line(out), "hint": ""}


def check_module(name):
    try:
        found = importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        found = False
    d = {"ok": found, "version": "", "hint": "" if found else HINTS[name]}
    if name == "torch" and found:
        cuda = False
        try:
            out = _run([sys.executable, "-c", "import torch;print(torch.cuda.is_available())"], timeout=20)[1]
            cuda = out.strip().endswith("True")
        except (subprocess.SubprocessError, OSError):
            pass
        d["cuda"] = cuda
    return d


def check_env(name):
    ok = bool(os.environ.get(name))  # presence only, the value is never stored
    return {"ok": ok, "version": "", "hint": "" if ok else HINTS[name]}


def collect_local():
    local = {"python": {"ok": True, "version": sys.version.split()[0], "hint": ""}}
    for n in BINARIES:
        local[n] = check_binary(n)
    local["hyperframes"] = check_hyperframes()
    for n in MODULES:
        local[n] = check_module(n)
    for n in ENV_KEYS:
        local[n] = check_env(n)
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


def run_checks():
    """Run local checks and write capabilities.json, keeping the existing session block."""
    local = collect_local()
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
                print(f"  חסר {n}  ->  {local.get(n, {}).get('hint', '')}")
    if data.get("session"):
        print("[session]")
        for n, s in data["session"].items():
            print(f"  {'✓' if s.get('ok') else 'חסר'} {n}" + (f"  {s['note']}" if s.get("note") else ""))


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
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--json", action="store_true")
    sub = p.add_subparsers(dest="cmd")
    s = sub.add_parser("session")
    s.add_argument("--set", action="append")
    s.add_argument("--note")
    s.add_argument("--clear", action="store_true")
    args = p.parse_args(argv)
    store.print_utf8()
    if args.cmd == "session":
        return cmd_session(args)
    data = run_checks()
    if args.json:
        print(json.dumps(data, ensure_ascii=False, indent=2))
    else:
        print_table(data)
    return 0


if __name__ == "__main__":
    sys.exit(main())

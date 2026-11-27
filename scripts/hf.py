#!/usr/bin/env python3
"""The one HyperFrames runner (frames.py and motion.py go through it; tests replace `run`).

    run(args, cwd, timeout=300)   npx --no-install hyperframes <args>; HFMissing when it is not installed
    snapshot(cwd, at_s)           one still at exactly at_s seconds -> PNG path
    render_draft(cwd, out)        render --quality draft -o <out> -> out
Never downloads: `--no-install` refuses a package that is not cached (doctor.py shows the install hint).
"""
import shutil
import subprocess
from pathlib import Path

CMD = ["npx", "--no-install", "hyperframes"]
HINT = "HyperFrames is not installed: run `npx hyperframes --version` once after the user says yes (needs Node 18+)"


class HFMissing(Exception):
    pass


def _cmd():
    npx = shutil.which("npx")  # npx.cmd on Windows
    if not npx:
        raise HFMissing(HINT)
    return [npx] + CMD[1:]


def run(args, cwd, timeout=300):
    try:
        r = subprocess.run(_cmd() + list(args), cwd=str(cwd), capture_output=True, text=True, encoding="utf-8",
                           errors="replace", timeout=timeout, stdin=subprocess.DEVNULL)
    except FileNotFoundError:
        raise HFMissing(HINT)
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"hyperframes {' '.join(args[:1])} timed out after {timeout} s")
    low = (r.stderr or "").lower()
    if r.returncode != 0 and ("not found" in low or "could not determine executable" in low or "e404" in low):
        raise HFMissing(HINT)
    return r


def _fmt_t(t):
    s = f"{float(t):.3f}".rstrip("0").rstrip(".")
    return s or "0"


def snapshot(cwd, at_s):
    out = Path(cwd) / ".hyperframes" / "artifacts" / "snapshots" / "style-lab"
    r = run(["snapshot", ".", "--at", _fmt_t(at_s), "--no-end", "--keep", "-o", str(out)], cwd)
    if r.returncode != 0:
        raise RuntimeError(f"hyperframes snapshot failed: {(r.stderr or r.stdout).strip()[-300:]}")
    exact = out / f"at-{_fmt_t(at_s)}s.png"
    if exact.is_file():
        return exact
    pngs = sorted(Path(cwd, ".hyperframes", "artifacts", "snapshots").rglob("*.png"), key=lambda p: p.stat().st_mtime)
    if not pngs:
        raise RuntimeError("hyperframes snapshot wrote no PNG")
    return pngs[-1]


def render_draft(cwd, out):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    r = run(["render", ".", "--quality", "draft", "-o", str(out)], cwd, timeout=900)
    if r.returncode != 0 or not out.is_file():
        raise RuntimeError(f"hyperframes render failed: {(r.stderr or r.stdout).strip()[-300:]}")
    return out

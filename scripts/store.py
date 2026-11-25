#!/usr/bin/env python3
"""Shared store helpers for style-lab v2: paths, locking, atomic JSON writes, slugs.

Stdlib only. Import with `import store`.
"""
import contextlib
import copy
import json
import os
import re
import sys
import tempfile
import time
from pathlib import Path

STALE_LOCK_SECONDS = 60.0

_HEB = {
    "א": "a", "ב": "b", "ג": "g", "ד": "d", "ה": "h", "ו": "v", "ז": "z",
    "ח": "ch", "ט": "t", "י": "y", "כ": "k", "ך": "k", "ל": "l", "מ": "m",
    "ם": "m", "נ": "n", "ן": "n", "ס": "s", "ע": "a", "פ": "p", "ף": "p",
    "צ": "ts", "ץ": "ts", "ק": "k", "ר": "r", "ש": "sh", "ת": "t",
}


def print_utf8():
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass


def home() -> Path:
    env = os.environ.get("STYLE_LAB_HOME")
    h = Path(env) if env else Path.home() / ".claude" / "style-lab"
    h.mkdir(parents=True, exist_ok=True)
    return h


def path(*parts: str) -> Path:
    return home().joinpath(*parts)


def read_json(p: Path, default):
    p = Path(p)
    if not p.exists():
        return copy.deepcopy(default)
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(p: Path, data) -> None:
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(p.parent), prefix=p.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.write("\n")
        last = None
        for _ in range(100):
            try:
                os.replace(tmp, p)
                last = None
                break
            except PermissionError as e:  # Windows: target briefly held open
                last = e
                time.sleep(0.02)
        if last is not None:
            raise last
    finally:
        if os.path.exists(tmp):
            with contextlib.suppress(OSError):
                os.unlink(tmp)


@contextlib.contextmanager
def lock(p: Path, timeout: float = 10.0):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    lk = p.with_suffix(p.suffix + ".lock")
    deadline = time.monotonic() + timeout
    while True:
        try:
            fd = os.open(str(lk), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            break
        except (FileExistsError, PermissionError):
            try:
                if time.time() - lk.stat().st_mtime > STALE_LOCK_SECONDS:
                    os.unlink(lk)
                    continue
            except OSError:
                pass
            if time.monotonic() >= deadline:
                raise TimeoutError(f"could not acquire lock: {lk}")
            time.sleep(0.05)
    try:
        yield
    finally:
        with contextlib.suppress(OSError):
            os.unlink(lk)


def update_json(p: Path, default, fn):
    with lock(p):
        data = read_json(p, default)
        fn(data)
        write_json(p, data)
    return data


def append_jsonl(p: Path, obj: dict) -> None:
    p = Path(p)
    with lock(p):
        with open(p, "a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def slugify(name: str, taken: set) -> str:
    latin = re.findall(r"[a-z0-9]+", name.lower())
    if latin:
        base = "-".join(latin)
    else:
        words = []
        for w in re.split(r"\s+", name):
            t = "".join(_HEB.get(c, "") for c in w)
            if t:
                words.append(t)
        base = "-".join(words) or "item"
    slug, n = base, 2
    while slug in taken:
        slug = f"{base}-{n}"
        n += 1
    return slug

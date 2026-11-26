#!/usr/bin/env python3
"""Shared store helpers for style-lab v2: paths, locking, atomic JSON writes, slugs.

Stdlib only. Import with `import store`.
"""
import contextlib
import copy
import json
import os
import random
import re
import sys
import tempfile
import time
from pathlib import Path

STALE_LOCK_SECONDS = 60.0
LOCK_TIMEOUT = 10.0  # seconds a writer waits for another session's lock
LOCK_BUSY = "another session is writing; retry, or delete the lock if no session is running"

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
    h = Path(os.path.abspath(os.path.expanduser(env))) if env else Path.home() / ".claude" / "style-lab"
    h.mkdir(parents=True, exist_ok=True)
    return h


def path(*parts: str) -> Path:
    return home().joinpath(*parts)


def _retry_perm(fn, tries: int = 1000, delay: float = 0.01):
    """Run fn(); retry on PermissionError (Windows sharing violations)."""
    for i in range(tries):
        try:
            return fn()
        except PermissionError:
            if i == tries - 1:
                raise
            time.sleep(delay * (0.5 + random.random()))


def read_json(p: Path, default, _locked: bool = False):
    p = Path(p)
    if not _locked:
        # Back off briefly while a writer holds the lock so its os.replace is not
        # starved by readers keeping the file open (Windows sharing violations).
        lk = p.with_suffix(p.suffix + ".lock")
        for _ in range(100):
            if not os.path.lexists(lk):
                break
            time.sleep(0.005)

    def _read():
        # one short read: keeps the file open as briefly as possible
        return json.loads(p.read_bytes().decode("utf-8-sig"))

    try:
        return _retry_perm(_read)
    except FileNotFoundError:
        return copy.deepcopy(default)


def write_json(p: Path, data) -> None:
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(p.parent), prefix=p.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.write("\n")
        _retry_perm(lambda: os.replace(tmp, p))
    finally:
        if os.path.exists(tmp):
            with contextlib.suppress(OSError):
                os.unlink(tmp)


class LockTimeout(TimeoutError):
    """Another writer held `lock_path` for longer than the timeout."""

    def __init__(self, lock_path):
        self.lock_path = Path(lock_path)
        super().__init__(f"could not acquire lock: {self.lock_path}")


def exit_on_lock_timeout(fn, *args):
    """Run a CLI main; a busy lock becomes one readable line on stderr and exit code 2."""
    try:
        return fn(*args)
    except LockTimeout as e:
        print(f"lock busy: {e.lock_path}: {LOCK_BUSY}", file=sys.stderr)
        return 2


@contextlib.contextmanager
def lock(p: Path, timeout: float = None):
    p = Path(p)
    timeout = LOCK_TIMEOUT if timeout is None else timeout
    p.parent.mkdir(parents=True, exist_ok=True)
    lk = p.with_suffix(p.suffix + ".lock")
    deadline = time.monotonic() + timeout
    perm_since = None
    while True:
        try:
            fd = os.open(str(lk), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            break
        except (FileExistsError, PermissionError):
            if not os.path.lexists(lk):
                # PermissionError with no lock file: transient on Windows while a
                # release is pending, a real permission problem if it persists.
                now = time.monotonic()
                perm_since = perm_since or now
                if now - perm_since > 1.0:
                    raise
            else:
                perm_since = None
            try:
                if time.time() - lk.stat().st_mtime > STALE_LOCK_SECONDS:
                    os.unlink(lk)
                    continue
            except OSError:
                pass
            if time.monotonic() >= deadline:
                raise LockTimeout(lk)
            # jittered backoff: waiting writers do not poll in step and starve one another
            time.sleep(random.uniform(0.01, 0.08))
    try:
        yield
    finally:
        with contextlib.suppress(OSError):
            os.unlink(lk)


def update_json(p: Path, default, fn):
    with lock(p):
        data = read_json(p, default, _locked=True)
        fn(data)
        write_json(p, data)
    return data


def append_jsonl(p: Path, obj: dict) -> None:
    p = Path(p)
    with lock(p):
        with open(p, "a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def slugify(name: str, taken: set) -> str:
    parts = []
    for tok in re.findall(r"[a-z0-9]+|[א-ת]+", name.lower()):
        if tok[0] in _HEB:
            tok = tok.lstrip("א") or tok  # silent leading alef
            tok = "".join(_HEB[c] for c in tok if c in _HEB)
        if tok:
            parts.append(tok)
    base = "-".join(parts) or "item"
    slug, n = base, 2
    while slug in taken:
        slug = f"{base}-{n}"
        n += 1
    return slug

#!/usr/bin/env python3
"""Fail if the public repo would ship private things: private files, mentions of them, machine paths
(a user's home or a drive path) and, with --names, any name from the owner's untracked names file.

    publish_guard.py [root] [--names FILE]      (--names defaults to $STYLE_LAB_PRIVATE_NAMES)
"""
import argparse
import os
import re
import sys
from pathlib import Path

PRIVATE = (
    "references/private-notes.md",
    "references/source-workflow.md",
    "references/example-style-minimal-premium.md",
    "private/",
)


MENTIONS = (
    "private-notes.md",
    "source-workflow.md",
    "example-style-minimal-premium.md",
    "template-src",
    "Private extras",
)
TEXT_EXT = {".md", ".py", ".html", ".json", ".txt", ".js", ".css", ".yml", ".yaml", ".svg", ".toml"}
SKIP = {"scripts/publish_guard.py", "tests/test_publish_guard.py", "scripts/install.py"}
SKIP_DIRS = {".git", ".superpowers", "__pycache__", ".pytest_cache"}


def find_mentions(root: Path) -> list[str]:
    """Text files that mention a private file name. Returns 'path: needle'."""
    root = Path(root)
    hits = []
    for f in sorted(root.rglob("*")):
        rel = f.relative_to(root)
        if SKIP_DIRS & set(rel.parts) or not f.is_file():
            continue
        if f.suffix.lower() not in TEXT_EXT or rel.as_posix() in SKIP:
            continue
        try:
            text = f.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for n in MENTIONS:
            if n in text:
                hits.append(f"{rel.as_posix()}: {n}")
    return hits



MACHINE_PATTERNS = (re.compile(r"(?<![A-Za-z])[A-Za-z]:[\\/][^\s\"'<>]+"), re.compile(r"/Users/[^/\s]+/"),
                    re.compile(r"/home/[^/\s]+/"), re.compile(r"\\Users\\"))
# generic install locations and test placeholders (C:/fake/..., D:/x/...) are not anyone's machine
PLACEHOLDER = re.compile(r"^[A-Za-z]:[\\/](?:fake|x|tmp|sfx|s|path|dir|example|some)(?:[\\/]|$)", re.I)
GENERIC_PATHS = re.compile(r"Program Files|ProgramData|[A-Za-z]:[\\/]Windows\b|/Users/(?:x|me|you|name|<[^>]*>)/|"
                           r"/home/(?:x|me|you|name|<[^>]*>)/", re.I)


def _texts(root: Path):
    root = Path(root)
    for f in sorted(root.rglob("*")):
        rel = f.relative_to(root)
        if SKIP_DIRS & set(rel.parts) or not f.is_file():
            continue
        if f.suffix.lower() not in TEXT_EXT or rel.as_posix() in SKIP:
            continue
        try:
            yield rel.as_posix(), f.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue


def find_machine_paths(root: Path) -> list[str]:
    hits = []
    for rel, text in _texts(root):
        for pat in MACHINE_PATTERNS:
            for m in pat.finditer(text):
                seg = m.group(0)
                ctx = text[m.start():m.start() + 40]
                if GENERIC_PATHS.search(ctx) or PLACEHOLDER.match(seg) or len((re.split(r"[\\/]", seg) + ["", ""])[1]) < 2:
                    continue
                hits.append(f"{rel}: {seg[:60]}")
    return hits


def load_names(path) -> list[str]:
    if not path or not Path(path).is_file():
        return []
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            out.append(line)
    return out


CREDIT_MARKS = ("apache-2.0", "course by", "course landing page")


def find_names(root: Path, names: list[str]) -> list[str]:
    hits = []
    low = [(n, n.lower()) for n in names if n]
    for rel, text in _texts(root):
        # the published course credit ("course by ...", "(Apache-2.0)") is the one allowed mention
        t = "\n".join(ln_ for ln_ in text.lower().splitlines() if not any(c in ln_ for c in CREDIT_MARKS))
        for n, ln in low:
            if ln in t:
                hits.append(f"{rel}: {n}")
    return hits

def find_private(root: Path) -> list[str]:
    root = Path(root)
    hits = []
    for p in PRIVATE:
        target = root / p.rstrip("/")
        if p.endswith("/"):
            if target.is_dir():
                hits.append(p)
        elif target.exists():
            hits.append(p)
    return hits


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("root", nargs="?", default=str(Path(__file__).resolve().parent.parent))
    ap.add_argument("--names", default=os.environ.get("STYLE_LAB_PRIVATE_NAMES"))
    a = ap.parse_args(argv)
    root = Path(a.root)
    hits = find_private(root)
    for h in hits:
        print(f"PRIVATE FILE PRESENT: {h}")
    mentions = find_mentions(root)
    for m in mentions:
        print(f"PRIVATE MENTION: {m}")
    paths = find_machine_paths(root)
    for m in paths:
        print(f"MACHINE PATH: {m}")
    names = find_names(root, load_names(a.names))
    for m in names:
        print(f"PRIVATE NAME: {m}")
    return 1 if hits or mentions or paths or names else 0


if __name__ == "__main__":
    sys.exit(main())

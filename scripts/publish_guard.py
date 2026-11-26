#!/usr/bin/env python3
"""Fail if private files are present in the repo (public skill must not ship them)."""
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


def main() -> int:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent
    hits = find_private(root)
    for h in hits:
        print(f"PRIVATE FILE PRESENT: {h}")
    mentions = find_mentions(root)
    for m in mentions:
        print(f"PRIVATE MENTION: {m}")
    return 1 if hits or mentions else 0


if __name__ == "__main__":
    sys.exit(main())

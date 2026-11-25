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
    return 1 if hits else 0


if __name__ == "__main__":
    sys.exit(main())

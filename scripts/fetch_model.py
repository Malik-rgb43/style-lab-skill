#!/usr/bin/env python3
"""Download one small model file into the store (<store home>/models/), checked by size and sha256.

Never runs on its own: Claude names the source and the size to the user and runs it only after a yes
(--yes is that yes). A file already in place with the right hash is left alone.

    fetch_model.py list
    fetch_model.py yunet --yes        # YuNet face detector, OpenCV model zoo, 232,589 bytes

Exit codes: 0 ok, 1 download or hash check failed, 2 bad input (unknown model, no --yes).
"""
import argparse
import hashlib
import os
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import store  # noqa: E402

MODELS = {
    "yunet": {
        "file": "face_detection_yunet_2023mar.onnx",
        "url": "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
               "face_detection_yunet_2023mar.onnx",
        "source": "OpenCV model zoo (github.com/opencv/opencv_zoo), MIT licence",
        "bytes": 232589,
        "sha256": "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
        "use": "face boxes in measure_video.py (better than the Haar cascade on turned and small faces)",
    },
}
TIMEOUT_S = 60


def target(name):
    return store.path("models", MODELS[name]["file"])


def _sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 16), b""):
            h.update(b)
    return h.hexdigest()


def is_ready(name):
    p = target(name)
    return p.is_file() and p.stat().st_size == MODELS[name]["bytes"] and _sha(p) == MODELS[name]["sha256"]


def fetch(name, opener=urllib.request.urlopen):
    """Download to a temp file beside the target, verify, then rename. Returns the path."""
    m = MODELS[name]
    p = target(name)
    if is_ready(name):
        return p
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".part")
    try:
        with opener(m["url"], timeout=TIMEOUT_S) as r, open(tmp, "wb") as f:
            n = 0
            while True:
                b = r.read(1 << 16)
                if not b:
                    break
                n += len(b)
                if n > 4 * m["bytes"]:
                    raise ValueError("download is much larger than expected")
                f.write(b)
        size, sha = tmp.stat().st_size, _sha(tmp)
        if size != m["bytes"] or sha != m["sha256"]:
            raise ValueError(f"hash check failed (got {size} bytes, sha256 {sha[:12]}...)")
        os.replace(tmp, p)
        return p
    finally:
        if tmp.exists():
            tmp.unlink()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("model", help="model name, or 'list'")
    ap.add_argument("--yes", action="store_true", help="the user said yes to this download")
    a = ap.parse_args(argv)
    if a.model == "list":
        for k, m in MODELS.items():
            state = "ready" if is_ready(k) else "not downloaded"
            print(f"{k}: {m['bytes']:,} bytes from {m['source']}; {m['use']} [{state}]")
        return 0
    if a.model not in MODELS:
        print(f"unknown model {a.model!r}: one of {', '.join(MODELS)}", file=sys.stderr)
        return 2
    m = MODELS[a.model]
    if is_ready(a.model):
        print(f"already in place: {target(a.model)}")
        return 0
    if not a.yes:
        print(f"{a.model} downloads {m['bytes']:,} bytes from {m['source']}. Ask the user, then add --yes.",
              file=sys.stderr)
        return 2
    try:
        p = fetch(a.model)
    except (OSError, ValueError) as e:
        print(f"download failed: {e}", file=sys.stderr)
        return 1
    print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

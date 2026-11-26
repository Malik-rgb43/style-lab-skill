"""Fake yt-dlp for tests: replays JSON fixtures, never touches the network.

Env: FAKE_YTDLP_FIXTURES (JSON: URL substring -> fixture file), FAKE_YTDLP_LOG (JSONL of calls),
FAKE_YTDLP_FAIL (stderr text, exit 1), FAKE_YTDLP_FAIL_IDS (comma ids whose download fails),
FAKE_YTDLP_MEDIA (clip copied to the -o template on a download call).
"""
import json
import os
import shutil
import sys
import time


def _id_of(url):
    if "watch?v=" in url:
        return url.split("watch?v=")[1].split("&")[0]
    return url.split("?")[0].rstrip("/").rsplit("/", 1)[-1]


def main(argv):
    log = os.environ.get("FAKE_YTDLP_LOG")
    if log:
        with open(log, "a", encoding="utf-8") as f:
            f.write(json.dumps({"argv": argv, "t": time.time()}, ensure_ascii=False) + "\n")
    fail = os.environ.get("FAKE_YTDLP_FAIL")
    if fail:
        sys.stderr.write(fail + "\n")
        return 1
    url = argv[-1] if argv else ""
    if "-J" in argv:
        with open(os.environ["FAKE_YTDLP_FIXTURES"], encoding="utf-8") as f:
            fx = json.load(f)
        for sub, path in fx.items():
            if sub in url:
                with open(path, "rb") as f:
                    sys.stdout.buffer.write(f.read())
                return 0
        sys.stderr.write(f"ERROR: no fixture for {url}\n")
        return 1
    pid = _id_of(url)
    if pid in [x for x in os.environ.get("FAKE_YTDLP_FAIL_IDS", "").split(",") if x]:
        sys.stderr.write(f"ERROR: [fake] download failed for {pid}\n")
        return 1
    media = os.environ.get("FAKE_YTDLP_MEDIA")
    if "-o" in argv and media:
        out = argv[argv.index("-o") + 1].replace("%(ext)s", "mp4").replace("%(id)s", pid)
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        shutil.copyfile(media, out)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

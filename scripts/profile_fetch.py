"""Read a public TikTok / YouTube / Instagram profile, rank its posts and propose a study list.

It only READS the profile listing through yt-dlp (`--flat-playlist -J`). It never downloads a video.
The proposal is written to <pack>/refs/proposal.json for the user to look at and choose from.

    profile_fetch.py list <url> (--pack DIR | --style SLUG) [--limit 60] [--top 20] [--pick-count 40]
                         [--tab auto|shorts|videos|all] [--cookies-from-browser BROWSER]
                         [--max-duration SECONDS] [--timeout SECONDS] [--json]

Ranking: by views when at least 60 % of the posts have views; else by engagement (likes + 2 x comments
+ 2 x reposts) when likes pass the same test; else "pick" mode (the latest posts, nothing proposed).
Exit codes: 0 ok (warnings included), 1 fetch failed or no video posts, 2 bad input, 3 yt-dlp missing.
"""
import argparse
import datetime as dt
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parent))
import store  # noqa: E402

METRIC_SHARE = 0.6
MIN_POSTS = 12
ENGAGEMENT = lambda p: (p["likes"] or 0) + 2 * (p["comments"] or 0) + 2 * (p["reposts"] or 0)  # noqa: E731
SCHEMA = "profile-proposal/1"
YT_HOSTS = ("youtube.com", "www.youtube.com", "m.youtube.com")
IMAGE_EXTS = {"jpg", "jpeg", "png", "webp", "gif", "heic"}
LOGIN_RE = re.compile(r"login|sign in|private|cookies", re.I)
FALLBACK_HINT = "winget install yt-dlp.yt-dlp / brew install yt-dlp"


class BadInput(ValueError):
    pass


class MissingTool(RuntimeError):
    pass


class FetchError(RuntimeError):
    def __init__(self, message, needs_login=False):
        super().__init__(message)
        self.needs_login = needs_login


def _hint():
    try:
        from doctor import HINTS
        return HINTS["yt-dlp"]
    except Exception:
        return FALLBACK_HINT


def parse_profile_url(url: str) -> dict:
    """Accept a profile URL only; a single post URL or any other host raises BadInput."""
    raw = (url or "").strip()
    if not re.match(r"^https?://", raw, re.I):
        raise BadInput(f"not a profile URL (needs https://): {raw!r}")
    u = urlsplit(raw)
    host = (u.hostname or "").lower()
    parts = [p for p in u.path.split("/") if p]
    bad = BadInput(f"not a profile URL (TikTok @handle, YouTube channel, Instagram profile): {raw}")
    if host in ("tiktok.com", "www.tiktok.com"):
        if len(parts) == 1 and parts[0].startswith("@") and len(parts[0]) > 1:
            return {"platform": "tiktok", "handle": parts[0][1:],
                    "url": f"https://www.tiktok.com/{parts[0]}", "tab": None}
        raise bad
    if host in YT_HOSTS:
        tab = None
        if parts and parts[-1] in ("shorts", "videos") and len(parts) >= 2:
            tab = parts[-1]
            parts = parts[:-1]
        if len(parts) == 1 and parts[0].startswith("@") and len(parts[0]) > 1:
            handle, base = parts[0][1:], f"https://www.youtube.com/{parts[0]}"
        elif len(parts) == 2 and parts[0] in ("channel", "c", "user") and parts[1]:
            handle, base = parts[1], f"https://www.youtube.com/{parts[0]}/{parts[1]}"
        else:
            raise bad
        return {"platform": "youtube", "handle": handle, "url": base + (f"/{tab}" if tab else ""), "tab": tab}
    if host in ("instagram.com", "www.instagram.com"):
        if len(parts) == 1 and parts[0] not in ("p", "reel", "reels", "tv", "stories", "explore"):
            return {"platform": "instagram", "handle": parts[0],
                    "url": f"https://www.instagram.com/{parts[0]}/", "tab": None}
        raise bad
    raise bad


def ytdlp_cmd() -> list:
    env = os.environ.get("STYLE_LAB_YTDLP")
    if env:
        return [sys.executable, env] if env.lower().endswith(".py") else [env]
    found = shutil.which("yt-dlp")
    if not found:
        raise MissingTool(f"yt-dlp is not installed: {_hint()}")
    return [found]


def _entries(node):
    """Flatten a yt-dlp playlist JSON (nested tab playlists included) into raw entries."""
    out = []
    for e in (node or {}).get("entries") or []:
        if not isinstance(e, dict):
            continue
        if e.get("entries") is not None:
            out.extend(_entries(e))
        else:
            out.append(e)
    return out


def _fetch(url, *, limit, cookies_browser, run, timeout=120):
    cookies = ["--cookies-from-browser", cookies_browser] if cookies_browser else []
    cmd = [*ytdlp_cmd(), "--flat-playlist", "-J", "--playlist-end", str(limit), *cookies, url]
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    try:
        r = run(cmd, capture_output=True, env=env, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise FetchError(f"yt-dlp timed out after {timeout:g} s", False)
    except FileNotFoundError as e:
        raise MissingTool(f"yt-dlp could not run ({e}): {_hint()}")

    def txt(b):
        return b.decode("utf-8", errors="replace") if isinstance(b, (bytes, bytearray)) else (b or "")

    if r.returncode != 0:
        lines = [x.strip() for x in txt(r.stderr).splitlines() if x.strip()]
        last = lines[-1] if lines else f"yt-dlp exited with {r.returncode}"
        raise FetchError(last, bool(LOGIN_RE.search(" ".join(lines[-3:]) or last)))
    try:
        data = json.loads(txt(r.stdout))
    except json.JSONDecodeError:
        raise FetchError("yt-dlp returned output that is not JSON", False)
    if not isinstance(data, dict):
        raise FetchError("yt-dlp returned JSON that is not an object", False)
    return _entries(data)


def list_posts(profile, *, limit, cookies_browser=None, run=subprocess.run, tab="auto",
               timeout=120, tabs_failed=None) -> list:
    """Raw entries of a profile. A YouTube channel without a tab in the URL reads /shorts and /videos.

    If one tab fails and another works, the listing continues and `tabs_failed` (a list, when given)
    gets `{"tab", "error"}`; it only raises when every tab failed."""
    url = profile["url"]
    if profile["platform"] == "youtube" and not profile.get("tab"):
        base = re.sub(r"/(shorts|videos)$", "", url)
        tabs = ["shorts", "videos"] if tab in ("auto", "all") else [tab]
        out, failed, last = [], [], None
        for t in tabs:
            try:
                got = _fetch(f"{base}/{t}", limit=limit, cookies_browser=cookies_browser, run=run, timeout=timeout)
            except FetchError as e:
                failed.append({"tab": t, "error": str(e)})
                last = e
                continue
            for e in got:
                e = dict(e)
                e["tab"] = t
                out.append(e)
        if len(failed) == len(tabs):
            raise last
        if tabs_failed is not None:
            tabs_failed.extend(failed)
        return out
    entries = _fetch(url, limit=limit, cookies_browser=cookies_browser, run=run, timeout=timeout)
    if profile["platform"] == "youtube":
        for e in entries:
            e["tab"] = profile["tab"]
    return entries


def _int(v):
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    return int(v)


def normalise(raw: dict, platform: str):
    """A Post, or None for images, lives, playlists and entries without a video URL."""
    url = raw.get("url") or raw.get("webpage_url")
    pid = raw.get("id")
    if not url or not pid or not str(url).startswith("http"):
        return None
    if raw.get("_type") == "playlist" or raw.get("is_live") or raw.get("live_status") in ("is_live", "is_upcoming"):
        return None
    if str(raw.get("ext") or "").lower() in IMAGE_EXTS or raw.get("media_type") in ("image", "photo"):
        return None
    if platform == "instagram" and "/p/" in url and not raw.get("duration"):
        return None
    if platform == "tiktok" and "/photo/" in url:
        return None
    ts = _int(raw.get("timestamp"))
    date = None
    if ts is not None:
        date = dt.datetime.fromtimestamp(ts, dt.timezone.utc).date().isoformat()
    elif re.fullmatch(r"\d{8}", str(raw.get("upload_date") or "")):
        d = str(raw["upload_date"])
        try:
            day = dt.date(int(d[:4]), int(d[4:6]), int(d[6:]))
            date = day.isoformat()
            ts = int(dt.datetime(day.year, day.month, day.day, tzinfo=dt.timezone.utc).timestamp())
        except ValueError:
            pass
    thumbs = [t for t in (raw.get("thumbnails") or []) if isinstance(t, dict) and t.get("url")]
    thumb = (max(thumbs, key=lambda t: (t.get("width") or 0) * (t.get("height") or 0))["url"]
             if thumbs else raw.get("thumbnail"))
    dur = raw.get("duration")
    return {
        "id": str(pid), "url": url, "title": raw.get("title") or raw.get("description") or "",
        "upload_date": date, "timestamp": ts,
        "duration_s": dur if isinstance(dur, (int, float)) and not isinstance(dur, bool) else None,
        "views": _int(raw.get("view_count")), "likes": _int(raw.get("like_count")),
        "comments": _int(raw.get("comment_count")), "reposts": _int(raw.get("repost_count")),
        "thumbnail": thumb, "tab": raw.get("tab"),
    }


def _by_newest(posts):
    return sorted(posts, key=lambda p: -(p["timestamp"] or 0))


def rank(posts, *, top=20, pick_count=40, max_duration=None) -> dict:
    richness = lambda p: sum(p[k] is not None for k in ("views", "likes", "comments", "reposts"))  # noqa: E731
    pos, uniq = {}, []
    for p in posts:
        if p["id"] not in pos:
            pos[p["id"]] = len(uniq)
            uniq.append(p)
        elif richness(p) > richness(uniq[pos[p["id"]]]):
            uniq[pos[p["id"]]] = p
    warnings = []
    if max_duration is not None:
        kept = [p for p in uniq if p["duration_s"] is None or p["duration_s"] <= max_duration]
        if len(kept) < len(uniq):
            warnings.append(f"{len(uniq) - len(kept)} posts longer than {max_duration:g} s were left out")
        uniq = kept
    if len(uniq) < MIN_POSTS:
        warnings.append(
            f"only {len(uniq)} video posts found (fewer than {MIN_POSTS}): the style will have a thin library")
    n = len(uniq)
    views_ok = n and sum(1 for p in uniq if (p["views"] or 0) > 0) / n >= METRIC_SHARE
    likes_ok = n and sum(1 for p in uniq if (p["likes"] or 0) > 0) / n >= METRIC_SHARE
    if views_ok or likes_ok:
        mode, score = ("views", lambda p: p["views"] or 0) if views_ok else ("engagement", ENGAGEMENT)
        if mode == "engagement":
            warnings.append("not enough view counts: ranked by engagement (likes + 2 x comments + 2 x reposts)")
        scored = sorted([p for p in uniq if score(p) > 0], key=lambda p: (-score(p), -(p["timestamp"] or 0)))
        rest = _by_newest([p for p in uniq if score(p) <= 0])
        entries = []
        for i, p in enumerate(scored + rest, 1):
            s = score(p)
            entries.append({**p, "rank": i, "score": s, "proposed": s > 0 and i <= top})
    else:
        mode = "pick"
        warnings.append("no usable view or like counts: pick mode (latest posts, nothing proposed; the user chooses)")
        ordered = _by_newest(uniq)  # dated posts newest first, undated ones after, in playlist order
        entries = [{**p, "rank": i, "score": None, "proposed": False} for i, p in enumerate(ordered[:pick_count], 1)]
    return {"mode": mode, "entries": entries, "warnings": warnings}


def _fmt(v):
    return "-" if v is None else f"{v:,}"


def _table(entries):
    lines = [f"{'#':>3}  {'date':<10}  {'views':>11}  {'likes':>9}  {'dur':>5}  title"]
    for e in entries:
        d = e["duration_s"]
        dur = "-" if d is None else f"{int(d)}s"
        title = " ".join((e["title"] or "").split())[:40]
        mark = "*" if e["proposed"] else " "
        lines.append(f"{e['rank']:>3}{mark} {(e['upload_date'] or '-'):<10}  {_fmt(e['views']):>11}  "
                     f"{_fmt(e['likes']):>9}  {dur:>5}  {title}")
    return "\n".join(lines)


def _resolve_pack(args):
    if args.pack:
        return Path(args.pack)
    import styles
    try:
        s = styles.find(styles.load_registry(), args.style)
    except (ValueError, OSError, KeyError, TypeError, AttributeError) as e:
        raise BadInput(f"cannot read the style registry: {e}")
    if not s:
        raise BadInput(f"style not found: {args.style}")
    if not s.get("pack"):
        raise BadInput(f"style {args.style} has no pack folder")
    return Path(s["pack"])


def cmd_list(args) -> int:
    tabs_failed = []
    try:
        pack = _resolve_pack(args)
        profile = parse_profile_url(args.url)
        raw = list_posts(profile, limit=args.limit, cookies_browser=args.cookies_from_browser, tab=args.tab,
                         timeout=args.timeout, tabs_failed=tabs_failed)
    except BadInput as e:
        print(f"bad input: {e}", file=sys.stderr)
        return 2
    except MissingTool as e:
        print(f"missing tool: {e}", file=sys.stderr)
        return 3
    except FetchError as e:
        print(f"could not read the profile: {e}", file=sys.stderr)
        if e.needs_login:
            print("try --cookies-from-browser <your browser>: your own logged-in browser, nothing is stored",
                  file=sys.stderr)
        return 1
    posts = [p for p in (normalise(r, profile["platform"]) for r in raw) if p]
    if not posts:
        print("no video posts found on this profile", file=sys.stderr)
        return 1
    res = rank(posts, top=args.top, pick_count=args.pick_count, max_duration=args.max_duration)
    if not res["entries"]:
        print(f"no video posts left after --max-duration {args.max_duration:g}", file=sys.stderr)
        return 1
    for t in tabs_failed:
        res["warnings"].append(f"tab {t['tab']} could not be read ({t['error']}): the list may be incomplete")
    proposal = {
        "schema": SCHEMA, "url": profile["url"], "platform": profile["platform"], "handle": profile["handle"],
        "fetched": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "mode": res["mode"], "total_posts": len({p["id"] for p in posts}),
        "entries": res["entries"], "warnings": res["warnings"], "tabs_failed": tabs_failed,
        "selected": [], "consent": None,
    }
    out = pack / "refs" / "proposal.json"
    try:
        store.write_json(out, proposal)
    except OSError as e:
        print(f"bad input: cannot write {out}: {e}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(proposal, ensure_ascii=False, indent=2))
    else:
        print(f"{profile['platform']} @{profile['handle']}: {proposal['total_posts']} video posts, mode {res['mode']}")
        print(_table(res["entries"]))
        print(f"proposal: {out}")
    for w in res["warnings"]:
        print(f"warning: {w}", file=sys.stderr)
    return 0


def _limit(v):
    n = int(v)
    if not 1 <= n <= 200:
        raise argparse.ArgumentTypeError("must be between 1 and 200")
    return n


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("list", help="read a profile and propose posts to study (no download)")
    a.add_argument("url")
    g = a.add_mutually_exclusive_group(required=True)
    g.add_argument("--pack")
    g.add_argument("--style")
    a.add_argument("--limit", type=_limit, default=60, help="posts to read per tab, 1-200")
    a.add_argument("--timeout", type=float, default=120, help="seconds before a yt-dlp call is stopped")
    a.add_argument("--top", type=int, default=20)
    a.add_argument("--pick-count", type=int, default=40)
    a.add_argument("--tab", choices=("auto", "shorts", "videos", "all"), default="auto")
    a.add_argument("--cookies-from-browser")
    a.add_argument("--max-duration", type=float)
    a.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    store.print_utf8()
    return cmd_list(args)


if __name__ == "__main__":
    sys.exit(store.exit_on_lock_timeout(main))

"""Read a public TikTok / YouTube / Instagram profile, rank its posts and propose a study list.

`list` only READS the profile listing (yt-dlp `--flat-playlist -J`; Instagram falls back to instaloader,
anonymous). The proposal goes to <pack>/refs/proposal.json. Videos are downloaded only by `download`, after
the user picked them and confirmed private study use; they are local study copies, never published.

    profile_fetch.py list <url> (--pack DIR | --style SLUG) [--limit 60] [--top 20] [--pick-count 40]
                         [--tab auto|shorts|videos|all] [--cookies-from-browser BROWSER]
                         [--max-duration SECONDS] [--timeout SECONDS] [--json]
    profile_fetch.py page (--pack DIR | --style SLUG) --out FILE.html      the tick page (no video in it)
    profile_fetch.py pick (--pack | --style) (--ranks "1-5,8" | --ids a,b | --proposed)
    profile_fetch.py download (--pack | --style) --i-confirm-private-study [--cookies-from-browser B]
                         [--max-height 1080]       into refs/V01.mp4 ..., with refs/sources.json and sources.md

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



def profile_ok(args):
    try:
        return parse_profile_url(args.url)["platform"]
    except Exception:
        return None


def _fetch_failed(e):
    print(f"could not read the profile: {e}", file=sys.stderr)
    if getattr(e, "needs_login", False):
        print("try --cookies-from-browser <your browser>: your own logged-in browser, nothing is stored",
              file=sys.stderr)
    return 1

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
        raw = None
        if profile_ok(args) == "instagram":
            if _has_instaloader():
                try:
                    raw = instaloader_posts(parse_profile_url(args.url)["handle"], args.limit)
                except Exception as e2:  # instaloader raises its own types
                    print(f"instaloader could not read the profile: {e2}", file=sys.stderr)
                    raw = None
            if raw is None:
                print(f"could not read the profile: {e}", file=sys.stderr)
                print("missing tool: pip install instaloader, or pass --cookies-from-browser <your browser>",
                      file=sys.stderr)
                return 3
        if raw is None:
            return _fetch_failed(e)
    if not raw and profile_ok(args) == "instagram" and _has_instaloader():
        try:
            raw = instaloader_posts(parse_profile_url(args.url)["handle"], args.limit)
        except Exception:
            pass
    profile = parse_profile_url(args.url)
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



# ---------------------------------------------------------------- Instagram fallback

def instaloader_posts(handle: str, limit: int, module=None) -> list:
    """Raw entries in the yt-dlp shape, read anonymously with instaloader (no password, ever)."""
    if module is None:
        import instaloader as module  # noqa: N813
    L = module.Instaloader(download_pictures=False, download_videos=False, download_video_thumbnails=False,
                           save_metadata=False, quiet=True)
    prof = module.Profile.from_username(L.context, handle)
    out = []
    for post in prof.get_posts():
        if len(out) >= limit:
            break
        if not getattr(post, "is_video", False):
            continue
        when = getattr(post, "date_utc", None)
        out.append({"id": post.shortcode, "url": f"https://www.instagram.com/reel/{post.shortcode}/",
                    "title": (getattr(post, "caption", None) or "")[:200],
                    "view_count": getattr(post, "video_view_count", None), "like_count": getattr(post, "likes", None),
                    "comment_count": getattr(post, "comments", None),
                    "timestamp": int(when.replace(tzinfo=dt.timezone.utc).timestamp()) if when else None,
                    "duration": getattr(post, "video_duration", None), "thumbnail": getattr(post, "url", None)})
    return out


def _has_instaloader():
    import importlib.util
    try:
        return importlib.util.find_spec("instaloader") is not None
    except (ImportError, ValueError):
        return False


# ---------------------------------------------------------------- the page and the pick

PAGE_TEMPLATE = Path(__file__).resolve().parent.parent / "assets" / "profile-pick-template.html"


def _proposal_path(pack):
    return Path(pack) / "refs" / "proposal.json"


def load_proposal(pack):
    p = _proposal_path(pack)
    data = store.read_json(p, None)
    if not isinstance(data, dict) or data.get("schema") != SCHEMA:
        raise BadInput(f"no proposal in {p} (run profile_fetch.py list first)")
    return data


def build_page(proposal: dict, template: str) -> str:
    keep = {k: proposal.get(k) for k in ("url", "platform", "handle", "fetched", "mode", "total_posts")}
    keep["entries"] = [{k: e.get(k) for k in ("id", "rank", "title", "upload_date", "views", "likes", "duration_s",
                                              "thumbnail", "proposed", "url")} for e in proposal.get("entries", [])]
    block = json.dumps(keep, ensure_ascii=False, indent=1).replace("<", "\\u003c")
    out, n = re.subn(r'(<script type="application/json" id="profile-data">)(.*?)(</script>)',
                     lambda m: m.group(1) + "\n" + block + "\n" + m.group(3), template, count=1, flags=re.S)
    if n != 1:
        raise BadInput("the page template has no profile-data block")
    return out


def _parse_ranks(spec, n):
    out = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        m = re.fullmatch(r"(\d+)(?:\s*-\s*(\d+))?", part)
        if not m:
            raise BadInput(f"bad rank '{part}' (use 1-5,8)")
        a, b = int(m.group(1)), int(m.group(2) or m.group(1))
        if a < 1 or b > n or a > b:
            raise BadInput(f"rank {part} is out of range 1-{n}")
        out.extend(range(a, b + 1))
    return list(dict.fromkeys(out))


def pick(pack, *, ranks=None, ids=None, proposed=False) -> list:
    data = load_proposal(pack)
    entries = data.get("entries", [])
    if proposed:
        if data.get("mode") == "pick":
            raise BadInput("pick mode proposes nothing: pass --ranks or --ids")
        chosen = [e["id"] for e in entries if e.get("proposed")]
        how = "proposed"
    elif ranks:
        by_rank = {e["rank"]: e["id"] for e in entries}
        chosen = [by_rank[r] for r in _parse_ranks(ranks, len(entries)) if r in by_rank]
        how = f"ranks {ranks}"
    elif ids:
        known = {e["id"] for e in entries}
        chosen = [i.strip() for i in str(ids).split(",") if i.strip()]
        bad = [i for i in chosen if i not in known]
        if bad:
            raise BadInput(f"unknown id(s): {', '.join(bad)}")
        how = "ids"
    else:
        raise BadInput("give --ranks, --ids or --proposed")
    if not chosen:
        raise BadInput("nothing selected")

    def go(d):
        d["selected"] = list(dict.fromkeys(chosen))
        d["selected_at"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
        d["selected_how"] = how
    store.update_json(_proposal_path(pack), {}, go)
    return chosen


# ---------------------------------------------------------------- download

SOURCES_START, SOURCES_END = "<!-- profile_fetch:start -->", "<!-- profile_fetch:end -->"
PRIVATE_LINE = "Study copies for private style training only. Never publish, upload or share them."
CONSENT_HE = "ההורדה רק אחרי אישור שהיא ללימוד פרטי בלבד (--i-confirm-private-study)"


def next_ref_name(refs: Path) -> str:
    n = 0
    for p in Path(refs).glob("V*.*"):
        m = re.fullmatch(r"V(\d+)", p.stem)
        if m:
            n = max(n, int(m.group(1)))
    n += 1
    return f"V{n:02d}.mp4" if n < 100 else f"V{n:03d}.mp4"


def _sha256(p):
    import hashlib
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _duration(p):
    ffp = shutil.which("ffprobe")
    if not ffp:
        return None
    r = subprocess.run([ffp, "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(p)],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    try:
        return float(r.stdout.strip())
    except ValueError:
        return 0.0


def render_sources_md(existing: str, sources: list, proposal: dict) -> str:
    fetched = str(proposal.get("fetched") or "")[:10]
    rows = ["| file | post | posted | views | downloaded | sha256 |", "|---|---|---|---|---|---|"]
    for s in sources:
        rows.append(f"| {s['file']} | {s['url']} | {s.get('posted') or '-'} | "
                    f"{_fmt(s.get('views'))} | {str(s.get('downloaded') or '')[:10]} | {s['sha256'][:12]} |")
    block = "\n".join([SOURCES_START, PRIVATE_LINE, "",
                       f"Profile: {proposal.get('url')} (fetched {fetched}, ranked by {proposal.get('mode')})", "",
                       *rows, SOURCES_END])
    text = existing or ""
    if SOURCES_START in text and SOURCES_END in text:
        a, b = text.index(SOURCES_START), text.index(SOURCES_END) + len(SOURCES_END)
        return text[:a] + block + text[b:]
    return (text.rstrip("\n") + "\n\n" if text.strip() else "") + block + "\n"


def download_selected(pack: Path, *, cookies_browser=None, max_height=1080, run=subprocess.run) -> dict:
    pack = Path(pack)
    refs = pack / "refs"
    data = load_proposal(pack)
    sel = data.get("selected") or []
    if not sel:
        raise BadInput("nothing selected: run profile_fetch.py pick first")
    if not data.get("consent"):
        store.update_json(_proposal_path(pack), {}, lambda d: d.update(
            consent={"private_study": True, "at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}))
    by_id = {e["id"]: e for e in data.get("entries", [])}
    src_path = refs / "sources.json"
    done = {s["id"] for s in (store.read_json(src_path, {"sources": []}) or {}).get("sources", [])}
    partial = refs / ".partial"
    out = {"downloaded": [], "skipped": [], "failed": []}
    base = ytdlp_cmd()
    cookies = ["--cookies-from-browser", cookies_browser] if cookies_browser else []
    h = int(max_height)
    try:
        for pid in sel:
            if pid in done:
                out["skipped"].append(pid)
                continue
            e = by_id.get(pid)
            if e is None:
                out["failed"].append({"id": pid, "error": "not in the proposal"})
                continue
            partial.mkdir(parents=True, exist_ok=True)
            cmd = [*base, "--no-playlist", "-f", f"bv*[height<={h}]+ba/b[height<={h}]", "--merge-output-format",
                   "mp4", "-o", f"{partial}/{pid}.%(ext)s", *cookies, e["url"]]
            try:
                r = run(cmd, capture_output=True, timeout=1800)
            except subprocess.TimeoutExpired:
                out["failed"].append({"id": pid, "error": "yt-dlp timed out"})
                continue
            files = sorted(partial.glob(f"{pid}.*"))
            got = next((f for f in files if f.suffix.lower() == ".mp4"), files[0] if files else None)
            if getattr(r, "returncode", 1) != 0 or got is None or got.stat().st_size == 0:
                err = getattr(r, "stderr", b"")
                err = err.decode("utf-8", "replace") if isinstance(err, bytes) else str(err or "")
                out["failed"].append({"id": pid, "error": (err.strip().splitlines() or ["download failed"])[-1][:200]})
                continue
            dur = _duration(got)
            if dur is not None and dur <= 0:
                out["failed"].append({"id": pid, "error": "the downloaded file has no duration"})
                continue
            name = next_ref_name(refs)
            os.replace(got, refs / name)
            entry = {"file": name, "id": pid, "url": e["url"], "posted": e.get("upload_date"), "views": e.get("views"),
                     "likes": e.get("likes"), "downloaded": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                     "sha256": _sha256(refs / name)}

            def add(d, entry=entry):
                d.setdefault("sources", []).append(entry)
            store.update_json(src_path, {"sources": []}, add)
            done.add(pid)
            out["downloaded"].append(name)
            md = refs / "sources.md"
            old = md.read_text(encoding="utf-8") if md.is_file() else ""
            new = render_sources_md(old, store.read_json(src_path, {"sources": []})["sources"], data)
            tmp = md.with_name(md.name + ".tmp")
            tmp.write_text(new, encoding="utf-8")
            os.replace(tmp, md)
    finally:
        if partial.exists():
            shutil.rmtree(partial, ignore_errors=True)
    return out


def cmd_page(args) -> int:
    pack = _resolve_pack(args)
    html = build_page(load_proposal(pack), PAGE_TEMPLATE.read_text(encoding="utf-8"))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(html, encoding="utf-8")
    print(f"page: {args.out} (publish it privately; it holds no video)")
    return 0


def cmd_pick(args) -> int:
    pack = _resolve_pack(args)
    chosen = pick(pack, ranks=args.ranks, ids=args.ids, proposed=args.proposed)
    print(f"selected {len(chosen)}: {', '.join(chosen)}")
    return 0


def cmd_download(args) -> int:
    if not args.i_confirm_private_study:
        print(CONSENT_HE, file=sys.stderr)
        print("Download only after confirming private study use (--i-confirm-private-study)", file=sys.stderr)
        return 2
    pack = _resolve_pack(args)
    res = download_selected(pack, cookies_browser=args.cookies_from_browser, max_height=args.max_height)
    print(json.dumps(res, ensure_ascii=False))
    for f in res["failed"]:
        print(f"failed: {f['id']}: {f['error']}", file=sys.stderr)
    return 1 if res["failed"] else 0

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
    for name, helptext in (("page", "the tick page"), ("pick", "choose posts"), ("download", "download the picks")):
        a = sub.add_parser(name, help=helptext)
        g = a.add_mutually_exclusive_group(required=True)
        g.add_argument("--pack")
        g.add_argument("--style")
        if name == "page":
            a.add_argument("--out", required=True)
        elif name == "pick":
            h = a.add_mutually_exclusive_group(required=True)
            h.add_argument("--ranks")
            h.add_argument("--ids")
            h.add_argument("--proposed", action="store_true")
        else:
            a.add_argument("--i-confirm-private-study", action="store_true")
            a.add_argument("--cookies-from-browser")
            a.add_argument("--max-height", type=int, default=1080)
    args = ap.parse_args(argv)
    store.print_utf8()
    if args.cmd == "list":
        return cmd_list(args)
    try:
        return {"page": cmd_page, "pick": cmd_pick, "download": cmd_download}[args.cmd](args)
    except BadInput as e:
        print(f"bad input: {e}", file=sys.stderr)
        return 2
    except MissingTool as e:
        print(f"missing tool: {e}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    sys.exit(store.exit_on_lock_timeout(main))

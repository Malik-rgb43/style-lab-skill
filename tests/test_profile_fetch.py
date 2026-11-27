import copy
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import profile_fetch as pf
from conftest import REPO_ROOT, run_cli

FX = Path(__file__).resolve().parent / "fixtures"
PROFILES = FX / "profiles"
FAKE = FX / "fake_ytdlp.py"


def load(name):
    return json.loads((PROFILES / name).read_text(encoding="utf-8"))


def posts_of(name, platform):
    return [p for p in (pf.normalise(e, platform) for e in load(name)["entries"]) if p]


@pytest.fixture
def fake(tmp_path, monkeypatch):
    """Point the module at the fake yt-dlp; returns a setter for the URL -> fixture map."""
    log = tmp_path / "calls.jsonl"
    monkeypatch.setenv("STYLE_LAB_YTDLP", str(FAKE))
    monkeypatch.setenv("FAKE_YTDLP_LOG", str(log))
    for k in ("FAKE_YTDLP_FAIL", "FAKE_YTDLP_FAIL_IDS", "FAKE_YTDLP_MEDIA"):
        monkeypatch.delenv(k, raising=False)

    def setmap(mapping):
        f = tmp_path / "fixtures.json"
        f.write_text(json.dumps({k: str(PROFILES / v) for k, v in mapping.items()}), encoding="utf-8")
        monkeypatch.setenv("FAKE_YTDLP_FIXTURES", str(f))
        return log

    return setmap


def test_parse_profile_urls():
    cases = [
        ("https://www.tiktok.com/@example.creator", "tiktok", "example.creator", None),
        ("https://www.youtube.com/@ExampleChannel", "youtube", "ExampleChannel", None),
        ("https://www.youtube.com/@ExampleChannel/shorts", "youtube", "ExampleChannel", "shorts"),
        ("https://www.youtube.com/channel/UCexample123/videos", "youtube", "UCexample123", "videos"),
        ("https://www.youtube.com/c/ExampleChannel", "youtube", "ExampleChannel", None),
        ("https://www.youtube.com/user/ExampleChannel", "youtube", "ExampleChannel", None),
        ("https://www.instagram.com/example.creator/", "instagram", "example.creator", None),
    ]
    for url, platform, handle, tab in cases:
        got = pf.parse_profile_url(url)
        assert (got["platform"], got["handle"], got["tab"]) == (platform, handle, tab), url
    assert "?" not in pf.parse_profile_url("https://www.tiktok.com/@example.creator?lang=en")["url"]
    for bad in ("https://www.instagram.com/p/x", "https://www.youtube.com/watch?v=x",
                "https://www.tiktok.com/@example.creator/video/1", "https://www.instagram.com/reel/x/",
                "https://example.com/@x", "file:///x", ""):
        with pytest.raises(pf.BadInput):
            pf.parse_profile_url(bad)


def test_rank_views_top20():
    res = pf.rank(posts_of("tiktok_flat.json", "tiktok"))
    assert res["mode"] == "views"
    proposed = [e for e in res["entries"] if e["proposed"]]
    assert len(proposed) == 20
    scores = [e["score"] for e in proposed]
    assert scores == sorted(scores, reverse=True)
    assert res["entries"][0]["rank"] == 1 and res["entries"][0]["id"].endswith("017")


def test_rank_engagement_when_views_missing():
    posts = posts_of("tiktok_flat.json", "tiktok")
    for p in posts:
        p["views"] = None
    res = pf.rank(posts)
    assert res["mode"] == "engagement"
    e = res["entries"][0]
    assert e["score"] == e["likes"] + 2 * e["comments"] + 2 * e["reposts"]


def test_rank_pick_without_metrics():
    posts = posts_of("instagram_flat_nometrics.json", "instagram")
    assert len(posts) == 42  # 3 images dropped
    res = pf.rank(posts)
    assert res["mode"] == "pick"
    assert len(res["entries"]) == 40
    assert not any(e["proposed"] for e in res["entries"])
    ts = [e["timestamp"] for e in res["entries"]]
    assert ts == sorted(ts, reverse=True)


def test_posts_without_views_never_proposed():
    res = pf.rank(posts_of("youtube_shorts_flat.json", "youtube"))
    assert res["mode"] == "views"
    nulls = [e for e in res["entries"] if not e["views"]]
    assert len(nulls) == 5
    assert not any(e["proposed"] for e in nulls)
    last_scored = max(e["rank"] for e in res["entries"] if e["views"])
    assert all(e["rank"] > last_scored for e in nulls)


def test_youtube_tabs_merge_dedupe():
    outputs = [(PROFILES / "youtube_shorts_flat.json").read_bytes(),
               (PROFILES / "youtube_videos_flat.json").read_bytes()]
    calls = []

    def run(cmd, **kw):
        calls.append(cmd)
        return SimpleNamespace(returncode=0, stdout=outputs[len(calls) - 1], stderr=b"")

    os.environ["STYLE_LAB_YTDLP"] = "yt-dlp-stub"
    try:
        raw = pf.list_posts(pf.parse_profile_url("https://www.youtube.com/@ExampleChannel"), limit=60, run=run)
    finally:
        del os.environ["STYLE_LAB_YTDLP"]
    assert [c[-1].rsplit("/", 1)[-1] for c in calls] == ["shorts", "videos"]
    assert all(c[:1] == ["yt-dlp-stub"] and "-J" in c and "--flat-playlist" in c for c in calls)
    posts = [p for p in (pf.normalise(e, "youtube") for e in raw) if p]
    res = pf.rank(posts)
    ids = [e["id"] for e in res["entries"]]
    assert len(ids) == len(set(ids)) == 30
    assert {e["tab"] for e in res["entries"]} == {"shorts", "videos"}


def test_fewer_than_12_warns_exit0(fake, tmp_path):
    fake({"tiktok.com": "tiktok_small.json"})
    pack = tmp_path / "pack"
    r = run_cli("profile_fetch.py", "list", "https://www.tiktok.com/@example.creator", "--pack", pack)
    assert r.returncode == 0, r.stderr
    assert "fewer than 12" in r.stderr
    prop = json.loads((pack / "refs" / "proposal.json").read_text(encoding="utf-8"))
    assert any("fewer than 12" in w for w in prop["warnings"])
    assert prop["schema"] == "profile-proposal/1" and prop["selected"] == [] and prop["consent"] is None
    assert prop["total_posts"] == 8


def test_cli_list_hebrew_pack_no_download(fake, tmp_path):
    log = fake({"tiktok.com": "tiktok_flat.json"})
    pack = tmp_path / "חבילה בדיקה" / "pack"
    r = run_cli("profile_fetch.py", "list", "https://www.tiktok.com/@example.creator", "--pack", pack)
    assert r.returncode == 0, r.stderr
    assert "טיפ קצר לעריכה" in r.stdout
    prop = json.loads((pack / "refs" / "proposal.json").read_text(encoding="utf-8"))
    assert "טיפ קצר לעריכה" in {e["title"] for e in prop["entries"]}
    assert [p.name for p in (pack / "refs").iterdir()] == ["proposal.json"]
    calls = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert calls and all("-J" in c["argv"] for c in calls)


def test_cli_list_style_resolves_pack(fake, tmp_path):
    fake({"tiktok.com": "tiktok_flat.json"})
    import styles
    pack = tmp_path / "stylepack"
    reg = {"version": 2, "styles": [{"slug": "demo", "name": "Demo", "pack": str(pack)}]}
    styles.save_registry(reg)
    r = run_cli("profile_fetch.py", "list", "https://www.tiktok.com/@example.creator", "--style", "demo")
    assert r.returncode == 0, r.stderr
    assert (pack / "refs" / "proposal.json").exists()
    r = run_cli("profile_fetch.py", "list", "https://www.tiktok.com/@example.creator", "--style", "nope")
    assert r.returncode == 2


def test_bad_url_exit2(tmp_path):
    r = run_cli("profile_fetch.py", "list", "https://www.instagram.com/p/x", "--pack", tmp_path / "p")
    assert r.returncode == 2 and "Traceback" not in r.stderr


def test_missing_ytdlp_exit3(tmp_path, monkeypatch):
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.delenv("STYLE_LAB_YTDLP", raising=False)
    monkeypatch.setenv("PATH", str(empty))
    r = run_cli("profile_fetch.py", "list", "https://www.tiktok.com/@example.creator", "--pack", tmp_path / "p")
    assert r.returncode == 3
    assert "yt-dlp" in r.stderr and "install" in r.stderr
    assert "Traceback" not in r.stderr + r.stdout


def test_fetch_error_login_hint(fake, tmp_path, monkeypatch):
    fake({"tiktok.com": "tiktok_flat.json"})
    monkeypatch.setenv("FAKE_YTDLP_FAIL", "ERROR: login required")
    r = run_cli("profile_fetch.py", "list", "https://www.tiktok.com/@example.creator", "--pack", tmp_path / "p")
    assert r.returncode == 1
    assert "--cookies-from-browser" in r.stderr + r.stdout
    assert "login required" in r.stderr


def test_ytdlp_cmd_py_and_exe(monkeypatch):
    monkeypatch.setenv("STYLE_LAB_YTDLP", "x/fake.py")
    assert pf.ytdlp_cmd() == [sys.executable, "x/fake.py"]
    monkeypatch.setenv("STYLE_LAB_YTDLP", "x/yt-dlp.exe")
    assert pf.ytdlp_cmd() == ["x/yt-dlp.exe"]


def test_normalise_drops_non_video_and_picks_largest_thumb():
    assert pf.normalise({"id": "a", "title": "t"}, "tiktok") is None
    assert pf.normalise({"id": "a", "url": "https://example.com/a", "is_live": True}, "youtube") is None
    p = pf.normalise(load("tiktok_small.json")["entries"][0], "tiktok")
    assert p["thumbnail"].endswith("_l.jpg") and p["upload_date"] and p["duration_s"]


def test_max_duration_filters():
    posts = posts_of("tiktok_flat.json", "tiktok")
    res = pf.rank(posts, max_duration=30)
    assert all(e["duration_s"] <= 30 for e in res["entries"])
    assert any("longer than" in w for w in res["warnings"])


def test_no_http_client_imports():
    src = (REPO_ROOT / "scripts" / "profile_fetch.py").read_text(encoding="utf-8")
    for needle in ("import requests", "urllib.request", "httpx", "http.client"):
        assert needle not in src


def _run_by_url(outputs, fail=()):
    def run(cmd, **kw):
        tab = cmd[-1].rsplit("/", 1)[-1]
        if tab in fail:
            return SimpleNamespace(returncode=1, stdout=b"", stderr=b"ERROR: HTTP Error 404: Not Found\n")
        return SimpleNamespace(returncode=0, stdout=(PROFILES / outputs[tab]).read_bytes(), stderr=b"")
    return run


def test_one_missing_tab_does_not_fail(monkeypatch):
    monkeypatch.setenv("STYLE_LAB_YTDLP", "yt-dlp-stub")
    prof = pf.parse_profile_url("https://www.youtube.com/@ExampleChannel")
    outs = {"shorts": "youtube_shorts_flat.json", "videos": "youtube_videos_flat.json"}
    failed = []
    raw = pf.list_posts(prof, limit=60, run=_run_by_url(outs, fail=("videos",)), tabs_failed=failed)
    assert len(raw) == 25 and failed == [{"tab": "videos", "error": "ERROR: HTTP Error 404: Not Found"}]
    with pytest.raises(pf.FetchError):
        pf.list_posts(prof, limit=60, run=_run_by_url(outs, fail=("videos", "shorts")))


def test_cli_missing_tab_records_tabs_failed(fake, tmp_path):
    fake({"/shorts": "youtube_shorts_flat.json"})  # no fixture for /videos: that call fails
    pack = tmp_path / "pack"
    r = run_cli("profile_fetch.py", "list", "https://www.youtube.com/@ExampleChannel", "--pack", pack)
    assert r.returncode == 0, r.stderr
    prop = json.loads((pack / "refs" / "proposal.json").read_text(encoding="utf-8"))
    assert [t["tab"] for t in prop["tabs_failed"]] == ["videos"]
    assert "videos" in r.stderr


def test_max_duration_before_warning_and_empty_exit1(fake, tmp_path):
    posts = posts_of("tiktok_flat.json", "tiktok")
    res = pf.rank(posts, max_duration=20)  # keeps only a handful
    assert any("fewer than 12" in w for w in res["warnings"]) and any("longer than" in w for w in res["warnings"])
    assert f"fewer than {pf.MIN_POSTS}" in res["warnings"][-1]
    fake({"tiktok.com": "tiktok_flat.json"})
    r = run_cli("profile_fetch.py", "list", "https://www.tiktok.com/@example.creator", "--pack", tmp_path / "p",
                "--max-duration", "1")
    assert r.returncode == 1 and "--max-duration" in r.stderr and "Traceback" not in r.stderr


def test_upload_date_is_timestamp_fallback():
    raw = {"id": "a", "url": "https://www.youtube.com/watch?v=a", "upload_date": "20250102"}
    p = pf.normalise(raw, "youtube")
    assert p["upload_date"] == "2025-01-02" and p["timestamp"] == 1735776000
    undated = pf.normalise({"id": "b", "url": "https://www.youtube.com/watch?v=b"}, "youtube")
    res = pf.rank([undated, p])
    assert res["mode"] == "pick" and [e["id"] for e in res["entries"]] == ["a", "b"]


def test_duplicate_keeps_richer_entry():
    base = {"id": "a", "url": "https://www.youtube.com/watch?v=a", "view_count": None}
    thin = pf.normalise(base, "youtube")
    rich = pf.normalise({**base, "view_count": 50, "like_count": 5}, "youtube")
    for order in ([thin, rich], [rich, thin]):
        res = pf.rank(order)
        assert len(res["entries"]) == 1 and res["entries"][0]["views"] == 50


def test_tiktok_photo_dropped():
    assert pf.normalise({"id": "a", "url": "https://www.tiktok.com/@example.creator/photo/1"}, "tiktok") is None


def test_json_not_object_exit1(tmp_path, monkeypatch):
    class R:
        returncode = 0
        stdout = b"[1, 2]"
        stderr = b""
    monkeypatch.setenv("STYLE_LAB_YTDLP", "yt-dlp-stub")
    with pytest.raises(pf.FetchError, match="not an object"):
        pf.list_posts(pf.parse_profile_url("https://www.tiktok.com/@example.creator"), limit=5, run=lambda c, **k: R)


def test_timeout_is_fetch_error(monkeypatch):
    def run(cmd, **kw):
        assert kw["timeout"] == 7
        raise subprocess.TimeoutExpired(cmd, 7)
    monkeypatch.setenv("STYLE_LAB_YTDLP", "yt-dlp-stub")
    with pytest.raises(pf.FetchError, match="timed out"):
        pf.list_posts(pf.parse_profile_url("https://www.tiktok.com/@example.creator"), limit=5, run=run, timeout=7)


def test_limit_validated_and_bad_registry_exit2(tmp_path):
    for bad in ("0", "201"):
        r = run_cli("profile_fetch.py", "list", "https://www.tiktok.com/@example.creator", "--pack", tmp_path / "p",
                    "--limit", bad)
        assert r.returncode == 2
    import store
    reg = store.home() / "styles.json"
    reg.parent.mkdir(parents=True, exist_ok=True)
    reg.write_text("{not json", encoding="utf-8")
    r = run_cli("profile_fetch.py", "list", "https://www.tiktok.com/@example.creator", "--style", "x")
    assert r.returncode == 2 and "Traceback" not in r.stderr


def test_unwritable_pack_exit2(fake, tmp_path):
    fake({"tiktok.com": "tiktok_flat.json"})
    blocker = tmp_path / "file"
    blocker.write_text("x", encoding="utf-8")
    r = run_cli("profile_fetch.py", "list", "https://www.tiktok.com/@example.creator", "--pack", blocker / "pack")
    assert r.returncode == 2 and "Traceback" not in r.stderr


# ---------------------------------------------------------------- Task 2: page, pick, download

def _listed(fake, tmp_path, pack=None):
    log = fake({"tiktok.com": "tiktok_flat.json"})
    pack = pack or tmp_path / "כונן אחר" / "חבילה"
    r = run_cli("profile_fetch.py", "list", "https://www.tiktok.com/@example.creator", "--pack", pack)
    assert r.returncode == 0, r.stderr
    return pack, log


def _media(tmp_path, monkeypatch):
    import shutil
    import subprocess
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg missing")
    clip = tmp_path / "clip.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc2=s=64x64:d=1", str(clip)], check=True)
    monkeypatch.setenv("FAKE_YTDLP_MEDIA", str(clip))
    return clip


def _calls(log):
    rows = [json.loads(x) for x in log.read_text(encoding="utf-8").splitlines()] if log.exists() else []
    return [c for c in rows if "-J" not in c["argv"]]


def test_download_requires_consent_flag(fake, tmp_path):
    pack, log = _listed(fake, tmp_path)
    assert run_cli("profile_fetch.py", "pick", "--pack", pack, "--ranks", "1-3").returncode == 0
    r = run_cli("profile_fetch.py", "download", "--pack", pack)
    assert r.returncode == 2 and "--i-confirm-private-study" in r.stderr
    assert not list((pack / "refs").glob("V*.mp4")) and _calls(log) == []


def test_download_requires_selection(fake, tmp_path):
    pack, _ = _listed(fake, tmp_path)
    r = run_cli("profile_fetch.py", "download", "--pack", pack, "--i-confirm-private-study")
    assert r.returncode == 2


def test_download_writes_refs_and_sources(fake, tmp_path, monkeypatch):
    import datetime as dt
    import hashlib
    pack, log = _listed(fake, tmp_path)
    _media(tmp_path, monkeypatch)
    run_cli("profile_fetch.py", "pick", "--pack", pack, "--ranks", "1-3")
    r = run_cli("profile_fetch.py", "download", "--pack", pack, "--i-confirm-private-study")
    assert r.returncode == 0, r.stderr
    refs = pack / "refs"
    assert sorted(p.name for p in refs.glob("V*.mp4")) == ["V01.mp4", "V02.mp4", "V03.mp4"]
    src = json.loads((refs / "sources.json").read_text(encoding="utf-8"))["sources"]
    for s in src:
        assert hashlib.sha256((refs / s["file"]).read_bytes()).hexdigest() == s["sha256"]
    md = (refs / "sources.md").read_text(encoding="utf-8")
    assert md.count("| V0") == 3 and "private style training only" in md
    prop = json.loads((refs / "proposal.json").read_text(encoding="utf-8"))
    first = min(c["t"] for c in _calls(log))
    assert dt.datetime.fromisoformat(prop["consent"]["at"]).timestamp() <= first + 1
    assert not (refs / ".partial").exists()


def test_download_resume_no_renumber(fake, tmp_path, monkeypatch):
    pack, log = _listed(fake, tmp_path)
    _media(tmp_path, monkeypatch)
    (pack / "refs" / "V01.mp4").write_bytes(b"hand file")
    run_cli("profile_fetch.py", "pick", "--pack", pack, "--ranks", "1-3")
    assert run_cli("profile_fetch.py", "download", "--pack", pack, "--i-confirm-private-study").returncode == 0
    names = sorted(p.name for p in (pack / "refs").glob("V*.mp4"))
    assert names == ["V01.mp4", "V02.mp4", "V03.mp4", "V04.mp4"]
    n = len(_calls(log))
    before = {p.name: p.read_bytes() for p in (pack / "refs").glob("V*.mp4")}
    assert run_cli("profile_fetch.py", "download", "--pack", pack, "--i-confirm-private-study").returncode == 0
    assert len(_calls(log)) == n
    assert {p.name: p.read_bytes() for p in (pack / "refs").glob("V*.mp4")} == before


def test_partial_failure_continues_exit1(fake, tmp_path, monkeypatch):
    pack, _ = _listed(fake, tmp_path)
    _media(tmp_path, monkeypatch)
    run_cli("profile_fetch.py", "pick", "--pack", pack, "--ranks", "1-3")
    prop = json.loads((pack / "refs" / "proposal.json").read_text(encoding="utf-8"))
    bad = prop["selected"][1]
    monkeypatch.setenv("FAKE_YTDLP_FAIL_IDS", bad)
    r = run_cli("profile_fetch.py", "download", "--pack", pack, "--i-confirm-private-study")
    assert r.returncode == 1 and bad in r.stderr
    assert len(list((pack / "refs").glob("V*.mp4"))) == 2 and not (pack / "refs" / ".partial").exists()


def test_sources_md_hand_text_kept():
    prop = {"url": "https://www.tiktok.com/@example.creator", "fetched": "2026-01-01T00:00:00", "mode": "views"}
    src = [{"file": "V01.mp4", "url": "https://www.tiktok.com/@example.creator/video/1", "posted": "2026-01-01",
            "views": 5, "downloaded": "2026-01-02T00:00:00", "sha256": "a" * 64}]
    hand = "# my notes\nabove\n"
    once = pf.render_sources_md(hand, src, prop)
    twice = pf.render_sources_md(once + "below\n", src, prop)
    assert twice.startswith(hand) and twice.endswith("below\n") and twice.count(pf.SOURCES_START) == 1


def test_pick_errors_exit2(fake, tmp_path):
    pack, _ = _listed(fake, tmp_path)
    assert run_cli("profile_fetch.py", "pick", "--pack", pack, "--ranks", "99").returncode == 2
    assert run_cli("profile_fetch.py", "pick", "--pack", pack, "--ids", "nope").returncode == 2
    prop = json.loads((pack / "refs" / "proposal.json").read_text(encoding="utf-8"))
    prop["mode"] = "pick"
    (pack / "refs" / "proposal.json").write_text(json.dumps(prop), encoding="utf-8")
    assert run_cli("profile_fetch.py", "pick", "--pack", pack, "--proposed").returncode == 2


def test_style_flag_resolves_pack_for_pick(fake, tmp_path):
    import styles
    pack = tmp_path / "stylepack"
    styles.save_registry({"version": 2, "styles": [{"slug": "demo", "name": "Demo", "pack": str(pack)}]})
    _listed(fake, tmp_path, pack)
    assert run_cli("profile_fetch.py", "pick", "--style", "demo", "--ranks", "1").returncode == 0
    assert json.loads((pack / "refs" / "proposal.json").read_text(encoding="utf-8"))["selected"]


class _Post:
    def __init__(self, k, video=True):
        import datetime as dt
        self.shortcode, self.is_video = f"SC{k:02d}", video
        self.video_view_count, self.likes, self.comments = 1000 * (k + 1), 10 * k, k
        self.date_utc = dt.datetime(2026, 1, (k % 28) + 1)
        self.video_duration, self.caption, self.url = 20.0, f"post {k}", f"https://example.cdn/{k}.jpg"


class _FakeInstaloader:
    class Instaloader:
        def __init__(self, **kw):
            self.context = object()

    class Profile:
        @staticmethod
        def from_username(ctx, handle):
            class _P:
                def get_posts(self):
                    return [_Post(k) for k in range(15)] + [_Post(99, video=False)]
            return _P()


def test_instagram_falls_back_to_instaloader(fake, tmp_path, monkeypatch):
    import sys
    monkeypatch.setenv("FAKE_YTDLP_FAIL", "ERROR: login required")
    monkeypatch.setitem(sys.modules, "instaloader", _FakeInstaloader)
    monkeypatch.setattr(pf, "_has_instaloader", lambda: True)
    pack = tmp_path / "pack"
    assert pf.main(["list", "https://www.instagram.com/example.creator/", "--pack", str(pack)]) == 0
    prop = json.loads((pack / "refs" / "proposal.json").read_text(encoding="utf-8"))
    assert prop["mode"] == "views" and all("/reel/" in e["url"] for e in prop["entries"])
    assert len(prop["entries"]) == 15


def test_instagram_no_fallback_exit3(fake, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_YTDLP_FAIL", "ERROR: login required")
    monkeypatch.setattr(pf, "_has_instaloader", lambda: False)
    assert pf.main(["list", "https://www.instagram.com/example.creator/", "--pack", str(tmp_path / "p")]) == 3


def test_page_has_no_media_or_scripts(fake, tmp_path):
    import re
    pack, _ = _listed(fake, tmp_path)
    out = tmp_path / "page.html"
    assert run_cli("profile_fetch.py", "page", "--pack", pack, "--out", out).returncode == 0
    html = out.read_text(encoding="utf-8")
    prop = json.loads((pack / "refs" / "proposal.json").read_text(encoding="utf-8"))
    assert all(e["id"] in html for e in prop["entries"])
    assert "<video" not in html and "<script src" not in html
    thumbs = {e["thumbnail"] for e in prop["entries"] if e.get("thumbnail")}
    thumbs |= {prop["url"]} if isinstance(prop.get("url"), str) else set()
    thumbs |= {e[k] for e in prop["entries"] for k in ("url", "link", "permalink") if isinstance(e.get(k), str)}
    for url in re.findall(r"https://[^\"'\s)<>]+", html):
        assert url in thumbs or "fonts.googleapis.com" in url or "fonts.gstatic.com" in url

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

import json
import shutil
import subprocess
import sys

import pytest

import ideas
import taste
from conftest import run_cli

SNIPPET = '<div style="color:{{brand}}">{{title}}</div>'
PARAMS = {"brand": {"type": "color", "default": "#123456", "client_bound": True, "role": "accent"},
          "title": {"type": "text", "default": "Hi", "client_bound": False},
          "logo": {"type": "image", "default": None, "client_bound": True}}


def snippet(tmp_path, text=SNIPPET, name="s.html"):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


def mk(tmp_path, title="t", kind="b-roll", beats=("hook",), tags=(), style=None, client=None, **kw):
    return ideas.add(title, kind, list(beats), "9:16", snippet(tmp_path, name=title + ".html"), PARAMS,
                     tags=tags, made_for={"style": style, "client": client, "project": "p1"}, **kw)


def add_client(name="example-shop"):
    r = run_cli("clients.py", "add", "--name", name, "--color", "#008754")
    assert r.returncode == 0, r.stderr
    r = run_cli("clients.py", "palette", name, "--pick", "accent")
    assert r.returncode == 0, r.stderr


def test_add_and_search_filters(home, tmp_path):
    a = mk(tmp_path, "a", "b-roll", tags=["warm"], style="paper-3d")
    mk(tmp_path, "b", "text-animation", tags=["warm"], style="paper-3d")
    mk(tmp_path, "c", "b-roll", tags=["cool"])
    assert a["id"] == "I0001"
    assert [i["id"] for i in ideas.search(kind="b-roll", tags=["warm"])] == ["I0001"]
    assert {i["id"] for i in ideas.search(style="paper-3d")} == {"I0001", "I0002"}
    assert ideas.search(beat_types=["nope"]) == []
    assert (home / "ideas" / "I0001" / "snippet.html").is_file()
    r = run_cli("ideas.py", "search", "--style", "paper-3d", "--json")
    assert r.returncode == 0 and len(json.loads(r.stdout)) == 2


def test_ranking_by_pick_rate(home, tmp_path):
    a, b = mk(tmp_path, "a"), mk(tmp_path, "b")
    for _ in range(4):
        ideas.offered(a["id"])
    for _ in range(3):
        ideas.picked(a["id"])
    ideas.offered(b["id"])
    assert [i["id"] for i in ideas.search()] == [a["id"], b["id"]]


def test_client_literal_refused(home, tmp_path):
    add_client()
    bad = snippet(tmp_path, '<div style="color:#008754">x</div>', "bad.html")
    with pytest.raises(ValueError, match="parameter"):
        ideas.add("t", "b-roll", ["hook"], "9:16", bad, {}, made_for={"client": "example-shop"})
    r = run_cli("ideas.py", "add", "t", "--kind", "b-roll", "--beat-type", "hook", "--aspect", "9:16",
                "--snippet", bad, "--client", "example-shop")
    assert r.returncode == 1 and "parameter" in r.stderr
    named = snippet(tmp_path, "<p>Welcome to Example-Shop</p>", "named.html")
    with pytest.raises(ValueError, match="parameter"):
        ideas.add("t", "b-roll", ["hook"], "9:16", named, {}, made_for={"client": "example-shop"})
    ok = snippet(tmp_path, '<div style="color:{{brand}}">x</div>', "ok.html")
    r = run_cli("ideas.py", "add", "t", "--kind", "b-roll", "--beat-type", "hook", "--aspect", "9:16",
                "--snippet", ok, "--client", "example-shop")
    assert r.returncode == 0, r.stderr


def test_render_with_client_tokens(home, tmp_path):
    add_client()
    i = mk(tmp_path)
    hexv = json.loads((home / "clients" / "example-shop" / "client.json").read_text(encoding="utf-8")
                      )["palette"]["roles"]["dark"]["accent"]
    out = ideas.render(i["id"], "example-shop", {"logo": "l.png"})
    assert hexv in out and ">Hi<" in out
    r = run_cli("ideas.py", "render", i["id"], "--client", "example-shop", "--out", tmp_path / "o.html")
    assert r.returncode == 2 and "logo" in r.stderr
    r = run_cli("ideas.py", "render", i["id"], "--client", "example-shop", "--param", "logo=l.png",
                "--param", "title=Yo", "--out", tmp_path / "o.html")
    assert r.returncode == 0 and ">Yo<" in (tmp_path / "o.html").read_text(encoding="utf-8")


def test_render_without_client_uses_brand_defaults(home, tmp_path):
    i = mk(tmp_path)
    out = ideas.render(i["id"], None, {"logo": "x"})
    assert ideas.brand_defaults()["roles"]["dark"]["accent"] in out
    assert ideas.render(i["id"], None, {"logo": "x"}, theme="light").count(
        ideas.brand_defaults()["roles"]["light"]["accent"]) == 1


@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="needs ffmpeg")
def test_clip_capped(home, tmp_path):
    src = tmp_path / "src.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=1080x1920:rate=10:duration=6",
                    "-pix_fmt", "yuv420p", str(src)], check=True)
    i = mk(tmp_path, clip=src)
    clip = home / i["clip"]
    pr = json.loads(subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height:format=duration",
         "-of", "json", str(clip)], capture_output=True, text=True).stdout)
    assert float(pr["format"]["duration"]) <= 4.05
    s = pr["streams"][0]
    assert min(s["width"], s["height"]) == 720 and s["height"] > s["width"]


def test_remove_archives(home, tmp_path):
    i = mk(tmp_path)
    ideas.remove(i["id"])
    assert ideas.search() == []
    assert (home / "ideas" / "_removed" / i["id"] / "snippet.html").is_file()
    assert not (home / "ideas" / i["id"]).exists()
    assert ideas.get(i["id"])["removed"] is True


def test_avoid_rejected_twice(home, tmp_path):
    add_client()
    a, b = mk(tmp_path, "a"), mk(tmp_path, "b")
    for proj in ("p1", "p2"):
        taste.log("example-shop", None, "idea", "x", [f"idea:{a['id']}"], proj)
    ids = lambda **k: [i["id"] for i in ideas.search(**k)]  # noqa: E731
    assert set(ids()) == {a["id"], b["id"]}
    assert ids(avoid_client="example-shop") == [b["id"]]
    r = run_cli("ideas.py", "search", "--avoid-client", "example-shop", "--json")
    assert [i["id"] for i in json.loads(r.stdout)] == [b["id"]]
    assert ids(exclude_sigs=frozenset({f"idea:{b['id']}"})) == [a["id"]]


def test_style_link_not_copied(home, tmp_path):
    pack = tmp_path / "pack"
    pack.mkdir()
    sn = pack / "x.html"
    sn.write_text("{{title}}", encoding="utf-8")
    i = ideas.add("l", "caption", ["hook"], "9:16", sn, {"title": PARAMS["title"]},
                  link={"style": "paper-3d", "entry": "x", "path": "x.html"})
    assert i["source"] == "style-link" and i["snippet"] == str(sn)
    assert not (home / "ideas" / i["id"] / "snippet.html").exists()
    assert ideas.render(i["id"], None, {"title": "A"}) == "A"


def test_no_network_imports():
    src = open(ideas.__file__, encoding="utf-8").read()
    for mod in ("requests", "urllib.request", "http.client", "httpx", "socket", "aiohttp"):
        assert f"import {mod}" not in src
    assert sys.modules  # sanity

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


# ---- fix round 1

def _raises_param(tmp_path, text, params=None, **kw):
    sn = snippet(tmp_path, text, "x.html")
    with pytest.raises(ideas.Refused, match="parameter"):
        ideas.add("t", "b-roll", ["hook"], "9:16", sn, params or {}, **kw)


@pytest.mark.parametrize("lit", ["#008754", "#abc", "#008754ff", "#abcd", "rgb(0,135,84)",
                                 "rgba(0, 135, 84, .5)", "rgb(0 135 84)", "hsl(157, 100%, 26.5%)"])
def test_literal_forms_refused(home, tmp_path, lit):
    add_client()
    cp = home / "clients" / "example-shop" / "client.json"
    d = json.loads(cp.read_text(encoding="utf-8"))
    d["brand_colors"].append("#aabbcc")
    cp.write_text(json.dumps(d), encoding="utf-8")
    _raises_param(tmp_path, f'<i style="color:{lit}">x</i>', made_for={"client": "example-shop"})


def test_literal_in_param_default_and_other_clients(home, tmp_path):
    add_client()
    p = {"c": {"type": "color", "default": "#008754", "client_bound": False}}
    _raises_param(tmp_path, "{{c}}", p)  # no made-for client: every client is checked


def test_name_normalised_and_logo(home, tmp_path):
    add_client()
    for t in ("<p>EXAMPLE_SHOP</p>", "<p>example shop</p>", "<p>exampleshop</p>"):
        _raises_param(tmp_path, t)
    c = home / "clients" / "example-shop" / "client.json"
    d = json.loads(c.read_text(encoding="utf-8"))
    d["logo"] = ["C:/x/Brand Logo v2.svg"]
    d["aliases"] = ["ex"]
    c.write_text(json.dumps(d), encoding="utf-8")
    _raises_param(tmp_path, '<img src="assets/brand logo v2.svg">')
    ok = snippet(tmp_path, "<p>ex</p> <p>shopping</p>", "ok.html")  # short alias and part-words are not names
    assert ideas.add("t", "b-roll", ["hook"], "9:16", ok, {})["id"]


def test_neutrals_not_refused(home, tmp_path):
    add_client()
    p = home / "clients" / "example-shop" / "client.json"
    d = json.loads(p.read_text(encoding="utf-8"))
    d["palette"]["roles"]["light"]["background"] = "#ffffff"
    d["palette"]["roles"]["dark"]["text"] = "#000000"
    p.write_text(json.dumps(d), encoding="utf-8")
    sn = snippet(tmp_path, "<b style='color:#fff;background:#000'>x</b>", "n.html")
    assert ideas.add("t", "b-roll", ["hook"], "9:16", sn, {}, made_for={"client": "example-shop"})["id"]


def test_unknown_client_exit_2(home, tmp_path):
    r = run_cli("ideas.py", "add", "t", "--kind", "b-roll", "--aspect", "9:16", "--snippet",
                snippet(tmp_path), "--client", "ghost")
    assert r.returncode == 2 and "ghost" in r.stderr


def test_style_link_checked_validated_and_idempotent(home, tmp_path):
    add_client()
    pack = tmp_path / "pack"
    pack.mkdir()
    bad = pack / "b.html"
    bad.write_text("example-shop", encoding="utf-8")
    link = {"style": "paper-3d", "entry": "e1", "path": "b.html"}
    with pytest.raises(ideas.Refused):
        ideas.add("l", "caption", ["hook"], "9:16", bad, {}, link=link)
    with pytest.raises(ValueError, match="not found"):
        ideas.add("l", "caption", ["hook"], "9:16", pack / "none.html", {}, link=link)
    with pytest.raises(ValueError, match="not found"):
        ideas.add("l", "caption", ["hook"], "9:16", None, {}, still=pack / "none.png", link=link)
    good = pack / "g.html"
    good.write_text("{{t}}", encoding="utf-8")
    a = ideas.add("l", "caption", ["hook"], "9:16", good, {}, link=link, made_for={"style": "paper-3d"})
    ideas.picked(a["id"], "example-shop")
    b = ideas.add("l2", "caption", ["hook"], "9:16", good, {}, link=link, made_for={"style": "paper-3d"},
                  approved=True)
    assert b["id"] == a["id"] and b["title"] == "l2" and b["approved"] and b["picked"] == 1
    assert len(ideas.load_index()["ideas"]) == 1
    assert ideas.find_by_link("paper-3d", "e1")["id"] == a["id"] and ideas.find_by_link("paper-3d", "zz") is None
    assert [i["id"] for i in ideas.find(made_for_style="paper-3d")] == [a["id"]]
    ideas.set_approved(a["id"], False)
    assert ideas.get(a["id"])["approved"] is False


def test_link_without_snippet_renders_still(home, tmp_path):
    still = tmp_path / "s.png"
    still.write_bytes(b"x")
    i = ideas.add("l", "caption", ["hook"], "9:16", None, {}, still=still, link={"style": "s", "entry": "e"})
    assert "<img" in ideas.render(i["id"])
    with pytest.raises(ValueError):
        ideas.add("l", "caption", ["hook"], "9:16", None, {}, link={"style": "s", "entry": "f"})


def test_include_generic_and_per_client_counters(home, tmp_path):
    add_client()
    a = mk(tmp_path, "a", client="example-shop")
    g = mk(tmp_path, "g")
    ids = lambda **k: {i["id"] for i in ideas.search(**k)}  # noqa: E731
    assert ids(client="example-shop") == {a["id"]}
    assert ids(client="example-shop", include_generic=True) == {a["id"], g["id"]}
    ideas.offered(g["id"], "example-shop")
    ideas.picked(g["id"], "example-shop")
    got = ideas.get(g["id"])
    assert got["offered"] == 1 and got["by_client"]["example-shop"] == {"offered": 1, "picked": 1}


def test_missing_still_and_missing_role(home, tmp_path):
    r = run_cli("ideas.py", "add", "t", "--kind", "b-roll", "--aspect", "9:16", "--snippet",
                snippet(tmp_path), "--still", tmp_path / "nope.png")
    assert r.returncode == 2 and "still" in r.stderr
    i = mk(tmp_path)
    with pytest.raises(ValueError, match="no role"):
        ideas.render(i["id"], {"theme": "dark", "palette": {"roles": {"dark": {}}}}, {"logo": "x"})


def test_failed_add_leaves_no_orphans(home, tmp_path, monkeypatch):
    sn = snippet(tmp_path)
    monkeypatch.setattr(ideas.store, "write_json", lambda *a, **k: (_ for _ in ()).throw(OSError("boom")))
    with pytest.raises(OSError):
        ideas.add("t", "b-roll", ["hook"], "9:16", sn, {})
    monkeypatch.undo()
    assert not list((home / "ideas").glob("I*")) and not list((home / "ideas").glob("_new-*"))
    assert ideas.load_index()["ideas"] == []


# ---------------------------------------------------------------- Task 3: seeding

def _pack(home, tmp_path):
    import store
    pack = tmp_path / "pack"
    lib = pack / "library"
    anims = []
    for k, (cat, reusable) in enumerate((("Kinetic Text", True), ("B-roll", False), ("Transition", None)), 1):
        rel = f"V01-A0{k} t{k}"
        (lib / rel).mkdir(parents=True)
        (lib / rel / "still.png").write_bytes(b"png")
        e = {"id": f"V01-A0{k}", "title": f"t{k}", "category": cat, "scene_type": "overlay",
             "folder": rel, "still": f"{rel}/still.png", "clip": f"{rel}/clip.mp4"}
        if reusable:
            e["reusable"] = True
        anims.append(e)
    store.write_json(lib / "index.json", {"animations": anims})
    store.write_json(pack / "animations.json", {"animations": [{"id": "V01-A03", "reusable": True}]})
    home.mkdir(parents=True, exist_ok=True)
    (home / "styles.json").write_text(json.dumps({"version": 2, "styles": [
        {"slug": "paper-3d", "name": "Paper 3D", "pack": str(pack), "aspect": "9:16"}]}), encoding="utf-8")
    return pack


def test_seed_style_links_not_copies(home, tmp_path):
    pack = _pack(home, tmp_path)
    r = ideas.seed_style("paper-3d")
    assert len(r["added"]) == 2 and r["skipped"] == ["V01-A02"]
    for iid in r["added"]:
        i = ideas.get(iid)
        assert i["source"] == "style-link" and str(pack) in i["still"]
        assert [f.name for f in (home / "ideas" / iid).iterdir()] == ["meta.json"]
    assert {ideas.get(i)["kind"] for i in r["added"]} == {"text-animation", "transition"}
    assert ideas.seed_style("paper-3d")["added"] == []
    assert len(ideas.seed_style("paper-3d", categories=["b-roll"])["added"]) == 1
    rc = run_cli("ideas.py", "seed-style", "paper-3d")
    assert rc.returncode == 0 and "0" in rc.stdout


def _closing_project(home, tmp_path):
    import project
    import store
    home.mkdir(parents=True, exist_ok=True)
    (home / "styles.json").write_text(json.dumps({"version": 2, "styles": [{"slug": "paper-3d",
                                                  "name": "Paper 3D"}]}), encoding="utf-8")
    d = tmp_path / "proj"
    project.init(d, "launch", "clip.mp4", "paper-3d")
    (d / "build").mkdir(parents=True, exist_ok=True)
    (d / "build" / "b1.html").write_text('<div style="color:{{brand}}">{{title}}</div>', encoding="utf-8")
    (d / "build" / "b2.html").write_text('<div>{{title}}</div>', encoding="utf-8")
    beats = {"1": {"built": True, "snippet": "build/b1.html", "params": PARAMS, "title": "one", "kind": "b-roll",
                   "beat_types": ["number"]},
             "2": {"built": True, "snippet": "build/b2.html", "params": {"title": PARAMS["title"]},
                   "title": "two", "kind": "text-animation", "beat_types": []},
             "3": {"built": True, "snippet": None, "params": None},
             "4": {"built": True, "snippet": "build/b2.html", "params": {}}}
    store.write_json(d / "build" / "state.json", {"version": 1, "beats": beats})
    store.write_json(d / "plan.json", {"beats": [{"n": 1, "start": 0.0, "end": 2.0}, {"n": 2, "start": 2.0, "end": 8.0},
                                                 {"n": 3, "start": 8.0, "end": 9.0}, {"n": 4, "start": 9.0, "end": 10.0}]})
    return d


def test_final_candidates_only_snippets(home, tmp_path):
    d = _closing_project(home, tmp_path)
    c = ideas.final_candidates(d)
    assert [x["beat"] for x in c] == [1, 2] and all(x["ticked"] for x in c)
    assert c[1]["end_s"] - c[1]["start_s"] == pytest.approx(4.0)
    assert (d / "closing.json").is_file()
    r = run_cli("ideas.py", "from-final", d)
    assert r.returncode == 0 and "one" in r.stdout and "two" in r.stdout


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg missing")
def test_commit_final_untick_and_close(home, tmp_path):
    import project
    d = _closing_project(home, tmp_path)
    final = tmp_path / "final.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc2=s=1080x1920:d=10:r=25",
                    "-pix_fmt", "yuv420p", str(final)], check=True)
    r = run_cli("ideas.py", "commit-final", d, "--final", final, "--untick", "2")
    assert r.returncode == 0, r.stderr
    banked = [i for i in ideas.load_index()["ideas"] if i["source"] == "final"]
    assert len(banked) == 1 and banked[0]["approved"] and banked[0]["made_for"]["style"] == "paper-3d"
    clip = ideas._abs(banked[0]["clip"])
    dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                                str(clip)], capture_output=True, text=True).stdout)
    assert dur <= 4.05
    assert project.load(d)["status"] == "final"

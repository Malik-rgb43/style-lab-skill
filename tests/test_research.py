import json
import re
from pathlib import Path

import pytest

import font_match
import store

from conftest import run_cli


def _make_font(path, shape):
    from fontTools.fontBuilder import FontBuilder
    from fontTools.pens.ttGlyphPen import TTGlyphPen

    def glyph(kind):
        pen = TTGlyphPen(None)
        if kind == "tall":
            pen.moveTo((200, 0)); pen.lineTo((200, 700)); pen.lineTo((300, 700)); pen.lineTo((300, 0))
            pen.closePath()
        elif kind == "box":
            pen.moveTo((50, 0)); pen.lineTo((50, 500)); pen.lineTo((550, 500)); pen.lineTo((550, 0))
            pen.closePath()
        elif kind == "tri":
            pen.moveTo((50, 0)); pen.lineTo((300, 700)); pen.lineTo((550, 0)); pen.closePath()
        return pen.glyph()

    names = [".notdef", "space", "A", "b", "c", "one", "two", "three"]
    cmap = {32: "space", 65: "A", 98: "b", 99: "c", 49: "one", 50: "two", 51: "three"}
    fb = FontBuilder(1000, isTTF=True)
    fb.setupGlyphOrder(names)
    fb.setupCharacterMap(cmap)
    fb.setupGlyf({n: (glyph("none") if n == "space" else glyph(shape)) for n in names})
    fb.setupHorizontalMetrics({n: (600, 0) for n in names})
    fb.setupHorizontalHeader(ascent=800, descent=-200)
    fb.setupNameTable({"familyName": f"Syn{shape}", "styleName": "Regular"})
    fb.setupOS2(sTypoAscender=800, sTypoDescender=-200)
    fb.setupPost()
    fb.save(str(path))


def test_font_match_ranks_true_font_first(tmp_path):
    PIL = pytest.importorskip("PIL")
    pytest.importorskip("fontTools")
    from PIL import Image, ImageDraw, ImageFont
    fdir = tmp_path / "fonts"
    fdir.mkdir()
    for shape in ("tall", "box", "tri"):
        _make_font(fdir / f"{shape}.ttf", shape)
    font = ImageFont.truetype(str(fdir / "box.ttf"), 60)
    im = Image.new("L", (500, 120), 255)
    ImageDraw.Draw(im).text((10, 20), "Abc 123", font=font, fill=0)
    crop = tmp_path / "crop.png"
    im.save(crop)
    top = font_match.rank(str(crop), "Abc 123", font_match.list_fonts([fdir]))
    assert top[0]["file"].endswith("box.ttf")
    assert top[0]["family"] == "Synbox"
    r = run_cli("font_match.py", crop, "--text", "Abc 123", "--fonts-dir", fdir)
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout)[0]["file"].endswith("box.ttf")


def test_assets_log_table_and_json(tmp_path):
    pack = tmp_path / "pack"
    for n in ("one", "two"):
        r = run_cli("assets_log.py", "add", pack, "--kind", "font", "--name", n, "--source", "web",
                    "--licence", "OFL")
        assert r.returncode == 0, r.stderr
    md = (pack / "ASSETS.md").read_text(encoding="utf-8")
    assert len([ln for ln in md.splitlines() if ln.startswith("| font")]) == 2
    assert len(json.loads((pack / "assets.json").read_text(encoding="utf-8"))) == 2
    r = run_cli("assets_log.py", "add", pack, "--kind", "font", "--name", "x", "--source", "web")
    assert r.returncode == 2


def test_requires_proposal_from_assets(tmp_path):
    pack = tmp_path / "pack"
    run_cli("assets_log.py", "add", pack, "--kind", "broll", "--name", "city", "--source", "ai",
            "--licence", "generated")
    r = run_cli("assets_log.py", "requires", pack)
    assert r.returncode == 0
    props = json.loads(r.stdout)
    assert props and props[0]["why"] and props[0]["fallback"] and props[0]["level"] == "recommended"
    assert not (pack / "styles.json").exists()


def test_sound_libraries_in_store(home):
    assert store.settings()["sound_libraries"] == []
    r = run_cli("assets_log.py", "sound-lib", "add", "lib1", "D:/sfx", "--kind", "sfx", "--kind", "music")
    assert r.returncode == 0, r.stderr
    libs = json.loads(run_cli("assets_log.py", "sound-lib", "list").stdout)
    assert libs == [{"name": "lib1", "path": "D:/sfx", "kinds": ["sfx", "music"]}]
    assert store.settings()["sound_libraries"] == libs
    assert run_cli("assets_log.py", "sound-lib", "remove", "lib1").returncode == 0
    assert store.settings()["sound_libraries"] == []


def _caps(tmp_path, ok):
    p = tmp_path / "caps.json"
    p.write_text(json.dumps({"local": {n: {"ok": True} for n in ok}, "session": {}}), encoding="utf-8")
    return p


def test_recipe_check_missing_required(tmp_path):
    pack = tmp_path / "pack"
    r = run_cli("assets_log.py", "recipe", "add", pack, "--kind", "ai_still_animated",
                "--how", "ai_image+i2v", "--tool", "image_generator:required:stock photo + zoom")
    assert r.returncode == 0, r.stderr
    r = run_cli("assets_log.py", "recipe", "check", pack, "--caps", _caps(tmp_path, ["ffmpeg"]))
    assert r.returncode == 1
    assert "image_generator" in r.stdout and "stock photo + zoom" in r.stdout


def test_recipe_roundtrip_and_requires(tmp_path):
    pack = tmp_path / "pack"
    r = run_cli("assets_log.py", "recipe", "add", pack, "--kind", "ai_still_animated",
                "--how", "ai_image+i2v", "--look", '{"grain": 0.2, "lens": "35mm"}',
                "--prompt-template", "{subject} in {era}", "--negative", "text",
                "--motion", '{"camera_ease": "easeInOut", "dur_s": 3.2}',
                "--tool", "image_generator:required:stock", "--tool", "ffmpeg:required",
                "--test", "paid_ok_pending", "--sample", "a.png", "--evidence", "v1@12.5")
    assert r.returncode == 0, r.stderr
    rec = json.loads(run_cli("assets_log.py", "recipe", "list", pack).stdout)["recipes"][0]
    assert rec["how"] == "ai_image+i2v" and rec["look"]["grain"] == 0.2 and rec["look"]["lens"] == "35mm"
    assert rec["prompt_template"] == "{subject} in {era}" and rec["negative"] == "text"
    assert rec["motion"]["dur_s"] == 3.2 and len(rec["tools"]) == 2
    assert rec["test"] == {"status": "paid_ok_pending", "sample_paths": ["a.png"], "notes": ""}
    assert rec["source_evidence"] == ["v1@12.5"]
    r = run_cli("assets_log.py", "recipe", "check", pack, "--caps", _caps(tmp_path, ["ffmpeg", "higgsfield"]))
    assert r.returncode == 0, r.stdout + r.stderr
    assert "costs money" in r.stdout
    props = json.loads(r.stdout[r.stdout.index("{"):])["requires_proposals"]
    assert {p["name"] for p in props} == {"image_generator", "ffmpeg"}
    assert not (pack / "styles.json").exists()


def test_recipe_never_runs_paid():
    for name in ("assets_log.py", "font_match.py"):
        src = (Path(__file__).resolve().parent.parent / "scripts" / name).read_text(encoding="utf-8")
        assert not re.search(r"^\s*(import|from)\s+(requests|httpx|urllib\.request|http\.client)", src, re.M)

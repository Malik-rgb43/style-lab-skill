import json
import re
from pathlib import Path

import pytest

import assets_log
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


def _make_he_font(path, shape, with_hebrew=True):
    from fontTools.fontBuilder import FontBuilder
    from fontTools.pens.ttGlyphPen import TTGlyphPen

    def glyph(kind, w):
        pen = TTGlyphPen(None)
        pen.moveTo((50, 0)); pen.lineTo((50, 700)); pen.lineTo((50 + w, 700)); pen.lineTo((50 + w, 0))
        pen.closePath()
        if kind == "hook":
            pen.moveTo((50 + w, 700)); pen.lineTo((550, 700)); pen.lineTo((550, 600)); pen.lineTo((50 + w, 600))
            pen.closePath()
        return pen.glyph()

    chars = {32: "space", 0x5E9: "shin", 0x5DC: "lamed", 0x5D5: "vav", 0x5DD: "finalmem"}
    if not with_hebrew:
        chars = {32: "space", 65: "A"}
    names = [".notdef"] + list(chars.values())
    fb = FontBuilder(1000, isTTF=True)
    fb.setupGlyphOrder(names)
    fb.setupCharacterMap(chars)
    gl = {}
    for i, n in enumerate(names):
        gl[n] = TTGlyphPen(None).glyph() if n == "space" else glyph(shape, 100 + 60 * (i % 3))
    fb.setupGlyf(gl)
    fb.setupHorizontalMetrics({n: (650, 0) for n in names})
    fb.setupHorizontalHeader(ascent=800, descent=-200)
    fb.setupNameTable({"familyName": f"He{shape}{int(with_hebrew)}", "styleName": "Regular"})
    fb.setupOS2(sTypoAscender=800, sTypoDescender=-200)
    fb.setupPost()
    fb.save(str(path))


def test_font_match_hebrew_rtl_and_coverage(tmp_path):
    pytest.importorskip("PIL")
    pytest.importorskip("fontTools")
    from PIL import Image, ImageDraw, ImageFont
    fdir = tmp_path / "fonts"
    fdir.mkdir()
    _make_he_font(fdir / "plain.ttf", "plain")
    _make_he_font(fdir / "hook.ttf", "hook")
    _make_he_font(fdir / "latin.ttf", "plain", with_hebrew=False)
    text = "\u05e9\u05dc\u05d5\u05dd"
    font = ImageFont.truetype(str(fdir / "hook.ttf"), 60)
    raqm = font_match.has_raqm()
    kw = {"direction": "rtl"} if raqm else {}
    shown = text if raqm else font_match.reverse_visual(text)
    im = Image.new("L", (400, 120), 255)
    ImageDraw.Draw(im).text((10, 20), shown, font=font, fill=0, **kw)
    crop = tmp_path / "crop.png"
    im.save(crop)
    out = font_match.rank_full(str(crop), text, font_match.list_fonts([fdir]), lang="he")
    assert out["bidi"] in ("raqm", "reversed_fallback")
    assert any(x.endswith("latin.ttf") for x in out["skipped_no_glyphs"])
    assert out["results"][0]["file"].endswith("hook.ttf")
    assert all(not r["file"].endswith("latin.ttf") for r in out["results"])


def test_font_match_bad_crop_exit_2(tmp_path):
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"not an image")
    r = run_cli("font_match.py", bad, "--text", "x", "--fonts-dir", tmp_path)
    assert r.returncode == 2 and "Traceback" not in r.stderr


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
    out = json.loads(r.stdout)
    assert out["results"][0]["file"].endswith("box.ttf")
    assert "skipped_no_glyphs" in out and "bidi" in out


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
    assert {p["name"] for p in props} == {"image_generator", "video_generator", "ffmpeg"}
    assert not (pack / "styles.json").exists()


def test_recipe_never_runs_paid():
    import ast
    banned = {"urllib", "urllib3", "http", "requests", "httpx", "socket", "aiohttp"}
    root = Path(__file__).resolve().parent.parent / "scripts"
    for name in ("assets_log.py", "font_match.py", "store.py"):
        tree = ast.parse((root / name).read_text(encoding="utf-8"))
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                mods = [a.name for a in n.names]
            elif isinstance(n, ast.ImportFrom):
                mods = [n.module or ""]
            else:
                mods = []
                if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "__import__":
                    raise AssertionError(f"{name}: __import__ call")
            for m in mods:
                assert m.split(".")[0] not in banned, f"{name} imports {m}"


def test_requires_ai_enum_and_upgrade():
    rows = [{"kind": "broll", "name": "a", "source": "aircraft footage"},
            {"kind": "broll", "name": "b", "source": "ai_video"}]
    props = assets_log.propose_requires(rows)
    assert [p["name"] for p in props] == ["video_generator"]
    out = []
    assets_log.merge_proposal(out, "x", "w", "recommended", "")
    assets_log.merge_proposal(out, "x", "w", "required", "fb")
    assert out[0]["level"] == "required" and out[0]["fallback"] == "fb"


def test_recipe_how_implies_generators(tmp_path):
    pack = tmp_path / "pack"
    assert run_cli("assets_log.py", "recipe", "add", pack, "--kind", "k", "--how", "ai_image+i2v").returncode == 0
    r = run_cli("assets_log.py", "recipe", "check", pack, "--caps", _caps(tmp_path, ["ffmpeg"]))
    props = json.loads(r.stdout[r.stdout.index("{"):])["requires_proposals"]
    assert {p["name"] for p in props} == {"image_generator", "video_generator"}


def test_recipe_first_add_needs_how(tmp_path):
    r = run_cli("assets_log.py", "recipe", "add", tmp_path / "pack", "--kind", "k")
    assert r.returncode == 2 and "--how" in r.stderr


def test_corrupt_files_exit_2_one_line(tmp_path):
    pack = tmp_path / "pack"
    pack.mkdir()
    (pack / "recipes.json").write_text("{nope", encoding="utf-8")
    (pack / "assets.json").write_text('{"a": 1}', encoding="utf-8")
    for args in (("recipe", "list", pack), ("recipe", "check", pack), ("requires", pack),
                 ("add", pack, "--kind", "font", "--name", "n", "--source", "s", "--licence", "l")):
        r = run_cli("assets_log.py", *args)
        assert r.returncode == 2 and "Traceback" not in r.stderr, args
    (pack / "recipes.json").write_text('{"recipes": []}', encoding="utf-8")
    caps = tmp_path / "c.json"
    caps.write_text('{"local": []}', encoding="utf-8")
    r = run_cli("assets_log.py", "recipe", "check", pack, "--caps", caps)
    assert r.returncode == 2 and "Traceback" not in r.stderr


def test_resolver_tokens_and_paid_flag(tmp_path):
    import doctor
    caps = {"local": {"stockfish": {"ok": True}}, "session": {"mcp__higgsfield__generate": {"ok": True}}}
    assert not doctor.resolve("stock_footage", caps)["ok"]
    assert doctor.resolve("image_generator", caps)["ok"]
    assert doctor.is_paid("higgsfield") and doctor.is_paid("image_generator") and not doctor.is_paid("ffmpeg")
    pack = tmp_path / "pack"
    run_cli("assets_log.py", "recipe", "add", pack, "--kind", "k", "--how", "code", "--tool", "ffmpeg:required",
            "--paid", "ffmpeg")
    caps_f = tmp_path / "c.json"
    caps_f.write_text(json.dumps({"local": {"ffmpeg": {"ok": True}}}), encoding="utf-8")
    r = run_cli("assets_log.py", "recipe", "check", pack, "--caps", caps_f)
    assert "costs money" in r.stdout


def test_doctor_style_and_recipe_check_agree():
    import doctor
    caps = {"local": {"ffmpeg": {"ok": True}},
            "session": {"mcp_higgsfield": {"ok": True}, "stockfish": {"ok": True}}}
    for cap in ("image_generator", "stock_footage", "ffmpeg", "blender", "higgsfield"):
        rows, _p, _f = assets_log.check_recipes(
            [{"kind": "k", "how": "code", "tools": [{"capability": cap, "level": "required"}]}], caps)
        assert rows[0]["ready"] == doctor.resolve(cap, caps)["ok"]

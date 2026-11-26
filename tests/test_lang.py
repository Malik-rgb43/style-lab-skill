import json

import pytest

import lang
import store
from conftest import run_cli


def _law(anchor, median=0.18):
    return {"when": "a", "then": "b", "anchor": anchor,
            "lag": {"median": median, "p10": median / 2, "p90": median * 2, "n": 5, "seen_in": 4}}


def test_profiles_load_and_store_override():
    he, en = lang.load_profile("he"), lang.load_profile("en")
    assert he["dir"] == "rtl" and he["caption"]["no_italics"] is True
    assert he["speech_wps_typical"] == 2.6
    assert he["align_model"] == "imvladikon/wav2vec2-large-xlsr-53-hebrew"
    assert en["dir"] == "ltr" and en["speech_wps_typical"] == 2.7 and en["align_model"] == "MMS_FA"
    for p in (he, en):
        t = p["triggers"]
        assert len(t["months"]) == 12 and "?" in t["question_marks"]
        assert set(lang.ROLES) <= set(p["fonts"])
    assert {"מאה", "אלף", "מיליון"} <= set(he["triggers"]["number_words"])
    assert "Gveret Levin" in he["fonts"]["handwritten"]
    mod = dict(he, speech_wps_typical=3.0)
    store.write_json(store.path("lang", "he.json"), mod)
    assert lang.load_profile("he")["speech_wps_typical"] == 3.0
    with pytest.raises(KeyError):
        lang.load_profile("xx")


def test_scale_timing_word_lags_only():
    word, vis = _law("word"), _law("visual")
    noanchor = {"when": "a", "then": "b", "lag": {"median": 0.5}}
    motion = {"when": "a", "then": "b", "anchor": "word", "duration_frames": 15, "lag": {"median": 0.18}}
    out = lang.scale_timing([word, vis, noanchor, motion], 2.7 / 2.6)
    assert out[0]["lag"]["median"] == pytest.approx(0.187, abs=0.001)
    assert out[1] == vis and out[2] == noanchor
    assert out[3]["duration_frames"] == 15
    assert word["lag"]["median"] == 0.18  # input not mutated


def test_mirror_table():
    plan = {r["element"]: r for r in lang.mirror_plan(
        ["progress_bar", {"kind": "slide_in", "side": "left"}, "clock", "digits", "code_block", "logo", "chart_axes",
         "arrow", "list_order", "timecode", "reading_order"])}
    assert plan["progress_bar"]["mirror"] is True
    assert plan["slide_in"]["mirror"] is True and plan["slide_in"]["side"] == "right"
    for k in ("clock", "digits", "code_block", "logo", "chart_axes", "timecode"):
        assert plan[k]["mirror"] is False and plan[k]["why"]
    for k in ("arrow", "list_order", "reading_order"):
        assert plan[k]["mirror"] is True
    same = lang.mirror_plan(["progress_bar"], target_dir="ltr", source_dir="ltr")
    assert same[0]["mirror"] is False


def test_italics_swapped_for_he():
    he, en = lang.load_profile("he"), lang.load_profile("en")
    sw = lang.emphasis_swaps(["italic", "weight"], he)
    assert sw == [{"italic": "weight"}]
    assert lang.emphasis_swaps(["italic"], en) == []
    assert lang.emphasis_swaps(["weight"], he) == []


def test_font_candidates_without_fonttools(monkeypatch):
    monkeypatch.setattr(lang, "_fonttools", lambda: None)
    he = lang.load_profile("he")
    res = lang.font_candidates("caption", {"weight": 700}, he, [])
    assert [c["family"] for c in res] == he["fonts"]["caption"]
    assert "font_metrics" in res.not_run
    assert all(set(c) == {"family", "file", "score", "why"} for c in res)


def _make_font(path, family, weight, x_height):
    from fontTools.fontBuilder import FontBuilder
    from fontTools.pens.ttGlyphPen import TTGlyphPen
    fb = FontBuilder(1000, isTTF=True)
    fb.setupGlyphOrder([".notdef", "A"])
    fb.setupCharacterMap({65: "A"})
    pen = TTGlyphPen(None)
    pen.moveTo((0, 0)); pen.lineTo((0, 700)); pen.lineTo((500, 700)); pen.lineTo((500, 0)); pen.closePath()
    g = pen.glyph()
    fb.setupGlyf({".notdef": g, "A": g})
    fb.setupHorizontalMetrics({".notdef": (500, 0), "A": (500, 0)})
    fb.setupHorizontalHeader(ascent=800, descent=-200)
    fb.setupNameTable({"familyName": family, "styleName": "Regular"})
    fb.setupOS2(usWeightClass=weight, sxHeight=x_height, sCapHeight=700)
    fb.setupPost()
    fb.save(str(path))


def test_font_candidates_with_fonttools(tmp_path):
    pytest.importorskip("fontTools")
    _make_font(tmp_path / "a.ttf", "Heebo", 400, 350)
    _make_font(tmp_path / "b.ttf", "Rubik", 800, 500)
    prof = {"fonts": {"caption": ["Rubik", "Heebo"]}}
    src = {"x_height_ratio": 0.7, "weight": 800, "avg_advance": 0.5, "italic_angle": 0}
    res = lang.font_candidates("caption", src, prof, [tmp_path])
    assert [c["family"] for c in res] == ["Rubik", "Heebo"] and res.not_run == []
    assert res[0]["score"] > res[1]["score"] and res[0]["file"].endswith("b.ttf")
    src2 = {"x_height_ratio": 0.5, "weight": 400, "avg_advance": 0.5, "italic_angle": 0}
    assert lang.font_candidates("caption", src2, prof, [tmp_path])[0]["family"] == "Heebo"


def test_adapt_cli_and_add(tmp_path):
    pack = tmp_path / "pack"
    pack.mkdir()
    (pack / "laws.json").write_text(json.dumps({"laws": [_law("word"), _law("visual")]}), encoding="utf-8")
    r = run_cli("lang.py", "adapt", pack, "--to", "he", "--from-lang", "en")
    assert r.returncode == 0, r.stderr
    out = json.loads((pack / "adapt-he.json").read_text(encoding="utf-8"))
    assert out["timing_ratio"] == pytest.approx(2.7 / 2.6, abs=1e-3) and out["laws_scaled"] == 1
    assert out["recreation_needed"] is True and out["emphasis_swaps"] == [{"italic": "weight"}]
    assert {m["element"]: m["mirror"] for m in out["mirror"]}["progress_bar"] is True
    assert set(out["fonts"]) == set(lang.ROLES)
    assert run_cli("lang.py", "adapt", pack, "--to", "zz").returncode == 2
    assert run_cli("lang.py", "adapt", tmp_path / "nope", "--to", "he").returncode == 2
    src = tmp_path / "es.json"
    es = dict(lang.load_profile("en"), code="es")
    src.write_text(json.dumps(es), encoding="utf-8")
    assert run_cli("lang.py", "add", "es", "--from", src).returncode == 0
    shown = run_cli("lang.py", "show", "es")
    assert shown.returncode == 0 and json.loads(shown.stdout)["code"] == "es"
    assert run_cli("lang.py", "add", "fr", "--from", src).returncode == 2

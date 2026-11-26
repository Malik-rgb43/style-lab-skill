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
        assert len(t["months"]) >= 12 and "?" in t["question_marks"]
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


def _kinds(words, code):
    return [(t["i"], t["j"], t["kind"]) for t in lang.find_triggers(words, lang.load_profile(code))]


def test_find_triggers_he_phrases_prefix_digits():
    assert _kinds("שתיים עשרה אנשים".split(), "he") == [(0, 2, "number")]
    assert _kinds("בשלוש ימים".split(), "he") == [(0, 1, "number")]
    assert _kinds("שנים עשר חודשים".split(), "he") == [(0, 2, "number")]
    assert _kinds("ב-2020 במרס".split(), "he") == [(0, 1, "number"), (1, 2, "month")]
    assert _kinds("מה זה?".split(), "he") == [(1, 2, "question")]
    assert _kinds("שני מיליארד אלפים".split(), "he") == [(0, 1, "number"), (1, 2, "number"), (2, 3, "number")]
    assert _kinds("שלום עולם".split(), "he") == []


def test_find_triggers_en_ambiguous_months():
    assert _kinds("in May 2020".split(), "en") == [(1, 2, "month"), (2, 3, "number")]
    assert _kinds("you may go".split(), "en") == []
    assert _kinds("we march on".split(), "en") == []
    assert _kinds("in june we ship".split(), "en") == [(1, 2, "month")]
    assert _kinds("twenty one dogs".split(), "en") == [(0, 1, "number"), (1, 2, "number")]


def test_scale_ratio_one_adds_no_marker():
    assert "lag_scaled_by" not in lang.scale_timing([_law("word")], 1.0)[0]


def test_mirror_whole_tokens_and_side_keys():
    plan = lang.mirror_plan([{"kind": "slide_in", "from": "left"}, {"kind": "slide", "direction": "right"},
                             "encoded_banner", "lists"])
    assert plan[0]["side"] == "right" and plan[1]["side"] == "left"
    assert plan[2]["mirror"] is False  # "code" is inside "encoded" but not a token
    assert plan[3]["mirror"] is True


def test_weight_closeness_variable_range():
    m = {"weight": 400, "weight_range": [100, 900]}
    assert lang.weight_closeness(700, m) == 1.0
    fixed = {"weight": 400, "weight_range": None}
    assert lang.weight_closeness(700, fixed) < 1.0
    assert lang.weight_closeness(950, {"weight": 400, "weight_range": [100, 900]}) < 1.0


def test_variable_font_range(tmp_path):
    pytest.importorskip("fontTools")
    from fontTools.fontBuilder import FontBuilder
    from fontTools.pens.ttGlyphPen import TTGlyphPen
    fb = FontBuilder(1000, isTTF=True)
    fb.setupGlyphOrder([".notdef", "A"])
    fb.setupCharacterMap({65: "A"})
    pen = TTGlyphPen(None)
    pen.moveTo((0, 0)); pen.lineTo((0, 700)); pen.lineTo((500, 700)); pen.closePath()
    g = pen.glyph()
    fb.setupGlyf({".notdef": g, "A": g})
    fb.setupHorizontalMetrics({".notdef": (500, 0), "A": (500, 0)})
    fb.setupHorizontalHeader(ascent=800, descent=-200)
    fb.setupNameTable({"familyName": "Rubik", "styleName": "Regular"})
    fb.setupOS2(usWeightClass=400, sxHeight=350, sCapHeight=700)
    fb.setupPost()
    fb.setupFvar([("wght", 300, 400, 900, "Weight")], [])
    fb.save(str(tmp_path / "v.ttf"))
    m = lang.font_metrics(tmp_path / "v.ttf")
    assert m["weight"] == 400 and m["weight_range"] == [300, 900]
    prof = {"fonts": {"caption": ["Rubik"]}}
    res = lang.font_candidates("caption", {"weight": 800, "x_height_ratio": 0.5, "avg_advance": 0.5}, prof, [tmp_path])
    assert res[0]["score"] > 0.99


def test_adapt_input_errors(tmp_path):
    pack = tmp_path / "p"
    pack.mkdir()
    r = run_cli("lang.py", "adapt", pack, "--to", "he")
    assert r.returncode == 2 and "Traceback" not in r.stderr and "lang" in r.stderr
    (pack / "fingerprint.json").write_text("[1]", encoding="utf-8")
    assert run_cli("lang.py", "adapt", pack, "--to", "he", "--from", "en").returncode == 2
    (pack / "fingerprint.json").write_text(json.dumps({"lang": "en", "elements": "x"}), encoding="utf-8")
    r = run_cli("lang.py", "adapt", pack, "--to", "he")
    assert r.returncode == 2 and "Traceback" not in r.stderr
    (pack / "fingerprint.json").write_text(json.dumps({"lang": "en"}), encoding="utf-8")
    assert run_cli("lang.py", "adapt", pack, "--to", "he").returncode == 0
    out = json.loads((pack / "adapt-he.json").read_text(encoding="utf-8"))
    assert "font_metrics_source" in out["not_run"] or "font_metrics" in out["not_run"]


def test_mirror_naming_styles_and_plurals():
    names = ["progressBar", "progress_bar", "progress-bar", "slidein", "captions", "logos", "timecodes", "slideIn"]
    got = {r["element"]: r["mirror"] for r in lang.mirror_plan(names)}
    for k in ("progressBar", "progress_bar", "progress-bar", "slidein", "captions", "slideIn"):
        assert got[k] is True, k
    assert got["logos"] is False and got["timecodes"] is False


def test_find_triggers_false_positives():
    assert _kinds(["השני"], "he") == [] and _kinds(["המאה"], "he") == []
    assert _kinds("שני אנשים".split(), "he") == [(0, 1, "number")]
    assert _kinds("I may have 2 dogs".split(), "en") == [(3, 4, "number")]
    assert _kinds("May 5".split(), "en") == [(0, 1, "month"), (1, 2, "number")]
    assert _kinds("5 March".split(), "en") == [(0, 1, "number"), (1, 2, "month")]

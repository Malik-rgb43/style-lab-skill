import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

import check_cut as cc


def W(text, start_ms, end_ms):
    return {"w": text, "start_ms": start_ms, "end_ms": end_ms}


def heard_of(words, step=400, dur=300):
    return [W(w, k * step, k * step + dur) for k, w in enumerate(words)]


EXPECTED = ["שלום", "לכולם", "היום", "נדבר", "על", "עריכה"]


# ---------------------------------------------------------------- word diff

def test_clean_diff_has_no_items():
    assert cc.diff_words(EXPECTED, heard_of(EXPECTED), "he") == []


def test_inserted_repeat_is_extra():
    got = ["שלום", "לכולם", "היום", "היום", "נדבר", "על", "עריכה"]
    items = cc.diff_words(EXPECTED, heard_of(got), "he")
    assert [i["kind"] for i in items] == ["EXTRA"]
    assert items[0]["heard"] == "היום"
    assert items[0]["t"] in (pytest.approx(0.8), pytest.approx(1.2))  # one of the two equal words


def test_dropped_word_is_missing_with_expected_time():
    got = ["שלום", "לכולם", "נדבר", "על", "עריכה"]
    times = [k * 0.5 for k in range(len(EXPECTED))]
    items = cc.diff_words(EXPECTED, heard_of(got), "he", times)
    assert [i["kind"] for i in items] == ["MISSING"]
    assert items[0]["expected"] == "היום"
    assert items[0]["t"] == pytest.approx(1.0)


def test_misspelling_is_diff_but_close_spelling_is_ignored():
    got = ["שלום", "לכולם", "היום", "נדבר", "על", "מוזיקה"]
    items = cc.diff_words(EXPECTED, heard_of(got), "he")
    assert [i["kind"] for i in items] == ["DIFF"]
    assert items[0]["expected"] == "עריכה" and items[0]["heard"] == "מוזיקה"
    near = ["שלום", "לכולם", "היום", "נדבר", "על", "עריכות"]
    assert cc.diff_words(EXPECTED, heard_of(near), "he") == []


def test_digits_and_niqqud_are_normalised():
    assert cc.diff_words(["שָׁלוֹם", "5"], heard_of(["שלום", "חמש"]), "he") == []


# ---------------------------------------------------------------- audio rules

def _db(seconds, level=-60.0):
    return np.full(int(seconds * 100), level)


def test_join_inside_burst_fails_and_quiet_join_passes():
    db = _db(4)
    db[190:260] = -20.0  # speech burst 1.9 s - 2.6 s
    cuts = {"joins": [{"n": 1, "dst": 1.0, "src_out": 3.0, "src_in": 4.0, "before": "a", "after": "b"},
                      {"n": 2, "dst": 2.2, "src_out": 5.0, "src_in": 6.0, "before": "c", "after": "d"}]}
    items = cc.join_energy(cuts, db, 12.0)
    assert [i["n"] for i in items] == [2]
    assert items[0]["over_db"] == pytest.approx(40.0, abs=1)


def test_tighten_joins_judged_on_the_crossfade_and_stack_joins_skipped():
    db = _db(4)
    db[200:260] = -20.0  # speech from 2.0 s
    # a stack join whose picture switch sits on the next onset (2.0 s): never judged here
    stack = {"n": 1, "dst": 2.0, "type": "stack", "audio_xfade_ms": [2000.0, 2010.0]}
    # a keep join whose +-30 ms slices reach the onset, but whose crossfade is in the quiet before it
    keep = {"n": 2, "dst": 1.98, "type": "keep", "audio_xfade_ms": [1965.0, 1980.0]}
    # a keep join whose crossfade is in speech
    loud = {"n": 3, "dst": 2.3, "type": "keep", "audio_xfade_ms": [2292.5, 2307.5], "src_out": 9.0,
            "src_in": 9.5}
    cuts = {"joins": [stack, keep, loud]}
    items = cc.join_energy(cuts, db, 12.0)
    assert [i["n"] for i in items] == [3] and items[0]["src"] == "9.00->9.50"
    assert cc.stack_joins(cuts) == [stack]
    # the same keep join without the crossfade field (cut.py) is judged on +-30 ms and fails
    assert [i["n"] for i in cc.join_energy({"joins": [{"n": 2, "dst": 1.98}]}, db, 12.0)] == [2]
    # with the samples, the window level is exact: a quiet crossfade right before an onset that
    # starts inside the same 10 ms slice (1970-1980 ms) passes; on the slices alone it would not
    db2 = db.copy()
    db2[197] = -25.0
    mono = np.full(400 * 480, 10 ** (-60 / 20), dtype=np.float32)
    mono[int(1976.5 * 48):] = 0.1
    edge = {"n": 4, "dst": 1.976, "type": "bridge", "audio_xfade_ms": [1966.0, 1976.0]}
    assert [i["n"] for i in cc.join_energy({"joins": [edge]}, db2, 12.0)] == [4]
    assert cc.join_energy({"joins": [edge]}, db2, 12.0, mono) == []


def test_long_pause_needs_quiet_audio_to_confirm():
    heard = [W("a", 0, 300), W("b", 1500, 1800)]
    quiet = _db(2)
    quiet[0:30] = -20.0
    quiet[150:180] = -20.0
    items = cc.long_pauses(heard, quiet, 450)
    assert len(items) == 1 and items[0]["gap_ms"] > 450
    loud = _db(3)
    loud[0:200] = -20.0  # ASR says a gap but the audio is speech all along; room tone follows
    assert cc.long_pauses(heard, loud, 450) == []
    assert cc.long_pauses([W("a", 0, 300), W("b", 500, 800)], quiet, 450) == []


def test_long_word_flagged():
    items = cc.long_words([W("רגיל", 0, 400), W("ארוךךך", 500, 2100)], 1200)
    assert len(items) == 1 and items[0]["word"] == "ארוךךך" and items[0]["dur_ms"] == 1600


# ---------------------------------------------------------------- synthetic media

def _ffmpeg(args):
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-nostdin"] + args, check=True)


def make_stem(tmp_path, pan=None, words=EXPECTED):
    stem = tmp_path / "cut"
    af = ["-af", pan] if pan else []
    _ffmpeg(["-f", "lavfi", "-i", "sine=f=300:d=3", "-ac", "2"] + af + [str(stem) + ".mp4"])
    (tmp_path / "cut.words.json").write_text(json.dumps(
        {"words": [dict(w, i=k) for k, w in enumerate(heard_of(words))]}, ensure_ascii=False), encoding="utf-8")
    (tmp_path / "cut.cuts.json").write_text(json.dumps({"joins": []}), encoding="utf-8")
    return stem


def write_heard(tmp_path, words):
    p = tmp_path / "heard.json"
    p.write_text(json.dumps({"words": heard_of(words)}, ensure_ascii=False), encoding="utf-8")
    return p


def test_channel_balance_reports_six_db(tmp_path):
    stem = make_stem(tmp_path, pan="pan=stereo|c0=c0|c1=0.5*c0")
    assert cc.channel_balance(str(stem) + ".mp4") == pytest.approx(6.0, abs=0.3)
    even = tmp_path / "even"
    even.mkdir()
    stem2 = make_stem(even)
    assert abs(cc.channel_balance(str(stem2) + ".mp4")) < 0.2


def test_cli_clean_exits_zero(tmp_path, capsys):
    stem = make_stem(tmp_path)
    heard = write_heard(tmp_path, EXPECTED)
    assert cc.main([str(stem), "--heard", str(heard)]) == 0
    assert "0 item(s) left" in capsys.readouterr().out
    assert (tmp_path / "cut.check.txt").is_file()


def test_cli_items_then_explained_exit_zero(tmp_path, capsys):
    stem = make_stem(tmp_path)
    heard = write_heard(tmp_path, ["שלום", "לכולם", "היום", "היום", "נדבר", "על", "עריכה"])
    assert cc.main([str(stem), "--heard", str(heard)]) == 1
    out = capsys.readouterr().out
    assert "EXTRA" in out and "1 item(s) left" in out
    explain = tmp_path / "explain.json"
    explain.write_text(json.dumps({"items": [{"match": "EXTRA: expected [] heard [היום]",
                                              "why": "speaker says it twice on purpose"}]},
                                  ensure_ascii=False), encoding="utf-8")
    assert cc.main([str(stem), "--heard", str(heard), "--explain", str(explain)]) == 0
    out = capsys.readouterr().out
    assert "EXPLAINED" in out and "0 item(s) left" in out


def test_cli_reports_channel_imbalance(tmp_path, capsys):
    stem = make_stem(tmp_path, pan="pan=stereo|c0=c0|c1=0.5*c0")
    heard = write_heard(tmp_path, EXPECTED)
    assert cc.main([str(stem), "--heard", str(heard)]) == 1
    assert "BALANCE" in capsys.readouterr().out


def test_cli_missing_model_exits_three_with_one_line(tmp_path, monkeypatch, capsys):
    stem = make_stem(tmp_path)
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "emptycache"))
    monkeypatch.setattr(cc.importlib.util, "find_spec", lambda name: object())
    assert cc.main([str(stem)]) == 3
    err = capsys.readouterr().err.strip()
    assert len(err.splitlines()) == 1
    assert cc.HE_ASR_MODEL in err and "--allow-download" in err and "GB" in err


def test_cli_missing_package_exits_three(tmp_path, monkeypatch, capsys):
    stem = make_stem(tmp_path)
    monkeypatch.setattr(cc.importlib.util, "find_spec", lambda name: None)
    assert cc.main([str(stem)]) == 3
    err = capsys.readouterr().err.strip()
    assert len(err.splitlines()) == 1 and "faster-whisper" in err and "pip install" in err


def test_model_choice_by_language():
    assert cc.resolve_model("he") == cc.HE_ASR_MODEL
    assert cc.resolve_model("en") == cc.OTHER_ASR_MODEL
    assert cc.resolve_model("he", "custom/model") == "custom/model"


def test_cli_missing_input_is_exit_one(tmp_path, capsys):
    assert cc.main([str(tmp_path / "nope")]) == 1


def test_cli_missing_ffmpeg_exits_3_one_line(tmp_path, home, monkeypatch):
    from conftest import run_cli
    stem = make_stem(tmp_path)
    empty = tmp_path / "nobin"
    empty.mkdir()
    monkeypatch.setenv("PATH", str(empty))
    r = run_cli("check_cut.py", stem, "--heard", write_heard(tmp_path, EXPECTED))
    assert r.returncode == 3, r.stdout + r.stderr
    lines = [x for x in r.stderr.splitlines() if x.strip()]
    assert len(lines) == 1 and "ffmpeg" in lines[0] and "install" in lines[0], lines


def test_cli_video_that_does_not_match_its_plan_exits_2(tmp_path, capsys):
    stem = make_stem(tmp_path)
    _ffmpeg(["-f", "lavfi", "-i", "testsrc2=s=64x36:r=25:d=2", "-f", "lavfi", "-i", "sine=f=300:d=2",
             "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-shortest", str(stem) + ".mp4"])
    (tmp_path / "cut.cuts.json").write_text(json.dumps({"joins": [], "frames": 80}), encoding="utf-8")
    assert cc.main([str(stem), "--heard", str(write_heard(tmp_path, EXPECTED))]) == 2
    err = capsys.readouterr().err.strip().splitlines()
    assert len(err) == 1 and "50 frames" in err[0] and "80" in err[0], err
    (tmp_path / "cut.cuts.json").write_text(json.dumps({"joins": [], "frames": 51}), encoding="utf-8")
    assert cc.main([str(stem), "--heard", str(write_heard(tmp_path, EXPECTED))]) == 0  # +-1 frame is fine


def test_cli_tighten_edge_bad_fails_the_check(tmp_path, capsys):
    stem = make_stem(tmp_path)
    (tmp_path / "cut.cuts.json").write_text(json.dumps({"joins": [], "edge_bad": [{"edge": "end"}]}),
                                            encoding="utf-8")
    assert cc.main([str(stem), "--heard", str(write_heard(tmp_path, EXPECTED))]) == 1
    assert "TIGHTEN 1 edge problem" in capsys.readouterr().out


# ---------------------------------------------------------------- end to end: cut.py -> tighten.py -> check_cut.py

@pytest.mark.slow  # about 80 s on a slow machine (5 tool runs); not env-gated: it runs by default
def test_vsl_cut_then_tighten_then_check_is_clean(tmp_path, home):
    """the documented VSL flow on real cut.py output: cut --preset vsl, tighten, check --heard (the
    tightened words as the transcript) exits 0; every tighten join carries its crossfade."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent / "fixtures"))
    import make_media as mm
    from conftest import run_cli

    spans = [(300, 800), (1650, 2100), (2900, 3400), (4300, 4800), (5000, 5400), (6300, 6800)]
    texts = {0: "שלום", 1: "לכולם", 2: "היום", 3: "נדבר", 4: "על", 5: "עריכה"}
    d = tmp_path / "תיקייה"
    d.mkdir()
    src = mm.make_clip(d / "raw.mkv", spans, 7500, fps=30)
    words = mm.write_json(d / "words.aligned.json", mm.words_doc(spans, texts=texts))
    keep = mm.write_json(d / "keep.json", {"keep": [[0, 0], [2, 5]]})
    stem = d / "vsl"
    r = run_cli("cut.py", src, words, keep, "--out", stem, "--preset", "vsl")
    assert r.returncode == 0, r.stdout + r.stderr
    tight = d / "vsl.tight"
    r = run_cli("tighten.py", stem, "--out", tight)
    assert r.returncode == 0, r.stdout + r.stderr
    cuts = json.loads(Path(f"{tight}.cuts.json").read_text(encoding="utf-8"))
    assert cuts["joins"] and all(len(j["audio_xfade_ms"]) == 2 and j["type"] in ("stack", "keep", "bridge")
                                 for j in cuts["joins"])
    assert any(j["type"] == "stack" for j in cuts["joins"])
    r = run_cli("check_cut.py", tight, "--heard", f"{tight}.words.json")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "0 item(s) left" in r.stdout
    # the plain cut checks clean too (its joins have no crossfade field: the +-30 ms rule)
    r = run_cli("check_cut.py", stem, "--heard", f"{stem}.words.json")
    assert r.returncode == 0, r.stdout + r.stderr


# ---------------------------------------------------------------- slow: real model on a real cut

def _slow_ready():
    stem = os.environ.get("STYLE_LAB_TEST_STEM")
    return bool(os.environ.get("STYLE_LAB_SLOW") == "1" and stem and Path(stem + ".mp4").is_file())


@pytest.mark.slow
@pytest.mark.skipif(not _slow_ready(), reason="needs STYLE_LAB_SLOW=1 and STYLE_LAB_TEST_STEM (a cut stem)")
def test_slow_real_cut_is_clean_or_every_item_listed(monkeypatch):
    from conftest import REAL_HOME
    if not (os.environ.get("HF_HUB_CACHE") or os.environ.get("HF_HOME")):
        monkeypatch.setenv("HF_HUB_CACHE", str(REAL_HOME / ".cache" / "huggingface" / "hub"))
    if not cc.align.hf_model_cached(cc.HE_ASR_MODEL):
        pytest.skip(f"{cc.HE_ASR_MODEL} is not in the Hugging Face cache")
    code, txt = cc.check(os.environ["STYLE_LAB_TEST_STEM"])
    assert code in (0, 1)
    if code == 1:
        listed = [ln for ln in txt.splitlines()[1:] if ln.strip() and "EXPLAINED" not in ln]
        assert listed and all("s " in ln for ln in listed)

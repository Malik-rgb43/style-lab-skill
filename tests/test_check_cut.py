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

import importlib.util
import json
import os
import statistics
from pathlib import Path

import numpy as np
import pytest

import align


def _words(n, gap_ms=100, word_ms=200, long_pause_at=None, long_pause_ms=900):
    out, t = [], 0
    for i in range(n):
        out.append({"i": i, "w": "מילה", "start_ms": t, "end_ms": t + word_ms, "p": 0.9})
        t += word_ms + (long_pause_ms if i == long_pause_at else gap_ms)
    return out


# ---------- make_chunks ----------

def test_make_chunks_covers_every_word_once():
    words = _words(200)
    dur = words[-1]["end_ms"] + 2000
    chunks = align.make_chunks(words, 30, 0.6, dur)
    assert len(chunks) > 1
    seen = []
    for i, j, a, b in chunks:
        assert i <= j
        seen.extend(range(i, j + 1))
        assert a == max(0, words[i]["start_ms"] - 600)
        assert b == min(dur, words[j]["end_ms"] + 600)
    assert seen == list(range(200))


def test_make_chunks_splits_at_longest_pause():
    # 300 ms per word: a 30 s window holds ~100 words; one long pause late in the first window
    words = _words(200, long_pause_at=80)
    chunks = align.make_chunks(words, 30, 0.6, words[-1]["end_ms"] + 1000)
    assert chunks[0][1] == 80
    assert chunks[1][0] == 81


def test_make_chunks_margin_clamped_to_duration():
    words = _words(3)
    chunks = align.make_chunks(words, 30, 0.6, words[-1]["end_ms"] + 100)
    assert chunks == [(0, 2, 0, words[-1]["end_ms"] + 100)]


# ---------- refine_edges ----------

def _curve(n_slices, bursts, floor=-70.0, loud=-20.0):
    db = np.full(n_slices, floor)
    for a, b in bursts:  # slice indices, end exclusive
        db[a:b] = loud
    return db


def _ctc_word(i, c0, c1):
    return {"i": i, "w": "x", "asr_start_ms": c0 - 150, "asr_end_ms": c1 + 150,
            "start_ms": c0 - 150, "end_ms": c1 + 150, "ctc_start_ms": c0, "ctc_end_ms": c1}


def test_refine_moves_edges_onto_bursts():
    # burst 1: slices 100-150 (1000-1500 ms), CTC says 1040-1420
    # burst 2: slices 200-260 (2000-2600 ms), CTC says 2060-2500
    db = _curve(400, [(100, 150), (200, 260)])
    words = [_ctc_word(0, 1040, 1420), _ctc_word(1, 2060, 2500)]
    align.refine_edges(words, db, floor_db=-50.0)
    assert (words[0]["start_ms"], words[0]["end_ms"]) == (1000, 1500)
    assert (words[1]["start_ms"], words[1]["end_ms"]) == (2000, 2600)


def test_refine_local_floor_default():
    db = _curve(400, [(100, 150)])
    words = [_ctc_word(0, 1040, 1420)]
    align.refine_edges(words, db)
    assert (words[0]["start_ms"], words[0]["end_ms"]) == (1000, 1500)


def test_refine_never_crosses_neighbour_or_gap_midpoint():
    # two words joined by continuous energy (no silence between them)
    db = _curve(400, [(100, 260)])
    words = [_ctc_word(0, 1040, 1500), _ctc_word(1, 1700, 2500)]
    align.refine_edges(words, db, floor_db=-50.0)
    mid = (1500 + 1700) // 2
    assert words[0]["end_ms"] <= mid <= words[1]["start_ms"]
    assert words[0]["end_ms"] <= words[1]["start_ms"]
    for w in words:
        assert w["start_ms"] < w["end_ms"]


def test_refine_reach_is_bounded():
    # energy runs long before the word: start walks back at most 120 ms
    db = _curve(400, [(0, 300)])
    words = [_ctc_word(0, 2000, 2200)]
    align.refine_edges(words, db, floor_db=-50.0)
    assert words[0]["start_ms"] == 2000 - 120
    assert words[0]["end_ms"] == 2200 + 300


def test_refine_unaligned_word_keeps_asr_times():
    db = _curve(400, [(100, 150)])
    w = {"i": 0, "w": "x", "asr_start_ms": 900, "asr_end_ms": 1600, "start_ms": 1, "end_ms": 2}
    align.refine_edges([w], db, floor_db=-50.0)
    assert (w["start_ms"], w["end_ms"]) == (900, 1600)


# ---------- normalisation per backend ----------

def test_word_norm_flags_non_latin_script():
    assert align.word_norm("Привет", "ru") == ("", "unaligned_script")
    assert align.word_norm("...", "en") == ("", "unaligned")
    assert align.word_norm("Café!", "fr") == ("cafe", None)
    assert align.word_norm("שלום,", "he") == ("שלום", None)


# ---------- CLI ----------

def _tiny_inputs(tmp_path, lang="he"):
    src = tmp_path / "clip.wav"
    src.write_bytes(b"\0" * 64)
    words = tmp_path / "words.json"
    words.write_text(json.dumps({"language": lang, "words": [
        {"i": 0, "w": "שלום", "start_ms": 0, "end_ms": 400, "p": 0.9}]}), encoding="utf-8")
    return src, words, tmp_path / "out.json"


def test_help_works(home):
    from conftest import run_cli
    r = run_cli("align.py", "--help")
    assert r.returncode == 0
    assert "--allow-download" in r.stdout


def test_missing_torch_exits_3(tmp_path, monkeypatch, capsys):
    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec",
                        lambda name, *a, **k: None if name.split(".")[0] == "torch" else real(name, *a, **k))
    src, words, out = _tiny_inputs(tmp_path)
    rc = align.main([str(src), str(words), "--out", str(out)])
    err = capsys.readouterr().err.strip().splitlines()
    assert rc == 3
    assert len(err) == 1
    assert "torch" in err[0] and "pip install torch" in err[0] and "MB" in err[0]
    assert not out.exists()


def test_absent_model_exits_3(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a, **k: object())
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "empty-cache"))
    src, words, out = _tiny_inputs(tmp_path)
    rc = align.main([str(src), str(words), "--out", str(out)])
    err = capsys.readouterr().err.strip().splitlines()
    assert rc == 3
    assert len(err) == 1
    assert align.HE_MODEL_ID in err[0] and "--allow-download" in err[0] and "GB" in err[0]
    assert not out.exists()


def test_absent_mms_model_exits_3(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a, **k: object())
    monkeypatch.setenv("TORCH_HOME", str(tmp_path / "empty-torch"))
    src, words, out = _tiny_inputs(tmp_path, lang="en")
    words.write_text(json.dumps({"language": "en", "words": [
        {"i": 0, "w": "hello", "start_ms": 0, "end_ms": 400, "p": 0.9}]}), encoding="utf-8")
    rc = align.main([str(src), str(words), "--out", str(out), "--lang", "auto"])
    err = capsys.readouterr().err.strip().splitlines()
    assert rc == 3 and len(err) == 1
    assert "MMS_FA" in err[0] and "--allow-download" in err[0]
    assert not out.exists()


def test_model_cache_detection(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))
    assert not align.hf_model_cached(align.HE_MODEL_ID)
    snap = tmp_path / "hub" / "models--imvladikon--wav2vec2-large-xlsr-53-hebrew" / "snapshots" / "abc"
    snap.mkdir(parents=True)
    (snap / "config.json").write_text("{}")
    assert align.hf_model_cached(align.HE_MODEL_ID)


def test_non_latin_doc_needs_no_model(tmp_path, monkeypatch, capsys):
    # every word unalignable: no package or model check, ASR times kept with flag
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a, **k: None)
    monkeypatch.setattr(align, "load_audio", lambda src: np.zeros(16000, dtype=np.float32))
    src, words, out = _tiny_inputs(tmp_path, lang="ru")
    words.write_text(json.dumps({"language": "ru", "segments": [], "words": [
        {"i": 0, "w": "Привет", "start_ms": 100, "end_ms": 500, "p": 0.9}]}), encoding="utf-8")
    rc = align.main([str(src), str(words), "--out", str(out)])
    assert rc == 0
    doc = json.loads(out.read_text(encoding="utf-8"))
    w = doc["words"][0]
    assert (w["start_ms"], w["end_ms"], w["asr_start_ms"], w["asr_end_ms"]) == (100, 500, 100, 500)
    assert w["flag"] == "unaligned_script"
    assert "segments" not in doc
    assert doc["align_stats"]["aligned"] == 0 and doc["align_stats"]["unaligned"] == 1


def test_auto_lang_without_language_field_fails(tmp_path, capsys):
    src, words, out = _tiny_inputs(tmp_path)
    words.write_text(json.dumps({"words": []}), encoding="utf-8")
    rc = align.main([str(src), str(words), "--out", str(out)])
    assert rc == 2
    assert "--lang" in capsys.readouterr().err
    assert not out.exists()


# ---------- slow: real model on a real clip (controller runs it) ----------

def _slow_ready():
    if os.environ.get("STYLE_LAB_SLOW") != "1":
        return False
    clip, words = os.environ.get("STYLE_LAB_TEST_CLIP"), os.environ.get("STYLE_LAB_TEST_WORDS")
    return bool(clip and words and Path(clip).is_file() and Path(words).is_file())


@pytest.mark.slow
@pytest.mark.skipif(not _slow_ready(), reason="needs STYLE_LAB_SLOW=1, STYLE_LAB_TEST_CLIP, STYLE_LAB_TEST_WORDS")
def test_slow_hebrew_alignment_moves_edges(monkeypatch):
    from conftest import REAL_HOME
    # the isolation fixture moves HOME; point the model cache back at the real one (read only)
    if not (os.environ.get("HF_HUB_CACHE") or os.environ.get("HF_HOME")):
        monkeypatch.setenv("HF_HUB_CACHE", str(REAL_HOME / ".cache" / "huggingface" / "hub"))
    if not align.hf_model_cached(align.HE_MODEL_ID):
        pytest.skip(f"{align.HE_MODEL_ID} is not in the Hugging Face cache")
    doc = json.loads(Path(os.environ["STYLE_LAB_TEST_WORDS"]).read_text(encoding="utf-8"))
    out = align.align(os.environ["STYLE_LAB_TEST_CLIP"], doc, "he")
    al = [w for w in out["words"] if "ctc_start_ms" in w]
    assert al
    shifts = [abs(w["start_ms"] - w["asr_start_ms"]) for w in al] + \
             [abs(w["end_ms"] - w["asr_end_ms"]) for w in al]
    assert statistics.median(shifts) > 20
    assert all(w["start_ms"] < w["end_ms"] for w in al)

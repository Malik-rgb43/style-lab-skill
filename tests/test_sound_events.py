"""sound_events.py: SFX onsets + class guesses, music and voice profile, sync offsets.

All audio is synthetic (tests/fixtures/make_sfx.py), so every onset time, level, BPM and duck depth
is known exactly. Most tests call the pure functions on numpy arrays; the CLI tests decode a small
file through ffmpeg (and run with --no-librosa so a subprocess does not pay librosa's import).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent / "fixtures"))
import make_media as mm  # noqa: E402
import make_sfx as ms  # noqa: E402

import sound_events as se  # noqa: E402
from conftest import run_cli  # noqa: E402

SR = mm.SR
SYNTH = [(1.0, "whoosh"), (2.5, "hit"), (3.2, "click"), (4.0, "riser")]  # riser runs 4.0-5.5 s


def mono(a):
    return a[:, 1].astype(np.float64)


@pytest.fixture
def no_librosa(monkeypatch):
    """librosa missing: its import fails, so the numpy onset envelope is used."""
    monkeypatch.setitem(sys.modules, "librosa", None)
    assert se._get_librosa() is None


def _has_librosa():
    try:
        import librosa  # noqa: F401
        return True
    except Exception:
        return False


# ---------------------------------------------------------------- SFX

def test_classes_on_synthetic(no_librosa):
    y = mono(ms.sfx_audio(6.5, SYNTH))
    found = se.onsets(y, SR)
    for t, _ in SYNTH:
        assert any(abs(o - t) <= 0.03 for o in found), (t, found)
    sfx = se.detect_sfx(y, SR)
    assert len(sfx) == len(SYNTH), [(e["t"], e["class_guess"]) for e in sfx]
    for (t, kind), ev in zip(SYNTH, sfx):
        assert abs(ev["t"] - t) <= 0.03, (kind, ev["t"])
        assert ev["class_guess"] == kind, (kind, ev["class_guess"], ev["features"])
        assert 0.5 <= ev["confidence"] <= 0.95
    riser = sfx[-1]
    assert 1.3 <= riser["dur_s"] <= 1.6  # the riser body is one event, not re-listed per step


def test_classify_api_shape(no_librosa):
    y = mono(ms.sfx_audio(3.0, [(1.0, "hit")]))
    c = se.classify(y, SR, 1.0, voice_db=-12.0)
    assert {"class_guess", "confidence", "dur_s", "level_vs_voice_db"} <= set(c)
    assert c["class_guess"] == "hit" and c["class_guess"] in se.CLASSES
    assert c["level_vs_voice_db"] == pytest.approx(-10.0 - -12.0, abs=1.0)  # hit peak -10 dBFS
    # a stretch of room tone matches no rule: unknown, never a forced label
    room = se.classify(y, SR, 2.2)
    assert room["class_guess"] == "unknown" and room["confidence"] == 0.0


def test_typing_is_one_event(no_librosa):
    y = mono(ms.sfx_audio(3.0, [(1.0, "typing")]))
    sfx = se.detect_sfx(y, SR)
    assert [e["class_guess"] for e in sfx] == ["typing"], [(e["t"], e["class_guess"]) for e in sfx]
    assert abs(sfx[0]["t"] - 1.0) <= 0.03
    assert sfx[0]["count"] == 9 and sfx[0]["rate_hz"] == pytest.approx(9.0, abs=0.3)


def test_voice_onsets_not_sfx(no_librosa):
    spans = [(300, 700), (900, 1400), (1700, 2100), (2600, 3200)]
    y = mono(ms.sfx_audio(5.0, [(4.0, "hit")], voice_spans=spans))
    words = mm.words_doc(spans)
    with_words = se.analyse(y, SR, words)["sfx"]
    assert [(round(e["t"], 1), e["class_guess"]) for e in with_words] == [(4.0, "hit")]
    # without the words the tone words still read as onsets; they are dropped as voice-like
    # (tonal, centred on the formant range, no air), so the sfx list stays the hit alone
    without = se.analyse(y, SR, None)
    assert without["sfx_dropped"]["voice_like"] >= 3
    assert [(round(e["t"], 1), e["class_guess"]) for e in without["sfx"]] == [(4.0, "hit")]
    assert "voice_filter" in without["not_run"]


def test_sfx_over_voice_kept(no_librosa):
    """A thump clearly above a word is still an SFX (marked over_voice, with its signature)."""
    spans = [(500, 2500)]
    y = mono(ms.sfx_audio(3.5, [(1.5, "hit", {"db": -4.0})], voice_spans=spans))
    sfx = se.analyse(y, SR, mm.words_doc(spans))["sfx"]
    assert [(round(e["t"], 1), e["class_guess"], e["over_voice"]) for e in sfx] == [(1.5, "hit", True)]
    assert sfx[0]["non_voice"] == "thump" and sfx[0]["rise_over_speech_db"] >= 6
    assert sfx[0]["level_vs_voice_db"] == pytest.approx(8.0, abs=1.5)  # hit -4 dBFS vs voice -12
    # the same hit at -10 dBFS (only 2 dB over the word) does not stand out of the speech: dropped
    y2 = mono(ms.sfx_audio(3.5, [(1.5, "hit")], voice_spans=spans))
    assert se.analyse(y2, SR, mm.words_doc(spans))["sfx"] == []


def test_speech_plosives_not_sfx(no_librosa):
    """Speech-like words (formant buzz + noise, a 5 ms plosive click at every word onset) and one
    broadband whoosh over a word: the whoosh is the only SFX."""
    spans = ms.speech_words(8.0, seed=3)
    target = next(a for a, b in spans if a > 3000 and b - a > 400)  # a word to put the whoosh on
    t_w = target / 1000 + 0.1
    a = ms.sfx_audio(8.0, [(t_w, "whoosh", {"db": -8.0, "lo": 400, "hi": 12000})],
                     voice_spans=spans, speech=True, word_db=-20, seed=3)
    doc = se.analyse(a, SR, mm.words_doc(spans))
    got = [(e["t"], e["class_guess"], e["over_voice"]) for e in doc["sfx"]]
    assert len(got) == 1, (got, doc["sfx_dropped"])
    assert abs(got[0][0] - t_w) <= 0.05 and got[0][1] == "whoosh" and got[0][2] is True
    assert doc["sfx"][0]["non_voice"] == "air"
    # without the whoosh: nothing (the plosives, syllables and voiced bursts are all speech)
    b = ms.sfx_audio(8.0, [], voice_spans=spans, speech=True, word_db=-20, seed=3)
    assert se.analyse(b, SR, mm.words_doc(spans))["sfx"] == []


def test_over_voice_cap(no_librosa):
    """At most 0.5 over-voice SFX per second of speech, the strongest kept."""
    spans = [(500, 2500)]  # 2 s of speech -> 1 allowed
    ev = [(1.0, "hit", {"db": -6.0}), (1.8, "hit", {"db": 0.0})]
    y = mono(ms.sfx_audio(3.5, ev, voice_spans=spans))
    doc = se.analyse(y, SR, mm.words_doc(spans))
    assert [round(e["t"], 1) for e in doc["sfx"]] == [1.8]
    assert doc["sfx_dropped"]["over_voice_cap"] == 1


# ---------------------------------------------------------------- music

@pytest.mark.parametrize("backend", ["numpy", "librosa"])
def test_bpm_estimate(backend, monkeypatch):
    if backend == "numpy":
        monkeypatch.setitem(sys.modules, "librosa", None)
    elif not _has_librosa():
        pytest.skip("librosa not installed")
    y = mono(ms.sfx_audio(10.0, [], music_bpm=120))
    mp = se.music_profile(y, SR)
    assert 118 <= mp["bpm"] <= 122, mp  # 60 or 240 (half / double) would fail here
    assert mp["bpm_method"].startswith("tempogram")
    assert ("librosa" in mp["bpm_method"]) == (backend == "librosa")
    oc = mp["bpm_octave_check"]  # the half / double decision is recorded
    assert oc["half_bpm"]["bpm"] == pytest.approx(mp["bpm"] / 2, abs=0.1)
    assert oc["double_bpm"]["bpm"] == pytest.approx(mp["bpm"] * 2, abs=0.2)
    assert "prior" in oc["rule"]
    assert mp["present"] is True and mp["bpm_conf"] >= 0.5


@pytest.mark.parametrize("seed", [3, 9])
def test_bpm_under_speech(seed, no_librosa):
    """A 128 BPM kick-and-hat bed under speech-like noise at an equal level reads 126-130."""
    spans = ms.speech_words(20.0, seed=seed)
    a = ms.sfx_audio(20.0, [], voice_spans=spans, speech=True, word_db=-20, seed=seed, music_bpm=128,
                     music_db=-20, music_kw={"hats": True, "pad": False})
    mp = se.music_profile(a, SR, mm.words_doc(spans))
    assert mp["bpm"] is not None and 126 <= mp["bpm"] <= 130, mp["bpm_octave_check"]
    assert mp["bpm_octave_check"]["chosen"] in ("picked", "half_bpm", "double_bpm")
    # speech alone has no tempo
    b = ms.sfx_audio(20.0, [], voice_spans=spans, speech=True, word_db=-20, seed=seed)
    assert se.music_profile(b, SR, mm.words_doc(spans))["bpm"] is None


def test_bpm_not_120_biased(no_librosa):
    y = mono(ms.sfx_audio(10.0, [], music_bpm=90))
    assert 88 <= se.music_profile(y, SR)["bpm"] <= 92


def test_music_beats_not_sfx(no_librosa):
    """The kicks of a music bed are music, not SFX; a hit well above them is still an SFX."""
    y = mono(ms.sfx_audio(10.0, [(5.1, "hit", {"db": -6.0})], music_bpm=120))
    sfx = se.analyse(y, SR)["sfx"]
    assert [(round(e["t"], 1), e["class_guess"]) for e in sfx] == [(5.1, "hit")]


@pytest.mark.parametrize("duck, lo, hi", [(8.0, 6.0, 10.0), (0.0, -1.5, 1.5)])
def test_ducking(duck, lo, hi, no_librosa):
    spans = [(1000, 3000), (4500, 6500), (8000, 9500)]
    y = mono(ms.sfx_audio(11.0, [], voice_spans=spans, music_bpm=120, music_db=-30, duck_db=duck))
    mp = se.music_profile(y, SR, mm.words_doc(spans))
    assert lo <= mp["ducking_db"] <= hi, mp["ducking_db"]
    assert len(mp["energy_per_s"]) == 11


def test_beat_cut_share(no_librosa):
    y = mono(ms.sfx_audio(10.0, [], music_bpm=120))  # beats at 0.25 + k * 0.5
    mp = se.music_profile(y, SR, cuts=[1.25, 2.27, 3.5, 4.2, 6.1])  # 3 on the beat (+-70 ms), 2 off
    assert mp["beat_cut_share"] == pytest.approx(0.6)
    assert se.music_profile(y, SR)["beat_cut_share"] is None


# ---------------------------------------------------------------- voice + loudness

@pytest.mark.parametrize("sr", [48000, se.AR])
def test_lufs_reference_tone(sr):
    """A 997 Hz sine at -20 dBFS RMS in one channel reads -20 LUFS (BS.1770 calibration), at the
    reference rate and at the analysis rate; two equal channels read 3 dB more."""
    t = np.arange(int(3 * sr)) / sr
    y = np.sin(2 * np.pi * 997 * t) * 10 ** (-20 / 20) * np.sqrt(2)
    assert se.lufs(y, sr) == pytest.approx(-20.0, abs=0.2)
    assert se.lufs(np.stack([y, y], axis=1), sr) == pytest.approx(-20.0 + 3.01, abs=0.2)
    assert se.lufs(y[: int(0.3 * sr)], sr) is None  # shorter than one 400 ms block


def test_voice_profile():
    spans = [(500, 900), (950, 1350), (1850, 2250), (2850, 3250), (3650, 4000)]
    a = ms.sfx_audio(4.5, [], voice_spans=spans)
    vp = se.voice_profile(a, SR, mm.words_doc(spans))
    assert vp["words_per_s"] == pytest.approx(5 / 3.5, abs=0.01)
    assert vp["pause_median_s"] == pytest.approx(0.5, abs=0.01)  # gaps 0.05 (not a pause), 0.5, 0.6, 0.4
    assert vp["lufs_scope"] == "words" and vp["lufs"] is not None
    nv = se.voice_profile(a, SR, None)
    assert nv["words_per_s"] is None and nv["lufs_scope"] == "mix"


# ---------------------------------------------------------------- coupling

def test_sync_offsets_pure():
    cuts = [1.0, 2.2, 3.6, 4.8]
    off = se.sync_offsets([c - 2 / 25 for c in cuts], cuts)
    assert off["n"] == 4 and off["median"] == pytest.approx(-0.08, abs=0.001)
    assert se.sync_offsets([], cuts) == {"median": None, "p10": None, "p90": None, "n": 0}
    assert se.sync_offsets([1.0], [])["n"] == 0
    assert se.coupled_share(cuts, [0.92, 2.12]) == 0.5 and se.coupled_share([], [1.0]) is None


def test_move_onsets_reader():
    doc = {"moves": [{"t0": 1.0}, {"t0": 1.0005}, {"kind": "x"}],
           "elements": [{"id": "a", "moves": [{"t0": 2.5, "t1": 3.0}, {"t1": 4.0}]},
                        {"id": "b", "moves": [{"t0": True}]}]}
    assert se.move_onsets(doc) == [1.0, 2.5]


def test_sync_offsets(tmp_path):
    """CLI: SFX placed 2 frames (25 fps) before the cuts give a median offset of -0.08 s."""
    cuts = [1.0, 2.2, 3.6, 4.8]
    src = ms.make_sfx_track(tmp_path / "ref ממדים.wav", 6.0, [(c - 0.08, "hit") for c in cuts])
    meas = mm.write_json(tmp_path / "measure.json", {"cuts": [{"t": c, "kind": "hard", "dur_f": 0} for c in cuts]})
    motion = mm.write_json(tmp_path / "motion.json", {"elements": [{"moves": [{"t0": 2.12}, {"t0": 5.5}]}]})
    out = tmp_path / "analysis" / "sound.json"
    r = run_cli("sound_events.py", src, "--out", out, "--measure", meas, "--motion", motion, "--no-librosa")
    assert r.returncode == 0, r.stderr
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert doc["schema"] == "sound/1" and doc["labels_are_guesses"] is True
    assert len(doc["source"]["sha256"]) == 64
    off = doc["coupling"]["offsets"]["sfx_vs_cuts"]
    assert off["n"] == 4 and off["median"] == pytest.approx(-0.08, abs=0.02)
    assert doc["coupling"]["cut_with_sfx_share"] == 1.0
    assert doc["coupling"]["move_with_sfx_share"] == 0.5
    assert doc["coupling"]["offsets"]["sfx_vs_moves"]["n"] == 1
    assert [e["class_guess"] for e in doc["sfx"]] == ["hit"] * 4
    assert "coupling_cuts" not in doc["not_run"] and "voice_filter" in doc["not_run"]
    assert "song" not in json.dumps(doc).lower()


# ---------------------------------------------------------------- edge cases

def test_no_audio_stream(tmp_path):
    src = ms.make_silent_video(tmp_path / "silent.mkv", 2.0)
    out = tmp_path / "sound.json"
    r = run_cli("sound_events.py", src, "--out", out, "--no-librosa")
    assert r.returncode == 0, r.stderr
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert doc["schema"] == "sound/1" and "sound" in doc["not_run"]
    assert doc["sfx"] == [] and doc["music"]["bpm"] is None and doc["source"]["has_audio"] is False
    assert doc["coupling"]["offsets"]["sfx_vs_cuts"]["n"] == 0
    assert doc["source"]["duration_s"] == pytest.approx(2.0, abs=0.1)


@pytest.mark.parametrize("kind", ["room", "digital_silence", "tiny"])
def test_static_or_short_audio(kind, no_librosa):
    """Room tone only, digital silence or 30 ms: a valid body, empty lists, n=0, no crash."""
    if kind == "room":
        a = ms.sfx_audio(6.0, [])
    elif kind == "digital_silence":
        a = np.zeros((6 * SR, 2), np.float32)
    else:
        a = np.zeros((int(0.03 * SR), 2), np.float32)
    doc = se.analyse(a, SR, None, cuts=[1.0, 2.0], moves=[])
    assert doc["sfx"] == [] and doc["music"]["bpm"] is None
    assert doc["coupling"]["cut_with_sfx_share"] == 0.0
    assert doc["coupling"]["move_with_sfx_share"] is None
    assert all(v["n"] == 0 for v in doc["coupling"]["offsets"].values())
    json.dumps(doc)  # serialisable: plain floats, no numpy types, no NaN


def test_stems_never_download(monkeypatch, tmp_path):
    """Demucs runs only when its weights are cached (or --allow-download): no subprocess otherwise."""
    calls = []
    monkeypatch.setattr(se.subprocess, "run", lambda *a, **k: calls.append(a) or None)
    monkeypatch.setattr(se, "demucs_status", lambda model=se.DEMUCS_MODEL: ("not_cached", "weights not cached"))
    stem, kind, why = se.separate_no_vocals(np.zeros((SR, 2), np.float32), SR)
    assert (stem, kind, calls) == (None, "none", []) and "cached" in why
    monkeypatch.setattr(se, "demucs_status", lambda model=se.DEMUCS_MODEL: ("missing", "demucs is not installed"))
    assert se.separate_no_vocals(np.zeros((SR, 2), np.float32), SR, allow_download=True)[1] == "none"
    assert calls == []


def test_demucs_cache_check(monkeypatch, tmp_path):
    import importlib.util
    if importlib.util.find_spec("demucs") is None:
        assert se.demucs_status()[0] == "missing"
        return
    monkeypatch.setenv("TORCH_HOME", str(tmp_path / "torch"))
    assert se.demucs_status()[0] == "not_cached"
    ck = tmp_path / "torch" / "hub" / "checkpoints"
    ck.mkdir(parents=True)
    yml = Path(list(importlib.util.find_spec("demucs").submodule_search_locations)[0]) / "remote" / "htdemucs.yaml"
    import re
    for sig in re.findall(r"\b([0-9a-f]{8})\b", yml.read_text(encoding="utf-8")):
        (ck / f"{sig}-00000000.th").write_bytes(b"")
    assert se.demucs_status()[0] == "cached"


def test_stem_is_used(no_librosa):
    """With a non-vocal stem, SFX and tempo come from it (stems recorded), voice from the mix."""
    spans = [(500, 2500)]
    mix = mono(ms.sfx_audio(3.5, [(1.5, "hit", {"db": -10.0})], voice_spans=spans))
    stem = mono(ms.sfx_audio(3.5, [(1.5, "hit", {"db": -10.0})]))
    doc = se.analyse(mix, SR, mm.words_doc(spans), stem=stem, stems="demucs")
    assert doc["stems"] == "demucs"
    assert [e["class_guess"] for e in doc["sfx"]] == ["hit"]  # under the word in the mix, clear in the stem
    assert doc["voice"]["lufs_scope"] == "words"


def test_coupling_skips_unknown(monkeypatch, no_librosa):
    fake = [{"t": 1.0, "class_guess": "unknown"}, {"t": 2.0, "class_guess": "hit"}]
    monkeypatch.setattr(se, "detect_sfx", lambda *a, **k: [dict(e) for e in fake])
    doc = se.analyse(np.zeros(5 * SR), SR, None, cuts=[1.0, 2.0])
    assert doc["coupling"]["cut_with_sfx_share"] == 0.5
    assert doc["coupling"]["offsets"]["sfx_vs_cuts"]["n"] == 1


def test_clean_exits(monkeypatch, tmp_path, capsys):
    """Write failure -> 2, any other error -> 1, one line and never a traceback; odd byte counts trimmed."""
    src = ms.make_sfx_track(tmp_path / "a.wav", 1.0, [])

    def boom(*a, **k):
        raise ValueError("synthetic failure")
    monkeypatch.setattr(se, "run", boom)
    assert se.main([str(src), "--out", str(tmp_path / "s.json"), "--no-librosa"]) == 1
    err = capsys.readouterr().err
    assert "synthetic failure" in err and "Traceback" not in err and len(err.strip().splitlines()) == 1

    def oserr(*a, **k):
        raise PermissionError("denied")
    monkeypatch.setattr(se, "run", oserr)
    assert se.main([str(src), "--out", str(tmp_path / "s.json"), "--no-librosa"]) == 2

    class R:
        returncode, stdout, stderr = 0, b"\x01\x00\x02\x00\x03\x00\x04\x00\x05\x00", b""
    monkeypatch.setattr(se.subprocess, "run", lambda *a, **k: R())
    assert se.load_audio("x").shape == (2, 2)


def test_bad_input(tmp_path):
    r = run_cli("sound_events.py", tmp_path / "missing.wav", "--out", tmp_path / "s.json", "--no-librosa")
    assert r.returncode == 2 and "Traceback" not in r.stderr
    src = ms.make_sfx_track(tmp_path / "a.wav", 1.0, [])
    bad = tmp_path / "w.json"
    bad.write_text("{nope", encoding="utf-8")
    r = run_cli("sound_events.py", src, "--out", tmp_path / "s.json", "--words", bad, "--no-librosa")
    assert r.returncode == 2 and "Traceback" not in r.stderr
    assert not (tmp_path / "s.json").exists()
    with pytest.raises(se.BadInput):
        se.word_spans({"words": [{"start": "x", "end": 1.0}]})
    with pytest.raises(se.BadInput):
        se.cut_times({"cuts": "nope"})


# ---------------------------------------------------------------- timbre + sections

def _bursts(sr, n=4, on=0.3, gap=1.0, seed=0):
    rng = np.random.default_rng(seed)
    y = np.zeros(int((n * (on + gap) + 0.5) * sr))
    spans = []
    for i in range(n):
        a = 0.3 + i * (on + gap)
        i0 = int(a * sr)
        y[i0:i0 + int(on * sr)] = rng.normal(0, 0.1, int(on * sr))
        spans.append((a, a + on))
    return y, spans


def test_voice_compression_and_room():
    sr = se.AR
    dry, spans = _bursts(sr)
    t = np.arange(int(1.0 * sr)) / sr
    ir = np.random.default_rng(1).normal(0, 1, len(t)) * np.exp(-6.91 * t / 0.4)
    ir[0] = 1.0
    wet = np.convolve(dry, ir)[: len(dry)]
    wet *= np.abs(dry).max() / np.abs(wet).max()
    r_dry = se.timbre(dry, sr, spans)["rt60_est_s"]
    r_wet = se.timbre(wet, sr, spans)["rt60_est_s"]
    assert r_dry is not None and r_wet is not None and r_wet - r_dry > 0.2
    assert r_wet == pytest.approx(0.4, abs=0.15)
    limited = np.clip(dry, -0.1, 0.1)
    assert se.timbre(limited, sr, spans)["crest_db"] < se.timbre(dry, sr, spans)["crest_db"] - 3
    # white noise is flat per Hz, so it rises ~3 dB per octave; de-ess reads the 5-9 kHz share
    tm = se.timbre(dry, sr, spans)
    assert tm["spectral_tilt_db_per_oct"] == pytest.approx(3.0, abs=1.0)
    assert tm["de_ess_db"] is not None and tm["timbre_scope"] == "words"
    vp = se.voice_profile(dry, sr, spans)
    assert {"spectral_tilt_db_per_oct", "crest_db", "rt60_est_s", "de_ess_db"} <= set(vp)


def test_timbre_silent_and_short():
    sr = se.AR
    out = se.timbre(np.zeros(sr), sr, [(0.1, 0.4)])
    assert out["crest_db"] is None and out["rt60_est_s"] is None
    assert se.timbre(np.zeros(100), sr, None)["spectral_tilt_db_per_oct"] is None


def test_music_sections_split_at_change():
    sr = se.AR
    t = np.arange(int(10 * sr)) / sr
    low = 0.2 * np.sin(2 * np.pi * 110 * t) * (1 + 0.5 * np.sin(2 * np.pi * 2 * t))
    high = 0.05 * np.random.default_rng(2).normal(0, 1, len(t))
    secs = se.music_sections(np.concatenate([low, high]), sr)
    assert len(secs) == 2 and secs[0]["t1"] == pytest.approx(10.0, abs=1.01)
    assert secs[0]["t0"] == 0.0 and secs[-1]["t1"] == pytest.approx(20.0, abs=0.05)
    one = se.music_sections(low, sr)
    assert len(one) == 1


def test_music_sections_energy_is_dbfs():
    sr = se.AR
    t = np.arange(int(6 * sr)) / sr
    y = np.sin(2 * np.pi * 440 * t) * 10 ** (-20 / 20) * np.sqrt(2)  # -20 dBFS RMS
    assert se.music_sections(y, sr)[0]["energy"] == pytest.approx(-20.0, abs=0.5)

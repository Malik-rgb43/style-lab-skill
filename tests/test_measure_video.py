import json
import sys

import numpy as np
import pytest

import measure_video as mv
from conftest import run_cli
from fixtures import make_media as mm

BOX = [(0.0, 4.0, "x", 0.2, 0.8, "linear")]


def _clip(tmp_path, name="c.mp4", **kw):
    return mm.make_motion_clip(tmp_path / name, **kw)


def _doc(out):
    return json.loads((out / "measure.json").read_text(encoding="utf-8"))


def test_hard_cuts_found(tmp_path):
    p = _clip(tmp_path, dur_s=4, fps=25, moves=BOX, cuts=[1.0, 2.5])
    frames, fps = mv.decode_gray(p, 320)
    cuts = mv.detect_cuts(frames, fps)
    assert fps == 25 and len(cuts) == 2
    assert all(c["kind"] == "hard" for c in cuts)
    for c, want in zip(cuts, (1.0, 2.5)):
        assert abs(c["t"] - want) <= 1 / 25 + 1e-6


def test_gradual_transition_is_one_cut(tmp_path):
    p = _clip(tmp_path, dur_s=4, fps=25, moves=BOX, cuts=[(1.5, 12)])
    frames, fps = mv.decode_gray(p, 320)
    cuts = mv.detect_cuts(frames, fps)
    assert len(cuts) == 1 and cuts[0]["kind"] == "gradual"
    assert 10 <= cuts[0]["dur_f"] <= 14


def test_cadence_on_twos(tmp_path):
    p = _clip(tmp_path, dur_s=3, fps=30, moves=[(0.0, 3.0, "x", 0.2, 0.8, "linear")], stepped=2)
    frames, fps = mv.decode_gray(p, 320)
    c = mv.cadence(frames, fps)
    assert c["on_twos"] is True
    assert 14 <= c["unique_fps"] <= 16


def test_cadence_full_rate_not_on_twos(tmp_path):
    p = _clip(tmp_path, dur_s=3, fps=30, moves=[(0.0, 3.0, "x", 0.2, 0.8, "linear")])
    frames, fps = mv.decode_gray(p, 320)
    c = mv.cadence(frames, fps)
    assert c["on_twos"] is False and c["unique_fps"] > 28


def test_static_clip_valid(tmp_path):
    p = _clip(tmp_path, dur_s=6, fps=25)
    out = tmp_path / "out"
    r = run_cli("measure_video.py", p, "--out", out)
    assert r.returncode == 0, r.stderr
    d = _doc(out)
    assert mv.validate(d) == []
    assert d["cuts"] == [] and len(d["shots"]) == 1
    assert d["longest_hold_s"] >= 5.9
    assert d["cadence"]["on_twos"] is False and d["cadence"]["unique_fps"] == 25
    assert d["text"]["size_pct_h"]["n"] == 0 and d["visual_event_rate"] == [0] * 6
    assert (out / "frames.csv").read_text(encoding="utf-8").startswith("frame,t_s")


def test_inputs_untouched_and_hashed(tmp_path):
    p = _clip(tmp_path, dur_s=2, fps=25, moves=BOX)
    before = mv.sha256_of(p)
    out = tmp_path / "out"
    assert run_cli("measure_video.py", p, "--out", out).returncode == 0
    assert mv.sha256_of(p) == before
    assert _doc(out)["video"]["sha256"] == before


def test_missing_ffmpeg_exit3(tmp_path, monkeypatch):
    empty = tmp_path / "nobin"
    empty.mkdir()
    monkeypatch.setenv("PATH", str(empty))
    r = run_cli("measure_video.py", tmp_path / "x.mp4", "--out", tmp_path / "o")
    assert r.returncode == 3
    assert "Traceback" not in r.stderr
    assert len([x for x in r.stderr.splitlines() if x.strip()]) == 1 and "ffmpeg" in r.stderr


def test_bad_input_exit2(tmp_path):
    r = run_cli("measure_video.py", tmp_path / "nope.mp4", "--out", tmp_path / "o")
    assert r.returncode == 2 and "Traceback" not in r.stderr


def test_from_analysis_import(tmp_path):
    p = _clip(tmp_path, dur_s=3, fps=25, moves=BOX)
    adir = tmp_path / "va"
    adir.mkdir()
    (adir / "measurements.json").write_text(json.dumps(
        {"edit_points": [{"t_s": 1.0, "kind": "cut", "type": "hard"},
                         {"t_s": 2.0, "kind": "cut", "type": "dissolve"}],
         "pacing": {"cuts_per_min": 40}}), encoding="utf-8")
    (adir / "audio.json").write_text(json.dumps({"loudness": {"integrated_lufs": -16.2, "true_peak": -1.1}}),
                                     encoding="utf-8")
    (adir / "transcript.json").write_text(json.dumps(
        {"words": [{"start": 0.1, "end": 0.4, "word": "a"}, {"start": 0.5, "end": 0.9, "word": "b"}]}),
        encoding="utf-8")
    out = tmp_path / "out"
    r = run_cli("measure_video.py", p, "--out", out, "--from-analysis", adir)
    assert r.returncode == 0, r.stderr
    d = _doc(out)
    assert [c["t"] for c in d["cuts"]] == [1.0, 2.0] and d["cuts"][1]["kind"] == "gradual"
    assert "cuts" not in d["not_run"]
    assert d["imported"] == {"cuts": 2, "pacing": True, "loudness": True, "words": 2}
    assert d["loudness"]["lufs"] == -16.2 and d["pacing"] == {"cuts_per_min": 40}
    assert len(d["shots"]) == 3 and abs(d["words_per_s"] - 2 / 3) < 0.01


def test_from_analysis_partial(tmp_path):
    p = _clip(tmp_path, dur_s=2, fps=25, moves=BOX, cuts=[1.0])
    adir = tmp_path / "va"
    adir.mkdir()
    out = tmp_path / "out"
    assert run_cli("measure_video.py", p, "--out", out, "--from-analysis", adir).returncode == 0
    d = _doc(out)
    assert d["imported"] == {} and len(d["cuts"]) == 1  # nothing to import: own detection


def test_words_file(tmp_path):
    p = _clip(tmp_path, dur_s=2, fps=25, moves=BOX)
    w = tmp_path / "w.json"
    w.write_text(json.dumps({"words": [{"start_ms": i * 100, "end_ms": i * 100 + 50} for i in range(10)]}))
    out = tmp_path / "out"
    assert run_cli("measure_video.py", p, "--out", out, "--words", w).returncode == 0
    assert _doc(out)["words_per_s"] == 5.0


def test_no_opencv_faces_not_run(tmp_path, monkeypatch):
    p = _clip(tmp_path, dur_s=2, fps=25, moves=BOX, cuts=[1.0])
    monkeypatch.setitem(sys.modules, "cv2", None)
    out = tmp_path / "out"
    assert mv.main([str(p), "--out", str(out)]) == 0
    d = _doc(out)
    assert "faces" in d["not_run"] and "text" in d["not_run"]
    assert len(d["cuts"]) == 1 and mv.validate(d) == []


def test_ocr_missing_is_not_run(tmp_path, monkeypatch):
    p = _clip(tmp_path, dur_s=2, fps=25, moves=BOX)
    monkeypatch.setattr(mv, "_ocr_engine", lambda: None)
    out = tmp_path / "out"
    assert mv.main([str(p), "--out", str(out), "--text", "ocr"]) == 0
    d = _doc(out)
    assert d["text"]["ocr"] == "not_run" and "ocr" in d["not_run"]


def test_shot_palette_oklab():
    img = np.zeros((10, 10, 3), np.uint8)
    img[:, :7] = (200, 30, 30)
    img[:, 7:] = (20, 20, 120)
    pal = mv.shot_palette(img, k=2)
    assert len(pal) == 2 and abs(pal[0]["share"] - 0.7) < 0.01 and abs(sum(e["share"] for e in pal) - 1) < 0.01
    assert pal[0]["hex"].upper().startswith("#C")


def test_text_boxes_finds_a_line_of_text():
    cv2 = pytest.importorskip("cv2")
    img = np.full((180, 320), 30, np.uint8)
    cv2.putText(img, "HELLO WORLD", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.4, 255, 3)
    boxes = mv.text_boxes(img)
    assert boxes and all(b[2] * 320 / (b[3] * 180) >= 2 for b in boxes)


def test_face_track_empty_on_synthetic(tmp_path):
    pytest.importorskip("cv2")
    if mv._haar() is None:
        pytest.skip("this cv2 build ships no Haar cascade")
    p = _clip(tmp_path, dur_s=1, fps=25, moves=BOX)
    frames, fps = mv.decode_gray(p, 320)
    assert mv.face_track(frames, fps) == []


def test_one_frame_clip_does_not_crash(tmp_path):
    p = _clip(tmp_path, dur_s=1 / 25, fps=25)
    out = tmp_path / "out"
    assert run_cli("measure_video.py", p, "--out", out).returncode == 0
    assert mv.validate(_doc(out)) == []


def test_non_ascii_paths(tmp_path):
    d = tmp_path / "תיקייה"
    d.mkdir()
    p = _clip(d, name="וידאו.mp4", dur_s=1, fps=25, moves=BOX)
    out = d / "פלט"
    assert run_cli("measure_video.py", p, "--out", out).returncode == 0
    assert (out / "measure.json").exists()


def test_missing_haar_cascade_faces_not_run(tmp_path, monkeypatch):
    p = _clip(tmp_path, dur_s=1, fps=25, moves=BOX)
    monkeypatch.setattr(mv, "_haar", lambda: None)
    out = tmp_path / "out"
    assert mv.main([str(p), "--out", str(out)]) == 0
    assert "faces" in _doc(out)["not_run"]

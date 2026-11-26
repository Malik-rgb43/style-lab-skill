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
    assert d["cadence"]["on_twos"] is False and d["cadence"]["unique_fps"] is None
    assert d["cadence"]["moving_frames"] == 0 and d["notes"] == [] and d["video"]["vfr"] is False
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
    monkeypatch.setattr(mv, "_ocr_available", lambda: False)
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


JUMP = [(0.0, 1.0, "x", 0.2, 0.4, "linear"), (1.0, 1.0, "x", 0.4, 0.8, "linear"),
        (1.0, 1.0, "scale", 1.0, 1.5, "linear")]


def test_jump_cut_found_inside_one_shot(tmp_path):
    p = _clip(tmp_path, dur_s=3, fps=25, moves=JUMP)
    frames, fps = mv.decode_gray(p, 320)
    cuts = mv.detect_cuts(frames, fps)
    assert [c["kind"] for c in cuts] == ["jump"]
    assert abs(cuts[0]["t"] - 1.0) <= 1 / 25 + 1e-6
    out = tmp_path / "out"
    assert mv.main([str(p), "--out", str(out)]) == 0
    d = _doc(out)
    assert len(d["shots"]) == 1 and d["cuts"][0]["kind"] == "jump" and mv.validate(d) == []


def test_smooth_motion_has_no_cuts(tmp_path):
    p = _clip(tmp_path, dur_s=4, fps=25, moves=[(0.0, 4.0, "x", 0.1, 0.9, "back.out"),
                                                 (0.5, 3.0, "scale", 1.0, 2.0, "power3.inout"),
                                                 (0.0, 4.0, "y", 0.3, 0.7, "power2.out")])
    frames, fps = mv.decode_gray(p, 320)
    assert mv.detect_cuts(frames, fps) == []


def test_cadence_16fps_in_30fps_file(tmp_path):
    src = _clip(tmp_path, "src16.mp4", dur_s=3, fps=16, moves=[(0.0, 3.0, "x", 0.2, 0.8, "linear")])
    dst = tmp_path / "v30.mp4"
    subprocess_run = __import__("subprocess").run
    subprocess_run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-r", "30", "-c:v", "libx264",
                    "-qp", "0", "-pix_fmt", "yuv420p", str(dst)], check=True)
    frames, fps = mv.decode_gray(dst, 320)
    assert fps == 30
    c = mv.cadence(frames, fps)
    assert 15 <= c["unique_fps"] <= 17


def test_probe_prefers_avg_and_clamps(monkeypatch):
    class R:
        returncode, stderr = 0, ""

        def __init__(self, avg, r):
            self.stdout = json.dumps({"streams": [{"codec_type": "video", "width": 320, "height": 180,
                                                    "avg_frame_rate": avg, "r_frame_rate": r}],
                                      "format": {"duration": "2"}})
    monkeypatch.setattr(mv.subprocess, "run", lambda *a, **k: R("25/1", "50/1"))
    i = mv.probe("x")
    assert i["fps"] == 25 and i["vfr"] is True and i["notes"] == []
    monkeypatch.setattr(mv.subprocess, "run", lambda *a, **k: R("0/0", "24000/1001"))
    assert abs(mv.probe("x")["fps"] - 23.976) < 0.01
    monkeypatch.setattr(mv.subprocess, "run", lambda *a, **k: R("90000/1", "90000/1"))
    i = mv.probe("x")
    assert i["fps"] == 30 and i["notes"] == ["vfr_clamped"]


def test_memory_error_is_one_line_exit1(tmp_path, monkeypatch, capsys):
    p = _clip(tmp_path, dur_s=1, fps=25)
    monkeypatch.setattr(mv, "measure", lambda *a, **k: (_ for _ in ()).throw(MemoryError()))
    assert mv.main([str(p), "--out", str(tmp_path / "o")]) == 1
    err = capsys.readouterr().err
    assert err.strip() == "error: too long for memory: use --analysis-width 160 or less"


def test_decode_streams_every_frame(tmp_path):
    p = _clip(tmp_path, dur_s=2, fps=25, moves=BOX)
    frames, fps = mv.decode_gray(p, 160)
    assert frames.shape == (50, 90, 160)


def test_text_file_renamed_mp4_exit2(tmp_path):
    f = tmp_path / "fake.mp4"
    f.write_text("this is not a video", encoding="utf-8")
    r = run_cli("measure_video.py", f, "--out", tmp_path / "o")
    assert r.returncode == 2 and "Traceback" not in r.stderr
    assert len([x for x in r.stderr.splitlines() if x.strip()]) == 1
    assert not (tmp_path / "o" / "measure.json").exists()


def test_face_branch_with_fake_detector(tmp_path, monkeypatch):
    p = _clip(tmp_path, dur_s=2, fps=25, moves=BOX)
    monkeypatch.setattr(mv, "_face_detector", lambda: (lambda gray: [(100, 18, 60, 60), (5, 5, 10, 10)]))
    out = tmp_path / "out"
    assert mv.main([str(p), "--out", str(out)]) == 0
    d = _doc(out)
    assert "faces" not in d["not_run"]
    assert d["faces_summary"]["share"] == 1.0 and d["faces_summary"]["headroom"]["median"] == 0.1
    assert d["shots"][0]["face_share"] == 1.0 and d["shots"][0]["headroom"] == 0.1


def test_no_face_detector_is_not_run(tmp_path, monkeypatch):
    p = _clip(tmp_path, dur_s=1, fps=25, moves=BOX)
    monkeypatch.setattr(mv, "_haar", lambda: None)
    monkeypatch.setattr(mv, "_mediapipe_detector", lambda: None)
    out = tmp_path / "out"
    assert mv.main([str(p), "--out", str(out)]) == 0
    assert "faces" in _doc(out)["not_run"]


def test_ocr_available_but_nothing_found_is_empty_list(tmp_path, monkeypatch):
    p = _clip(tmp_path, dur_s=1, fps=25, moves=BOX)
    monkeypatch.setattr(mv, "_ocr_available", lambda: True)
    monkeypatch.setattr(mv, "_ocr_engine", lambda: ("fake", None))
    monkeypatch.setattr(mv, "_run_ocr", lambda *a, **k: [])
    out = tmp_path / "out"
    assert mv.main([str(p), "--out", str(out), "--text", "ocr"]) == 0
    d = _doc(out)
    assert d["text"]["ocr"] == [] and "ocr" not in d["not_run"]


def test_event_rate_counts_a_hard_cut_once(tmp_path):
    p = _clip(tmp_path, dur_s=2, fps=25, cuts=[1.0])
    out = tmp_path / "out"
    assert mv.main([str(p), "--out", str(out), "--faces", "off", "--text", "off"]) == 0
    assert _doc(out)["visual_event_rate"] == [0, 1]


def test_bad_word_values_are_skipped(tmp_path):
    p = _clip(tmp_path, dur_s=2, fps=25, moves=BOX)
    w = tmp_path / "w.json"
    w.write_text(json.dumps({"words": [{"start": "x", "end": 1}, {"start": None}, {"start": 0.1, "end": "?"},
                                       {"start_ms": "bad"}, "junk", {"start": 0.5, "end": 0.9}]}))
    out = tmp_path / "out"
    assert run_cli("measure_video.py", p, "--out", out, "--words", w).returncode == 0
    assert _doc(out)["words_per_s"] == 1.0



def test_text_box_records_big_bar():
    pytest.importorskip("cv2")
    img = np.full((180, 320), 30, np.uint8)
    img[75:105, 40:280] = 255
    for x in range(40, 280):
        if x % 8 < 3:
            img[75:105, x] = 30  # slits make it text-like (strokes), not one flat block
    recs = mv.text_box_records(img, 1.5)
    assert len(recs) == 1 and recs[0]["t"] == 1.5
    assert abs(recs[0]["h_pct"] - 30 / 180 * 100) < 2.5 and recs[0]["h_pct"] >= 8
    assert abs(recs[0]["box"][3] * 100 - recs[0]["h_pct"]) < 0.01


def test_measure_json_has_text_boxes_list(tmp_path):
    p = _clip(tmp_path, dur_s=1, fps=25, moves=BOX)
    out = tmp_path / "out"
    assert mv.main([str(p), "--out", str(out), "--faces", "off"]) == 0
    d = _doc(out)
    assert d["text"]["boxes"] == [] and mv.validate(d) == []

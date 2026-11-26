import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from conftest import run_cli
import cover
import fingerprint
import measure_video as mv
import palette

sys.path.insert(0, str(Path(__file__).parent / "fixtures"))


def gray_of(rgb):
    return (rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)).astype(np.uint8)


def frame(hexc, seed=1, h=90, w=160):
    """A textured frame around one colour (the texture keeps it sharp and contrasty)."""
    rng = np.random.default_rng(seed)
    base = np.array([int(hexc[i:i + 2], 16) for i in (1, 3, 5)], dtype=np.float32)
    checker = ((np.indices((h, w)).sum(axis=0) // 8) % 2)[..., None] * 70.0
    img = base + checker - 35 + rng.normal(0, 3, (h, w, 3))
    return np.clip(img, 0, 255).astype(np.uint8)


def blur(rgb, k=5, axes=(0, 1)):
    x = rgb.astype(np.float32)
    for ax in axes:
        x = sum(np.roll(x, s, axis=ax) for s in range(-(k // 2), k // 2 + 1)) / k
    return np.clip(x, 0, 255).astype(np.uint8)


def wj(p, data):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data), encoding="utf-8")


def measure_doc(src=None, cuts=(), hexc="#5b3fd0", face=0.9, text_h=6.0, dur=3.0):
    """A measure.json as fingerprint.py reads it (same shape as tests/test_fingerprint.py)."""
    return {"schema": "measure/1", "source": src,
            "video": {"id": "x", "sha256": "%064x" % (abs(hash((src, hexc))) + 1), "duration_s": dur, "fps": 25,
                      "width": 160, "height": 90},
            "cuts": list(cuts),
            "shots": [{"t0": 0, "t1": dur, "dur_s": dur, "palette": [{"hex": hexc, "share": 1}],
                       "face_share": face, "headroom": 0.1}],
            "cadence": {"repeat_fraction": 0, "unique_fps": 25, "on_twos": False},
            "text": {"size_pct_h": {"median": text_h, "p10": text_h, "p90": text_h, "n": 1}},
            "loudness": {"lufs": -16.0, "tp": -1.0}, "words_per_s": 2.0, "longest_hold_s": 1.0, "not_run": []}


def real_fp(tmp_path, hexc="#5b3fd0", face=0.9, text_h=6.0, ids=("a", "b")):
    """A fingerprint built by the real fingerprint.py from synthetic analysis files."""
    pack = tmp_path / "fppack"
    for vid in ids:
        wj(pack / "analysis" / vid / "measure.json", measure_doc(None, hexc=hexc, face=face, text_h=text_h))
        # make the two videos different files so dedupe keeps both
        d = json.loads((pack / "analysis" / vid / "measure.json").read_text(encoding="utf-8"))
        d["video"]["sha256"] = ("%064x" % (ord(vid[0]) + 1))
        wj(pack / "analysis" / vid / "measure.json", d)
    return fingerprint.build(pack)


def test_score_prefers_on_palette_sharp_frame(tmp_path):
    fp = real_fp(tmp_path)
    assert fp["layers"]["colour"]["metrics"]["dominant_L"]["median"] is not None
    on, off = frame("#5b3fd0"), frame("#e8d020")
    s_on, why_on = cover.score_frame(on, gray_of(on), fp)
    s_off, _ = cover.score_frame(off, gray_of(off), fp)
    assert s_on > s_off and "palette" in why_on
    bl = blur(on, 5)
    s_bl, why = cover.score_frame(bl, gray_of(bl), fp)
    assert "rejected" not in why and why["sharpness"] < why_on["sharpness"]
    assert 0 <= s_on <= 1


def test_dominant_colour_not_mean(tmp_path):
    """A frame that is mostly violet with a bright strip: the dominant colour is violet, the mean is not."""
    fp = real_fp(tmp_path)
    f = frame("#5b3fd0")
    f[:, :40] = np.array([250, 250, 40], dtype=np.uint8)
    assert fp and cover._palette_closeness(f, fp) > cover._palette_closeness(frame("#e8d020"), fp)
    assert cover._palette_closeness(f, {"layers": {}}) is None


def test_score_uses_real_keys_text_face_look(tmp_path):
    fp = real_fp(tmp_path)
    fp["layers"]["look"] = {"metrics": {"grain_std": {"median": 0.02}, "tone_contrast": {"median": 0.3}}}
    f = frame("#5b3fd0")
    boxes_ok = [[0.1, 0.8, 0.5, 0.06]]  # 6 % tall = the style's text height
    full, why = cover.score_frame(f, gray_of(f), fp, faces=True, text=boxes_ok)
    assert {"palette", "look", "text", "face", "sharpness"} <= set(why)
    no_face, _ = cover.score_frame(f, gray_of(f), fp, faces=False, text=boxes_ok)
    no_text, w2 = cover.score_frame(f, gray_of(f), fp, faces=True, text=[])
    assert full > no_face and full > no_text and w2["text"] < why["text"]
    # unknown detectors (None) are left out, not penalised; no look layer -> no look part
    plain = real_fp(tmp_path)
    _, w3 = cover.score_frame(f, gray_of(f), plain)
    assert "face" not in w3 and "text" not in w3 and "look" not in w3
    # a style without text never rewards or punishes it
    notext = real_fp(tmp_path)
    notext["layers"]["type"]["metrics"] = {"text_height_pct": {"median": None}, "big_text_per_min": {"median": 0}}
    assert "text" not in cover.score_frame(f, gray_of(f), notext, text=[])[1]


def test_hard_rejects():
    dark = np.full((90, 160), 5, np.uint8)
    white = np.full((90, 160), 252, np.uint8)
    flat = np.full((90, 160), 120, np.uint8)
    assert cover.reject_reason(dark) == "too dark" and cover.reject_reason(white) == "too bright"
    assert cover.reject_reason(flat).startswith("flat")
    good = gray_of(frame("#5b3fd0"))
    assert cover.reject_reason(good, 0.5) is None
    noise = np.clip(128 + np.random.default_rng(3).normal(0, 90, (90, 160, 3)), 0, 255).astype(np.uint8)
    smear = blur(noise, 21, axes=(1,))  # horizontal motion smear: vertical detail survives
    smeared = gray_of(smear)
    assert cover.directional_ratio(smeared) > 3 and cover.directional_ratio(gray_of(noise)) < 1.5
    raw = cover.sharpness(smeared)
    # blur needs percentile < 0.3 AND raw < half the video median AND a one-sided ratio
    assert cover.reject_reason(smeared, 0.1, raw, raw * 3) == "directional motion blur"
    assert cover.reject_reason(smeared, 0.1, raw, raw * 1.5) is None   # not much softer than its video
    assert cover.reject_reason(smeared, 0.5, raw, raw * 3) is None     # sharp-ranked
    assert cover.reject_reason(smeared, 0.1) is None                   # no video statistics: no blur verdict
    s, why = cover.score_frame(smear, smeared, None, sharp_pct=0.1, raw_sharp=raw, video_median=raw * 3)
    assert s == 0.0 and why == {"rejected": "directional motion blur"}


def test_sharp_text_lines_are_kept():
    """Horizontal text lines are one-sided (ratio > 3) and rank low in a sharp video, but are not blurred."""
    lines = np.full((90, 160), 235, np.uint8)
    lines[4::8] = 20
    lines[5::8] = 20
    assert cover.directional_ratio(lines) > 3
    raw = cover.sharpness(lines)
    assert cover.reject_reason(lines, 0.05, raw, raw * 1.2) is None
    rgb = np.stack([lines] * 3, axis=-1)
    s, why = cover.score_frame(rgb, lines, None, sharp_pct=0.05, raw_sharp=raw, video_median=raw * 1.2)
    assert "rejected" not in why and s > 0


def test_relaxed_bars_keep_black_rejected():
    dark = (np.indices((90, 160)).sum(axis=0) // 8 % 2 * 14 + 6).astype(np.uint8)  # mean ~0.03, std ~0.027
    assert cover.reject_reason(dark) in ("too dark", "flat (no contrast)")
    assert cover.reject_reason(dark, relax=True) is None
    assert cover.reject_reason(np.zeros((90, 160), np.uint8), relax=True) == "too dark"
    assert cover.reject_reason(np.full((90, 160), 120, np.uint8), relax=True).startswith("flat")


def test_sharpness_percentile_within_video():
    p = cover.sharpness_percentiles([10, 20, 30, 40])
    assert p[0] == 0.0 and p[-1] == 1.0
    assert cover.sharpness_percentiles([5, 5, 5, 5]) == [0.5] * 4  # a flat graphic video is not punished
    assert cover.sharpness_percentiles([1, 2]) == [None, None]


def test_pick_one_per_video_then_fill_with_gap():
    c = [{"video": "a", "t": 1, "score": 0.9}, {"video": "a", "t": 2, "score": 0.8},
         {"video": "a", "t": 5, "score": 0.7}, {"video": "b", "t": 1, "score": 0.6},
         {"video": "b", "t": 2, "score": 0.5}, {"video": "c", "t": 1, "score": 0.4}]
    got = cover.pick(c, 3)
    assert [(x["video"], x["score"]) for x in got] == [("a", 0.9), ("b", 0.6), ("c", 0.4)]
    assert cover.pick([], 3) == [] and len(cover.pick(c, 10)) == 4  # t=2 of a and b skipped (< 2 s from t=1)
    single = [{"video": "a", "t": t, "score": s} for t, s in ((1, 0.9), (2, 0.85), (4, 0.8), (4.5, 0.7), (8, 0.6))]
    got = cover.pick(single, 3)
    assert [x["t"] for x in got] == [1, 4, 8]


def test_sample_times_skip_edges_cuts_and_gradual():
    ts = cover.sample_times(10.0, 25.0, [5.0], 8)
    assert all(0.5 <= t <= 9.5 for t in ts) and all(abs(t - 5.0) > 3 / 25 for t in ts)
    assert cover.sample_times(0.8, 25.0, [], 5) == []
    cuts = [{"t": 2.0, "kind": "gradual", "dur_f": 50}, {"t": 7.0, "kind": "jump", "dur_f": 1}]
    ts = cover.sample_times(10.0, 25.0, cuts, 40)
    assert ts and not any(2.0 - 3 / 25 <= t <= 4.0 + 3 / 25 for t in ts)
    assert not any(abs(t - 7.0) <= 3 / 25 for t in ts)


def test_set_cover_writes_and_registry(home, tmp_path):
    pack = tmp_path / "pack"
    pack.mkdir()
    assert run_cli("styles.py", "add", "x1", "--name", "X One", "--pack", pack).returncode == 0
    img = tmp_path / "pick.png"
    Image.fromarray(frame("#5b3fd0", h=720, w=1280)).save(img)
    r = run_cli("styles.py", "set-cover", "x1", img)
    assert r.returncode == 0, r.stderr
    out = pack / "cover.jpg"
    with Image.open(out) as im:
        assert im.format == "JPEG" and max(im.size) == 1080
    reg = json.loads((home / "styles.json").read_text(encoding="utf-8"))["styles"][0]
    assert reg["cover"] == "cover.jpg"
    assert str(out) in run_cli("styles.py", "list").stdout
    assert json.loads(run_cli("styles.py", "show", "x1").stdout)["cover_path"] == str(out)
    Image.fromarray(frame("#5b3fd0", h=400, w=200)).save(img)
    assert run_cli("styles.py", "set-cover", "x1", img).returncode == 0
    with Image.open(out) as im:
        assert im.size[1] == 1080 and im.size[0] == 540


def test_set_cover_bad_inputs(home, tmp_path):
    pack = tmp_path / "pack"
    pack.mkdir()
    run_cli("styles.py", "add", "x1", "--name", "X One", "--pack", pack)
    assert run_cli("styles.py", "set-cover", "x1", tmp_path / "nope.png").returncode == 1
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"not an image")
    r = run_cli("styles.py", "set-cover", "x1", bad)
    assert r.returncode == 1 and "Traceback" not in r.stderr
    assert not (pack / "cover.jpg").exists()
    assert run_cli("styles.py", "set-cover", "nostyle", bad).returncode == 1
    reg = json.loads((home / "styles.json").read_text(encoding="utf-8"))["styles"][0]
    assert "cover" not in reg


def make_pack(tmp_path, with_fp=False, face=0.9, ids=("v1", "v2"), dur_ms=3000):
    from make_media import make_clip
    tmp_path.mkdir(parents=True, exist_ok=True)
    pack = tmp_path / "pack"
    for k, vid in enumerate(ids):
        clip = make_clip(tmp_path / f"{vid}.mkv", [(300, dur_ms - 800)], dur_ms)
        d = measure_doc(str(clip), face=face, dur=dur_ms / 1000)
        d["video"]["sha256"] = "%064x" % (k + 1)
        wj(pack / "analysis" / vid / "measure.json", d)
    if with_fp:
        wj(pack / "fingerprint.json", fingerprint.build(pack))
    return pack


def test_candidates_without_fingerprint(tmp_path):
    pack = make_pack(tmp_path)
    (pack / "cover-candidates").mkdir()
    (pack / "cover-candidates" / "stale.jpg").write_bytes(b"x")
    r = run_cli("cover.py", "candidates", pack, "--n", "2", "--per-video", "4")
    assert r.returncode == 0, r.stderr
    doc = json.loads((pack / "cover-candidates.json").read_text(encoding="utf-8"))
    assert "fingerprint" in doc["not_run"]
    cands = doc["candidates"]
    assert len(cands) == 2 and {c["video"] for c in cands} == {"v1", "v2"}
    for c in cands:
        assert (pack / c["file"]).is_file() and 0.5 <= c["t"] <= 2.5
        assert set(c["why"]) == {"sharpness", "colourful"}
    assert len(r.stdout.strip().splitlines()) == 2
    assert not (pack / "cover-candidates" / "stale.jpg").exists() and not (pack / "cover-candidates.tmp").exists()
    assert not (pack / "cover.jpg").exists()  # the script never sets a cover


def test_candidates_with_real_fingerprint_and_single_video(tmp_path):
    pack = make_pack(tmp_path, with_fp=True, ids=("v1",), dur_ms=8000)
    r = run_cli("cover.py", "candidates", pack, "--n", "3", "--per-video", "12")
    assert r.returncode == 0, r.stderr
    doc = json.loads((pack / "cover-candidates.json").read_text(encoding="utf-8"))
    assert "fingerprint" not in doc["not_run"]
    ts = sorted(c["t"] for c in doc["candidates"])
    assert len(ts) >= 2 and all(b - a >= 2.0 - 1e-9 for a, b in zip(ts, ts[1:]))
    assert "palette" in doc["candidates"][0]["why"]


def test_missing_source_and_faces_not_run(tmp_path, monkeypatch):
    pack = make_pack(tmp_path, with_fp=True, face=0.9)
    d = json.loads((pack / "analysis" / "v2" / "measure.json").read_text(encoding="utf-8"))
    d["source"] = "gone.mkv"
    wj(pack / "analysis" / "v2" / "measure.json", d)
    monkeypatch.setattr(cover.mv, "_face_detector", lambda: None)
    assert cover.main(["candidates", str(pack), "--n", "2", "--per-video", "3"]) == 0
    doc = json.loads((pack / "cover-candidates.json").read_text(encoding="utf-8"))
    assert "video:v2:no_source" in doc["not_run"] and "faces" in doc["not_run"]
    assert [c["video"] for c in doc["candidates"]] == ["v1"] or {c["video"] for c in doc["candidates"]} == {"v1"}
    # a style with almost no faces does not report faces as not run
    pack2 = make_pack(tmp_path / "two", with_fp=True, face=0.1)
    assert cover.main(["candidates", str(pack2), "--n", "1", "--per-video", "3"]) == 0
    assert "faces" not in json.loads((pack2 / "cover-candidates.json").read_text(encoding="utf-8"))["not_run"]


def test_candidates_bad_pack(tmp_path):
    assert run_cli("cover.py", "candidates", tmp_path / "none").returncode == 2
    (tmp_path / "empty").mkdir()
    assert run_cli("cover.py", "candidates", tmp_path / "empty").returncode == 2


def fake_video(monkeypatch, pack, frame_fn, ids=("v1",)):
    """Run cover.main on synthetic frames: no ffmpeg, no decoding."""
    for vid in ids:
        src = pack.parent / f"{vid}.mkv"
        src.write_bytes(b"x")
        wj(pack / "analysis" / vid / "measure.json", {"source": str(src), "cuts": []})
    monkeypatch.setattr(cover.mv, "probe", lambda v: {"duration_s": 10.0, "fps": 25.0, "width": 160, "height": 90})
    monkeypatch.setattr(cover, "_grab_rgb", lambda video, t, w, h: frame_fn(t, h, w))
    monkeypatch.setattr(cover, "_save_jpg", lambda video, t, out: Path(out).write_bytes(b"jpg"))
    monkeypatch.setattr(cover.mv, "_face_detector", lambda: None)
    monkeypatch.setattr(cover.mv, "text_boxes", lambda g: [])
    monkeypatch.setattr(cover, "require_ffmpeg", lambda: None)


def test_dark_style_gets_candidates_through_relaxed_pass(tmp_path, monkeypatch):
    pack = tmp_path / "pack"
    def dark(t, h, w):
        g = (np.indices((h, w)).sum(axis=0) // 8 % 2 * 14 + 6 + int(t) % 3).astype(np.uint8)
        return np.stack([g, g, g], axis=-1)
    fake_video(monkeypatch, pack, dark)
    assert cover.main(["candidates", str(pack), "--n", "2", "--per-video", "8"]) == 0
    doc = json.loads((pack / "cover-candidates.json").read_text(encoding="utf-8"))
    assert doc["candidates"] and "video:v1:all_rejected" in doc["not_run"]
    assert "video:v1:no_candidates" not in doc["not_run"]
    assert sum(doc["rejects"]["v1"].values()) == 8 and set(doc["rejects"]["v1"]) <= {"too dark", "flat (no contrast)"}


def test_all_black_video_is_no_candidates_and_keeps_old_folder(tmp_path, monkeypatch, capsys):
    pack = tmp_path / "pack"
    fake_video(monkeypatch, pack, lambda t, h, w: np.zeros((h, w, 3), np.uint8))
    (pack / "cover-candidates").mkdir(parents=True)
    (pack / "cover-candidates" / "old.jpg").write_bytes(b"old")
    assert cover.main(["candidates", str(pack), "--n", "2", "--per-video", "5"]) == 1
    err = capsys.readouterr().err
    assert "video:v1:no_candidates" in err and "left as it is" in err and len(err.strip().splitlines()) == 1
    assert (pack / "cover-candidates" / "old.jpg").read_bytes() == b"old"
    assert not (pack / "cover-candidates.json").exists() and not (pack / "cover-candidates.tmp").exists()

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent / "fixtures"))
import make_media as mm  # noqa: E402

import cut  # noqa: E402
from conftest import run_cli  # noqa: E402

# seven "words" (400 Hz bursts) over a steady room tone; w1 -> w2 is a 0.8 s pause
SPANS = [(500, 900), (1000, 1400), (2200, 2600), (2700, 3100), (3600, 4000), (4700, 5100), (5200, 5500)]
DUR = 6500
KEEP = {"keep": [[0, 3], [5, 6]]}


def _db(spans=SPANS, dur=DUR, **audio_kw):
    a = mm.tone_track(dur, spans, **audio_kw)
    return cut.slice_db(a.mean(axis=1))


def _plan(spans=SPANS, keep=KEEP, preset="lesson", dur=DUR, flags=None, asr=None, audio_kw=None, **over):
    doc = mm.words_doc(spans, flags, asr)
    opts = cut.resolve_opts(preset, over)
    return cut.plan_pieces(doc["words"], keep, _db(spans, dur, **(audio_kw or {})), opts["fps"], opts)


def _quiet(db, t_ms):
    """the frame boundary at t_ms is in silence: loudest slice within +-30 ms below floor + 10 dB."""
    k = int(t_ms // 10)
    return db[max(0, k - 3): k + 3].max() < np.percentile(db, 10) + 10


def _inside(t, span):
    return span[0] < t < span[1]


# ---------------------------------------------------------------- presets

def test_presets_exact_values():
    assert cut.PRESETS["reels"] == {"fps": 30, "lead_ms": 60, "tail_ms": 80, "merge_gap_ms": 180,
                                    "pause_max_ms": 260, "pause_keep_ms": 70, "end_air_ms": 1000}
    assert cut.PRESETS["lesson"] == {"fps": 25, "lead_ms": 80, "tail_ms": 120, "merge_gap_ms": 250,
                                     "pause_max_ms": 450, "pause_keep_ms": 150, "end_air_ms": None}
    lesson = cut.resolve_opts("lesson")
    assert lesson == cut.resolve_opts(None)
    assert lesson["inside_ms"] == 60 and lesson["xfade_ms"] == 15.0 and lesson["sliver_ms"] == 240
    assert lesson["pause_min_cut_ms"] == 120 and lesson["sliver_frames"] is None


def test_explicit_flag_overrides_preset():
    o = cut.resolve_opts("reels", {"pause_keep_ms": 100, "fps": 25})
    assert o["pause_keep_ms"] == 100 and o["fps"] == 25
    assert o["lead_ms"] == 60 and o["end_air_ms"] == 1000


# ---------------------------------------------------------------- planning (pure)

def test_edges_on_frame_boundaries_and_in_silence():
    plan = _plan()
    db = _db()
    fms = 1000 / 25
    assert len(plan["pieces"]) == 3  # the 0.8 s pause splits the first span
    for p in plan["pieces"]:
        assert abs(p["src_in_ms"] - p["f_in"] * fms) < 0.001 and abs(p["src_out_ms"] - p["f_out"] * fms) < 0.001
        for t in (p["src_in_ms"], p["src_out_ms"]):
            assert _quiet(db, t), t
    assert plan["soft_joins"] == []


def test_long_pause_is_shortened_to_about_twice_pause_keep():
    plan = _plan()
    w = {x["src_i"]: x for x in plan["words"]}
    kept = w[2]["start_ms"] - w[1]["end_ms"]
    assert abs(kept - 2 * 150) <= 40, kept
    plan = _plan(preset="reels")
    w = {x["src_i"]: x for x in plan["words"]}
    kept = w[2]["start_ms"] - w[1]["end_ms"]
    assert abs(kept - 2 * 70) <= 1000 / 30 + 1, kept


def test_spans_closer_than_merge_gap_merge():
    # keep w0 and w2, drop w1; the source gap w0 -> w2 is 500 ms
    spans = [(500, 800), (1000, 1100), (1300, 1600), (2600, 2900)]
    plan = _plan(spans, {"keep": [[0, 0], [2, 2]]}, dur=3500)
    assert [(p["w0"], p["w1"]) for p in plan["pieces"]] == [(0, 0), (2, 2)]
    plan = _plan(spans, {"keep": [[0, 0], [2, 2]]}, dur=3500, merge_gap_ms=600)
    assert [(p["w0"], p["w1"]) for p in plan["pieces"]] == [(0, 2)]


def test_exclude_ms_splits_a_piece():
    plan = _plan(keep={"keep": [[2, 3]], "exclude_ms": [[2620, 2680]]})
    pcs = [(p["w0"], p["w1"], p["src_in_ms"], p["src_out_ms"]) for p in plan["pieces"]]
    assert len(pcs) == 2
    assert pcs[0][1] == 2 and pcs[0][3] == 2600
    assert pcs[1][0] == 3 and pcs[1][2] == 2680
    for p in plan["pieces"]:
        assert p["src_out_ms"] <= 2620 or p["src_in_ms"] >= 2680


def test_low_score_word_never_cut_inside_asr_or_refined_span():
    spans = [(500, 900), (1000, 1300), (1900, 2300)]
    # out-point: ASR thinks the low-score word runs to 1600
    plan = _plan(spans, {"keep": [[0, 1]]}, dur=3000, flags={1: "low_score"}, asr={1: (1000, 1600)})
    out = plan["pieces"][-1]["src_out_ms"]
    assert out >= 1600, out
    # in-point: ASR puts the start at 700, before the refined 1000
    spans = [(200, 500), (1000, 1300), (1900, 2300)]
    plan = _plan(spans, {"keep": [[1, 2]]}, dur=3000, flags={1: "low_score"}, asr={1: (700, 1300)})
    p0 = plan["pieces"][0]
    assert p0["src_in_ms"] <= 700, p0
    for p in plan["pieces"]:
        for t in (p["src_in_ms"], p["src_out_ms"]):
            assert not _inside(t, (700, 1300))
    # the remapped word keeps its refined times
    w1 = next(w for w in plan["words"] if w["src_i"] == 1)
    assert w1["end_ms"] - w1["start_ms"] == 300


@pytest.mark.parametrize("flag", ["unaligned", "unaligned_script"])
def test_unaligned_word_never_cut_inside_asr_span(flag):
    # w0 runs straight into w1 (ASR start 1000); the audio of w1 starts at 1100
    spans = [(500, 1000), (1000, 1400), (2000, 2300)]
    doc = mm.words_doc(spans, {1: flag}, {1: (1000, 1400)})
    db = cut.slice_db(mm.tone_track(3000, [(500, 1000), (1100, 1400), (2000, 2300)]).mean(axis=1))
    o = cut.resolve_opts("lesson")
    plan = cut.plan_pieces(doc["words"], {"keep": [[0, 0]]}, db, 25, o)
    out = plan["pieces"][0]["src_out_ms"]
    assert not _inside(out, (1000, 1400)), out
    # the same plan without the flag would cut at 1040, inside the ASR span
    doc2 = mm.words_doc(spans)
    plan2 = cut.plan_pieces(doc2["words"], {"keep": [[0, 0]]}, db, 25, o)
    assert _inside(plan2["pieces"][0]["src_out_ms"], (1000, 1400))
    # in-point side: keep the unaligned word itself, its ASR start is before the refined start
    spans = [(200, 500), (1100, 1400), (2000, 2300)]
    doc = mm.words_doc(spans, {1: flag}, {1: (800, 1500)})
    db = cut.slice_db(mm.tone_track(3000, spans).mean(axis=1))
    plan = cut.plan_pieces(doc["words"], {"keep": [[1, 1]]}, db, 25, o)
    p = plan["pieces"][0]
    assert not _inside(p["src_in_ms"], (800, 1500)) and not _inside(p["src_out_ms"], (800, 1500)), p


def test_edge_without_quiet_frame_lands_on_quietest_and_is_listed():
    spans = [(500, 1000), (1000, 1500), (1500, 2000)]
    kw = {"continuous": (500, 2000), "dips": [(1080, 1160, 12)]}
    plan = _plan(spans, {"keep": [[0, 0]]}, dur=3000, audio_kw=kw)
    p = plan["pieces"][0]
    assert p["src_out_ms"] == 1120  # the dip: quietest frame in reach, still voiced
    soft = [s for s in plan["soft_joins"] if s["edge"] == "out"]
    assert len(soft) == 1 and soft[0]["src_ms"] == 1120 and soft[0]["piece"] == p["id"]
    assert soft[0]["db"] >= soft[0]["quiet_thr_db"]


def test_retime_and_set_out_ms_move_the_edges():
    # the aligner says w1 ends at 1400; the audio (and the retime) say 1250
    db = _db([(500, 900), (1000, 1250)] + SPANS[2:])
    doc = mm.words_doc(SPANS)
    o = cut.resolve_opts("lesson")
    plan = cut.plan_pieces(doc["words"], {"keep": [[0, 1]], "retime": {"1": [1000, 1250]}}, db, 25, o)
    assert plan["pieces"][-1]["src_out_ms"] == 1360  # 1250 + 120 tail on the frame grid
    assert cut.plan_pieces(doc["words"], {"keep": [[0, 1]]}, db, 25, o)["pieces"][-1]["src_out_ms"] >= 1480
    base = _plan(keep={"keep": [[2, 3]]})
    old_out = base["pieces"][-1]["src_out_ms"]
    plan = _plan(keep={"keep": [[2, 3]], "set_out_ms": [[old_out, 3300]]})
    assert abs(plan["pieces"][-1]["src_out_ms"] - 3300) <= 20


def test_placeholder_after_a_word_that_does_not_end_a_piece_fails():
    with pytest.raises(ValueError, match="does not end a piece"):
        _plan(keep={"keep": [[0, 3]], "placeholders": [{"id": "x", "after_word": 2, "dur_s": 1.0}]})


# ---------------------------------------------------------------- CLI + render (real ffmpeg)

def _probe(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-show_entries",
                        "stream=codec_type,duration,nb_read_frames,r_frame_rate", "-of", "json", str(path)],
                       capture_output=True, text=True, check=True)
    return {s["codec_type"]: s for s in json.loads(r.stdout)["streams"]}


def _audio(path):
    r = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-vn", "-ac", "2", "-ar", "48000",
                        "-f", "f32le", "-"], capture_output=True, check=True)
    return np.frombuffer(r.stdout, dtype=np.float32).reshape(-1, 2)


def _setup(tmp_path, spans=SPANS, dur=DUR, keep=KEEP, fps=25, video="testsrc2", **audio_kw):
    src = mm.make_clip(tmp_path / "src.mkv", spans, dur, fps=fps, video=video, **audio_kw)
    words = mm.write_json(tmp_path / "words.aligned.json", mm.words_doc(spans))
    kp = mm.write_json(tmp_path / "keep.json", keep)
    return src, words, kp


@pytest.fixture(scope="module")
def base_clip(tmp_path_factory):
    """the default 25 fps clip (ramp video), its aligned words and keep.json, shared by the render tests."""
    d = tmp_path_factory.mktemp("cutclip")
    return _setup(d, video="ramp")


def _gray(path):
    r = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "gray", "-"],
                       capture_output=True, check=True)
    return np.frombuffer(r.stdout, dtype=np.uint8).reshape(-1, 160 * 90).mean(axis=1)


def _av_ok(mp4, fps, frames):
    s = _probe(mp4)
    assert int(s["video"]["nb_read_frames"]) == frames
    assert s["video"]["r_frame_rate"] == f"{fps}/1"
    assert abs(float(s["audio"]["duration"]) - float(s["video"]["duration"])) < 1 / fps


def test_render_lesson_writes_outputs_and_stays_in_sync(tmp_path, home, base_clip):
    src, words, kp = base_clip
    stem = tmp_path / "out" / "edit"
    r = run_cli("cut.py", src, words, kp, "--out", stem)
    assert r.returncode == 0, r.stderr
    cuts = json.loads(Path(f"{stem}.cuts.json").read_text(encoding="utf-8"))
    for k in ("pieces", "joins", "long_gaps_kept", "soft_joins", "placeholders", "channel_balance_db"):
        assert k in cuts
    p0 = cuts["pieces"][0]
    for k in ("src_in", "src_out", "dst_in", "frames", "f_in", "f_out", "w0", "w1", "text"):
        assert k in p0
    assert cuts["fps"] == 25 and cuts["settings"]["pause_keep_ms"] == 150
    wj = json.loads(Path(f"{stem}.words.json").read_text(encoding="utf-8"))
    assert [w["src_i"] for w in wj["words"]] == [0, 1, 2, 3, 5, 6]
    _av_ok(f"{stem}.mp4", 25, cuts["frames"])
    # the cut audio has its words where words.json says
    db = cut.slice_db(_audio(f"{stem}.mp4").mean(axis=1))
    for w in wj["words"]:
        mid = (w["start_ms"] + w["end_ms"]) // 20
        assert db[mid] > -20


def test_25fps_source_cut_at_30fps_keeps_av_in_sync(tmp_path, home, base_clip):
    src, words, kp = base_clip
    stem = tmp_path / "reel"
    r = run_cli("cut.py", src, words, kp, "--out", stem, "--preset", "reels", "--fps", "30")
    assert r.returncode == 0, r.stderr
    cuts = json.loads(Path(f"{stem}.cuts.json").read_text(encoding="utf-8"))
    assert cuts["fps"] == 30 and cuts["source_fps"] == 25
    fms = 1000 / 30
    for p in cuts["pieces"]:
        assert abs(p["src_in_ms"] - p["f_in"] * fms) < 0.001 and abs(p["src_out_ms"] - p["f_out"] * fms) < 0.001
    _av_ok(f"{stem}.mp4", 30, cuts["frames"])
    # every output frame shows the source frame of its source time (luma = 16 + source frame number)
    src_gray, out_gray = _gray(src), _gray(f"{stem}.mp4")
    for p in cuts["pieces"]:
        for m in range(p["dst_f"], p["dst_f"] + p["frames"]):
            t = p["src_in_ms"] / 1000 + (m - p["dst_f"]) / 30
            n = int(np.argmin(np.abs(src_gray - out_gray[m])))
            assert abs(n / 25 - t) <= 1 / 25 + 1e-6, (m, t, n)


def test_unequal_channels_are_balanced(tmp_path, home):
    src, words, kp = _setup(tmp_path, left_db=6.0)
    stem = tmp_path / "bal"
    r = run_cli("cut.py", src, words, kp, "--out", stem)
    assert r.returncode == 0, r.stderr
    cuts = json.loads(Path(f"{stem}.cuts.json").read_text(encoding="utf-8"))
    assert abs(cuts["channel_balance_db"] - 6.0) < 0.3
    a = _audio(f"{stem}.mp4")
    rms = np.sqrt((a.astype(np.float64) ** 2).mean(axis=0))
    assert abs(20 * np.log10(rms[0] / rms[1])) < 0.5


def test_balanced_source_is_left_alone(tmp_path, home, base_clip):
    src, words, kp = base_clip
    stem = tmp_path / "eq"
    r = run_cli("cut.py", src, words, kp, "--out", stem, "--no-render")
    assert r.returncode == 0, r.stderr
    cuts = json.loads(Path(f"{stem}.cuts.json").read_text(encoding="utf-8"))
    assert abs(cuts["channel_balance_db"]) < 0.5 and cuts["channel_balance"] == "none"
    assert not Path(f"{stem}.mp4").exists()


def test_placeholder_inserts_black_and_room_tone(tmp_path, home):
    keep = {"keep": [[0, 3], [5, 6]], "placeholders": [{"id": "ph1", "after_word": 3, "dur_s": 1.0}]}
    src, words, kp = _setup(tmp_path, dur=9500, keep=keep)
    stem = tmp_path / "ph"
    r = run_cli("cut.py", src, words, kp, "--out", stem)
    assert r.returncode == 0, r.stderr
    cuts = json.loads(Path(f"{stem}.cuts.json").read_text(encoding="utf-8"))
    ph = cuts["placeholders"][0]
    assert ph["id"] == "ph1" and abs(ph["end"] - ph["start"] - 1.0) < 1e-6
    rec = next(p for p in cuts["pieces"] if p.get("placeholder") == "ph1")
    assert rec["frames"] == 25
    _av_ok(f"{stem}.mp4", 25, cuts["frames"])
    mid = (ph["start"] + ph["end"]) / 2
    fr = subprocess.run(["ffmpeg", "-v", "error", "-ss", str(mid), "-i", f"{stem}.mp4", "-frames:v", "1",
                         "-f", "rawvideo", "-pix_fmt", "gray", "-"], capture_output=True, check=True).stdout
    assert np.frombuffer(fr, dtype=np.uint8).mean() < 20  # black
    a = _audio(f"{stem}.mp4").mean(axis=1)
    seg = a[int((ph["start"] + 0.1) * 48000): int((ph["end"] - 0.1) * 48000)]
    level = 20 * np.log10(np.sqrt((seg.astype(np.float64) ** 2).mean()) + 1e-12)
    assert -65 < level < -45  # room tone, not digital silence and not a word


def test_preset_equals_explicit_flags(tmp_path, home, base_clip):
    src, words, kp = base_clip
    a = tmp_path / "a"
    b = tmp_path / "b"
    r1 = run_cli("cut.py", src, words, kp, "--out", a, "--preset", "reels", "--no-render")
    r2 = run_cli("cut.py", src, words, kp, "--out", b, "--no-render", "--lead-ms", "60",
                 "--tail-ms", "80", "--merge-gap-ms", "180", "--pause-max-ms", "260", "--pause-keep-ms", "70",
                 "--end-air-ms", "1000")
    assert r1.returncode == 0 and r2.returncode == 0, r1.stderr + r2.stderr
    c1 = json.loads(Path(f"{a}.cuts.json").read_text(encoding="utf-8"))
    c2 = json.loads(Path(f"{b}.cuts.json").read_text(encoding="utf-8"))
    for k in ("fps", "frames", "settings", "pieces", "joins"):
        assert c1[k] == c2[k], k
    assert c1["fps"] == 25  # the 25 fps source keeps its own rate under every preset
    r3 = run_cli("cut.py", src, words, kp, "--out", tmp_path / "c", "--preset", "reels",
                 "--pause-keep-ms", "100", "--fps", "30", "--no-render")
    c3 = json.loads(Path(f"{tmp_path / 'c'}.cuts.json").read_text(encoding="utf-8"))
    assert c3["settings"]["pause_keep_ms"] == 100 and c3["fps"] == 30  # an explicit --fps wins


def test_missing_ffmpeg_exits_3_one_line(tmp_path, home, monkeypatch):
    kp = mm.write_json(tmp_path / "keep.json", KEEP)
    words = mm.write_json(tmp_path / "w.json", mm.words_doc(SPANS))
    empty = tmp_path / "nobin"
    empty.mkdir()
    monkeypatch.setenv("PATH", str(empty))
    r = run_cli("cut.py", tmp_path / "src.mkv", words, kp, "--out", tmp_path / "x")
    assert r.returncode == 3
    lines = [x for x in r.stderr.splitlines() if x.strip()]
    assert len(lines) == 1 and "ffmpeg" in lines[0]
    assert not list(tmp_path.glob("x*"))


# ---------------------------------------------------------------- review fixes

def test_last_word_past_the_video_end_is_clamped_and_listed():
    spans = [(500, 900), (1000, 1400), (1800, 2600)]
    doc = mm.words_doc(spans)
    db = _db(spans, 3000)
    plan = cut.plan_pieces(doc["words"], {"keep": [[0, 2]]}, db, 25, cut.resolve_opts("lesson"), dur_ms=2300)
    last = plan["pieces"][-1]
    assert last["src_out_ms"] <= 2300, last
    assert any(s["piece"] == last["id"] and s["edge"] == "out" for s in plan["soft_joins"])
    # the reviewer's probe: a last word starting inside but ending past the end never yields a frame past it
    spans = [(500, 900), (1000, 1400), (2200, 2600)]
    plan = cut.plan_pieces(mm.words_doc(spans)["words"], {"keep": [[0, 2]]}, _db(spans, 3000), 25,
                           cut.resolve_opts("lesson"), dur_ms=2300)
    assert all(p["src_out_ms"] <= 2300 for p in plan["pieces"])


def test_render_with_video_shorter_than_audio_stays_in_sync(tmp_path, home):
    spans = [(500, 900), (1000, 1400), (1800, 2600)]
    full = mm.make_clip(tmp_path / "full.mkv", spans, 3000)
    src = tmp_path / "short.mkv"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(full), "-filter_complex", "[0:v]trim=end=2.3[v]",
                    "-map", "[v]", "-map", "0:a", "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "copy", str(src)],
                   check=True)
    words = mm.write_json(tmp_path / "w.json", mm.words_doc(spans))
    kp = mm.write_json(tmp_path / "k.json", {"keep": [[0, 2]]})
    stem = tmp_path / "short_cut"
    r = run_cli("cut.py", src, words, kp, "--out", stem)
    assert r.returncode == 0, r.stderr
    cuts = json.loads(Path(f"{stem}.cuts.json").read_text(encoding="utf-8"))
    assert cuts["pieces"][-1]["src_out_ms"] <= cut.probe_video(src)[2]  # 58 frames = 2320 ms
    assert cuts["soft_joins"]
    _av_ok(f"{stem}.mp4", 25, cuts["frames"])


def test_render_api_balances_channels_and_replaces_atomically(tmp_path, home):
    src, _, _ = _setup(tmp_path, left_db=6.0)
    audio = cut.load_audio(src)
    db = cut.slice_db(audio.mean(axis=1))
    o = cut.resolve_opts("lesson")
    plan = cut.plan_pieces(mm.words_doc(SPANS)["words"], KEEP, db, 25, o)
    assert "channel_balance" not in plan
    stem = tmp_path / "api"
    cut.render(src, plan, stem, o)
    a = _audio(f"{stem}.mp4")
    rms = np.sqrt((a.astype(np.float64) ** 2).mean(axis=0))
    assert abs(20 * np.log10(rms[0] / rms[1])) < 0.5
    cuts = json.loads(Path(f"{stem}.cuts.json").read_text(encoding="utf-8"))
    assert abs(cuts["channel_balance_db"] - 6.0) < 0.3 and cuts["channel_balance"] == "gain"
    assert not Path(f"{stem}.tmp.mp4").exists()
    # a failed render leaves neither the output nor its temp file
    bad = tmp_path / "bad"
    plan2 = cut.plan_pieces(mm.words_doc(SPANS)["words"], KEEP, db, 25, o)
    with pytest.raises(RuntimeError):
        cut.render(tmp_path / "missing.mkv", plan2, bad, o, audio=audio)
    assert not Path(f"{bad}.mp4").exists() and not Path(f"{bad}.tmp.mp4").exists()
    # the plan files are moved in only with the video: a failed render writes none of them
    assert not list(tmp_path.glob("bad.*"))
    # ... and a failed re-render leaves the earlier set whole
    before = {p.name: p.read_bytes() for p in tmp_path.glob("api.*")}
    with pytest.raises(RuntimeError):
        cut.render(tmp_path / "missing.mkv", cut.plan_pieces(mm.words_doc(SPANS)["words"], {"keep": [[0, 1]]},
                                                             db, 25, o), stem, o, audio=audio)
    assert {p.name: p.read_bytes() for p in tmp_path.glob("api.*")} == before


def test_end_air_without_room_falls_back_to_the_normal_tail():
    spans = [(500, 900), (1000, 1400), (1900, 2300)]
    plan = cut.plan_pieces(mm.words_doc(spans)["words"], {"keep": [[0, 1]]}, _db(spans, 3000), 30,
                           cut.resolve_opts("reels"))
    last = plan["pieces"][-1]
    assert not last.get("tight")
    assert 1400 + 60 <= last["src_out_ms"] <= 1400 + 80 + 80, last
    assert plan["soft_joins"] == []


def test_end_air_zero_means_off():
    o = cut.resolve_opts("reels", {"end_air_ms": 0})
    assert o["end_air_ms"] is None and o["fps"] == 30
    with_air = _plan(preset="reels")
    without = _plan(preset="reels", end_air_ms=0)
    assert with_air["pieces"][-1]["src_out_ms"] >= 5500 + 980
    assert without["pieces"][-1]["src_out_ms"] <= 5500 + 80 + 80


def test_room_tone_never_comes_from_digital_silence():
    spans = SPANS
    db = _db(spans, 13000)
    db[950:] = -180.0  # 9.5 s on: digital silence (quieter than any room tone)
    keep = {"keep": [[0, 3], [5, 6]], "placeholders": [{"id": "p", "after_word": 3, "dur_s": 1.0}]}
    plan = cut.plan_pieces(mm.words_doc(spans)["words"], keep, db, 25, cut.resolve_opts("lesson"))
    rec = next(p for p in plan["pieces"] if p.get("placeholder") == "p")
    t0, t1 = rec["src_in_ms"], rec["src_out_ms"]
    assert db[int(t0 // 10): int(t1 // 10)].min() > cut.DIGITAL_SILENCE_DB, (t0, t1)
    assert 5500 <= t0 and t1 <= 9500


# ---------------------------------------------------------------- review round 2

def test_final_piece_cut_by_the_source_end_is_never_dropped_as_a_sliver():
    spans = [(500, 900), (1000, 1400), (2200, 2600)]
    # the last piece runs from the shortened pause (~2040) to the source end (2280): 6 frames, under 10
    plan = cut.plan_pieces(mm.words_doc(spans)["words"], {"keep": [[0, 2]]}, _db(spans, 3000), 25,
                           cut.resolve_opts("lesson", {"sliver_frames": 10}), dur_ms=2300)
    last = plan["pieces"][-1]
    assert last["w1"] == 2 and last["src_out_ms"] <= 2300
    assert 2 in [w["src_i"] for w in plan["words"]]
    assert plan["dropped_slivers"] == []
    soft = [s for s in plan["soft_joins"] if s["piece"] == last["id"] and s["edge"] == "out"]
    assert soft and soft[0]["why"] == "source ends inside the word"


def test_manual_points_and_exclusions_are_clamped_to_the_source_end():
    base = _plan(keep={"keep": [[2, 3]]})
    old_out = base["pieces"][-1]["src_out_ms"]
    plan = _plan(keep={"keep": [[2, 3]], "set_out_ms": [[old_out, 9000]]})
    last = plan["pieces"][-1]
    assert last["src_out_ms"] <= DUR
    assert any(s["piece"] == last["id"] and s["edge"] == "out" and s["why"] == "edge clamped to the source end"
               for s in plan["soft_joins"])
    # exclude_ms reaching past the source end (cut short at 3000): no edge past it either
    plan = cut.plan_pieces(mm.words_doc(SPANS)["words"], {"keep": [[2, 3]], "exclude_ms": [[3050, 3200]]},
                           _db(), 25, cut.resolve_opts("lesson"), dur_ms=3000)
    assert all(p["src_out_ms"] <= 3000 for p in plan["pieces"])
    assert any(s["edge"] == "out" for s in plan["soft_joins"])


def test_dropped_sliver_warning_names_each_lost_word(tmp_path, home, base_clip):
    src, words, _ = base_clip
    # w2 + w3 are excluded down to slivers: both words are lost and must be named
    kp = mm.write_json(tmp_path / "k.json", {"keep": [[0, 0], [2, 3]], "exclude_ms": [[2300, 3150]]})
    stem = tmp_path / "sl"
    r = run_cli("cut.py", src, words, kp, "--out", stem, "--no-render")
    assert r.returncode == 0, r.stderr
    cuts = json.loads(Path(f"{stem}.cuts.json").read_text(encoding="utf-8"))
    lost = [w["i"] for d in cuts["dropped_slivers"] for w in d["lost_words"]]
    assert sorted(lost) == [2, 3]
    out = r.stdout
    assert "w2 (2.20s)" in out and "w3 (2.70s)" in out


def test_vsl_preset_is_the_reels_cut_and_prints_the_next_step(tmp_path, home, base_clip):
    assert cut.PRESETS["vsl"] == dict(cut.PRESETS["reels"], tail_ms=200)  # tighten closes the air
    assert cut.resolve_opts("vsl") == dict(cut.resolve_opts("reels"), tail_ms=200)
    src, words, kp = base_clip
    stem = tmp_path / "v"
    r = run_cli("cut.py", src, words, kp, "--out", stem, "--preset", "vsl")
    assert r.returncode == 0, r.stderr
    nxt = [x for x in r.stdout.splitlines() if x.startswith("next:")]
    assert nxt == [f"next: python scripts/tighten.py {stem} --out {stem}.tight, "
                   f"then python scripts/check_cut.py {stem}.tight"]
    r = run_cli("cut.py", src, words, kp, "--out", tmp_path / "r", "--preset", "reels", "--no-render")
    assert "next:" not in r.stdout


# ---------------------------------------------------------------- final review fixes

def test_choose_fps_keeps_a_whole_source_rate_else_the_preset():
    from fractions import Fraction
    assert cut.choose_fps(Fraction(60), 30, "reels") == (60, None)
    assert cut.choose_fps(Fraction(24), 25) == (24, None)
    fps, note = cut.choose_fps(Fraction(30000, 1001), 30, "vsl")
    assert fps == 30 and "29.970" in note and "vsl" in note and "--fps" in note
    fps, note = cut.choose_fps(Fraction(7), 25)  # 48000 / 7 is not whole
    assert fps == 25 and note
    fps, note = cut.choose_fps(None, 25)
    assert fps == 25 and "could not be read" in note


def test_60fps_source_is_cut_at_60fps(tmp_path, home):
    src, words, kp = _setup(tmp_path, fps=60)
    stem = tmp_path / "s60"
    r = run_cli("cut.py", src, words, kp, "--out", stem, "--preset", "reels")
    assert r.returncode == 0, r.stderr
    assert not [x for x in r.stdout.splitlines() if x.startswith("note:") and "fps" in x], r.stdout
    cuts =json.loads(Path(f"{stem}.cuts.json").read_text(encoding="utf-8"))
    assert cuts["fps"] == 60 and cuts["source_fps"] == 60
    _av_ok(f"{stem}.mp4", 60, cuts["frames"])
    # an explicit --fps always wins over the source rate
    r = run_cli("cut.py", src, words, kp, "--out", tmp_path / "s30", "--preset", "reels", "--fps", "30",
                "--no-render")
    assert r.returncode == 0, r.stderr
    assert json.loads(Path(f"{tmp_path / 's30'}.cuts.json").read_text(encoding="utf-8"))["fps"] == 30


def test_29_97fps_source_uses_the_preset_rate_with_a_note(tmp_path, home):
    wav = tmp_path / "a.wav"
    mm.write_wav(wav, mm.tone_track(DUR, SPANS))
    src = tmp_path / "ntsc.mkv"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                    f"testsrc2=s=160x90:r=30000/1001:d={DUR / 1000}", "-i", str(wav), "-map", "0:v", "-map", "1:a",
                    "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-c:a", "pcm_s16le", str(src)],
                   check=True)
    assert abs(float(cut.source_fps(src)) - 29.97) < 0.01
    words = mm.write_json(tmp_path / "w.json", mm.words_doc(SPANS))
    kp = mm.write_json(tmp_path / "k.json", KEEP)
    for preset, want in (("lesson", 25), ("vsl", 30)):
        stem = tmp_path / preset
        r = run_cli("cut.py", src, words, kp, "--out", stem, "--preset", preset, "--no-render")
        assert r.returncode == 0, r.stderr
        notes = [x for x in r.stdout.splitlines() if x.startswith("note:") and "fps" in x]
        assert len(notes) == 1 and "29.970" in notes[0] and f"{want} fps" in notes[0], r.stdout
        assert json.loads(Path(f"{stem}.cuts.json").read_text(encoding="utf-8"))["fps"] == want


@pytest.mark.parametrize("entry, why", [([5], "must be"), ([3, 1], "reversed"), (["a", 2], "must be"),
                                        ([1, 2, 3], "must be")])
def test_malformed_or_reversed_keep_entry_exits_2_naming_it(tmp_path, home, base_clip, entry, why):
    src, words, _ = base_clip
    kp = mm.write_json(tmp_path / "k.json", {"keep": [[0, 1], entry]})
    r = run_cli("cut.py", src, words, kp, "--out", tmp_path / "x", "--no-render")
    assert r.returncode == 2, r.stdout + r.stderr
    lines = [x for x in r.stderr.splitlines() if x.strip()]
    assert len(lines) == 1 and "keep entry 1" in lines[0] and why in lines[0], lines
    assert json.dumps(entry) in lines[0]
    assert not list(tmp_path.glob("x*"))


def test_source_without_audio_is_one_plain_line(tmp_path, home):
    src = tmp_path / "silent.mkv"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"testsrc2=s=160x90:r=25:d={DUR / 1000}",
                    "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", str(src)], check=True)
    words = mm.write_json(tmp_path / "w.json", mm.words_doc(SPANS))
    kp = mm.write_json(tmp_path / "k.json", KEEP)
    r = run_cli("cut.py", src, words, kp, "--out", tmp_path / "x")
    assert r.returncode == 1
    lines = [x for x in r.stderr.splitlines() if x.strip()]
    assert len(lines) == 1 and "no audio stream" in lines[0] and "Traceback" not in r.stderr, lines


def test_no_render_renames_a_stale_video(tmp_path, home, base_clip):
    src, words, kp = base_clip
    stem = tmp_path / "st"
    assert run_cli("cut.py", src, words, kp, "--out", stem).returncode == 0
    old = Path(f"{stem}.mp4").read_bytes()
    kp2 = mm.write_json(tmp_path / "k2.json", {"keep": [[0, 1]]})
    r = run_cli("cut.py", src, words, kp2, "--out", stem, "--no-render")
    assert r.returncode == 0, r.stderr
    assert not Path(f"{stem}.mp4").exists()
    assert Path(f"{stem}.stale.mp4").read_bytes() == old
    assert "stale.mp4" in r.stdout


def test_replace_together_restores_the_earlier_set_on_failure(tmp_path, monkeypatch):
    finals = [str(tmp_path / n) for n in ("a.mp4", "a.cuts.json", "a.words.json")]
    for f in finals:
        Path(f).write_text("old " + f, encoding="utf-8")
    pairs = []
    for f in finals:
        Path(f + ".part").write_text("new " + f, encoding="utf-8")
        pairs.append((f + ".part", f))
    real, calls = cut.os.replace, []

    def flaky(a, b):
        calls.append((a, b))
        if a.endswith(".words.json.part"):  # the last move of the new set fails
            raise OSError("disk full")
        return real(a, b)

    monkeypatch.setattr(cut.os, "replace", flaky)
    with pytest.raises(OSError):
        cut.replace_together(pairs)
    monkeypatch.setattr(cut.os, "replace", real)
    for f in finals:
        assert Path(f).read_text(encoding="utf-8") == "old " + f
    assert sorted(p.name for p in tmp_path.iterdir()) == ["a.cuts.json", "a.mp4", "a.words.json"]
    # and a clean run replaces all three, leaving no backup behind
    for f in finals:
        Path(f + ".part").write_text("new " + f, encoding="utf-8")
    cut.replace_together(pairs)
    assert all(Path(f).read_text(encoding="utf-8") == "new " + f for f in finals)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["a.cuts.json", "a.mp4", "a.words.json"]


def test_replace_together_backups_are_unique_and_never_touch_a_users_bak(tmp_path, monkeypatch, capsys):
    final = str(tmp_path / "a.mp4")
    Path(final).write_text("old", encoding="utf-8")
    Path(final + ".bak").write_text("the user's own backup", encoding="utf-8")
    seen = []
    real = cut.os.replace

    def spy(a, b):
        seen.append(b)
        return real(a, b)

    monkeypatch.setattr(cut.os, "replace", spy)
    Path(final + ".part").write_text("new", encoding="utf-8")
    cut.replace_together([(final + ".part", final)])
    monkeypatch.setattr(cut.os, "replace", real)
    assert any(".bak-" in b for b in seen) and not any(b.endswith(".bak") for b in seen)
    assert Path(final).read_text(encoding="utf-8") == "new"
    assert Path(final + ".bak").read_text(encoding="utf-8") == "the user's own backup"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["a.mp4", "a.mp4.bak"]
    assert capsys.readouterr().err == ""


def test_replace_together_names_a_backup_it_could_not_restore(tmp_path, monkeypatch, capsys):
    finals = [str(tmp_path / "a.mp4"), str(tmp_path / "a.cuts.json")]
    for f in finals:
        Path(f).write_text("old", encoding="utf-8")
        Path(f + ".part").write_text("new", encoding="utf-8")
    real = cut.os.replace

    def flaky(a, b):
        if a.endswith(".cuts.json.part"):
            raise OSError("disk full")
        if ".bak-" in a and b.endswith("a.mp4"):  # the restore of a.mp4 fails too
            raise OSError("locked")
        return real(a, b)

    monkeypatch.setattr(cut.os, "replace", flaky)
    with pytest.raises(OSError):
        cut.replace_together([(f + ".part", f) for f in finals])
    monkeypatch.setattr(cut.os, "replace", real)
    err = [x for x in capsys.readouterr().err.splitlines() if x.strip()]
    left = [p for p in tmp_path.iterdir() if ".bak-" in p.name]
    assert len(err) == 1 and "could not restore" in err[0] and len(left) == 1 and str(left[0]) in err[0], err
    assert Path(finals[1]).read_text(encoding="utf-8") == "old"  # the other file was restored
    # the next run lists the leftover backup next to its outputs
    assert cut.stale_backups(finals) == [str(left[0])]
    cut.warn_stale_backups(finals)
    out = capsys.readouterr().out
    assert "leftover backup" in out and str(left[0]) in out


def test_frame_options_are_ms_and_behave_the_same_at_60_and_25_fps():
    assert (cut.frames_for(240, 25), cut.frames_for(240, 60), cut.frames_for(120, 25), cut.frames_for(120, 60)) \
        == (6, 15, 3, 8)
    plans = {}
    for fps in (25, 60):
        o = cut.resolve_opts("lesson")
        plans[fps] = [cut.plan_pieces(mm.words_doc(SPANS)["words"], {"keep": [[0, 0], [2, 3]], "exclude_ms": ex},
                                      _db(), fps, o) for ex in ([[2300, 3150]], [[2420, 3150]])]
        s = plans[fps][0]["settings"]
        assert s["sliver_ms"] == 240 and s["pause_min_cut_ms"] == 120
        assert s["sliver_frames"] == cut.frames_for(240, fps)
    for fps in (25, 60):
        tiny, kept = plans[fps]
        # 180 ms and 70 ms remainders: dropped at both rates; a 300 ms remainder: kept at both
        assert sorted(w["i"] for d in tiny["dropped_slivers"] for w in d["lost_words"]) == [2, 3], fps
        assert 2 in [w["src_i"] for w in kept["words"]], fps
        for p in kept["pieces"]:
            assert p["frames"] * 1000 / fps >= 240, (fps, p)
    # the deprecated frame options still work, in frames of the cut's fps
    o = cut.resolve_opts("lesson", {"sliver_frames": 15})
    p60 = cut.plan_pieces(mm.words_doc(SPANS)["words"], KEEP, _db(), 60, o)
    assert p60["settings"]["sliver_frames"] == 15 and p60["settings"]["sliver_ms"] == 250


def test_deprecated_frame_flags_print_a_note_and_conflict_with_the_ms_flags(tmp_path, home, base_clip):
    src, words, kp = base_clip
    r = run_cli("cut.py", src, words, kp, "--out", tmp_path / "d", "--sliver-frames", "8", "--no-render")
    assert r.returncode == 0, r.stderr
    assert "--sliver-frames is deprecated" in r.stdout
    c = json.loads(Path(f"{tmp_path / 'd'}.cuts.json").read_text(encoding="utf-8"))
    assert c["settings"]["sliver_frames"] == 8 and c["settings"]["sliver_ms"] == 320
    r = run_cli("cut.py", src, words, kp, "--out", tmp_path / "e", "--sliver-ms", "300", "--no-render")
    assert r.returncode == 0 and json.loads(Path(f"{tmp_path / 'e'}.cuts.json").read_text(
        encoding="utf-8"))["settings"]["sliver_frames"] == 8  # 300 ms at 25 fps
    r = run_cli("cut.py", src, words, kp, "--out", tmp_path / "f", "--sliver-ms", "300", "--sliver-frames", "8",
                "--no-render")
    assert r.returncode == 2 and "not both" in r.stderr

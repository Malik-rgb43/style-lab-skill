"""tighten.py: joins closed to the true voice end, nothing clipped, placeholders untouched.

Synthetic media built here (numpy audio + ffmpeg video): a steady 100 Hz room tone at -55 dBFS and
400 Hz "words" at -12 dBFS whose ends decay exponentially (tau 40 ms, about 160 ms down to the
room tone + 8 dB), so the aligner's word end (the end of the flat part) is well before the true
end of the sound. The video is a luma ramp (Y = 16 + 2 * frame number, lossless) so every output
frame can be traced back to its source frame. The cut is written in cut.py's output shape.
"""
import json
import math
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

import tighten  # noqa: E402
from conftest import run_cli  # noqa: E402

SR = 48000
FPS = 30
FMS = 1000 / FPS
SPF = SR // FPS
W, H = 160, 90
ROOM_DB, WORD_DB, TAU = -55.0, -12.0, 40.0
DUR = 3000
A = (300, 800)    # aligned word A (flat part); its tail dies away about 160 ms later
B = (1560, 2060)  # aligned word B: about 600 ms after A's tail has died away


def _amp(db):
    return 10 ** (db / 20.0) * math.sqrt(2)


def track(dur_ms, words, extra=(), noisy_room=False, drift_db=0.0):
    """words: (start, flat_end) at WORD_DB with a decaying tail; extra: (start, end, db) flat 700 Hz
    sound with 2 ms ramps and no tail (an unlisted word, a breath)."""
    n = int(dur_ms * SR / 1000)
    t = np.arange(n) / SR
    env = np.zeros(n)
    ramp = int(0.005 * SR)
    for a, b in words:
        i0, i1 = int(a * SR / 1000), int(b * SR / 1000)
        env[i0:i0 + ramp] = np.linspace(0, 1, ramp)
        env[i0 + ramp:i1] = 1.0
        tail = np.exp(-(np.arange(n - i1) / SR * 1000) / TAU)
        tail[tail < 1e-4] = 0
        env[i1:] = np.maximum(env[i1:], tail)
    room = _amp(ROOM_DB) * np.sin(2 * np.pi * 100 * t)
    if noisy_room:  # single-slice jumps: +5 dB on about half the odd slices, -5 dB on every 20th
        g = np.zeros(n // 480 + 1)
        g[1::2] = np.where(np.random.default_rng(3).random(len(g[1::2])) < 0.5, 5.0, 0.0)
        g[::20] = -5.0
        room = room * np.repeat(10 ** (g / 20), 480)[:n]
    if drift_db:  # the room's level drifts +-drift_db over 0.7 s, plus 1 dB jitter on every slice
        k = np.arange(n // 480 + 1)
        g = drift_db * np.sin(2 * np.pi * k * 0.01 / 0.7) + np.random.default_rng(5).normal(0, 1.0, len(k))
        room = room * np.repeat(10 ** (g / 20), 480)[:n]
    mono = room + _amp(WORD_DB) * env * np.sin(2 * np.pi * 400 * t)
    r2 = int(0.002 * SR)
    for a, b, db in extra:
        i0, i1 = int(a * SR / 1000), int(b * SR / 1000)
        e = np.ones(i1 - i0)
        e[:r2], e[-r2:] = np.linspace(0, 1, r2), np.linspace(1, 0, r2)
        mono[i0:i1] += _amp(db) * e * np.sin(2 * np.pi * 700 * t[i0:i1])
    return np.stack([mono, mono], axis=1).astype(np.float32)


def _write_wav(path, audio):
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes((np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes())


def make_source(tmp, words, dur_ms=DUR, extra=(), noisy_room=False, drift_db=0.0):
    audio = track(dur_ms, words, extra, noisy_room, drift_db)
    wav = tmp / "src.wav"
    _write_wav(wav, audio)
    src = tmp / "src.mkv"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                    f"nullsrc=s={W}x{H}:r={FPS}:d={dur_ms / 1000},geq=lum='16+2*N':cb=128:cr=128",
                    "-i", str(wav), "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-preset", "ultrafast",
                    "-qp", "0", "-pix_fmt", "yuv420p", "-c:a", "pcm_s16le", str(src)], check=True)
    wav.unlink()
    return src, audio


def make_cut(tmp, stem_name, src, audio, pieces, words):
    """Write <stem>.mp4/.cuts.json/.words.json in cut.py's shape.

    pieces: {"f_in", "f_out"} (source frames) or {"ph": id, "frames", "tone": (a_ms, b_ms)}.
    words: (text, src_start_ms, src_end_ms, piece_index)."""
    stem = tmp / stem_name
    out_pieces, segs, vlab, graph, dst = [], [], [], [], 0
    for n, p in enumerate(pieces):
        if "ph" in p:
            nf = p["frames"]
            base = audio[int(p["tone"][0] * 48): int(p["tone"][1] * 48)]
            segs.append(np.tile(base, (nf * SPF // len(base) + 1, 1))[: nf * SPF])
            graph.append(f"color=c=black:s={W}x{H}:r={FPS}:d={nf / FPS + 1},trim=end_frame={nf},setsar=1[v{n}]")
            out_pieces.append({"id": n, "placeholder": p["ph"], "src_in_ms": p["tone"][0],
                               "src_out_ms": p["tone"][1], "dst_in_ms": round(dst * FMS, 3), "dst_f": dst,
                               "frames": nf, "w0": None, "w1": None, "text": f"[{p['ph']}]"})
        else:
            nf = p["f_out"] - p["f_in"]
            segs.append(audio[p["f_in"] * SPF: p["f_out"] * SPF])
            graph.append(f"[0:v]trim=start_frame={p['f_in']}:end_frame={p['f_out']},setpts=PTS-STARTPTS,"
                         f"setsar=1[v{n}]")
            ws = [k for k, w in enumerate(words) if w[3] == n]
            out_pieces.append({"id": n, "src_in_ms": round(p["f_in"] * FMS, 3), "src_out_ms": round(p["f_out"] * FMS, 3),
                               "dst_in_ms": round(dst * FMS, 3), "dst_f": dst, "frames": nf,
                               "f_in": p["f_in"], "f_out": p["f_out"], "w0": ws[0] if ws else -1,
                               "w1": ws[-1] if ws else -1,
                               "text": " ".join(words[k][0] for k in ws)})
        vlab.append(f"[v{n}]")
        dst += nf
    # the source is split once per source piece
    nsrc = sum(1 for p in pieces if "ph" not in p)
    split = f"[0:v]split={nsrc}" + "".join(f"[s{j}]" for j in range(nsrc))
    j = 0
    for n, p in enumerate(pieces):
        if "ph" not in p:
            graph[n] = graph[n].replace("[0:v]", f"[s{j}]", 1)
            j += 1
    fg = ";".join([split] + graph + ["".join(vlab) + f"concat=n={len(pieces)}:v=1:a=0[v]"])
    wav = tmp / (stem_name + ".wav")
    _write_wav(wav, np.concatenate(segs))
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-i", str(wav), "-filter_complex", fg,
                    "-map", "[v]", "-map", "1:a", "-c:v", "libx264", "-preset", "ultrafast", "-qp", "0",
                    "-pix_fmt", "yuv420p", "-r", str(FPS), "-c:a", "aac", "-b:a", "192k", "-frames:v", str(dst),
                    str(stem) + ".mp4"], check=True)
    wav.unlink()
    cw, phs = [], []
    for k, (txt, s0, s1, pi) in enumerate(words):
        pc = out_pieces[pi]
        off = pc["dst_in_ms"] - pc["src_in_ms"]
        cw.append({"i": k, "src_i": k, "w": txt, "start_ms": int(round(s0 + off)), "end_ms": int(round(s1 + off)),
                   "p": 0.99, "score": 0.9})
    for pc in out_pieces:
        if pc["w0"] is None:
            phs.append({"id": pc["placeholder"], "start": round(pc["dst_in_ms"] / 1000, 3),
                        "end": round((pc["dst_f"] + pc["frames"]) * FMS / 1000, 3), "note": "test"})
    cuts = {"source": str(src.resolve()), "fps": FPS, "frames": dst, "duration_s": round(dst / FPS, 3),
            "pieces": out_pieces, "joins": [], "placeholders": phs}
    (tmp / (stem_name + ".cuts.json")).write_text(json.dumps(cuts, ensure_ascii=False), encoding="utf-8")
    (tmp / (stem_name + ".words.json")).write_text(json.dumps({"words": cw}, ensure_ascii=False), encoding="utf-8")
    return str(stem)


# ---------------------------------------------------------------- measuring helpers

def load_mono(path):
    r = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", str(SR), "-f", "s16le", "-"],
                       capture_output=True, check=True)
    return np.frombuffer(r.stdout, dtype=np.int16).astype(np.float64) / 32768.0


def slices_db(mono):
    n = len(mono) // 480
    x = mono[: n * 480].reshape(n, 480)
    return 20 * np.log10(np.sqrt((x * x).mean(axis=1)) + 1e-9)


def frame_ids(path):
    """source frame number of every frame (the raw Y plane, no range conversion) and its mean luma."""
    r = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:v", "-f", "rawvideo", "-pix_fmt", "yuv420p", "-"],
                       capture_output=True, check=True)
    fr = np.frombuffer(r.stdout, dtype=np.uint8).reshape(-1, W * H * 3 // 2)[:, : W * H]
    y = fr.astype(np.float64).mean(axis=1)
    return np.round((y - 16) / 2).astype(int), y


def stream_durations(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,duration", "-of", "json", str(path)],
                       capture_output=True, text=True, check=True)
    return {s["codec_type"]: float(s["duration"]) for s in json.loads(r.stdout)["streams"]}


def thr_of(db):
    """the room tone + 4 dB: below it for 3 slices = the sound has died away."""
    return float(np.percentile(db, 10)) + 4


def decay_end_ms(db, after_ms):
    """first 10 ms slice after after_ms where the level has fallen below the room tone + 8 dB."""
    thr = thr_of(db)
    for j in range(int(after_ms // 10), len(db) - 1):
        if db[j] < thr and db[j + 1] < thr and db[j + 2] < thr:
            return j * 10
    raise AssertionError("no decay end")


def onset_ms(db, after_ms):
    thr = thr_of(db)
    for j in range(int(after_ms // 10), len(db) - 1):
        if db[j] >= thr and db[j + 1] >= thr:
            return j * 10
    raise AssertionError("no onset")


def max_quiet_run_ms(db, a_ms, b_ms):
    """longest run of slices below the room tone + 4 dB between two times (dead air)."""
    thr, run, best = thr_of(db), 0, 0
    for j in range(int(a_ms // 10), int(b_ms // 10)):
        run = run + 1 if db[j] < thr else 0
        best = max(best, run)
    return best * 10


def join_gap(out, audio):
    """(A's decay end, B's onset) on the output timeline: both measured on the source (in the
    output they touch, there is no quiet between them to measure) and moved by each clip's offset."""
    cuts = _read(out, ".cuts.json")
    pa = [p for p in cuts["pieces"] if p["text"] == "A"][0]
    pb = [p for p in cuts["pieces"] if p["text"] == "B"][0]
    src_db = slices_db(audio.mean(axis=1).astype(np.float64))
    end_a, on_b = decay_end_ms(src_db, A[1]), onset_ms(src_db, B[0] - 200)
    return end_a - (pa["f_in"] - pa["dst_f"]) * FMS, on_b - (pb["f_in"] - pb["dst_f"]) * FMS


def _read(stem, ext):
    return json.loads(Path(stem + ext).read_text(encoding="utf-8"))


# ---------------------------------------------------------------- fixtures

@pytest.fixture(scope="module")
def straight(tmp_path_factory):
    """one piece = the whole source: A, a 600 ms pause after A's tail, B."""
    tmp_path = tmp_path_factory.mktemp("straight")
    src, audio = make_source(tmp_path, [A, B])
    stem = make_cut(tmp_path, "cut", src, audio, [{"f_in": 0, "f_out": 90}],
                    [("A", *A, 0), ("B", *B, 0)])
    return stem, src, audio


@pytest.fixture(scope="module")
def straight_out(straight):
    stem, src, audio = straight
    out = str(Path(stem).parent / "tight")
    res = tighten.tighten(stem, out)
    return stem, out, res, audio


# ---------------------------------------------------------------- tests

def test_long_pause_closes_to_the_decay_end(straight_out):
    stem, out, res, audio = straight_out
    end_a, on_b = join_gap(out, audio)
    assert -10 <= on_b - end_a <= FMS + 10, (end_a, on_b)
    db = slices_db(load_mono(out + ".mp4"))
    assert max_quiet_run_ms(db, A[0], on_b + 100) <= FMS + 10  # no dead air left at the join
    assert not Path(out + ".tmp.mp4").exists()
    cuts = _read(out, ".cuts.json")
    assert cuts["frames"] < 90 - 15  # about 600 ms closed
    assert res["frames"] == cuts["frames"]
    assert [w["w"] for w in _read(out, ".words.json")["words"]] == ["A", "B"]


def test_decaying_tail_is_never_cut(straight_out):
    stem, out, res, audio = straight_out
    cuts = _read(out, ".cuts.json")
    edges = cuts["edge_check"]
    assert len(edges) == 4 * sum(1 for p in cuts["pieces"] if p["w0"] is not None)  # start/end x 120/30 ms
    assert all(e["ok"] and e["uncovered_ms"] <= 2.5 for e in edges), edges
    assert all(abs(e["diff_db"]) < 0.05 for e in edges if e["win_ms"] == 120), edges
    assert cuts["edge_bad"] == [] and res["edge_bad"] == []
    # the whole of A, its tail included, is in the output at the source level
    src_db = slices_db(audio.mean(axis=1).astype(np.float64))
    out_db = slices_db(load_mono(out + ".mp4"))
    end_a = decay_end_ms(src_db, A[1])
    k0, k1 = A[0] // 10, end_a // 10
    assert np.abs(out_db[k0:k1] - src_db[k0:k1]).max() < 1.0
    # and so is B, its tail included, where the output put it
    pb = [p for p in cuts["pieces"] if p["text"] == "B"][0]
    off = pb["src_in_ms"] - pb["dst_in_ms"]
    end_b = decay_end_ms(src_db, B[1])
    j0, j1 = B[0] // 10, end_b // 10 + 1
    d0 = int(round((B[0] - off) / 10))
    # from the second slice on: AAC smears the 5 ms onset ramp of the first one (the edge check
    # above measures the onset on the mix itself)
    assert np.abs(out_db[d0 + 1:d0 + (j1 - j0)] - src_db[j0 + 1:j1]).max() < 1.0


def test_lip_sync_is_exact(straight_out):
    stem, out, res, _ = straight_out
    cuts = _read(out, ".cuts.json")
    ids, _ = frame_ids(out + ".mp4")
    assert len(ids) == cuts["frames"]
    for p in cuts["pieces"]:
        off_f = p["f_in"] - p["dst_f"]
        got = ids[p["dst_f"]: p["dst_f"] + p["frames"]]
        assert list(got) == list(range(p["f_in"], p["f_out"])), p
    # B's sound moved by exactly the same whole number of frames as its picture
    pb = [p for p in cuts["pieces"] if p["text"] == "B"][0]
    db = slices_db(load_mono(out + ".mp4"))
    src_db = slices_db(straight_out[3].mean(axis=1).astype(np.float64))
    k = int(round((B[0] - (pb["f_in"] - pb["dst_f"]) * FMS) / 10))
    assert np.abs(db[k + 1:k + 20] - src_db[B[0] // 10 + 1:B[0] // 10 + 20]).max() < 1.0
    wb = [w for w in _read(out, ".words.json")["words"] if w["w"] == "B"][0]
    assert abs(wb["start_ms"] - (B[0] - (pb["f_in"] - pb["dst_f"]) * FMS)) <= 1


def test_output_av_durations_match(straight_out):
    stem, out, res, _ = straight_out
    d = stream_durations(out + ".mp4")
    assert abs(d["video"] - d["audio"]) < 1 / FPS, d
    assert abs(d["video"] - _read(out, ".cuts.json")["frames"] / FPS) < 1 / FPS


def test_short_pause_is_left_alone(tmp_path):
    a2 = (A[1] + 160 + 150, A[1] + 160 + 150 + 300)  # 150 ms after A's tail: below --min-pause-ms
    src, audio = make_source(tmp_path, [A, a2])
    stem = make_cut(tmp_path, "cut", src, audio, [{"f_in": 0, "f_out": 90}], [("A", *A, 0), ("A2", *a2, 0)])
    res = tighten.tighten(stem, str(tmp_path / "tight"))
    assert res["frames"] == 90
    assert res["edge_bad"] == []


@pytest.fixture(scope="module")
def with_placeholder(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("placeholder")
    src, audio = make_source(tmp_path, [A, B])
    pieces = [{"f_in": 0, "f_out": 36}, {"ph": "demo", "frames": 30, "tone": (2400, 2900)}, {"f_in": 42, "f_out": 90}]
    stem = make_cut(tmp_path, "cut", src, audio, pieces, [("A", *A, 0), ("B", *B, 2)])
    return stem


def test_placeholder_keeps_its_length(with_placeholder, tmp_path):
    out = str(tmp_path / "tight")
    res = tighten.tighten(with_placeholder, out)
    cuts = _read(out, ".cuts.json")
    assert len(cuts["placeholders"]) == 1
    ph = cuts["placeholders"][0]
    assert abs((ph["end"] - ph["start"]) - 1.0) < 1e-6
    php = [p for p in cuts["pieces"] if p["w0"] is None]
    assert len(php) == 1 and php[0]["frames"] == 30 and php[0]["placeholder"] == "demo"
    # the air on both sides of the placeholder is kept: nothing else changed
    assert cuts["frames"] == 36 + 30 + 48
    ids, y = frame_ids(out + ".mp4")
    assert np.all(y[php[0]["dst_f"]: php[0]["dst_f"] + 30] < 20)  # black
    assert res["edge_bad"] == []


def test_no_placeholders_drops_them_and_closes_the_join(with_placeholder, tmp_path):
    out = str(tmp_path / "tight")
    res = tighten.tighten(with_placeholder, out, keep_placeholders=False)
    cuts = _read(out, ".cuts.json")
    assert cuts["placeholders"] == []
    assert all(p["w0"] is not None for p in cuts["pieces"])
    assert cuts["frames"] < 36 + 48 - 6
    assert res["edge_bad"] == []


def test_cli_clean_exits_0(straight, tmp_path):
    stem, _, _ = straight
    r = run_cli("tighten.py", stem, "--out", tmp_path / "cli", "--min-pause-ms", "250")
    assert r.returncode == 0, r.stdout + r.stderr
    assert Path(str(tmp_path / "cli") + ".mp4").is_file()


def test_cli_clipped_end_exits_1_and_names_the_piece(tmp_path):
    src, audio = make_source(tmp_path, [A, B])
    # piece 0's out-point (frame 26 = 867 ms) is inside A's decaying tail (it dies away about 960 ms)
    pieces = [{"f_in": 0, "f_out": 26}, {"f_in": 42, "f_out": 90}]
    stem = make_cut(tmp_path, "cut", src, audio, pieces, [("A", *A, 0), ("B", *B, 1)])
    r = run_cli("tighten.py", stem, "--out", tmp_path / "clip")
    assert r.returncode == 1, r.stdout + r.stderr
    text = r.stdout + r.stderr
    assert "piece 0 end" in text and "A" in text, text
    bad = _read(str(tmp_path / "clip"), ".cuts.json")["edge_bad"]
    assert {(e["piece"], e["edge"]) for e in bad} == {(0, "end")}
    # the clipped tail is reported in ms, apart from the dB sign: (nearly) the whole window is not played
    assert all(e["uncovered_ms"] >= 0.9 * min(e["win_ms"], 100) for e in bad), bad


def test_cli_usage_errors(tmp_path):
    r = run_cli("tighten.py", tmp_path / "missing", "--out", tmp_path / "x")
    assert r.returncode == 1
    assert "error" in (r.stdout + r.stderr).lower()


# ---------------------------------------------------------------- nothing swallowed

def _one_piece(tmp_path, words, extra=(), aligned=None):
    src, audio = make_source(tmp_path, words, extra=extra)
    aligned = aligned or [(f"W{k}", a, b) for k, (a, b) in enumerate(words)]
    stem = make_cut(tmp_path, "cut", src, audio, [{"f_in": 0, "f_out": 90}], [(w, a, b, 0) for w, a, b in aligned])
    return stem, audio


def test_gap_ms_leaves_that_much_room_tone(straight):
    stem, _, _ = straight
    out = str(Path(stem).parent / "gap100")
    res = tighten.tighten(stem, out, gap_ms=100)
    end_a, on_b = join_gap(out, straight[2])
    assert 90 <= on_b - end_a <= 100 + FMS + 10, (end_a, on_b)
    assert res["edge_bad"] == []


def test_voice_that_never_dies_away_is_never_cut(tmp_path):
    # A is sounding from 300 to 1400 ms but the aligner ends it at 800: 600 ms past the search limit
    stem, _ = _one_piece(tmp_path, [(300, 1400), (1800, 2300)], aligned=[("A", 300, 800), ("B", 1800, 2300)])
    r = run_cli("tighten.py", stem, "--out", tmp_path / "t")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "voice not died away" in r.stdout
    cuts = _read(str(tmp_path / "t"), ".cuts.json")
    assert cuts["frames"] == 90  # the pause after A is not closed
    v = cuts["voice_not_died_away"]
    assert [(x["cut_piece"], x["edge"], x["words"]) for x in v] == [(0, "end", "A")]
    assert cuts["edge_bad"] == []


@pytest.fixture
def sound_in_gap(tmp_path):
    # an unlisted -12 dB sound (a word the transcript does not have) alone in the pause between A and B
    return _one_piece(tmp_path, [A, (1900, 2400)], extra=[(1250, 1400, -12.0)],
                      aligned=[("A", *A), ("B", 1900, 2400)])


def test_sound_in_the_pause_keeps_the_join(sound_in_gap, tmp_path):
    stem, _ = sound_in_gap
    res = tighten.tighten(stem, str(tmp_path / "t"))
    assert res["frames"] == 90
    assert len(res["sound_in_pause"]) == 1
    assert 1250 <= res["sound_in_pause"][0]["sound_at_ms"] < 1400
    assert res["edge_bad"] == []


def test_missing_audit_catches_dropped_sound(sound_in_gap, tmp_path, monkeypatch):
    stem, _ = sound_in_gap
    monkeypatch.setattr(tighten, "_sound_between", lambda *a: [])  # a planner that misses the sound
    res = tighten.tighten(stem, str(tmp_path / "t"))
    assert res["frames"] < 90
    miss = [e for e in res["edge_bad"] if e["edge"] == "missing"]
    assert len(miss) == 1 and miss[0]["src_ms"][0] <= 1250 and miss[0]["src_ms"][1] >= 1400, res["edge_bad"]


def test_soft_onset_below_the_voice_threshold_is_kept(tmp_path):
    # a 40 ms breathy onset at about room tone + 6 dB right before B (below the +8 dB voice threshold)
    stem, audio = _one_piece(tmp_path, [A, B], extra=[(B[0] - 40, B[0], -50.0)], aligned=[("A", *A), ("B", *B)])
    out = str(tmp_path / "t")
    res = tighten.tighten(stem, out)
    assert res["edge_bad"] == []
    cuts = _read(out, ".cuts.json")
    pb = [p for p in cuts["pieces"] if p["text"] == "B"][0]
    assert pb["voice_src_ms"][0] == B[0] - 40
    assert cuts["frames"] < 90 - 12
    off = pb["f_in"] - pb["dst_f"]
    db = slices_db(load_mono(out + ".mp4"))
    k = int(round((B[0] - 40 - off * FMS) / 10))
    assert np.all(db[k:k + 4] > -52.5), db[k - 2:k + 6]  # the breath is there, not room tone (-55)


def test_refuses_out_equal_to_stem(straight):
    stem, _, _ = straight
    with pytest.raises(ValueError, match="differ"):
        tighten.tighten(stem, stem + ".mp4")
    r = run_cli("tighten.py", stem, "--out", stem)
    assert r.returncode == 1 and "differ" in r.stderr


def test_output_is_atomic(straight, monkeypatch):
    stem, _, _ = straight
    out = str(Path(stem).parent / "atomic")
    assert not Path(out + ".tmp.mp4").exists()
    real = tighten._run

    def failing_mux(cmd):
        if "concat" in cmd:
            real(cmd)  # let ffmpeg write the partial file, then report a failure
            return subprocess.CompletedProcess(cmd, 1, "", "boom")
        return real(cmd)

    monkeypatch.setattr(tighten, "_run", failing_mux)
    with pytest.raises(RuntimeError, match="mux"):
        tighten.tighten(stem, out)
    for ext in (".mp4", ".cuts.json", ".words.json", ".tmp.mp4", ".tmp.cuts.json", ".tmp.words.json"):
        assert not Path(out + ext).exists(), ext
    monkeypatch.setattr(tighten, "_run", real)
    tighten.tighten(stem, out)
    assert Path(out + ".mp4").is_file() and not Path(out + ".tmp.mp4").exists()


# ---------------------------------------------------------------- round 2

def test_quiet_unlisted_word_holds_the_join(tmp_path):
    # an unlisted word at about room tone + 6 dB: below the +8 dB voice level, above the +4 dB edge level
    stem, _ = _one_piece(tmp_path, [A, (1900, 2400)], extra=[(1100, 1300, -50.0)],
                         aligned=[("A", *A), ("B", 1900, 2400)])
    res = tighten.tighten(stem, str(tmp_path / "t"))
    assert res["frames"] == 90
    assert len(res["sound_in_pause"]) == 1 and 1100 <= res["sound_in_pause"][0]["sound_at_ms"] < 1300
    assert res["edge_bad"] == []


def test_noisy_room_single_slice_jumps_still_close(tmp_path):
    src, audio = make_source(tmp_path, [A, B], noisy_room=True)
    stem = make_cut(tmp_path, "cut", src, audio, [{"f_in": 0, "f_out": 90}], [("A", *A, 0), ("B", *B, 0)])
    res = tighten.tighten(stem, str(tmp_path / "t"))
    assert res["sound_in_pause"] == [] and res["voice_not_died_away"] == []
    assert res["frames"] < 90 - 12
    assert res["edge_bad"] == []


def _three(tmp_path, words, aligned, extra=()):
    """pieces 0-1500 ms (A), 2000-2300 ms (room tone only, no words), 2500-3500 ms (B)."""
    src, audio = make_source(tmp_path, words, dur_ms=4000, extra=extra)
    pieces = [{"f_in": 0, "f_out": 45}, {"f_in": 60, "f_out": 69}, {"f_in": 75, "f_out": 105}]
    stem = make_cut(tmp_path, "cut", src, audio, pieces, [(w, a, b, pi) for w, a, b, pi in aligned])
    return stem


B3 = (2700, 3200)


def test_non_consecutive_pieces_close_over_a_room_tone_piece(tmp_path):
    stem = _three(tmp_path, [A, B3], [("A", *A, 0), ("B", *B3, 2)])
    out = str(tmp_path / "t")
    res = tighten.tighten(stem, out)
    cuts = _read(out, ".cuts.json")
    assert [d["cut_piece"] for d in cuts["dropped_air_pieces"]] == [1]
    assert [j["type"] for j in cuts["joins"]] == ["stack"]
    assert res["frames"] < 45 + 30 - 15
    assert cuts["added_check"] == [] and res["edge_bad"] == []


def test_held_join_over_a_room_tone_piece_is_the_cuts_own(tmp_path):
    # an unlisted sound in piece 0's air after A: the join is held at piece 0's out / piece 2's in
    stem = _three(tmp_path, [A, B3], [("A", *A, 0), ("B", *B3, 2)], extra=[(1100, 1300, -12.0)])
    out = str(tmp_path / "t")
    res = tighten.tighten(stem, out)
    cuts = _read(out, ".cuts.json")
    assert len(res["sound_in_pause"]) == 1
    assert [j["type"] for j in cuts["joins"]] == ["keep"]
    assert res["frames"] == 45 + 30  # both pieces whole, only the room-tone piece is gone
    p0, p2 = cuts["pieces"]
    assert (p0["f_in"], p0["f_out"], p2["f_in"], p2["f_out"]) == (0, 45, 75, 105)
    assert cuts["added_check"] == [] and res["edge_bad"] == []


def test_voice_not_died_over_a_room_tone_piece_keeps_the_pieces(tmp_path):
    stem = _three(tmp_path, [(300, 1440), B3], [("A", 300, 800, 0), ("B", *B3, 2)])
    res = tighten.tighten(stem, str(tmp_path / "t"))
    assert [(v["cut_piece"], v["edge"]) for v in res["voice_not_died_away"]] == [(0, "end")]
    assert res["frames"] == 45 + 30
    assert res["edge_bad"] == []


def test_added_audit_flags_source_outside_the_pieces(tmp_path, monkeypatch):
    stem = _three(tmp_path, [A, B3], [("A", *A, 0), ("B", *B3, 2)])
    real = tighten.plan

    def leaky(*a, **k):  # a planner that lets piece 0's audio run 300 ms past its out-point
        pl = real(*a, **k)
        s = [it["S"] for it in pl["items"] if it["kind"] == "seg"][0]
        s.a1_tl = s.P["hi"] - s.off * pl["fms"] + 300
        return pl

    monkeypatch.setattr(tighten, "plan", leaky)
    res = tighten.tighten(stem, str(tmp_path / "t"))
    added = [e for e in res["edge_bad"] if e["edge"] == "added"]
    assert len(added) == 1 and added[0]["piece"] == 0, res["edge_bad"]


def test_missing_ffmpeg_exits_3(straight, tmp_path, monkeypatch):
    import cut
    stem, _, _ = straight
    monkeypatch.setattr(cut.shutil, "which", lambda name: None)
    assert tighten.main([stem, "--out", str(tmp_path / "x")]) == 3


def test_refuses_out_equal_to_the_raw_source_stem(straight):
    stem, src, _ = straight
    with pytest.raises(ValueError, match="raw source"):
        tighten.tighten(stem, str(src)[: -len(src.suffix)])


def test_gain_balance_without_gains_is_a_clean_error(straight, tmp_path):
    stem, _, _ = straight
    cuts = _read(stem, ".cuts.json")
    cuts["channel_balance"] = "gain"
    bad = tmp_path / "bad"
    Path(str(bad) + ".cuts.json").write_text(json.dumps(cuts), encoding="utf-8")
    Path(str(bad) + ".words.json").write_text(Path(stem + ".words.json").read_text(encoding="utf-8"), encoding="utf-8")
    Path(str(bad) + ".mp4").write_bytes(Path(stem + ".mp4").read_bytes())
    r = run_cli("tighten.py", bad, "--out", tmp_path / "o")
    assert r.returncode == 1 and "gains" in r.stderr and "Traceback" not in r.stderr


# ---------------------------------------------------------------- round 3

def above_room(x):
    """level of an added tone so that room tone + tone = room + x dB."""
    return ROOM_DB + 10 * math.log10(10 ** (x / 10) - 1)


def test_short_loud_click_holds_the_join(tmp_path, monkeypatch):
    # a 20 ms click at -20 dBFS (35 dB above the room) on exactly 2 slices, alone in the pause
    stem, _ = _one_piece(tmp_path, [A, (1900, 2400)], extra=[(1150, 1170, -20.0)],
                         aligned=[("A", *A), ("B", 1900, 2400)])
    res = tighten.tighten(stem, str(tmp_path / "t"))
    assert res["frames"] == 90
    assert [round(x["sound_at_ms"]) for x in res["sound_in_pause"]] == [1150]
    assert res["edge_bad"] == []
    monkeypatch.setattr(tighten, "_sound_between", lambda *a: [])  # forced: the audit must still see it
    res = tighten.tighten(stem, str(tmp_path / "t2"))
    miss = [e for e in res["edge_bad"] if e["edge"] == "missing"]
    assert len(miss) == 1 and miss[0]["src_ms"][0] <= 1150 and miss[0]["src_ms"][1] >= 1170, res["edge_bad"]


def test_drifting_room_still_closes_its_joins(tmp_path):
    src, audio = make_source(tmp_path, [A, B], drift_db=3.0)
    stem = make_cut(tmp_path, "cut", src, audio, [{"f_in": 0, "f_out": 90}], [("A", *A, 0), ("B", *B, 0)])
    out = str(tmp_path / "t")
    res = tighten.tighten(stem, out)
    assert res["sound_in_pause"] == [] and res["voice_not_died_away"] == [] and res["room_noise_warning"] is None
    assert [j["type"] for j in res["joins"]] == ["stack"]
    assert res["edge_bad"] == []
    # dead air left: B's onset vs A's tail reaching the room's nominal level (an analytic 198 ms after A)
    cuts = _read(out, ".cuts.json")
    pb = [p for p in cuts["pieces"] if p["text"] == "B"][0]
    on_b = B[0] - (pb["f_in"] - pb["dst_f"]) * FMS
    end_a = A[1] + TAU * (WORD_DB - ROOM_DB) / (20 * math.log10(math.e))
    assert -40 <= on_b - end_a <= FMS + 10, (end_a, on_b)


def test_soft_word_in_a_drifting_room_is_still_held(tmp_path):
    src, audio = make_source(tmp_path, [A, (1900, 2400)], extra=[(1150, 1350, above_room(6))], drift_db=3.0)
    stem = make_cut(tmp_path, "cut", src, audio, [{"f_in": 0, "f_out": 90}], [("A", *A, 0), ("B", 1900, 2400, 0)])
    res = tighten.tighten(stem, str(tmp_path / "t"))
    assert res["frames"] == 90
    assert len(res["sound_in_pause"]) == 1 and 1150 <= res["sound_in_pause"][0]["sound_at_ms"] < 1350
    assert res["edge_bad"] == []


def test_room_noise_warning_when_holds_are_noise_like(tmp_path):
    # a sound barely above the soft threshold (room + 5 dB) holds the only join: noise-like
    stem, _ = _one_piece(tmp_path, [A, (1900, 2400)], extra=[(1150, 1350, above_room(5))],
                         aligned=[("A", *A), ("B", 1900, 2400)])
    r = run_cli("tighten.py", stem, "--out", tmp_path / "t")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "warning:" in r.stdout and "room-noise" in r.stdout
    cuts = _read(str(tmp_path / "t"), ".cuts.json")
    assert cuts["room_noise_warning"] and cuts["sound_in_pause"][0]["noise_like"] is True


def test_short_window_catches_a_soft_onset_clipped_by_the_cut(tmp_path):
    # a breath at room + 6 dB from 1840 ms before B (1900 ms); the cut's in-point (frame 56 = 1867 ms)
    # is inside it. The 120 ms window is dominated by the loud word; the 30 ms one is not.
    src, audio = make_source(tmp_path, [A, (1900, 2400)], extra=[(1840, 1900, above_room(6))])
    stem = make_cut(tmp_path, "cut", src, audio, [{"f_in": 0, "f_out": 42}, {"f_in": 56, "f_out": 90}],
                    [("A", *A, 0), ("B", 1900, 2400, 1)])
    res = tighten.tighten(stem, str(tmp_path / "t"))
    bad = [(e["piece"], e["edge"], e["win_ms"]) for e in res["edge_bad"]]
    assert (1, "start", 30) in bad, res["edge_bad"]
    row = [e for e in res["edge_bad"] if e["piece"] == 1 and e["edge"] == "start"][0]
    assert row["uncovered_ms"] >= 20

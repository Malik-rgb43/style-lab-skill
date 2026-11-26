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


def track(dur_ms, words):
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
    mono = _amp(ROOM_DB) * np.sin(2 * np.pi * 100 * t) + _amp(WORD_DB) * env * np.sin(2 * np.pi * 400 * t)
    return np.stack([mono, mono], axis=1).astype(np.float32)


def _write_wav(path, audio):
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes((np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes())


def make_source(tmp, words, dur_ms=DUR):
    audio = track(dur_ms, words)
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
                               "f_in": p["f_in"], "f_out": p["f_out"], "w0": ws[0], "w1": ws[-1],
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
    return float(np.percentile(db, 10)) + 8


def decay_end_ms(db, after_ms):
    """first 10 ms slice after after_ms where the level has fallen below the room tone + 8 dB."""
    thr = thr_of(db)
    for j in range(int(after_ms // 10), len(db) - 1):
        if db[j] < thr and db[j + 1] < thr:
            return j * 10
    raise AssertionError("no decay end")


def onset_ms(db, after_ms):
    thr = thr_of(db)
    for j in range(int(after_ms // 10), len(db) - 1):
        if db[j] >= thr and db[j + 1] >= thr:
            return j * 10
    raise AssertionError("no onset")


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
    stem, out, res, _ = straight_out
    db = slices_db(load_mono(out + ".mp4"))
    end_a = decay_end_ms(db, A[1])
    on_b = onset_ms(db, end_a)
    assert -10 <= on_b - end_a <= FMS + 10, (end_a, on_b)
    cuts = _read(out, ".cuts.json")
    assert cuts["frames"] < 90 - 15  # about 600 ms closed
    assert res["frames"] == cuts["frames"]
    assert [w["w"] for w in _read(out, ".words.json")["words"]] == ["A", "B"]


def test_decaying_tail_is_never_cut(straight_out):
    stem, out, res, audio = straight_out
    cuts = _read(out, ".cuts.json")
    edges = cuts["edge_check"]
    assert len(edges) == 2 * sum(1 for p in cuts["pieces"] if p["w0"] is not None)
    assert all(abs(e["diff_db"]) < 0.05 for e in edges), edges
    assert cuts["edge_bad"] == [] and res["edge_bad"] == []
    # the whole of A, its tail included, is in the output at the source level
    src_db = slices_db(audio.mean(axis=1).astype(np.float64))
    out_db = slices_db(load_mono(out + ".mp4"))
    end_a = decay_end_ms(src_db, A[1])
    k0, k1 = A[0] // 10, end_a // 10 + 1
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
    on_b = onset_ms(db, decay_end_ms(db, A[1]))
    assert abs(on_b - (B[0] - (pb["f_in"] - pb["dst_f"]) * FMS)) <= 10
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
    assert [(e["piece"], e["edge"]) for e in bad] == [(0, "end")]


def test_cli_usage_errors(tmp_path):
    r = run_cli("tighten.py", tmp_path / "missing", "--out", tmp_path / "x")
    assert r.returncode == 1
    assert "error" in (r.stdout + r.stderr).lower()

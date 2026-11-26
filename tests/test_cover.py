import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from conftest import run_cli
import cover
import palette

sys.path.insert(0, str(Path(__file__).parent / "fixtures"))


def gray_of(rgb):
    return (rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)).astype(np.uint8)


def frame(hexc, seed=1, h=90, w=160):
    """A textured frame around one colour (the texture keeps it sharp)."""
    rng = np.random.default_rng(seed)
    base = np.array([int(hexc[i:i + 2], 16) for i in (1, 3, 5)], dtype=np.float32)
    checker = ((np.indices((h, w)).sum(axis=0) // 4) % 2)[..., None] * 24.0
    img = base + checker + rng.normal(0, 3, (h, w, 3))
    return np.clip(img, 0, 255).astype(np.uint8)


def blur(rgb, k=15):
    x = rgb.astype(np.float32)
    for _ in range(2):
        for ax in (0, 1):
            x = sum(np.roll(x, s, axis=ax) for s in range(-(k // 2), k // 2 + 1)) / k
    return np.clip(x, 0, 255).astype(np.uint8)


def violet_fp():
    L, C, _ = palette.hex_to_oklch("#5b3fd0")
    return {"trained_on": ["a"], "layers": {"colour": {"metrics": {
        "dominant_L": {"median": L}, "dominant_chroma": {"median": C}}}}}


def test_score_prefers_on_palette_sharp_frame():
    fp = violet_fp()
    on = frame("#5b3fd0")
    off = frame("#e8d020")
    s_on, _ = cover.score_frame(on, gray_of(on), fp, None)
    s_off, _ = cover.score_frame(off, gray_of(off), fp, None)
    assert s_on > s_off
    bl = blur(on)
    s_bl, why = cover.score_frame(bl, gray_of(bl), fp, None)
    assert s_bl < s_on and why["sharpness"] < 0.5
    assert 0 <= s_on <= 1


def test_score_uses_roles_look_text_and_face_when_known():
    fp = violet_fp()
    fp["layers"]["colour"]["palette"] = {"brand": "#5b3fd0"}
    fp["layers"]["text"] = {"metrics": {"text_share": {"median": 0.8}}}
    fp["layers"]["broll"] = {"metrics": {"face_share": {"median": 0.9}}}
    f = frame("#5b3fd0")
    look = {"grain": {"median": 0.02}, "tone": {"median": 0.4}}
    full, why = cover.score_frame(f, gray_of(f), fp, look, faces=True, text=True)
    none, _ = cover.score_frame(f, gray_of(f), fp, look, faces=False, text=False)
    assert {"palette", "look", "text", "face", "sharpness"} <= set(why)
    assert full > none
    # unknown detectors (None) are left out, not penalised
    _, why2 = cover.score_frame(f, gray_of(f), fp, None)
    assert "face" not in why2 and "text" not in why2 and "look" not in why2


def test_pick_one_per_video():
    c = [{"video": "a", "t": 1, "score": 0.9}, {"video": "a", "t": 2, "score": 0.8},
         {"video": "a", "t": 3, "score": 0.7}, {"video": "b", "t": 1, "score": 0.6},
         {"video": "b", "t": 2, "score": 0.5}, {"video": "c", "t": 1, "score": 0.4}]
    got = cover.pick(c, 3)
    assert [(x["video"], x["score"]) for x in got] == [("a", 0.9), ("b", 0.6), ("c", 0.4)]
    assert len(cover.pick(c, 2)) == 2 and cover.pick([], 3) == [] and len(cover.pick(c, 10)) == 3


def test_sample_times_skip_edges_and_cuts():
    ts = cover.sample_times(10.0, 25.0, [5.0], 8)
    assert all(0.5 <= t <= 9.5 for t in ts)
    assert all(abs(t - 5.0) > 3 / 25 for t in ts)
    assert cover.sample_times(0.8, 25.0, [], 5) == []


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
    lst = run_cli("styles.py", "list")
    assert str(out) in lst.stdout
    assert json.loads(run_cli("styles.py", "show", "x1").stdout)["cover_path"] == str(out)
    # a portrait image is scaled on its long (vertical) side too
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


def make_pack(tmp_path, fingerprint=None):
    from make_media import make_clip
    pack = tmp_path / "pack"
    for vid in ("v1", "v2"):
        clip = make_clip(tmp_path / f"{vid}.mkv", [(300, 2200)], 3000)
        d = pack / "analysis" / vid
        d.mkdir(parents=True)
        (d / "measure.json").write_text(json.dumps({"source": str(clip), "cuts": []}), encoding="utf-8")
    if fingerprint:
        (pack / "fingerprint.json").write_text(json.dumps(fingerprint), encoding="utf-8")
    return pack


def test_candidates_without_fingerprint(tmp_path):
    pack = make_pack(tmp_path)
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
    # the script never sets a cover
    assert not (pack / "cover.jpg").exists()


def test_candidates_with_fingerprint_and_missing_source(tmp_path):
    pack = make_pack(tmp_path, violet_fp())
    (pack / "analysis" / "v2" / "measure.json").write_text(json.dumps({"source": "gone.mkv"}), encoding="utf-8")
    fpj = json.loads((pack / "fingerprint.json").read_text(encoding="utf-8"))
    fpj["trained_on"] = ["v1", "v2"]
    (pack / "fingerprint.json").write_text(json.dumps(fpj), encoding="utf-8")
    r = run_cli("cover.py", "candidates", pack, "--n", "3", "--per-video", "3")
    assert r.returncode == 0, r.stderr
    doc = json.loads((pack / "cover-candidates.json").read_text(encoding="utf-8"))
    assert "fingerprint" not in doc["not_run"] and "video:v2:no_source" in doc["not_run"]
    assert [c["video"] for c in doc["candidates"]] == ["v1"]
    assert "palette" in doc["candidates"][0]["why"]


def test_candidates_bad_pack(tmp_path):
    assert run_cli("cover.py", "candidates", tmp_path / "none").returncode == 2
    (tmp_path / "empty").mkdir()
    assert run_cli("cover.py", "candidates", tmp_path / "empty").returncode == 2

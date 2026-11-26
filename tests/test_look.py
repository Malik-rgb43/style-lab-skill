import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

import fingerprint as fp
import look
from conftest import run_cli


# ---------------------------------------------------------------- synthetic frames

def _rgb(gray):
    return np.repeat(np.clip(gray, 0, 255)[..., None], 3, axis=2).astype(np.uint8)


def _glyphs(h=180, w=480, x0=60, y0=60, size=40, sw=6):
    """A line of block letters (H, T, L shapes) as a bool mask."""
    m = np.zeros((h, w), bool)
    x = x0
    for k in range(6):
        if k % 3 == 0:
            m[y0:y0 + size, x:x + sw] = True
            m[y0:y0 + size, x + 24:x + 24 + sw] = True
            m[y0 + 17:y0 + 17 + sw, x:x + 30] = True
        elif k % 3 == 1:
            m[y0:y0 + sw, x:x + 30] = True
            m[y0:y0 + size, x + 12:x + 12 + sw] = True
        else:
            m[y0:y0 + size, x:x + sw] = True
            m[y0 + size - sw:y0 + size, x:x + 28] = True
        x += 46
    return m


def _bbox(m):
    ys, xs = np.nonzero(m)
    H, W = m.shape
    return [xs.min() / W, ys.min() / H, (xs.max() + 1 - xs.min()) / W, (ys.max() + 1 - ys.min()) / H]


# ---------------------------------------------------------------- pure metrics

def test_grain_std_and_size():
    # a grain field of std 6 with the correlation of a sigma-1 Gaussian blur, on mid gray
    rng = np.random.default_rng(3)
    nz = look._blur(rng.normal(0, 1, (288, 384)).astype(np.float32), 1.0)
    nz = nz / nz.std() * 6.0
    g = look.grain(128 + nz)
    assert 5.0 <= g["std"] <= 7.0
    assert 1.5 <= g["size_px"] <= 4.0
    # per-pixel (white) noise reads about 1 px; a clean frame has no grain
    white = look.grain(128 + rng.normal(0, 6, (288, 384)))
    assert 5.0 <= white["std"] <= 7.0 and white["size_px"] < 1.5
    clean = look.grain(np.full((288, 384), 128.0))
    assert clean["std"] < 0.01 and clean["size_px"] is None
    # a gentle gradient is not grain, and black blocks (clipped grain) are skipped
    ramp = np.tile(np.linspace(60, 200, 384), (288, 1)) + nz
    ramp[:, :96] = 0
    assert 5.0 <= look.grain(ramp)["std"] <= 7.0


def test_vignette_detected():
    h, w = 180, 320
    r, _, _ = look._radius(h, w)
    v = look.vignette(200 * (1 - 0.35 * r ** 2))
    assert 0.30 <= v["falloff"] <= 0.40 and v["shape"] == "round"
    flat = look.vignette(np.full((h, w), 140.0))
    assert flat["shape"] == "none" and abs(flat["falloff"]) < 0.02
    # one dark corner is content, not a vignette
    one = np.full((h, w), 150.0)
    one[:50, :80] = 40
    assert look.vignette(one)["shape"] == "none"


def test_tone_points():
    g = np.clip(np.tile(np.linspace(0, 255, 320), (180, 1)), 16, 235)
    t = look.tone(_rgb(g))
    assert abs(t["black_pt"] - 16) <= 1 and abs(t["white_pt"] - 235) <= 1
    assert abs(t["midtone"] - 127) <= 2 and len(t["curve"]) == 5
    assert t["curve"] == sorted(t["curve"]) and 0 < t["contrast"] < 0.5


def test_split_tone():
    img = np.zeros((180, 320, 3), np.uint8)
    img[:, :160] = (20, 60, 70)      # teal shadows
    img[:, 160:] = (240, 170, 100)   # orange highlights
    gd = look.grade(img)
    assert gd["shadow_tint"][1] < 0 < gd["highlight_tint"][1]
    assert gd["highlight_tint"][0] > 0 and gd["sat"] > 0.03
    neutral = look.grade(_rgb(np.tile(np.linspace(0, 255, 320), (180, 1))))
    assert max(abs(x) for x in neutral["shadow_tint"] + neutral["highlight_tint"]) < 0.005


def test_text_shadow():
    m = _glyphs()
    sh = look._blur(np.roll(np.roll(m.astype(np.float32), 4, 0), 4, 1), 3.0)
    y = 110.0 * (1 - 0.7 * sh)
    y[m] = 250
    f = look.text_finish(_rgb(y), _bbox(m))
    s = f["shadow"]
    assert s is not None and 3 <= s["dx"] <= 5 and 3 <= s["dy"] <= 5
    assert 1.5 <= s["blur_px"] <= 4.5 and 0.5 <= s["alpha"] <= 0.9
    assert f["stroke_px"] == 0 and f["glow"] is False and f["fill_kind"] == "flat"
    # up-left shadow keeps its sign
    sh2 = look._blur(np.roll(np.roll(m.astype(np.float32), -3, 0), -3, 1), 1.5)
    y2 = 110.0 * (1 - 0.6 * sh2)
    y2[m] = 250
    s2 = look.text_finish(_rgb(y2), _bbox(m))["shadow"]
    assert -4 <= s2["dx"] <= -2 and -4 <= s2["dy"] <= -2


def test_text_stroke_glow_fill():
    m = _glyphs()
    st = look._dilate(m, 3)
    y = np.full(m.shape, 128.0)
    y[st] = 10
    y[m] = 245
    f = look.text_finish(_rgb(y), _bbox(st))
    assert 2 <= f["stroke_px"] <= 4 and f["shadow"] is None
    plain = np.full(m.shape, 60.0)
    plain[m] = 240
    p = look.text_finish(_rgb(plain), _bbox(m))
    assert p["stroke_px"] == 0 and p["shadow"] is None and p["glow"] is False and p["fill_hex"] == "#f0f0f0"
    glow = np.full(m.shape, 40.0) + look._blur(m.astype(np.float32), 4) * 120
    glow[m] = 250
    assert look.text_finish(_rgb(glow), _bbox(m))["glow"] is True
    rgb = _rgb(np.full(m.shape, 30.0)).astype(float)
    yy = np.mgrid[0:m.shape[0], 0:m.shape[1]][0].astype(float)
    rgb[m] = np.stack([255 - (yy - 60) * 3, 120 + (yy - 60) * 2, 80 + 0 * yy], -1)[m]
    assert look.text_finish(np.clip(rgb, 0, 255).astype(np.uint8), _bbox(m))["fill_kind"] == "gradient"
    # too small or empty boxes are not measured
    assert look.text_finish(_rgb(plain), [0.1, 0.1, 0.005, 0.005]) is None
    assert look.text_finish(_rgb(np.full(m.shape, 90.0)), _bbox(m)) is None


def test_sharpness_halation_chroma_letterbox_flicker():
    rng = np.random.default_rng(5)
    tex = look._blur(rng.normal(128, 60, (180, 320)).astype(np.float32), 1.0)
    shallow = look._blur(tex, 4.0)
    shallow[60:120, 100:220] = tex[60:120, 100:220]
    assert look.sharpness(shallow)["centre_edge_ratio"] > 5
    assert 0.5 <= look.sharpness(tex)["centre_edge_ratio"] <= 2
    assert look.sharpness(np.full((90, 160), 100.0)) == {"lap_var": 0.0, "centre_edge_ratio": 1.0}
    hi = np.full((200, 200), 40.0)
    hi[90:110, 90:110] = 255
    assert look.halation(_rgb(hi)) == 0.0
    glowing = hi + look._blur((hi > 200).astype(np.float32), 6) * 80
    assert look.halation(_rgb(glowing)) > 0.02
    assert look.halation(_rgb(np.full((100, 100), 90.0))) is None
    # B channel scaled 1 % around the centre = lateral chromatic aberration
    base = np.clip(look._blur(rng.normal(128, 60, (360, 640)).astype(np.float32), 2.0), 0, 255)
    yy, xx = np.mgrid[0:360, 0:640].astype(np.float32)
    sx, sy = 319.5 + (xx - 319.5) / 1.01, 179.5 + (yy - 179.5) / 1.01
    x0, y0 = np.clip(np.floor(sx).astype(int), 0, 638), np.clip(np.floor(sy).astype(int), 0, 358)
    fx, fy = sx - x0, sy - y0
    scaled = (base[y0, x0] * (1 - fx) * (1 - fy) + base[y0, x0 + 1] * fx * (1 - fy)
              + base[y0 + 1, x0] * (1 - fx) * fy + base[y0 + 1, x0 + 1] * fx * fy)
    ca = look.chroma_ab(np.stack([base, base, scaled], -1).astype(np.uint8))
    assert 1.0 <= ca <= 3.5
    assert look.chroma_ab(np.stack([base] * 3, -1).astype(np.uint8)) < 0.1
    assert look.letterbox(np.pad(np.full((140, 320), 120.0), ((20, 20), (0, 0)))) == [0.1111, 0.1111]
    assert look.letterbox(np.full((180, 320), 120.0)) == [0.0, 0.0]
    assert look.letterbox(np.full((180, 320), 5.0)) == [0.0, 0.0]   # a black frame is not a letterbox
    assert look.flicker([100.0] * 30) == 0.0
    assert look.flicker(list(np.linspace(50, 150, 30))) < 0.01        # a fade is a trend, not flicker
    assert look.flicker(list(100 + 3 * np.sin(np.arange(30) * 3.0))) > 1.0
    assert look.flicker([100.0, 101.0]) is None


# ---------------------------------------------------------------- CLI

def _clip(path, src, frames=None):
    args = ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", src]
    if frames:
        args += ["-frames:v", str(frames)]
    subprocess.run(args + ["-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", str(path)], check=True)
    return path


def _measure(path, shots, boxes=()):
    doc = {"schema": "measure/1", "shots": [{"t0": a, "t1": b} for a, b in shots],
           "text": {"boxes": [{"t": t, "box": b, "h_pct": round(b[3] * 100, 2)} for t, b in boxes]}}
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


def test_static_and_tiny_frames(tmp_path):
    d = tmp_path / "תיקייה"
    d.mkdir()
    one = _clip(d / "one.mp4", "color=c=gray:s=320x180:r=25:d=1", frames=1)
    tiny = _clip(d / "tiny.mp4", "testsrc2=s=16x16:r=25:d=1")
    m1 = _measure(d / "m1.json", [(0.0, 0.04)], [(0.0, [0.1, 0.1, 0.5, 0.2])])
    for vid, meas in ((one, m1), (tiny, None)):
        out = d / f"{vid.stem}.look.json"
        r = run_cli("look.py", vid, *(["--measure", meas] if meas else []), "--out", out)
        assert r.returncode == 0, r.stderr
        doc = json.loads(out.read_text(encoding="utf-8"))
        assert doc["schema"] == "look/1" and len(doc["video"]["sha256"]) == 64
        assert doc["per_shot"] and set(look.SHOT_KEYS) <= set(doc["summary"])
        for v in doc["summary"].values():
            if isinstance(v, dict):
                assert set(v) == {"median", "p10", "p90", "n"}
        assert isinstance(doc["not_run"], list)
        if vid is one:
            assert doc["video"]["frames"] == 1 and doc["per_shot"][0]["flicker"] is None
            assert doc["per_shot"][0]["n_frames"] == 1
    assert "text_finish (no --measure)" in doc["not_run"]
    assert doc["summary"]["grain_std"]["n"] == 0   # 16x16 is too small for grain


def test_cli_shots_text_and_bad_input(tmp_path):
    vid = _clip(tmp_path / "v.mp4", "testsrc2=s=320x180:r=25:d=2")
    meas = _measure(tmp_path / "m.json", [(0.0, 1.0), (1.0, 2.0)],
                    [(0.5, [0.05, 0.05, 0.6, 0.15]), (1.5, [0.05, 0.05, 0.6, 0.15])])
    out = tmp_path / "a" / "look.json"
    r = run_cli("look.py", vid, "--measure", meas, "--out", out)
    assert r.returncode == 0, r.stderr
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert [s["n_frames"] for s in doc["per_shot"]] == [8, 8]
    assert doc["analysis_width"] == 320 and doc["px_ref_width"] == 1920
    assert len(doc["text_finish"]["boxes"]) == 2 and doc["not_run"] == []
    assert run_cli("look.py", tmp_path / "missing.mp4", "--out", out).returncode == 2
    bad = tmp_path / "bad.json"
    bad.write_text("{}", encoding="utf-8")
    assert run_cli("look.py", vid, "--measure", bad, "--out", out).returncode == 2
    notvid = tmp_path / "x.mp4"
    notvid.write_text("not a video", encoding="utf-8")
    assert run_cli("look.py", notvid, "--out", out).returncode == 2


# ---------------------------------------------------------------- fingerprint

def test_fingerprint_reads_look(tmp_path):
    for k, g in enumerate((2.0, 3.0, 4.0)):
        d = tmp_path / "analysis" / f"V{k}"
        d.mkdir(parents=True)
        (d / "measure.json").write_text(json.dumps({"schema": "measure/1", "video": {
            "sha256": "%064x" % (k + 1), "duration_s": 10.0}, "cuts": [], "shots": [], "not_run": []}),
            encoding="utf-8")
        summ = {key: {"median": 1.0, "p10": 1.0, "p90": 1.0, "n": 1} for key in look.SHOT_KEYS}
        summ["grain_std"] = {"median": g, "p10": g, "p90": g, "n": 3}
        tf = {"n": 4, "stroke_px": {"median": 2.0, "p10": 2.0, "p90": 2.0, "n": 4}, "shadow_share": 0.5,
              "glow_share": 0.0}
        if k < 2:   # V2 has no look.json: it does not count for seen_in
            (d / "look.json").write_text(json.dumps({"schema": "look/1", "summary": summ, "text_finish": tf}),
                                         encoding="utf-8")
    out = fp.build(tmp_path)
    lk = out["layers"]["look"]
    assert lk["source"] == "refs"
    gs = lk["metrics"]["grain_std"]
    assert gs["median"] == 2.5 and gs["n"] == 2 and gs["seen_in"] == 2 and gs["p10"] <= gs["p90"]
    assert set(lk["metrics"]) == set(fp.LOOK_METRICS)
    assert lk["metrics"]["text_stroke_px"]["median"] == 2.0
    assert lk["metrics"]["text_shadow_share"]["median"] == 0.5
    assert lk["metrics"]["text_glow_share"]["median"] == 0.0


def test_fingerprint_reads_voice_timbre(tmp_path):
    for k, crest in enumerate((9.0, 11.0, 13.0)):
        d = tmp_path / "analysis" / f"V{k}"
        d.mkdir(parents=True)
        (d / "measure.json").write_text(json.dumps({"schema": "measure/1", "video": {
            "sha256": "%064x" % (k + 1), "duration_s": 60.0}, "cuts": [], "shots": [], "not_run": []}),
            encoding="utf-8")
        (d / "sound.json").write_text(json.dumps({"schema": "sound/1", "sfx": [], "voice": {
            "crest_db": crest, "rt60_est_s": 0.3, "spectral_tilt_db_per_oct": -4.0, "de_ess_db": -18.0},
            "music": {"present": True, "bpm": 120.0, "ducking_db": 6.0,
                      "sections": [{"t0": 0, "t1": 30}, {"t0": 30, "t1": 60}]}}), encoding="utf-8")
    m = fp.build(tmp_path)["layers"]["sound"]["metrics"]
    assert m["crest_db"]["median"] == 11.0 and m["rt60_est_s"]["median"] == 0.3
    assert m["music_sections_per_min"]["median"] == 2.0 and m["music_ducking_db"]["median"] == 6.0

import json
import re

import numpy as np
import pytest

import track_motion as tm
from conftest import run_cli
from fixtures import make_media as mm

# Reference eases as their standard cubic-bezier equivalents (the CSS / easings.net tables GSAP's
# eases are quoted with: power1 = quad, power2 = cubic, power3 = quart).
EASES = {
    "power1.out": (0.25, 0.46, 0.45, 0.94),
    "power2.out": (0.215, 0.61, 0.355, 1.0),
    "power3.out": (0.165, 0.84, 0.44, 1.0),
    "power2.inOut": (0.645, 0.045, 0.355, 1.0),  # easeInOutCubic
    "expo.out": (0.19, 1.0, 0.22, 1.0),
    "back.out": (0.175, 0.885, 0.32, 1.275),
    "sine.inOut": (0.445, 0.05, 0.55, 0.95),
    "linear": (0.0, 0.0, 1.0, 1.0),
}


def _sample(name, n):
    return tm.bezier_y(np.linspace(0, 1, n), EASES[name])


def _gsap_ease(name, p):
    """GSAP's own power curves (power2 = cubic), for the clips; make_media's names otherwise."""
    p = min(1.0, max(0.0, p))
    if name == "power2.out":
        return 1 - (1 - p) ** 3
    if name == "power2.in":
        return p ** 3
    return _ORIG_EASE(name, p)


_ORIG_EASE = mm._ease


@pytest.mark.parametrize("n", [18, 40])
@pytest.mark.parametrize("name", ["power3.out", "power2.inOut", "linear", "back.out"])
def test_fit_bezier_recovers_known_eases(name, n):
    bez, rmse = tm.fit_bezier(_sample(name, n))
    assert rmse <= 0.01
    assert tm.nearest_named(bez) == name
    assert len(bez) == 4 and all(round(v, 3) == v for v in bez)


def test_fit_bezier_non_uniform_times():
    t = np.sort(np.r_[0.0, np.random.default_rng(1).uniform(0, 1, 20), 1.0])
    bez, rmse = tm.fit_bezier(tm.bezier_y(t, EASES["expo.out"]), t)
    assert rmse <= 0.01 and tm.nearest_named(bez) == "expo.out"


def test_nearest_named_every_reference():
    for name, b in tm.NAMED.items():
        assert tm.nearest_named(b) == name


def test_analyse_move_peak_and_end():
    m = tm.analyse_move(_sample("power3.out", 15), 30)
    assert m["peak_at"] <= 0.2 and m["end"] == "soft"
    assert m["dur_f"] == 14 and m["ease_name"] == "power3.out"
    lin = np.r_[np.linspace(0.2, 0.8, 15), [0.8] * 5]  # linear, then a hard stop and a hold
    m = tm.analyse_move(lin, 30)
    assert 0.3 <= m["peak_at"] <= 0.7 and m["end"] == "hard"
    assert m["dur_f"] == 14 and abs(m["delta"] - 0.6) < 1e-6
    assert m["ease_name"] == "linear" and m["overshoot"] == 0 and m["settle_f"] == 0


def test_overshoot_measured():
    m = tm.analyse_move(_sample("back.out", 20) * 0.5 + 0.1, 30)
    assert 0.05 <= m["overshoot"] <= 0.15
    assert m["settle_f"] > 0
    assert m["ease_name"] == "back.out"


def test_custom_ease_string():
    assert tm.custom_ease([0.21549, 0.6104, 0.3551, 1.0]) == "M0,0 C0.215,0.61 0.355,1 1,1"
    assert tm.custom_ease([0.0004, -0.0004, 1, 1.2756]) == "M0,0 C0,0 1,1.276 1,1"
    m = tm.analyse_move(_sample("power2.out", 24), 30)
    num = r"(-?\d+(?:\.\d{1,3})?)"
    hit = re.fullmatch(rf"M0,0 C{num},{num} {num},{num} 1,1", m["custom_ease"])
    assert hit, m["custom_ease"]
    assert [float(g) for g in hit.groups()] == m["bezier"]
    assert m["bezier"] == [round(v, 3) for v in m["bezier"]]


def test_segment_moves_hold_between_two_moves():
    a = tm.bezier_y(np.linspace(0, 1, 12), EASES["power2.out"])
    v = np.r_[[0.0] * 5, a * 0.4, [0.4] * 10, 0.4 - a * 0.2, [0.2] * 6]
    moves = tm.segment_moves(v, 30)
    assert moves == [(5, 16), (27, 38)]  # a[0] == 0, so each move starts on its first sample
    noise = 0.5 + np.random.default_rng(0).normal(0, 1e-4, 60)
    assert tm.segment_moves(noise, 30, min_delta=0.003) == []
    assert tm.segment_moves(np.zeros(1), 30) == [] and tm.segment_moves(np.full(9, 0.3), 30) == []


def test_segment_moves_ignores_on_twos_holds():
    a = tm.bezier_y(np.linspace(0, 1, 21), EASES["power2.out"])
    held = np.repeat(a[::2], 2)[:21]  # on twos: every value shown twice
    v = np.r_[[0.0] * 4, held, [held[-1]] * 4, [1.0] * 6]
    moves = tm.segment_moves(v, 30)
    assert len(moves) == 1
    a, b = tm.move_slice(v, *moves[0])
    m = tm.analyse_move(v[a:b + 1], 30)
    assert m["rmse"] <= 0.02 and m["ease_name"] == "power2.out" and m["dur_f"] == 20
    # the other phase: the move starts between two pictures
    a2 = tm.bezier_y(np.linspace(0, 1, 22), EASES["power2.out"])
    v2 = np.r_[[0.0] * 5, np.repeat(a2[1::2], 2), [1.0] * 8]
    moves = tm.segment_moves(v2, 30)
    assert len(moves) == 1
    a, b = tm.move_slice(v2, *moves[0])
    m = tm.analyse_move(v2[a:b + 1], 30)
    assert m["rmse"] <= 0.02 and m["ease_name"] == "power2.out" and m["dur_f"] == 21


def test_stagger_uneven():
    s = tm.stagger([0, .08, .14, .18])
    assert s["even"] is False and s["trend"] == "accel"
    assert s["intervals"] == [0.08, 0.06, 0.04]
    e = tm.stagger([0.0, 0.1, 0.2, 0.3, 0.4])
    assert e["even"] is True and e["trend"] == "flat"
    assert tm.stagger([0.0, 0.05, 0.15, 0.3])["trend"] == "decel"
    assert tm.stagger([1.0]) == {"intervals": [], "even": True, "trend": "flat"}


def test_entrance_exit_mirrored():
    fps = 30
    up = tm.bezier_y(np.linspace(0, 1, 13), tm.NAMED["power2.out"])
    down = 1 - tm.bezier_y(np.linspace(0, 1, 13), tm.NAMED["power2.in"])
    op = np.r_[up, [1.0] * 30, down]
    n = len(op)
    track = {"id": "el0", "t0": 0.0, "fps": fps,
             "series": {"x": np.full(n, 0.5), "y": np.full(n, 0.5), "scale": np.ones(n), "opacity": op}}
    ee = tm.entrance_exit(track)
    assert ee["in"]["prop"] == "opacity" and ee["out"]["prop"] == "opacity"
    assert ee["in"]["ease_name"] == "power2.out" and ee["out"]["ease_name"] == "power2.in"
    assert ee["mirrored"] is True
    assert abs(ee["hold_s"] - 30 / fps) <= 2 / fps
    track["series"]["opacity"] = np.r_[up, [1.0] * 30, 1 - np.linspace(0, 1, 7)]  # a fast linear exit
    assert tm.entrance_exit(track)["mirrored"] is False


def _texture(h, w, seed=3):
    cv2 = pytest.importorskip("cv2")
    rng = np.random.default_rng(seed)
    big = rng.uniform(0, 255, (h, w)).astype(np.float32)
    big = cv2.GaussianBlur(big, (0, 0), 2.0)
    big = (big - big.min()) / (big.max() - big.min()) * 255
    return big.astype(np.uint8)


def test_camera_push_in_scale_curve():
    cv2 = pytest.importorskip("cv2")
    h, w, fps = 180, 320, 30
    tex = _texture(h, w)
    prog = tm.bezier_y(np.linspace(0, 1, 21), EASES["power2.inOut"])
    scales = np.r_[[1.0] * 5, 1.0 + 0.2 * prog, [1.2] * 8]  # 1.0 -> 1.2 over 20 frames
    frames = np.stack([cv2.warpAffine(tex, cv2.getRotationMatrix2D(((w - 1) / 2, (h - 1) / 2), 0, s), (w, h),
                                      flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT) for s in scales])
    cam = tm.camera_track(frames)
    assert cam.shape == (len(frames), 4)
    moves = tm.camera_moves(cam, fps)
    assert [m["prop"] for m in moves] == ["scale"], moves
    m = moves[0]
    assert 0.18 <= m["delta"] <= 0.22
    assert m["ease_name"] == "power2.inOut" and m["rmse"] <= 0.02
    assert 18 <= m["dur_f"] <= 21
    # a cut resets the track to the identity
    cam2 = tm.camera_track(frames, cut_frames=[15])
    assert np.allclose(cam2[15], [1, 0, 0, 0])


def _measure(tmp_path, clip):
    out = tmp_path / "m"
    r = run_cli("measure_video.py", clip, "--out", out, "--faces", "off", "--text", "off")
    assert r.returncode == 0, r.stderr
    return out / "measure.json"


def test_stepped_motion_fit_on_unique_frames(tmp_path, monkeypatch):
    monkeypatch.setattr(mm, "_ease", _gsap_ease)
    clip = mm.make_motion_clip(tmp_path / "steps.mp4", dur_s=2.4, fps=30, stepped=2,
                               moves=[(0.3, 1.8, "x", 0.3, 0.7, "power2.out")])
    doc = tm.track(clip)
    assert doc["cadence"]["unique_fps"] <= 17  # measure_video: about half the moving frames repeat
    assert doc["camera_moves"] == []
    moves = [m for e in doc["elements"] for m in e["moves"]]
    assert len(moves) == 1, moves  # hold frames must not read as extra moves
    m = moves[0]
    assert m["prop"] == "x" and m["rmse"] <= 0.02 and m["ease_name"] == "power2.out"
    assert 0.37 <= m["delta"] <= 0.43
    assert abs(m["t0"] - 0.3) <= 2 / 30
    assert len(doc["speed_graphs"][m["id"]]) == len(doc["value_graphs"][m["id"]]) - 1
    assert tm.validate(doc) == []


def test_no_motion_empty(tmp_path):
    clip = mm.make_motion_clip(tmp_path / "still.mp4", dur_s=2, fps=25)
    meas = _measure(tmp_path, clip)
    out = tmp_path / "an" / "motion.json"
    r = run_cli("track_motion.py", clip, "--measure", meas, "--out", out)
    assert r.returncode == 0, r.stderr
    d = json.loads(out.read_text(encoding="utf-8"))
    assert d["schema"] == "motion/1" and tm.validate(d) == []
    assert d["camera_moves"] == [] and d["elements"] == [] and d["stagger_groups"] == []
    assert d["speed_graphs"] == {} and d["value_graphs"] == {}
    assert d["summary"]["moves"] == 0 and d["summary"]["dur_s"]["n"] == 0
    assert d["video"]["sha256"] == json.loads(meas.read_text(encoding="utf-8"))["video"]["sha256"]
    assert not list(out.parent.glob("*.tmp"))


def test_cli_bad_input(tmp_path):
    out = tmp_path / "motion.json"
    r = run_cli("track_motion.py", tmp_path / "missing.mp4", "--out", out)
    assert r.returncode == 2 and "Traceback" not in r.stderr
    clip = mm.make_motion_clip(tmp_path / "c.mp4", dur_s=0.4, fps=25)
    bad = tmp_path / "bad.json"
    bad.write_text('{"schema": "other"}', encoding="utf-8")
    r = run_cli("track_motion.py", clip, "--measure", bad, "--out", out)
    assert r.returncode == 2 and "Traceback" not in r.stderr and not out.exists()


def test_very_short_clip(tmp_path):
    clip = mm.make_motion_clip(tmp_path / "short.mp4", dur_s=0.12, fps=25)
    doc = tm.track(clip)
    assert tm.validate(doc) == [] and doc["camera_moves"] == [] and doc["elements"] == []

import itertools
import json
import shutil
import subprocess

import pytest

import hf
import motion
import picks
import store
from conftest import run_cli
from fixtures import plan5 as P
from test_frames import PNG


def test_window_short_and_last_beat():
    assert motion.window(10.0, 11.2, 30)[:2] == (10.0, 12.0)
    assert motion.window(29.0, 30.0, 30)[:2] == (28.0, 30.0)
    t0, t1, _ = motion.window(3.0, 9.0, 30)
    assert t1 - t0 == pytest.approx(4.0)
    t0, t1, warn = motion.window(0.0, 1.5, 1.5)
    assert (t0, t1) == (0.0, 1.5) and warn


def test_palette_from_fingerprint_and_fallback():
    pal = motion.motion_palette(P.FINGERPRINT)
    assert pal["curve_source"] == "fingerprint" and pal["eases"][0]["name"] == "power2.out"
    assert pal["entrance_dur_s"]["median"] == pytest.approx(14 / 30, abs=1e-3)
    for fp in (None, {"layers": {}}, {"layers": {"motion": {"eases": [{"name": "linear", "bezier": [0, 0, 1, 1],
                                                                        "share": 0.1}]}}}):
        p = motion.motion_palette(fp)
        assert p["curve_source"] == "craft-defaults"
        assert all(not motion._is_linear(e["bezier"]) for e in p["eases"])
    lin = {"layers": {"motion": {"eases": [{"name": "linear", "bezier": [0, 0, 1, 1], "share": 0.4}]}}}
    assert motion.motion_palette(lin)["eases"][0]["name"] == "linear"  # only when the style really uses it


@pytest.fixture
def world(home, tmp_path):
    P.write_style(home, tmp_path)
    P.write_client(run_cli)
    d = P.write_project(tmp_path)
    P.set_options(d, {b: P.good_set(b) for b in (1, 2, 3)})
    for b in (1, 2, 3):
        vd = d / "frames" / f"B0{b}" / "s1" / "a"
        vd.mkdir(parents=True)
        (vd / "index.html").write_text(
            '<div id="root" data-composition-id="x" data-width="1080" data-height="1920" data-duration="2">'
            'יש 3 דרכים</div><script src="motion-params.js"></script>', encoding="utf-8")
        (vd / "tokens.css").write_text(":root{}", encoding="utf-8")
        store.write_json(d / "frames.json", {"version": 1, "beats": {
            str(k): {"sets": [{"set": 1, "variants": [{"id": f"B0{k}-f1-a", "dir": f"frames/B0{k}/s1/a"}]}]}
            for k in (1, 2, 3)}})
        picks.pick(d, "plan", b, f"B0{b}-p1")
        picks.pick(d, "frames", b, f"B0{b}-f1-a")
    return d


def _params(vd):
    return motion._read_params(vd)


def test_three_variants_differ_in_motion_only(world):
    n, dirs, src, _ = motion.variants(world, 1)
    assert n == 1 and len(dirs) == 3 and src == "fingerprint"
    assert len({motion._design_hash(v) for v in dirs}) == 1
    params = [(v / "motion-params.js").read_text(encoding="utf-8") for v in dirs]
    assert len(set(params)) == 3
    vs = [{"entrance": p["entrance"], "ease": p["ease"], "timing": p["timing"], "camera": p["camera"]}
          for p in map(_params, dirs)]
    for a, b in itertools.combinations(vs, 2):
        assert motion._differ(a, b)
    assert 'data-duration="2"' in (dirs[0] / "index.html").read_text(encoding="utf-8")
    assert motion.check_variants(world, 1, 1) == []


def test_eases_come_from_style(world):
    _, dirs, _, _ = motion.variants(world, 1)
    names = {e["name"] for e in P.FINGERPRINT["layers"]["motion"]["eases"]}
    assert all(_params(v)["ease"]["name"] in names for v in dirs)
    p = _params(dirs[0])
    p["ease"] = {"name": "linear", "bezier": [0, 0, 1, 1], "custom_ease": ""}
    (dirs[0] / "motion-params.js").write_text("window.SL_MOTION = " + json.dumps(p) + ";", encoding="utf-8")
    assert any("not one of the style's eases" in x for x in motion.check_variants(world, 1, 1))


def test_more_never_repeats(world):
    import project
    sigs = []
    for n in (1, 2, 3, 4):
        _, dirs, _, _ = motion.variants(world, 2)
        s = motion._set(world, 2, n)
        project.record_shown(world, "motion", 2, n, s["variants"])
        sigs.append({v["sig"] for v in s["variants"]})
    assert not (sigs[0] & sigs[1]) and len(sigs[1]) == 3
    r = run_cli("motion.py", "variants", world, "--beat", "2")
    assert r.returncode == 1


def test_mode_one_skips_render(world, monkeypatch):
    monkeypatch.setattr(hf, "render_draft", lambda *a, **k: pytest.fail("no render in mode one"))
    assert run_cli("motion.py", "mode", world, "one").returncode == 0
    motion.set_mode(world, "three", beat=2)
    _, dirs, _, _ = motion.variants(world, 1)
    assert len(dirs) == 1
    pk = picks.load(world)["motion"]
    assert pk["beats"]["1"]["decided_by"] == "claude" and pk["mode"] == "one" and pk["overrides"]["2"] == "three"
    _, dirs2, _, _ = motion.variants(world, 2)
    assert len(dirs2) == 3


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg missing")
def test_render_duration_bounds(world, monkeypatch):
    seconds = {"v": 5.0}

    def fake_render(cwd, out):
        out.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", f"color=c=black:s=64x64:d={seconds['v']}",
                        str(out)], check=True)
        return out
    monkeypatch.setattr(hf, "render_draft", fake_render)
    motion.variants(world, 1)
    r = motion.main(["render", str(world), "--beat", "1", "--set", "1"])
    assert r == 1
    seconds["v"] = 3.0
    clips = motion.render(world, 1, 1)
    assert len(clips) == 3
    shown = [i for i in store.read_json(world / "shown.json", {})["items"] if i["step"] == "motion"]
    assert len(shown) == 3


def test_beat_html_must_read_params(world):
    _, dirs, _, _ = motion.variants(world, 3)
    for v in dirs:
        h = (v / "index.html").read_text(encoding="utf-8").replace('<script src="motion-params.js"></script>', "")
        (v / "index.html").write_text(h, encoding="utf-8")
    assert any("does not read motion-params.js" in x for x in motion.check_variants(world, 3, 1))


def test_needs_frame_pick(home, tmp_path):
    P.write_style(home, tmp_path)
    P.write_client(run_cli)
    d = P.write_project(tmp_path)
    r = run_cli("motion.py", "variants", d, "--beat", "1")
    assert r.returncode == 1 and "frame pick" in r.stderr

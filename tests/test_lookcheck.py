import json
import shutil

import numpy as np
import pytest

import imgmetrics as im
import lookcheck
import store
from conftest import run_cli
from fixtures import plan5 as P
from fixtures.make_media import make_image

pytestmark = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg missing")


def _noisy(seed, sigma=0.04, base=(0.45, 0.4, 0.35)):
    rng = np.random.default_rng(seed)
    img = np.ones((120, 160, 3), np.float32) * np.array(base, np.float32)
    return np.clip(img + rng.normal(0, sigma, img.shape), 0, 1)


def test_judge_pass_and_fail(tmp_path):
    refs = [im.look_stats(im.decode_image(make_image(tmp_path / f"r{k}.png", _noisy(k)))) for k in range(3)]
    gen = im.look_stats(im.decode_image(make_image(tmp_path / "g.png", _noisy(9))))
    assert lookcheck.judge(gen, refs)["verdict"] == "pass"
    clean = im.look_stats(im.decode_image(make_image(tmp_path / "c.png", _noisy(9, sigma=0.0))))
    res = lookcheck.judge(clean, refs)
    assert res["verdict"] == "fail"
    g = next(c for c in res["checks"] if c["metric"] == "grain")
    assert not g["ok"] and g["lo"] < g["hi"]
    assert lookcheck.judge(gen, [], (), {})["verdict"] == "not_run"


@pytest.fixture
def world(home, tmp_path):
    pack = P.write_style(home, tmp_path)
    P.write_client(run_cli)
    d = P.write_project(tmp_path)
    return d, pack


def _item(d, file, recipe="paper-cutout", status="generated"):
    store.write_json(d / "gen" / "queue.json", {"version": 1, "items": [
        {"id": "G001", "beat": 1, "stage": "frames", "recipe": recipe, "kind": "image", "status": status,
         "file": file, "batch": "S01"}]})


def _ref_video(pack, tmp_path, sigma=0.04):
    import subprocess
    frames = tmp_path / "rf"
    for k in range(3):
        make_image(frames / f"{k:03d}.png", _noisy(k, sigma))
    (pack / "refs").mkdir(exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-framerate", "1", "-i", str(frames / "%03d.png"),
                    "-c:v", "libx264", "-crf", "0", "-pix_fmt", "yuv444p", str(pack / "refs" / "V01.mp4")], check=True)


def _status(d):
    return store.read_json(d / "gen" / "queue.json", {})["items"][0]["status"]


def test_needs_claude_verdict(world, tmp_path):
    d, pack = world
    _ref_video(pack, tmp_path)
    make_image(d / "gen" / "files" / "a.png", _noisy(5))
    _item(d, "gen/files/a.png")
    r = run_cli("lookcheck.py", d, "--item", "G001")
    assert r.returncode == 0, r.stdout + r.stderr
    assert _status(d) == "generated"
    assert (d / "gen" / "refcache").is_dir()
    r = run_cli("lookcheck.py", d, "--item", "G001", "--claude", "pass", "--why", "paper texture reads right")
    assert r.returncode == 0 and _status(d) == "look_pass"
    lines = [json.loads(x) for x in (d / "gen" / "log.jsonl").read_text(encoding="utf-8").splitlines()]
    assert lines[-1]["verdict"] == "pass" and lines[-1]["claude"]["verdict"] == "pass" and lines[-1]["pass"]


def test_metrics_fail_claude_pass_needs_override(world, tmp_path):
    d, pack = world
    _ref_video(pack, tmp_path)
    make_image(d / "gen" / "files" / "a.png", _noisy(5, sigma=0.0, base=(0.95, 0.95, 0.95)))
    _item(d, "gen/files/a.png")
    r = run_cli("lookcheck.py", d, "--item", "G001", "--claude", "pass", "--why", "looks fine to me")
    assert r.returncode == 1 and _status(d) == "look_fail" and "--override" in r.stderr
    r = run_cli("lookcheck.py", d, "--item", "G001", "--claude", "pass", "--why", "fine",
                "--override", "the owner wants it brighter here")
    assert r.returncode == 0 and _status(d) == "look_pass"
    last = json.loads((d / "gen" / "log.jsonl").read_text(encoding="utf-8").splitlines()[-1])
    assert last["override"] == "the owner wants it brighter here"


def test_no_refs_not_run(world, tmp_path):
    d, pack = world
    store.write_json(pack / "recipes.json", {"recipes": [{"kind": "bare", "how": "ai_image+i2v", "look": {},
                                                          "source_evidence": []}]})
    make_image(d / "gen" / "files" / "a.png", _noisy(5))
    _item(d, "gen/files/a.png", recipe="bare")
    r = run_cli("lookcheck.py", d, "--item", "G001")
    assert r.returncode == 0 and "לא נמדד" in r.stdout
    r = run_cli("lookcheck.py", d, "--item", "G001", "--claude", "fail", "--why", "a hand has six fingers")
    assert r.returncode == 1 and _status(d) == "look_fail"
    r = run_cli("lookcheck.py", d, "--item", "G001", "--claude", "pass")
    assert r.returncode == 2 and "--why" in r.stderr

import json
import shutil

import numpy as np
import pytest

import buildcheck as bc
import picks
import store
from conftest import run_cli
from fixtures import plan5 as P
from fixtures.make_media import make_image

BEATS = [{"n": k, "start": float(k - 1), "end": float(k), "script": "יש 3 דרכים", "beat_types": []}
         for k in range(1, 5)]


@pytest.fixture
def world(home, tmp_path):
    P.write_style(home, tmp_path)
    P.write_client(run_cli)
    d = P.write_project(tmp_path, beats=BEATS, length=4.0)
    sets = {}
    for b in range(1, 5):
        s = P.good_set(b)
        for o in s:
            o["scene_type"] = "cutaway"
        sets[b] = s
    P.set_options(d, sets)
    for b in range(1, 5):
        picks.pick(d, "plan", b, f"B0{b}-p1")
    return d


def _layout(kind):
    img = np.zeros((160, 90, 3), np.float32) + 0.1
    if kind == "a":
        img[20:70, 10:80] = (0.5, 0.3, 0.9)
        img[100:140, 20:60] = 0.9
    else:  # the same ground, the blocks moved: a different design
        img[90:150, 5:40] = (0.5, 0.3, 0.9)
        img[10:40, 45:85] = 0.9
    return img


def _frames(d, beats=(1, 2)):
    doc = {"version": 1, "beats": {}}
    for b in beats:
        still = d / "frames" / f"B0{b}" / "s1" / "a.png"
        make_image(still, _layout("a"))
        doc["beats"][str(b)] = {"sets": [{"set": 1, "variants": [
            {"id": f"B0{b}-f1-a", "still": still.relative_to(d).as_posix(), "at_s": 0.5}]}]}
        picks.pick(d, "frames", b, f"B0{b}-f1-a")
    store.write_json(d / "frames.json", doc)


def test_captions_first(world):
    r = run_cli("buildcheck.py", "start-beat", world, "--beat", "1")
    assert r.returncode == 1 and "הכתוביות" in r.stderr


def test_beats_in_order(world):
    bc.captions_done(world)
    bc.start_beat(world, 1)
    r = run_cli("buildcheck.py", "start-beat", world, "--beat", "3")
    assert r.returncode == 1 and "הבא בתור הוא 1" in r.stderr
    bc.beat_done(world, 1)
    assert run_cli("buildcheck.py", "start-beat", world, "--beat", "2").returncode == 0


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg missing")
def test_first_two_compared_gate(world, tmp_path):
    _frames(world)
    bc.captions_done(world)
    snap = make_image(tmp_path / "snap.png", _layout("a"))
    for b in (1, 2):
        bc.start_beat(world, b)
        bc.beat_done(world, b)
    r = run_cli("buildcheck.py", "start-beat", world, "--beat", "3")
    assert r.returncode == 1 and "להשוות" in r.stderr
    assert bc.compare(world, 1, snap)["pass"]
    assert run_cli("buildcheck.py", "start-beat", world, "--beat", "3").returncode == 1
    r = run_cli("buildcheck.py", "start-beat", world, "--beat", "3", "--override", "the owner asked to skip")
    assert r.returncode == 0
    assert bc.load(world)["override"][0]["reason"] == "the owner asked to skip"


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg missing")
def test_compare_thresholds(world, tmp_path):
    _frames(world)
    same = make_image(tmp_path / "same.png", _layout("a"))
    other = make_image(tmp_path / "other.png", _layout("b"))
    assert bc.compare(world, 1, same)["pass"] is True
    r = run_cli("buildcheck.py", "compare", world, "--beat", "1", "--snapshot", other)
    assert r.returncode == 1 and "SSIM" in r.stdout
    assert bc.load(world)["beats"]["1"]["compared"]["ssim"] < 0.6


def _rule(layer, text, *extra):
    r = run_cli("lessons.py", "add", "--text", text, "--measurable", f"m {text}", "--layer", layer,
                "--project", "proj-a", "--video", "v1", *extra)
    assert r.returncode == 0, r.stderr
    import lessons
    lid = lessons.load()["lessons"][-1]["id"]
    r = run_cli("lessons.py", "approve", lid, "--one-note", "--yes-no-conflict")
    assert r.returncode == 0, r.stderr
    return lid


def test_checklist_from_rules_and_auto(world):
    a = _rule("craft", "captions never on the chin", "--kind", "quality")
    b = _rule("client", "logo bottom right", "--target", "example-shop")
    store.write_json(world / "gen" / "queue.json", {"version": 1, "items": [
        {"id": "G001", "stage": "build", "kind": "image", "status": "generated", "file": "gen/files/a.png"}]})
    md = bc.checklist(world).read_text(encoding="utf-8")
    assert a in md and b in md
    pre = bc.load(world)["checklist"]["pre"]
    ai = next(i for i in pre if i.get("auto") == "ai_look_pass")
    assert ai["ticked"] is False
    r = run_cli("buildcheck.py", "tick", world, "--phase", "pre", "--n", str(ai["n"]))
    assert r.returncode == 1 and "אוטומטי" in r.stderr
    assert next(i for i in pre if i.get("auto") == "spend_ok")["ticked"] is True


def _tick_all(d, phase):
    for it in bc.load(d)["checklist"][phase]:
        if it["source"] != "auto":
            bc.tick(d, phase, it["n"])


def test_one_draft_per_round(world, tmp_path):
    f = tmp_path / "draft.mp4"
    f.write_bytes(b"x")
    bc.checklist(world)
    r = run_cli("buildcheck.py", "draft", world, "--file", f)
    assert r.returncode == 1
    _tick_all(world, "pre")
    assert run_cli("buildcheck.py", "draft", world, "--file", f).returncode == 0
    r = run_cli("buildcheck.py", "draft", world, "--file", f)
    assert r.returncode == 1 and "טיוטה אחת לסבב" in r.stderr
    assert bc.next_round(world) == 2
    assert not any(i["ticked"] for i in bc.load(world)["checklist"]["pre"] if i["source"] != "auto")
    _tick_all(world, "pre")
    assert run_cli("buildcheck.py", "draft", world, "--file", f).returncode == 0


def test_present_needs_post(world, tmp_path):
    f = tmp_path / "draft.mp4"
    f.write_bytes(b"x")
    bc.checklist(world)
    _tick_all(world, "pre")
    bc.draft(world, f)
    r = run_cli("buildcheck.py", "present", world)
    assert r.returncode == 1
    bc.tick(world, "post", 1, "prove qa clean")
    assert run_cli("buildcheck.py", "present", world).returncode == 1
    bc.tick(world, "post", 2)
    assert run_cli("buildcheck.py", "present", world).returncode == 0
    assert bc.load(world)["drafts"][0]["presented"] is True


def test_beat_done_keeps_snippet_for_ideas(world):
    import ideas
    bc.captions_done(world)
    bc.start_beat(world, 1)
    sn = world / "build" / "b1.html"
    sn.parent.mkdir(parents=True, exist_ok=True)
    sn.write_text("<div>{{title}}</div>", encoding="utf-8")
    bc.beat_done(world, 1, "build/b1.html", {"title": {"type": "text", "default": "x", "client_bound": False}})
    assert [c["beat"] for c in ideas.final_candidates(world)] == [1]

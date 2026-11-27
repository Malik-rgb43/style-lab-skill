import json
import subprocess
import sys

import pytest

import ideas
import picks
import store
import taste
from conftest import REPO_ROOT, run_cli
from fixtures import plan5 as P


@pytest.fixture
def world(home, tmp_path):
    P.write_style(home, tmp_path)
    P.write_client(run_cli)
    d = P.write_project(tmp_path)
    P.set_options(d, {1: P.good_set(1), 2: P.good_set(2), 3: P.good_set(3)})
    return d


def frames_doc(d, beats=(1, 2, 3)):
    doc = {"version": 1, "beats": {}}
    for b in beats:
        doc["beats"][str(b)] = {"sets": [{"set": 1, "variants": [
            {"id": f"B{b:02d}-f1-{x}", "title": f"frame {x}", "sig": f"frames:{b}{x}0000000000"[:19],
             "wild": x == "c", "tags": [x], "still": f"frames/B{b:02d}/s1/{x}.png"} for x in "abc"]}]}
    store.write_json(d / "frames.json", doc)
    for b in beats:
        for x in "abc":
            f = d / "frames" / f"B{b:02d}" / "s1" / f"{x}.png"
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_bytes(b"png")
    return doc


def test_sync_logs_owner_only_and_once(world, tmp_path):
    sn = tmp_path / "i.html"
    sn.write_text("<div>{{t}}</div>", encoding="utf-8")
    idea = ideas.add("counter", "b-roll", ["number"], "9:16", sn,
                     {"t": {"type": "text", "default": "x", "client_bound": False}})
    s = P.good_set(1, idea_id=idea["id"])
    P.set_options(world, {1: s})
    picks.pick(world, "plan", 1, "B01-p2")                 # the idea option, by the owner
    picks.pick(world, "plan", 2, "B02-p1", by="claude")    # Claude's own pick: never logged
    r = picks.sync(world)
    assert r["logged"] == ["plan:1:B01-p2"]
    lines = [json.loads(x) for x in (store.path("taste", "example-shop.jsonl")).read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 1
    rec = lines[0]
    assert rec["picked"] == "B01-p2" and sorted(rec["rejected"]) == ["B01-p1", "B01-p3"]
    assert rec["sigs"]["picked"].startswith("plan:") and len(rec["sigs"]["rejected"]) == 2
    assert ideas.get(idea["id"])["picked"] == 1
    assert picks.sync(world)["logged"] == []
    assert len((store.path("taste", "example-shop.jsonl")).read_text(encoding="utf-8").splitlines()) == 1


def test_more_and_own_reported(world):
    picks.pick(world, "plan", 1, more=True)
    picks.pick(world, "plan", 2, own="כרטיס עם הלוגו שלנו")
    r = run_cli("picks.py", "sync", world)
    assert r.returncode == 0 and "עוד רעיונות" in r.stdout and "הלוגו" in r.stdout


def test_validate_order_and_repeat(world):
    frames_doc(world)
    picks.pick(world, "frames", 1, "B01-f1-a")
    assert any("before the plan pick" in p for p in picks.validate(world))
    picks.pick(world, "plan", 1, "B01-p1")
    assert picks.validate(world) == []
    s2 = P.good_set(2)
    s2[0]["concept_key"] = P.good_set(1)[0]["concept_key"]
    s2[0]["category"] = P.good_set(1)[0]["category"]
    P.set_options(world, {2: s2})
    picks.pick(world, "plan", 2, "B02-p1")
    assert any("same concept as beat 1" in p for p in picks.validate(world))
    picks.pick(world, "motion", 3, "B03-m1-a")
    probs = picks.validate(world)
    assert any("motion B3" in p for p in probs)
    r = run_cli("picks.py", "validate", world)
    assert r.returncode == 1


def test_parallel_writers(world):
    code = ("import sys; sys.path.insert(0, r'%s'); import picks; picks.pick(sys.argv[1], 'plan', int(sys.argv[2]),"
            " 'B0%%s-p1' %% sys.argv[2])" % (REPO_ROOT / "scripts"))
    store.write_json(world / "plan.json", store.read_json(world / "plan.json", {}))
    beats = list(P.BEATS) + [{"n": 4, "start": 6.0, "end": 7.0, "script": "x"}]
    P.write_project(world.parent, name=world.name, beats=beats, length=7.0)
    P.set_options(world, {b: P.good_set(b) for b in (1, 2, 3, 4)})
    procs = [subprocess.Popen([sys.executable, "-c", code, str(world), str(b)]) for b in (1, 2, 3, 4)]
    assert all(p.wait(60) == 0 for p in procs)
    assert sorted(picks.load(world)["plan"]) == ["1", "2", "3", "4"]


def test_bad_pick_input(world):
    with pytest.raises(picks.BadInput):
        picks.pick(world, "frames", 1, own="x")
    r = run_cli("picks.py", "pick", world, "--step", "plan", "--beat", "1")
    assert r.returncode == 2

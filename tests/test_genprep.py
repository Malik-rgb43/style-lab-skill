import json

import pytest

import genprep
import store
from conftest import run_cli
from fixtures import plan5 as P


def caps(**ok):
    store.write_json(store.path("capabilities.json"), {"local": {}, "session": {
        k: {"ok": v, "note": ""} for k, v in ok.items()}})


@pytest.fixture
def world(home, tmp_path):
    P.write_style(home, tmp_path)
    P.write_client(run_cli)
    caps(higgsfield=True)
    return P.write_project(tmp_path)


def _items(d):
    return {i["id"]: i for i in store.read_json(d / "gen" / "queue.json", {})["items"]}


def test_prompt_from_template_words_palette_look(world):
    it = genprep.add_item(world, 1, "frames", "paper-cutout", "a", {"subject": "old newspaper"})
    assert "old newspaper" in it["prompt"] and "grain: fine" in it["prompt"] and "(#008754)" in it["prompt"]
    assert it["negative"].endswith(genprep.NEGATIVE_ALWAYS) and it["negative"].startswith("blurry")
    assert it["tool"] == "higgsfield" and it["status"] == "queued" and it["id"] == "G001"
    assert it["units"] == 1 and it["unit"] == "image"


def test_missing_slot_exit2(world):
    r = run_cli("genprep.py", "add", world, "--beat", "1", "--stage", "frames", "--recipe", "paper-cutout")
    assert r.returncode == 2 and "subject" in r.stderr
    caps(higgsfield=True)
    r = run_cli("genprep.py", "add", world, "--beat", "1", "--stage", "build", "--recipe", "era-scene",
                "--slot", "subject=a street")
    assert r.returncode == 2 and "era" in r.stderr
    r = run_cli("genprep.py", "add", world, "--beat", "1", "--stage", "build", "--recipe", "code-card")
    assert r.returncode == 2 and "not an AI recipe" in r.stderr


def test_pick_tool_and_fallback(world):
    rec = genprep.load_recipes(world.parent / "pack")[0]
    assert genprep.pick_tool(rec, store.read_json(store.path("capabilities.json"), {}))["capability"] == "higgsfield"
    caps(higgsfield=False)
    r = run_cli("genprep.py", "add", world, "--beat", "1", "--stage", "frames", "--recipe", "paper-cutout",
                "--slot", "subject=x")
    assert r.returncode == 1 and "stock" in r.stderr


def test_units_and_seed(world):
    it = genprep.add_item(world, 2, "build", "era-scene", None, {"subject": "a street", "era": "1950s"})
    assert it["unit"] == "second" and it["units"] == 5 and it["kind"] == "video"
    rec = dict(genprep.load_recipes(world.parent / "pack")[1], motion={"dur_s": {"median": 3.4}})
    assert genprep._units(rec, True) == (3, "second")
    assert genprep.seed_for("p", "G001") == genprep.seed_for("p", "G001") != genprep.seed_for("p", "G001.r2")
    assert 0 <= genprep.seed_for("p", "G001") < 2 ** 32


def _set_status(d, iid, status, file="gen/files/a.png"):
    def fn(q):
        for i in q["items"]:
            if i["id"] == iid:
                i["status"], i["file"] = status, file
    store.update_json(d / "gen" / "queue.json", {}, fn)


def test_animate_requires_look_pass(world):
    caps(higgsfield=True)
    it = genprep.add_item(world, 1, "frames", "paper-cutout", "a", {"subject": "x"})
    _set_status(world, it["id"], "generated")
    r = run_cli("genprep.py", "animate", world, "--from", it["id"])
    assert r.returncode == 1 and "look check" in r.stderr
    _set_status(world, it["id"], "look_pass")
    v = genprep.animate(world, it["id"])
    assert v["camera_curve"]["custom_ease"] == "M0,0 C0.165,0.84 0.44,1 1,1"
    assert v["camera_curve"]["dur_s"] == 3.4 and v["units"] == 3 and v["stage"] == "build"
    assert v["params"]["source_image"] == "gen/files/a.png" and "slow settle" in v["prompt"]
    assert "drift apart" in v["prompt"]


def test_refine_keeps_batch_and_bumps_attempt(world):
    it = genprep.add_item(world, 1, "frames", "paper-cutout", "a", {"subject": "old newspaper"})

    def fn(q):
        q["items"][0]["batch"] = "S01"
    store.update_json(world / "gen" / "queue.json", {}, fn)
    r = run_cli("genprep.py", "refine", world, "--item", it["id"], "--drop", "old newspaper")
    assert r.returncode == 2 and "--why" in r.stderr
    new = genprep.refine(world, it["id"], add=["torn edges"], drop=["old newspaper"], why="the subject read as text")
    assert new["id"] == "G001.r2" and new["attempt"] == 2 and new["retry_of"] == "G001"
    assert new["batch"] == "S01" and "old newspaper" not in new["prompt"] and new["prompt"].endswith("torn edges")
    assert new["seed"] != it["seed"] and _items(world)["G001.r2"]["status"] == "queued"
    r = run_cli("genprep.py", "refine", world, "--item", "G999", "--why", "x")
    assert r.returncode == 2 and "Traceback" not in r.stderr

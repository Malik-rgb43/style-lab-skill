import json
import os
import subprocess
import sys

from conftest import REPO_ROOT, run_cli

HE = "המהירות ליניארית, צריך להתחיל מהר ולסיים לאט"


def reg(home):
    return json.loads((home / "lessons" / "lessons.json").read_text(encoding="utf-8"))


def add(*extra, text=HE, layer="craft"):
    args = ["add", "--text", text, "--measurable", "easing curve is linear", "--layer", layer,
            "--project", "proj-a", "--video", "v1", *extra]
    return run_cli("lessons.py", *args)


def mkstyle(tmp_path, name="סגנון בדיקה"):
    pack = tmp_path / "pack"
    pack.mkdir(exist_ok=True)
    r = run_cli("styles.py", "add", "--name", name, "--pack", pack)
    assert r.returncode == 0, r.stderr


def test_add_candidate_and_one_off(home):
    r = add("--kind", "quality")
    assert r.returncode == 0, r.stderr
    r = add(layer="video")
    assert r.returncode == 0, r.stderr
    ls = reg(home)["lessons"]
    assert [x["id"] for x in ls] == ["L0001", "L0002"]
    assert ls[0]["status"] == "candidate" and ls[0]["kind"] == "quality"
    assert ls[1]["status"] == "one-off" and ls[1]["kind"] is None and ls[1]["target"] is None
    assert reg(home)["next_id"] == 3
    assert ls[0]["evidence"][0]["project"] == "proj-a"


def test_target_validation(home):
    r = add("--target", "nope", layer="style")
    assert r.returncode == 1 and "no style" in r.stderr
    r = add("--target", "nope", layer="client")
    assert r.returncode == 1 and "no client" in r.stderr
    r = add()
    assert r.returncode == 1 and "kind" in r.stderr
    assert not (home / "lessons" / "lessons.json").exists() or reg(home)["lessons"] == []


def test_style_target_resolved_to_slug(home, tmp_path):
    mkstyle(tmp_path)
    r = add("--target", "סגנון בדיקה", layer="style")
    assert r.returncode == 0, r.stderr
    assert reg(home)["lessons"][0]["target"] == "sgnvn-bdykh"


def test_hebrew_roundtrip(home):
    assert add("--kind", "taste").returncode == 0
    assert reg(home)["lessons"][0]["text"] == HE
    r = run_cli("lessons.py", "show", "L0001")
    assert r.returncode == 0 and HE in r.stdout
    r = run_cli("lessons.py", "list")
    assert HE in r.stdout


def test_rejected_readd_warns(home):
    assert add("--kind", "taste").returncode == 0
    assert run_cli("lessons.py", "reject", "L0001", "--reason", "x").returncode == 0
    assert reg(home)["lessons"][0]["status"] == "rejected"
    r = add("--kind", "taste", text="  " + HE + " ")
    assert r.returncode == 0 and "L0001" in r.stderr
    assert len(reg(home)["lessons"]) == 2


def test_edit_layer_revalidates(home):
    assert add("--kind", "quality").returncode == 0
    r = run_cli("lessons.py", "edit", "L0001", "layer=style")
    assert r.returncode == 1
    assert reg(home)["lessons"][0]["layer"] == "craft"
    r = run_cli("lessons.py", "edit", "L0001", "kind=")
    assert r.returncode == 1
    r = run_cli("lessons.py", "edit", "L0001", "scope=intro")
    assert r.returncode == 0 and reg(home)["lessons"][0]["scope"] == "intro"


def test_list_filters(home):
    add("--kind", "quality")
    add(layer="video")
    r = run_cli("lessons.py", "list", "--status", "one-off", "--json")
    assert [x["id"] for x in json.loads(r.stdout)] == ["L0002"]


def test_parallel_adds(home):
    env = dict(os.environ)
    ps = [subprocess.Popen(
        [sys.executable, str(REPO_ROOT / "scripts" / "lessons.py"), "add", "--text", f"t{i}",
         "--measurable", "m", "--layer", "video", "--project", "p", "--video", "v"],
        cwd=REPO_ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE) for i in range(6)]
    assert all(p.wait() == 0 for p in ps)
    ids = [x["id"] for x in reg(home)["lessons"]]
    assert len(ids) == 6 and len(set(ids)) == 6

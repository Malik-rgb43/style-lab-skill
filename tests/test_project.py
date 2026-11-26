import json
import sys

import pytest

import project
import taste
from conftest import REPO_ROOT, run_cli  # noqa: F401


def setup_env(home, clients=("example-shop",)):
    home.mkdir(parents=True, exist_ok=True)
    (home / "styles.json").write_text(
        json.dumps({"version": 2, "styles": [{"slug": "paper-3d", "name": "Paper 3D"}]}), encoding="utf-8")
    for c in clients:
        r = run_cli("clients.py", "add", "--name", c, "--color", "#008754")
        assert r.returncode == 0, r.stderr


def mk(tmp_path, name, client="example-shop", title="launch"):
    d = tmp_path / name
    project.init(d, title, "clip.mp4", "paper-3d", client)
    return d


def write(d, name, data):
    (d / name).write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def plan_with(sigs_by_id):
    return {"beats": [{"n": 1, "options": [{"id": i, "sig": s} for i, s in sigs_by_id.items()]}]}


def test_init_registers_and_dedupes(home, tmp_path):
    setup_env(home)
    a = project.init(tmp_path / "a", "השקה", "c.mp4", "paper-3d", "example-shop")
    b = project.init(tmp_path / "b", "השקה", "c.mp4", "paper-3d", "example-shop")
    assert a["id"] != b["id"]
    reg = json.loads((home / "projects.json").read_text(encoding="utf-8"))
    assert reg["version"] == 1 and len(reg["projects"]) == 2
    assert all(p["stage"] == "B0" and p["status"] == "active" for p in reg["projects"])
    plan = json.loads((tmp_path / "a" / "plan.json").read_text(encoding="utf-8"))
    assert plan["client"] == "example-shop" and plan["style"] == "paper-3d"
    assert project.load(tmp_path / "a")["id"] == a["id"]
    with pytest.raises(KeyError):
        project.load(tmp_path / "nope")


def test_sig_stable_and_normalised():
    s = project.sig("plan", "Captions", "Gift box")
    assert s == project.sig("plan", " captions ", "gift  box")
    assert s.startswith("plan:") and len(s.split(":")[1]) == 12
    assert s != project.sig("frames", "Captions", "Gift box")


def test_blocked_union(home, tmp_path):
    setup_env(home)
    cur, other = mk(tmp_path, "cur"), mk(tmp_path, "other", title="other")
    project.record_shown(cur, "plan", 3, 1, [{"id": "B03-p1", "sig": "plan:shown", "idea_id": "I0001"}])
    taste.log("example-shop", None, "plan", "B01-p1", ["B01-p2"], "old",
              sigs={"picked": "plan:pk", "rejected": ["plan:rej"]})
    write(other, "plan.json", plan_with({"B01-p1": "plan:usedother"}))
    write(other, "picks.json", {"version": 1, "plan": {"1": {"picked": "B01-p1", "decided_by": "claude"}}})
    write(cur, "plan.json", {"beats": [{"n": 5, "options": [{"id": "B05-p2", "sig": "plan:beat5"}]}]})
    write(cur, "picks.json", {"version": 1, "plan": {"5": {"picked": "B05-p2", "decided_by": "owner"}}})
    b = project.blocked(cur, "plan", 3)
    assert {"plan:shown", "idea:I0001", "plan:rej", "plan:usedother", "plan:beat5"} <= b
    assert "plan:pk" not in b
    # beat 5 itself is not blocked by its own pick; shown sigs of beat 3 do not leak to beat 5
    assert "plan:beat5" not in project.blocked(cur, "plan", 5)
    assert "plan:shown" not in project.blocked(cur, "plan", 5)
    taste.log("example-shop", None, "frames", "x", [], "old", sigs={"picked": None, "rejected": ["frames:r"]})
    f = project.blocked(cur, "frames", 3)
    assert "plan:usedother" not in f and "plan:beat5" not in f and "frames:r" in f


def test_own_project_taste_not_blocking(home, tmp_path):
    setup_env(home)
    cur = mk(tmp_path, "cur")
    write(cur, "plan.json", {"beats": [{"n": 3, "options": [{"id": "B03-p2", "sig": "plan:mine"}]}]})
    write(cur, "picks.json", {"version": 1, "plan": {"3": {"picked": "B03-p2", "decided_by": "owner"}}})
    taste.log("example-shop", None, "frames", "B03-f1-a", [], cur.name, sigs={"picked": "frames:mine", "rejected": []})
    b = project.blocked(cur, "frames", 3)
    assert "plan:mine" not in b and "frames:mine" not in b
    assert "plan:mine" not in project.blocked(cur, "plan", 3)


def fill_sets(d, step, beat, n):
    for k in range(1, n + 1):
        project.record_shown(d, step, beat, k, [{"id": f"B{beat}-{k}", "sig": f"{step}:s{k}"}])


def test_budget_three_extra_sets(home, tmp_path):
    setup_env(home)
    d = mk(tmp_path, "cur")
    assert project.next_set(d, "plan", 3) == 1 and project.next_set(d, "plan", 3) == 1  # not burned
    fill_sets(d, "plan", 3, 4)
    assert project.next_set(d, "plan", 4) == 1
    with pytest.raises(project.BudgetExceeded):
        project.next_set(d, "plan", 3)
    with pytest.raises(project._Bad):
        project.next_set(d, "plan", 3, owner_ok_more=True)
    with pytest.raises(project._Bad):
        project.next_set(d, "plan", 3, owner_ok_more=True, owner_words="‏ ")
    assert project.next_set(d, "plan", 3, owner_ok_more=True, owner_words="כן, עוד סט") == 5
    log_ = json.loads((d / "shown.json").read_text(encoding="utf-8"))["owner_ok_more"]
    assert log_[0]["beat"] == 3 and log_[0]["step"] == "plan" and log_[0]["words"] == "כן, עוד סט" and log_[0]["at"]
    r = run_cli("project.py", "next-set", d, "--step", "plan", "--beat", 3)
    assert r.returncode == 1 and "לשאול את הבעלים" in r.stderr
    assert run_cli("project.py", "next-set", d, "--step", "plan", "--beat", 3, "--owner-ok-more").returncode == 2
    assert run_cli("project.py", "next-set", d, "--step", "plan", "--beat", 3, "--owner-ok-more",
                   "--owner-words", "כן").returncode == 0


def test_owner_words_rtl_marks(home, tmp_path):
    setup_env(home)
    d = mk(tmp_path, "cur")
    fill_sets(d, "plan", 3, 4)
    assert project.next_set(d, "plan", 3, owner_ok_more=True, owner_words="כן, תמשיך‏") == 5
    with pytest.raises(project._Bad):
        project.next_set(d, "plan", 3, owner_ok_more=True, owner_words="‏ ‎")


def test_sig_float_and_punctuation():
    assert project.sig("motion", "x", 0.15000000000000002) == project.sig("motion", "x", 0.15)
    assert project.sig("plan", "a", "קופסת־מתנה") == project.sig("plan", "a", "קופסת מתנה")
    assert project.sig("plan", "a-b", "c_d") == project.sig("plan", "a b", "c d")
    assert project.sig("plan", True, 3) == project.sig("plan", "1", "3")
    assert project.sig("plan", "a|b", "c") != project.sig("plan", "a", "b|c")


def test_pick_keeps_sig_without_plan_option(home, tmp_path):
    setup_env(home)
    cur = mk(tmp_path, "cur")
    write(cur, "plan.json", {"beats": []})
    write(cur, "picks.json", {"version": 1, "plan": {"5": {"picked": "B05-p2", "sig": "plan:kept",
                                                             "decided_by": "owner"}}})
    assert "plan:kept" in project.blocked(cur, "plan", 3)


def test_reinit_keeps_client(home, tmp_path):
    setup_env(home)
    d = mk(tmp_path, "cur")
    project.init(d, "launch", "clip.mp4", "paper-3d")
    assert project.load(d)["client"] == "example-shop"


def test_record_shown_advances_counter(home, tmp_path):
    setup_env(home)
    d = mk(tmp_path, "cur")
    project.record_shown(d, "frames", 2, 3, [{"id": "B02-f3-a", "sig": "frames:x"}])
    assert project.next_set(d, "frames", 2) == 4
    items = json.loads((d / "shown.json").read_text(encoding="utf-8"))["items"]
    assert items[0]["id"] == "B02-f3-a" and items[0]["beat"] == 2 and items[0]["set"] == 3


def test_hebrew_path(home, tmp_path):
    setup_env(home)
    d = tmp_path / "עריכה 1"
    r = run_cli("project.py", "init", d, "--title", "עריכה", "--clip", tmp_path / "קליפ.mp4", "--style", "paper-3d")
    assert r.returncode == 0, r.stderr
    e = project.load(d)
    assert "\\" not in e["path"] and e["path"].endswith("עריכה 1") and "\\" not in e["clip"]
    assert run_cli("project.py", "stage", d, "B3").returncode == 0
    assert project.load(d)["stage"] == "B3"
    assert run_cli("project.py", "stage", d, "B9").returncode == 2
    assert run_cli("project.py", "close", d, "--final", tmp_path / "f.mp4").returncode == 0
    assert project.load(d)["status"] == "final"
    assert run_cli("project.py", "show", tmp_path / "missing").returncode == 2
    assert run_cli("project.py", "list", "--json").returncode == 0
    r = run_cli("project.py", "blocked", d, "--step", "plan", "--beat", 3)
    assert r.returncode == 0

import json
import os
import subprocess
import sys

import taste as taste_mod
from conftest import REPO_ROOT, run_cli


def add_client(name="Shop Demo"):
    r = run_cli("clients.py", "add", "--name", name, "--color", "#008754")
    assert r.returncode == 0, r.stderr
    return name.lower().replace(" ", "-")


def log(client, project, picked, rejected, tags, step="frames"):
    args = ["log", "--step", step, "--picked", picked, "--project", project, "--tags", *tags]
    if client:
        args[1:1] = ["--client", client]
    for x in rejected:
        args += ["--rejected", x]
    return run_cli("taste.py", *args)


def test_log_and_summary(home):
    slug = add_client()
    for i in range(3):
        r = log("Shop Demo", f"p{i}", f"A{i}", [f"N{i}", "X"], ["picked=warm,flat", "rejected=neon,3d"])
        assert r.returncode == 0, r.stderr
    f = home / "taste" / f"{slug}.jsonl"
    lines = [json.loads(x) for x in f.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 3 and lines[0]["tags"] == {"picked": ["warm", "flat"], "rejected": ["neon", "3d"]}
    r = run_cli("taste.py", "summary", "--client", "Shop Demo", "--json")
    assert r.returncode == 0, r.stderr
    s = json.loads(r.stdout)
    assert "warm" in s["prefers"] and "neon" in s["avoids"]
    assert s["picked_tags"]["warm"] == 3 and s["rejected_tags"]["neon"] == 3
    assert s["rejected_twice"] == ["X"]


def test_no_client_bucket(home):
    r = log(None, "p1", "A", ["B"], ["picked=warm"])
    assert r.returncode == 0, r.stderr
    assert (home / "taste" / "_none.jsonl").exists()
    r = run_cli("taste.py", "summary", "--json")
    assert r.returncode == 0 and json.loads(r.stdout)["picked_tags"] == {"warm": 1}


def test_unknown_client_rejected(home):
    r = log("ghost", "p1", "A", [], [])
    assert r.returncode == 1 and "ghost" in r.stderr
    assert not (home / "taste").exists()
    r = run_cli("taste.py", "log", "--step", "frames", "--picked", "A", "--project", "p", "--style", "nostyle")
    assert r.returncode == 1


def test_bad_step_rejected(home):
    assert log(None, "p", "A", [], [], step="bogus").returncode != 0


def test_feedback_add_list_done(home):
    for t in ("first", "second"):
        r = run_cli("lessons.py", "feedback", "add", "--text", t, "--step", "frames", "--project", "x")
        assert r.returncode == 0, r.stderr
    txt = (home / "skill-feedback.md").read_text(encoding="utf-8")
    assert "- [ ] " in txt and "(step: frames, project: x) first" in txt
    r = run_cli("lessons.py", "feedback", "list")
    assert "first" in r.stdout and "second" in r.stdout
    assert run_cli("lessons.py", "feedback", "done", "1").returncode == 0
    txt = (home / "skill-feedback.md").read_text(encoding="utf-8")
    assert "- [x] " in txt and txt.count("- [ ] ") == 1
    r = run_cli("lessons.py", "feedback", "list")
    assert "first" not in r.stdout and "second" in r.stdout
    assert run_cli("lessons.py", "feedback", "done", "5").returncode == 1


def test_parallel_taste_logs(home):
    env = dict(os.environ)
    code = ("import sys; sys.path.insert(0, 'scripts'); import taste; "
            "[taste.log(None, None, 'frames', f'A{sys.argv[1]}-{i}', ['B'], 'p') for i in range(10)]")
    ps = [subprocess.Popen([sys.executable, "-c", code, str(n)], cwd=REPO_ROOT, env=env) for n in range(4)]
    assert all(p.wait() == 0 for p in ps)
    lines = (home / "taste" / "_none.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 40 and all(json.loads(x) for x in lines)


def test_summary_last_must_not_be_negative(home):
    r = run_cli("taste.py", "summary", "--last", "-1")
    assert r.returncode == 1 and "--last" in r.stderr
    assert run_cli("taste.py", "summary", "--last", "0").returncode == 0


def test_feedback_list_empty(home):
    r = run_cli("lessons.py", "feedback", "list")
    assert r.returncode == 0 and "no open items" in r.stdout


def test_taste_sigs_roundtrip(home):
    add_client("example-shop")
    taste_mod.log("example-shop", None, "plan", "B1", ["B2"], "p1", sigs={"picked": "plan:a", "rejected": ["plan:b"]})
    taste_mod.log("example-shop", None, "plan", "B3", [], "p2")  # old line, no sigs
    taste_mod.log("example-shop", None, "frames", "F1", [], "p1", sigs={"picked": "frames:z", "rejected": []})
    u = taste_mod.used_sigs("example-shop", "plan")
    assert u == {"picked": {"plan:a"}, "rejected": {"plan:b"}}
    assert taste_mod.used_sigs("example-shop", "plan", exclude_project="p1")["rejected"] == set()
    assert taste_mod.summary("example-shop")["entries"] == 3
    r = run_cli("taste.py", "log", "--client", "example-shop", "--step", "motion", "--picked", "M1",
                "--project", "p", "--sig-picked", "motion:q", "--sig-rejected", "motion:r", "--sig-rejected", "motion:s")
    assert r.returncode == 0, r.stderr
    assert taste_mod.used_sigs("example-shop", "motion") == {"picked": {"motion:q"}, "rejected": {"motion:r", "motion:s"}}

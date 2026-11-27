import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "evals"))
import checks  # noqa: E402
import run_evals  # noqa: E402


def cli(*args):
    return subprocess.run([sys.executable, str(REPO / "evals" / "run_evals.py"), *map(str, args)],
                          capture_output=True, text=True, encoding="utf-8")


def test_all_scenarios_validate():
    r = cli("validate", "--all")
    assert r.returncode == 0, r.stdout + r.stderr


def test_selftest_prepare_check_and_leak(tmp_path):
    real = tmp_path / "real"
    (real / ".claude" / "style-lab").mkdir(parents=True)
    (real / ".claude" / "skills").mkdir()
    work = tmp_path / "w"
    assert cli("prepare", "_selftest", "--work", work, "--real-home", real).returncode == 0
    prompt = (work / "PROMPT.md").read_text(encoding="utf-8")
    assert "{work}" not in prompt and "{env}" not in prompt and "STYLE_LAB_HOME=" in prompt and "STUCK" in prompt
    (work / "transcript.md").write_text("ASK: שלום\nSAY: done\n", encoding="utf-8")
    assert cli("check", "_selftest", "--work", work, "--real-home", real).returncode == 0
    (real / ".claude" / "style-lab" / "x.json").write_text("{}", encoding="utf-8")
    r = cli("check", "_selftest", "--work", work, "--real-home", real, "--json")
    assert r.returncode == 1 and any(c["type"] == "no_leak" and not c["ok"] for c in json.loads(r.stdout)["criteria"])
    (real / ".claude" / "style-lab" / "x.json").unlink()
    (real / ".claude" / "skills" / "new-skill").mkdir()
    assert cli("check", "_selftest", "--work", work, "--real-home", real).returncode == 1
    rep = cli("report", tmp_path)
    assert "FAIL _selftest" in rep.stdout


def test_prepare_refuses_work_inside_real_claude(tmp_path):
    real = tmp_path / "real"
    (real / ".claude").mkdir(parents=True)
    assert cli("prepare", "_selftest", "--work", real / ".claude" / "w", "--real-home", real).returncode == 2


def test_needs_missing_skips_exit3(tmp_path, monkeypatch):
    sc = run_evals.load("_selftest")
    sc["needs_scripts"] = ["no_such_script.py"]
    monkeypatch.setattr(run_evals, "load", lambda sid: sc)
    work = tmp_path / "w"
    assert run_evals.prepare("_selftest", work) == 3
    assert run_evals.check("_selftest", work) == 3
    assert run_evals.report(tmp_path) == 0
    assert json.loads((work / "result.json").read_text(encoding="utf-8"))["passed"] is None


def _ctx(tmp_path, transcript="", state=None):
    w = tmp_path / "w"
    (w / "home").mkdir(parents=True, exist_ok=True)
    (w / "transcript.md").write_text(transcript, encoding="utf-8")
    (w / "state.json").write_text(json.dumps(state or {}), encoding="utf-8")
    return checks.Ctx(w)


def test_each_check_pass_and_fail(tmp_path):
    ctx = _ctx(tmp_path, "ASK: רשימה של 20\nUSER: כן\nSAY: הורדתי\n")
    (ctx.work / "a.json").write_text(json.dumps({"x": {"y": [3, 2, 1]}, "a": [1, 2], "b": [3]}), encoding="utf-8")
    f = "{work}/a.json"
    cases = [
        ("file_exists", {"path": f}, {"path": "{work}/nope"}),
        ("file_count", {"glob": "{work}/*.json", "min": 1}, {"glob": "{work}/*.json", "min": 3}),
        ("json_path", {"file": f, "path": "x.y", "op": "sorted_desc"}, {"file": f, "path": "a", "op": "sorted_desc"}),
        ("json_path", {"file": f, "path": "a", "op": "disjoint_with", "value": "b"},
         {"file": f, "path": "a", "op": "disjoint_with", "value": "a"}),
        ("json_path", {"file": f, "path": "x.y", "op": "len_eq", "value": 3}, {"file": f, "path": "x.y", "op": "len_eq", "value": 2}),
        ("transcript_has", {"regex": "^ASK:", "min": 1}, {"regex": "^STUCK:", "min": 1}),
        ("transcript_lacks", {"regex": "^STUCK:"}, {"regex": "^USER:"}),
        ("transcript_order", {"before": "^ASK:", "after": "^USER:"}, {"before": "^USER:", "after": "^ASK:"}),
        ("lessons", {"max": 0}, {"min": 1}),
        ("ytdlp_calls", {"kind": "download", "max": 0}, {"kind": "download", "min": 1}),
    ]
    for t, good, bad in cases:
        assert checks.run(ctx, {"type": t, "args": good})[0], (t, good)
        assert not checks.run(ctx, {"type": t, "args": bad})[0], (t, bad)


def test_unchanged(tmp_path):
    w = tmp_path / "w"
    (w / "d").mkdir(parents=True)
    (w / "d" / "f.txt").write_text("a", encoding="utf-8")
    hashes = {"{work}/d": checks.tree_hashes(w / "d")}
    ctx = _ctx(tmp_path, state={"hashes": hashes})
    assert checks.run(ctx, {"type": "unchanged", "args": {"path": "{work}/d"}})[0]
    (w / "d" / "f.txt").write_text("b", encoding="utf-8")
    assert not checks.run(ctx, {"type": "unchanged", "args": {"path": "{work}/d"}})[0]


def test_ytdlp_after_consent(tmp_path):
    ctx = _ctx(tmp_path)
    pack = tmp_path / "pack"
    (pack / "refs").mkdir(parents=True)
    now = time.time()
    at = time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime(now))
    (pack / "refs" / "proposal.json").write_text(json.dumps({"consent": {"at": at}}), encoding="utf-8")
    log = ctx.work / "ytdlp.log.jsonl"
    log.write_text(json.dumps({"argv": ["-o", "x", "u"], "t": now - 100}) + "\n", encoding="utf-8")
    assert not checks.ytdlp_after_consent(ctx, proposal=str(pack / "refs" / "proposal.json"))[0]
    log.write_text(json.dumps({"argv": ["-o", "x", "u"], "t": now + 5}) + "\n", encoding="utf-8")
    assert checks.ytdlp_after_consent(ctx, proposal=str(pack / "refs" / "proposal.json"))[0]

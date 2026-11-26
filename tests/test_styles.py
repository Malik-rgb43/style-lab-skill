import os
import json
import shutil
import subprocess
import sys
from pathlib import Path

from conftest import REPO_ROOT, run_cli

FIX = Path(__file__).parent / "fixtures" / "styles_v1.json"
NEW = ("kind", "aspect", "name_source", "version", "rules_count")


def seed(home):
    home.mkdir(parents=True, exist_ok=True)
    shutil.copy(FIX, home / "styles.json")


def test_migrate_v1_to_v2(home):
    seed(home)
    orig = json.loads(FIX.read_text(encoding="utf-8"))
    r = run_cli("styles.py", "migrate")
    assert r.returncode == 0, r.stderr
    d = json.loads((home / "styles.json").read_text(encoding="utf-8"))
    assert d["version"] == 2
    for new, old in zip(d["styles"], orig["styles"]):
        for k, v in old.items():
            assert new[k] == v
        assert (new["kind"], new["aspect"], new["name_source"], new["version"], new["rules_count"]) == ("trained", "9:16", "user", 1, 0)
    assert list(home.glob("styles.json.bak-*"))


def test_migrate_twice_noop(home):
    seed(home)
    run_cli("styles.py", "migrate")
    r = run_cli("styles.py", "migrate")
    assert r.returncode == 0
    assert "already v2" in r.stdout
    assert len(list(home.glob("styles.json.bak-*"))) == 1


def test_migrate_missing_registry(home):
    r = run_cli("styles.py", "migrate")
    assert r.returncode == 0
    assert "no registry yet" in r.stdout
    assert not (home / "styles.json").exists()


def test_add_hebrew_name_autoslug(home, tmp_path):
    r = run_cli("styles.py", "add", "--name", "סגנון חדש", "--pack", tmp_path)
    assert r.returncode == 0, r.stderr
    d = json.loads((home / "styles.json").read_text(encoding="utf-8"))
    assert d["version"] == 2
    assert d["styles"][0]["slug"] == "sgnvn-chdsh"
    r = run_cli("styles.py", "show", "סגנון חדש")
    assert r.returncode == 0
    assert "sgnvn-chdsh" in r.stdout and '"clients"' in r.stdout


def test_env_registry_still_wins(home, tmp_path, monkeypatch):
    reg = tmp_path / "elsewhere" / "styles.json"
    monkeypatch.setenv("STYLE_LAB_REGISTRY", str(reg))
    r = run_cli("styles.py", "add", "x1", "--name", "X One", "--pack", tmp_path)
    assert r.returncode == 0, r.stderr
    assert reg.exists()
    assert not (home / "styles.json").exists()


def test_set_aspect_validates(home, tmp_path):
    run_cli("styles.py", "add", "x1", "--name", "X One", "--pack", tmp_path)
    r = run_cli("styles.py", "set", "x1", "aspect=3:2")
    assert r.returncode == 1
    assert "9:16" in r.stderr and "16:9" in r.stderr
    assert run_cli("styles.py", "set", "x1", "aspect=1:1", "rules_count=3").returncode == 0


def test_parallel_adds(home, tmp_path):
    procs = [subprocess.Popen(
        [sys.executable, str(Path("scripts") / "styles.py"), "add", f"p{i}", "--name", f"P {i}", "--pack", str(tmp_path)],
        cwd=REPO_ROOT, env=dict(os.environ), stdout=subprocess.PIPE, stderr=subprocess.PIPE) for i in range(6)]
    for p in procs:
        p.communicate()
        assert p.returncode == 0
    d = json.loads((home / "styles.json").read_text(encoding="utf-8"))
    assert len(d["styles"]) == 6


def test_list_has_clients_column(home):
    seed(home)
    r = run_cli("styles.py", "list")
    assert r.returncode == 0
    assert all(line.split("\t")[4].startswith("clients:") for line in r.stdout.strip().splitlines())


def test_write_on_v1_creates_backup(home):
    seed(home)
    r = run_cli("styles.py", "set", "gabi-organic", "aspect=16:9")
    assert r.returncode == 0, r.stderr
    baks = list(home.glob("styles.json.bak-*"))
    assert len(baks) == 1
    assert baks[0].read_bytes() == FIX.read_bytes()
    d = json.loads((home / "styles.json").read_text(encoding="utf-8"))
    assert d["version"] == 2 and d["styles"][0]["aspect"] == "16:9"


def test_clients_lookup_failure_does_not_crash(home, monkeypatch, capsys):
    import types
    import styles
    seed(home)

    def boom(slug):
        raise ValueError("corrupt")
    monkeypatch.setitem(sys.modules, "clients", types.SimpleNamespace(linked_clients=boom))
    monkeypatch.setattr(sys, "argv", ["styles.py", "list"])
    assert styles.main() == 0
    out = capsys.readouterr()
    assert "gabi-organic" in out.out and "warning" in out.err


def _seed_styles(home, styles):
    home.mkdir(parents=True, exist_ok=True)
    rows = [{"slug": s, "name": n, "aliases": al, "pack": str(home), "status": "training"} for s, n, al in styles]
    (home / "styles.json").write_text(json.dumps({"version": 2, "styles": rows}), encoding="utf-8")


def _slugs(home):
    return [s["slug"] for s in json.loads((home / "styles.json").read_text(encoding="utf-8"))["styles"]]


def test_find_prefers_slug_over_alias(home):
    # aaa carries the alias "bbb"; `remove bbb` must remove the style whose slug is bbb
    _seed_styles(home, [("aaa", "Alpha", ["bbb"]), ("bbb", "Beta", [])])
    r = run_cli("styles.py", "remove", "bbb")
    assert r.returncode == 0, r.stderr
    assert _slugs(home) == ["aaa"]


def test_find_prefers_name_over_alias():
    import styles
    data = {"styles": [{"slug": "aaa", "name": "Alpha", "aliases": ["Zed"]},
                       {"slug": "ccc", "name": "Zed", "aliases": []}]}
    assert styles.find(data, "zed")["slug"] == "ccc"
    assert styles.find(data, "AAA")["slug"] == "aaa"


def test_alias_collision_rejected(home, tmp_path):
    run_cli("styles.py", "add", "aaa", "--name", "Alpha", "--pack", tmp_path)
    run_cli("styles.py", "add", "bbb", "--name", "Beta", "--pack", tmp_path)
    run_cli("styles.py", "set", "bbb", "aliases=bee")
    for bad in ("bbb", "beta", "BEE"):
        r = run_cli("styles.py", "set", "aaa", f"aliases=ok,{bad}")
        assert r.returncode == 1 and "already" in r.stderr and "Traceback" not in r.stderr, bad
    d = json.loads((home / "styles.json").read_text(encoding="utf-8"))
    assert d["styles"][0]["aliases"] == []
    assert run_cli("styles.py", "set", "aaa", "aliases=alpha,a1").returncode == 0  # own name is fine
    r = run_cli("styles.py", "set", "bbb", "aliases=a1")
    assert r.returncode == 1 and "already" in r.stderr


def test_empty_name_rejected(home, tmp_path):
    r = run_cli("styles.py", "add", "--name", "  ", "--pack", tmp_path)
    assert r.returncode == 1 and "name" in r.stderr
    assert not (home / "styles.json").exists()
    run_cli("styles.py", "add", "aaa", "--name", "Alpha", "--pack", tmp_path)
    r = run_cli("styles.py", "set", "aaa", "name=")
    assert r.returncode == 1 and "name" in r.stderr
    assert json.loads((home / "styles.json").read_text(encoding="utf-8"))["styles"][0]["name"] == "Alpha"


def _dangling_setup(home, tmp_path):
    run_cli("styles.py", "add", "aaa", "--name", "Alpha", "--pack", tmp_path, "--status", "ready")
    run_cli("styles.py", "add", "bbb", "--name", "Beta", "--pack", tmp_path)
    assert run_cli("clients.py", "add", "--name", "Demo Shop").returncode == 0
    assert run_cli("clients.py", "link", "demo-shop", "aaa").returncode == 0
    assert run_cli("clients.py", "link", "demo-shop", "bbb").returncode == 0


def test_remove_reports_linked_clients(home, tmp_path):
    _dangling_setup(home, tmp_path)
    r = run_cli("styles.py", "remove", "aaa")
    assert r.returncode == 0 and "1 client(s) still link to it" in r.stdout
    run_cli("styles.py", "add", "ccc", "--name", "Gamma", "--pack", tmp_path)
    r = run_cli("styles.py", "remove", "ccc")
    assert r.returncode == 0 and "still link" not in r.stdout


def test_list_and_check_report_dangling_links(home, tmp_path):
    _dangling_setup(home, tmp_path)
    run_cli("styles.py", "remove", "aaa")
    line = "dangling: client Demo Shop -> style aaa (missing style)"
    r = run_cli("styles.py", "list")
    assert r.returncode == 0 and line in r.stderr.splitlines() and line not in r.stdout
    assert r.stdout.strip().splitlines()[0].startswith("bbb\t")
    r = run_cli("styles.py", "check")
    assert r.returncode == 0 and line in r.stdout.splitlines()  # a dangling link alone does not fail
    assert "ok" not in r.stdout.splitlines()
    run_cli("styles.py", "remove", "bbb")
    r = run_cli("styles.py", "list")
    assert "no styles yet" in r.stdout and "style bbb (missing style)" in r.stderr


def test_check_still_fails_on_missing_ready_path(home, tmp_path):
    _dangling_setup(home, tmp_path)
    run_cli("styles.py", "remove", "bbb")
    run_cli("styles.py", "set", "aaa", f"pack={tmp_path / 'gone'}")
    r = run_cli("styles.py", "check")
    assert r.returncode == 1 and "dangling: client Demo Shop -> style bbb (missing style)" in r.stdout


def test_require_unrequire_roundtrip(home, tmp_path):
    run_cli("styles.py", "add", "--name", "Demo", "--pack", tmp_path)
    r = run_cli("styles.py", "require", "demo", "--name", "higgsfield", "--why", "AI B-roll",
                "--level", "recommended", "--fallback", "stock footage")
    assert r.returncode == 0, r.stderr
    d = json.loads((home / "styles.json").read_text(encoding="utf-8"))
    assert d["styles"][0]["requires"] == [{"name": "higgsfield", "why": "AI B-roll",
                                           "level": "recommended", "fallback": "stock footage"}]
    assert "higgsfield" in run_cli("styles.py", "show", "demo").stdout
    run_cli("styles.py", "require", "demo", "--name", "higgsfield", "--why", "x", "--level", "required")
    d = json.loads((home / "styles.json").read_text(encoding="utf-8"))
    assert len(d["styles"][0]["requires"]) == 1 and d["styles"][0]["requires"][0]["level"] == "required"
    assert run_cli("styles.py", "unrequire", "demo", "--name", "higgsfield").returncode == 0
    d = json.loads((home / "styles.json").read_text(encoding="utf-8"))
    assert d["styles"][0]["requires"] == []
    assert run_cli("styles.py", "unrequire", "demo", "--name", "higgsfield").returncode == 1
    assert run_cli("styles.py", "require", "demo", "--name", "a", "--why", "b", "--level", "bad").returncode != 0


def test_migrate_adds_requires(home):
    seed(home)
    run_cli("styles.py", "migrate")
    d = json.loads((home / "styles.json").read_text(encoding="utf-8"))
    assert all(s["requires"] == [] for s in d["styles"])
    d["styles"][0]["requires"].append({"name": "x"})
    assert d["styles"][1]["requires"] == []

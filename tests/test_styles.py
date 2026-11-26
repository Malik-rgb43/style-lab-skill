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
    import conftest
    conftest._ENV["env"] = dict(__import__("os").environ)
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
        cwd=REPO_ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE) for i in range(6)]
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

import json
import subprocess

import pytest

import doctor
import store


class _Done:
    def __init__(self, out="tool 1.0\nline2", rc=0):
        self.stdout, self.stderr, self.returncode = out, "", rc


@pytest.fixture
def everything(monkeypatch):
    monkeypatch.setattr(doctor.shutil, "which", lambda n: f"/bin/{n}")
    monkeypatch.setattr(doctor.subprocess, "run", lambda *a, **k: _Done())
    monkeypatch.setattr(doctor.importlib.util, "find_spec", lambda n: object())
    monkeypatch.setenv("RUNPOD_API_KEY", "x")
    monkeypatch.setenv("RUNPOD_ENDPOINT_ID", "y")


def caps():
    return store.read_json(store.path("capabilities.json"), None)


def test_all_present(home, everything):
    assert doctor.main([]) == 0
    c = caps()
    assert c["checked"]
    assert set(c["stages"]) == set(doctor.STAGES)
    assert all(s["missing"] == [] for s in c["stages"].values())


def test_missing_ffmpeg(home, everything, monkeypatch):
    monkeypatch.setattr(doctor.shutil, "which", lambda n: None if n == "ffmpeg" else f"/bin/{n}")
    doctor.main([])
    c = caps()
    assert c["stages"]["train"]["missing"] == ["ffmpeg"]
    assert c["local"]["ffmpeg"]["hint"]


def test_hyperframes_timeout(home, everything, monkeypatch):
    def boom(*a, **k):
        raise subprocess.TimeoutExpired("npx", 20)
    monkeypatch.setattr(doctor.subprocess, "run", boom)
    doctor.main([])
    assert caps()["local"]["hyperframes"]["ok"] is False


def test_env_values_never_written(home, everything, monkeypatch):
    monkeypatch.setenv("RUNPOD_API_KEY", "secret123")
    doctor.main([])
    assert "secret123" not in store.path("capabilities.json").read_text(encoding="utf-8")
    assert caps()["stages"]["cloud_asr"]["paid"] == ["RUNPOD_API_KEY", "RUNPOD_ENDPOINT_ID"]


def test_session_set_preserved(home, everything):
    assert doctor.main(["session", "--set", "higgsfield=ok", "--note", "mcp"]) == 0
    doctor.main([])
    s = caps()["session"]["higgsfield"]
    assert s["ok"] is True and s["note"] == "mcp"


def test_session_clear_and_bad_value(home, everything):
    doctor.main(["session", "--set", "pexels=ok"])
    assert doctor.main(["session", "--set", "pexels=maybe"]) == 1
    doctor.main(["session", "--clear"])
    assert caps()["session"] == {}


def test_json_output(home, everything, capsys):
    doctor.main(["--json"])
    assert "stages" in json.loads(capsys.readouterr().out)

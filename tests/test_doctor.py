import json
import subprocess

import pytest

import doctor
import store


CALLS = []


class FakePopen:
    out, rc, timeout = "tool 1.0\nline2", 0, False

    def __init__(self, cmd, **kw):
        CALLS.append((list(cmd), kw))
        self.pid, self.returncode = 1, type(self).rc

    def communicate(self, timeout=None):
        if type(self).timeout and timeout == 45:
            type(self).timeout = False
            raise subprocess.TimeoutExpired("npx", timeout)
        return type(self).out, ""

    def kill(self):
        pass


@pytest.fixture
def everything(monkeypatch):
    CALLS.clear()
    FakePopen.out, FakePopen.rc, FakePopen.timeout = "tool 1.0\nline2", 0, False
    monkeypatch.setattr(doctor.subprocess, "Popen", FakePopen)
    monkeypatch.setattr(doctor, "_kill_tree", lambda p: None)
    monkeypatch.setattr(doctor.shutil, "which", lambda n: f"/bin/{n}")
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
    FakePopen.timeout = True
    doctor.main([])
    h = caps()["local"]["hyperframes"]
    assert h["ok"] is False and h["note"] == "check timed out"


def test_hyperframes_noninteractive(home, everything, monkeypatch):
    monkeypatch.setattr(doctor.shutil, "which", lambda n: None if n == "hyperframes" else f"/bin/{n}")
    doctor.main([])
    cmd, kw = next(c for c in CALLS if "hyperframes" in c[0])
    assert "--no" in cmd and "--yes" not in cmd and kw["stdin"] == subprocess.DEVNULL
    assert caps()["local"]["hyperframes"]["ok"] is True


def test_hyperframes_npx_not_cached_never_downloads(home, everything, monkeypatch):
    monkeypatch.setattr(doctor.shutil, "which", lambda n: None if n == "hyperframes" else f"/bin/{n}")
    FakePopen.rc = 1
    h = doctor.check_hyperframes()
    cmd = next(c for c, _ in CALLS if "hyperframes" in c)
    assert cmd[1] == "--no" and "--yes" not in cmd
    assert h["ok"] is False and h["note"].startswith("not installed")
    assert "ask the user first" in h["note"]


def test_hyperframes_failed_note(home, everything, monkeypatch):
    FakePopen.rc = 3  # a global hyperframes binary that fails its --version
    h = doctor.check_hyperframes()
    assert h["note"] == "check failed (exit 3)"


def test_blender_found_in_program_files(home, everything, monkeypatch, tmp_path):
    monkeypatch.setattr(doctor.shutil, "which", lambda n: None if n == "blender" else f"/bin/{n}")
    monkeypatch.delenv("BLENDER_PATH", raising=False)
    for v in ("4.2", "5.2"):
        d = tmp_path / "Blender Foundation" / f"Blender {v}"
        d.mkdir(parents=True)
        (d / "blender.exe").write_text("x")
    monkeypatch.setenv("ProgramFiles", str(tmp_path))
    doctor.main([])
    b = caps()["local"]["blender"]
    assert b["ok"] is True and "Blender 5.2" in b["path"]
    assert caps()["local"]["ffmpeg"]["path"] == "/bin/ffmpeg"


def test_torch_cuda_not_probed_by_default(home, everything):
    doctor.main([])
    assert caps()["local"]["torch"]["cuda"] == "not probed"


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

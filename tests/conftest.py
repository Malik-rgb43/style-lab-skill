import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

# Real user profile, captured before any patching.
REAL_HOME = Path.home().resolve()
# On Windows the temp dir lives inside the profile, so guard the real config dir.
REAL_CLAUDE = REAL_HOME / ".claude"


def _inside(path, parent):
    try:
        Path(path).resolve().relative_to(parent)
        return True
    except ValueError:
        return False


@pytest.fixture(autouse=True)
def _isolate_user_dirs(tmp_path_factory, monkeypatch):
    """Every test gets a fresh fake user profile; nothing can reach the real one."""
    fake = tmp_path_factory.mktemp("userprofile")
    monkeypatch.setenv("HOME", str(fake))
    monkeypatch.setenv("USERPROFILE", str(fake))
    if os.name == "nt":
        monkeypatch.setenv("HOMEDRIVE", fake.drive)
        monkeypatch.setenv("HOMEPATH", str(fake)[len(fake.drive):])
    monkeypatch.setenv("STYLE_LAB_HOME", str(fake / ".claude" / "style-lab"))
    monkeypatch.delenv("STYLE_LAB_REGISTRY", raising=False)
    return fake


@pytest.fixture
def home(tmp_path, monkeypatch):
    h = tmp_path / "home"
    monkeypatch.setenv("STYLE_LAB_HOME", str(h))
    monkeypatch.delenv("STYLE_LAB_REGISTRY", raising=False)
    return h


def run_cli(script, *args):
    env = dict(os.environ)
    sl = env.get("STYLE_LAB_HOME")
    assert sl, "run_cli: STYLE_LAB_HOME is unset (test isolation broken)"
    assert not _inside(sl, REAL_CLAUDE), f"run_cli: STYLE_LAB_HOME {sl} is inside the real user profile (~/.claude)"
    return subprocess.run(
        [sys.executable, str(Path("scripts") / script), *map(str, args)],
        cwd=REPO_ROOT, env=env, capture_output=True, text=True, encoding="utf-8",
    )


_LOCAL_HOSTS = ("127.0.0.1", "::1", "localhost")


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    """Tests never reach the network: connecting anywhere but localhost raises."""
    real_connect, real_connect_ex = socket.socket.connect, socket.socket.connect_ex

    def _allowed(address):
        if isinstance(address, tuple) and address:
            return address[0] in _LOCAL_HOSTS
        return True  # AF_UNIX paths and the like are not network

    def connect(self, address):
        if not _allowed(address):
            raise RuntimeError("network blocked in tests")
        return real_connect(self, address)

    def connect_ex(self, address):
        if not _allowed(address):
            raise RuntimeError("network blocked in tests")
        return real_connect_ex(self, address)

    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(socket.socket, "connect_ex", connect_ex)

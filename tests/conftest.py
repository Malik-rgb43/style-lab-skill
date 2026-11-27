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
    env = child_env()
    sl = env.get("STYLE_LAB_HOME")
    assert sl, "run_cli: STYLE_LAB_HOME is unset (test isolation broken)"
    assert not _inside(sl, REAL_CLAUDE), f"run_cli: STYLE_LAB_HOME {sl} is inside the real user profile (~/.claude)"
    return subprocess.run(
        [sys.executable, str(Path("scripts") / script), *map(str, args)],
        cwd=REPO_ROOT, env=env, capture_output=True, text=True, encoding="utf-8",
    )


_LOCAL_HOSTS = ("127.0.0.1", "::1", "localhost")
_BIND_OK = (None, "", "0.0.0.0", "::")
NETGUARD_DIR = Path(__file__).resolve().parent / "_netguard"


def child_env(env=None):
    """Env for a child Python: the network guard (tests/_netguard/sitecustomize.py) loads first."""
    env = dict(os.environ if env is None else env)
    old = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(NETGUARD_DIR) + (os.pathsep + old if old else "")
    return env


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    """Tests never reach the network, in-process or in a child Python: only localhost is allowed."""
    real_connect, real_connect_ex = socket.socket.connect, socket.socket.connect_ex
    real_sendto, real_gai = socket.socket.sendto, socket.getaddrinfo

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

    def sendto(self, data, *args):
        if args and not _allowed(args[-1]):
            raise RuntimeError("network blocked in tests")
        return real_sendto(self, data, *args)

    def getaddrinfo(host, *args, **kw):
        if isinstance(host, bytes):
            host = host.decode("ascii", "ignore")
        if host not in _LOCAL_HOSTS and host not in _BIND_OK:
            raise RuntimeError("network blocked in tests")
        return real_gai(host, *args, **kw)

    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(socket.socket, "connect_ex", connect_ex)
    monkeypatch.setattr(socket.socket, "sendto", sendto)
    monkeypatch.setattr(socket, "getaddrinfo", getaddrinfo)
    # every child Python the test spawns (run_cli or a plain subprocess) inherits the guard
    monkeypatch.setenv("PYTHONPATH", child_env()["PYTHONPATH"])

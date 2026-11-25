import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

_ENV = {}


@pytest.fixture
def home(tmp_path, monkeypatch):
    h = tmp_path / "home"
    monkeypatch.setenv("STYLE_LAB_HOME", str(h))
    monkeypatch.delenv("STYLE_LAB_REGISTRY", raising=False)
    _ENV["env"] = dict(os.environ)
    return h


def run_cli(script, *args):
    env = _ENV.get("env", dict(os.environ))
    return subprocess.run(
        [sys.executable, str(Path("scripts") / script), *map(str, args)],
        cwd=REPO_ROOT, env=env, capture_output=True, text=True, encoding="utf-8",
    )

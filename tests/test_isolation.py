import os
import socket
import subprocess
import sys
from pathlib import Path

import conftest


def test_home_is_not_real_home():
    assert Path.home().resolve() != conftest.REAL_HOME
    assert not conftest._inside(os.environ["STYLE_LAB_HOME"], conftest.REAL_CLAUDE)


def test_subprocess_default_home_is_tmp():
    r = subprocess.run(
        [sys.executable, "-c", "from pathlib import Path;print(Path.home() / '.claude' / 'style-lab')"],
        env=dict(os.environ), capture_output=True, text=True, encoding="utf-8",
    )
    assert r.returncode == 0, r.stderr
    default = Path(r.stdout.strip())
    assert not conftest._inside(default, conftest.REAL_CLAUDE)
    assert default == Path(os.environ["STYLE_LAB_HOME"])


def test_tests_block_network():
    import pytest
    s = socket.socket()
    try:
        with pytest.raises(RuntimeError, match="network blocked"):
            s.connect(("203.0.113.1", 80))
    finally:
        s.close()

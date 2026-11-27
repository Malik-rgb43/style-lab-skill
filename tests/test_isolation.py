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


def test_dns_blocked_but_localhost_works():
    import pytest
    with pytest.raises(RuntimeError, match="network blocked"):
        socket.getaddrinfo("example.com", 80)
    assert socket.getaddrinfo("localhost", 80)
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    c = socket.socket()
    try:
        c.connect(srv.getsockname())
    finally:
        c.close()
        srv.close()


def test_child_python_cannot_reach_network():
    code = (
        "import urllib.request,sys\n"
        "try:\n"
        "    urllib.request.urlopen('http://example.com', timeout=5)\n"
        "except BaseException as e:\n"
        "    print('BLOCKED', type(e).__name__, e); sys.exit(0)\n"
        "print('REACHED'); sys.exit(1)\n"
    )
    r = subprocess.run([sys.executable, "-c", code], env=dict(os.environ), capture_output=True,
                       text=True, encoding="utf-8", timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "network blocked" in r.stdout


def test_child_python_localhost_works():
    code = (
        "import socket\n"
        "s=socket.socket(); s.bind(('127.0.0.1',0)); s.listen(1)\n"
        "c=socket.create_connection(('localhost', s.getsockname()[1]), timeout=5); print('OK')\n"
    )
    r = subprocess.run([sys.executable, "-c", code], env=dict(os.environ), capture_output=True,
                       text=True, encoding="utf-8", timeout=60)
    assert "OK" in r.stdout, r.stdout + r.stderr

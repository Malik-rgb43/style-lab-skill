"""Loaded by every Python child process the tests spawn (tests/conftest.py puts this folder on PYTHONPATH).

Blocks network use outside localhost, so a CLI under test cannot reach the internet either.
"""
import socket

_LOCAL = ("127.0.0.1", "::1", "localhost")
_BIND_OK = (None, "", "0.0.0.0", "::")
_MSG = "network blocked in tests"


def _addr_ok(address):
    if isinstance(address, tuple) and address:
        return address[0] in _LOCAL
    return True  # AF_UNIX paths are not network


def _install():
    real_connect, real_connect_ex = socket.socket.connect, socket.socket.connect_ex
    real_sendto, real_gai = socket.socket.sendto, socket.getaddrinfo

    def connect(self, address):
        if not _addr_ok(address):
            raise RuntimeError(_MSG)
        return real_connect(self, address)

    def connect_ex(self, address):
        if not _addr_ok(address):
            raise RuntimeError(_MSG)
        return real_connect_ex(self, address)

    def sendto(self, data, *args):
        if args and not _addr_ok(args[-1]):
            raise RuntimeError(_MSG)
        return real_sendto(self, data, *args)

    def getaddrinfo(host, *args, **kw):
        if isinstance(host, bytes):
            host = host.decode("ascii", "ignore")
        if host not in _LOCAL and host not in _BIND_OK:
            raise RuntimeError(_MSG)
        return real_gai(host, *args, **kw)

    socket.socket.connect, socket.socket.connect_ex, socket.socket.sendto = connect, connect_ex, sendto
    socket.getaddrinfo = getaddrinfo


_install()

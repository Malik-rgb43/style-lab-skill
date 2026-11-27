import hashlib
import io

import pytest

import fetch_model as fm


class Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


def fake(data):
    def opener(url, timeout=None):
        assert url.startswith("https://")
        return Resp(data)
    return opener


@pytest.fixture
def model(monkeypatch):
    data = b"onnx" * 100
    m = dict(fm.MODELS["yunet"], bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
    monkeypatch.setitem(fm.MODELS, "yunet", m)
    return data


def test_needs_yes_and_known_name(home, model, capsys):
    assert fm.main(["yunet"]) == 2 and "--yes" in capsys.readouterr().err
    assert fm.main(["nope", "--yes"]) == 2
    assert not fm.target("yunet").exists()


def test_fetch_verifies_hash(home, model):
    p = fm.fetch("yunet", opener=fake(model))
    assert p.read_bytes() == model and fm.is_ready("yunet")
    assert fm.fetch("yunet", opener=lambda *a, **k: pytest.fail("no second download")) == p


def test_bad_hash_leaves_nothing(home, model):
    with pytest.raises(ValueError):
        fm.fetch("yunet", opener=fake(b"x" * len(model)))
    assert not fm.target("yunet").exists()
    assert not list(fm.target("yunet").parent.glob("*.part"))


def test_list(home, capsys):
    assert fm.main(["list"]) == 0
    out = capsys.readouterr().out
    assert "yunet" in out and "232,589" in out and "not downloaded" in out

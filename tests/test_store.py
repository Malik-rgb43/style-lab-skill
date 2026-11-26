import json
import os
import re
import threading
import time

import pytest

import store


def test_home_uses_env(home):
    assert store.home() == home
    assert home.is_dir()


def test_write_then_read_roundtrip(home):
    p = store.path("a.json")
    store.write_json(p, {"name": "קורס"})
    assert store.read_json(p, {}) == {"name": "קורס"}
    assert "קורס" in p.read_text(encoding="utf-8")


def test_read_missing_returns_copy_of_default(home):
    default = {"items": []}
    got = store.read_json(store.path("missing.json"), default)
    got["items"].append(1)
    assert default == {"items": []}


def test_update_json_parallel(home):
    p = store.path("list.json")

    def worker(base):
        for i in range(25):
            store.update_json(p, [], lambda d, v=base * 100 + i: d.append(v))

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    data = store.read_json(p, [])
    assert len(data) == 200 and len(set(data)) == 200


def test_stale_lock_taken_over(home):
    p = store.path("x.json")
    lk = p.with_suffix(p.suffix + ".lock")
    lk.write_text("")
    old = time.time() - 120
    os.utime(lk, (old, old))
    with store.lock(p, timeout=1):
        pass


def test_lock_timeout(home):
    p = store.path("y.json")
    with store.lock(p):
        with pytest.raises(TimeoutError):
            with store.lock(p, timeout=0.2):
                pass


def test_append_jsonl(home):
    p = store.path("log.jsonl")
    store.append_jsonl(p, {"a": "קורס"})
    store.append_jsonl(p, {"a": 2})
    lines = p.read_text(encoding="utf-8").splitlines()
    assert [json.loads(x) for x in lines] == [{"a": "קורס"}, {"a": 2}]


def test_slugify():
    assert store.slugify("Gabi Organic", set()) == "gabi-organic"
    assert store.slugify("קורס גבי", set()) == "kvrs-gby"
    assert store.slugify("Gabi Organic", {"gabi-organic"}) == "gabi-organic-2"
    assert re.match(r"^[a-z0-9][a-z0-9-]*$", store.slugify("!!!", set()))
    assert store.slugify("!!!", {"item"}) == "item-2"


def test_slugify_mixed_and_hebrew_words():
    assert store.slugify("קורס AI", set()) == "kvrs-ai"
    assert store.slugify("גבי-אורגני", set()) == "gby-vrgny"
    assert store.slugify("קורס 2", set()) == "kvrs-2"


def test_read_json_tolerates_bom(home):
    p = store.path("bom.json")
    p.write_bytes(b"\xef\xbb\xbf" + json.dumps({"a": "קורס"}, ensure_ascii=False).encode("utf-8"))
    assert store.read_json(p, {}) == {"a": "קורס"}


def test_lock_real_permission_error_not_timeout(home, monkeypatch):
    def boom(*a, **k):
        raise PermissionError("denied")
    monkeypatch.setattr(store.os, "open", boom)
    with pytest.raises(PermissionError):
        with store.lock(store.path("z.json"), timeout=5):
            pass


def test_readers_during_writers(home):
    import subprocess
    import sys
    from pathlib import Path
    scripts = str(Path(store.__file__).resolve().parent)
    p = store.path("rw.json")
    code = (
        "import sys; sys.path.insert(0, sys.argv[1]); import store\n"
        "p = store.path('rw.json'); w = int(sys.argv[2])\n"
        "for i in range(25):\n"
        "    store.update_json(p, [], lambda d, v=w*100+i: d.append(v))\n"
    )
    procs = [subprocess.Popen([sys.executable, "-c", code, scripts, str(w)], env=dict(os.environ)) for w in range(4)]
    errors = []
    stop = threading.Event()

    def reader():
        while not stop.is_set():
            try:
                store.read_json(p, [])
            except Exception as e:  # noqa
                errors.append(repr(e))

    readers = [threading.Thread(target=reader) for _ in range(3)]
    for t in readers:
        t.start()
    codes = [pr.wait(timeout=120) for pr in procs]
    stop.set()
    for t in readers:
        t.join()
    assert errors == []
    assert codes == [0, 0, 0, 0]
    data = store.read_json(p, [])
    assert len(data) == 100 and len(set(data)) == 100

import json

from conftest import run_cli


def mkstyle(tmp_path, name="Alpha"):
    pack = tmp_path / ("pack-" + name)
    pack.mkdir(exist_ok=True)
    r = run_cli("styles.py", "add", "--name", name, "--pack", pack)
    assert r.returncode == 0, r.stderr
    return name.lower()


def profile(home, slug):
    return json.loads((home / "clients" / slug / "client.json").read_text(encoding="utf-8"))


def add_client(name="חנות הדגמה", *extra):
    r = run_cli("clients.py", "add", "--name", name, "--color", "#008754", *extra)
    assert r.returncode == 0, r.stderr
    return r


def test_add_and_show_hebrew(home):
    add_client()
    p = profile(home, "chnvt-hdgmh")
    assert p["fonts"] == {"heading": "Karantina 700", "body": "Heebo", "mono": "Consolas"}
    assert p["brand_colors"] == ["#008754"]
    assert p["palette"] is None and p["styles"] == []
    r = run_cli("clients.py", "show", "חנות הדגמה")
    assert r.returncode == 0 and "חנות הדגמה" in r.stdout


def test_link_requires_existing_style(home):
    add_client()
    r = run_cli("clients.py", "link", "chnvt-hdgmh", "ghost")
    assert r.returncode == 1 and "no style" in r.stderr
    assert profile(home, "chnvt-hdgmh")["styles"] == []


def test_link_default_is_single(home, tmp_path):
    a, b = mkstyle(tmp_path, "Alpha"), mkstyle(tmp_path, "Beta")
    add_client()
    assert run_cli("clients.py", "link", "chnvt-hdgmh", a, "--default").returncode == 0
    assert run_cli("clients.py", "link", "chnvt-hdgmh", b, "--default").returncode == 0
    st = {s["slug"]: s["default"] for s in profile(home, "chnvt-hdgmh")["styles"]}
    assert st == {a: False, b: True}


def test_linked_clients_computed(home, tmp_path):
    a = mkstyle(tmp_path)
    add_client()
    run_cli("clients.py", "link", "chnvt-hdgmh", a, "--default")
    r = run_cli("styles.py", "show", a)
    assert json.loads(r.stdout)["clients"] == [
        {"slug": "chnvt-hdgmh", "name": "חנות הדגמה", "default": True}]


def test_dangling_link_reported(home, tmp_path):
    a = mkstyle(tmp_path)
    add_client()
    run_cli("clients.py", "link", "chnvt-hdgmh", a)
    assert run_cli("styles.py", "remove", a).returncode == 0
    r = run_cli("clients.py", "show", "chnvt-hdgmh")
    assert r.returncode == 0 and "(missing style)" in r.stdout
    assert run_cli("styles.py", "list").returncode == 0


def test_remove_keeps_folder(home):
    add_client()
    r = run_cli("clients.py", "remove", "chnvt-hdgmh")
    assert r.returncode == 0 and "folder kept" in r.stdout
    assert (home / "clients" / "chnvt-hdgmh").exists()
    assert json.loads((home / "clients.json").read_text(encoding="utf-8"))["clients"] == []


def test_bad_color_rejected(home):
    r = run_cli("clients.py", "add", "--name", "X", "--color", "rgb(1,2,3)")
    assert r.returncode == 1 and "hex colour like #008754" in r.stderr
    assert not (home / "clients.json").exists()


def test_set_and_unlink(home, tmp_path):
    a = mkstyle(tmp_path)
    add_client()
    r = run_cli("clients.py", "set", "chnvt-hdgmh", "languages=he,en", "fonts.body=Rubik", "tone=warm")
    assert r.returncode == 0, r.stderr
    p = profile(home, "chnvt-hdgmh")
    assert p["languages"] == ["he", "en"] and p["fonts"]["body"] == "Rubik" and p["tone"] == "warm"
    assert run_cli("clients.py", "set", "chnvt-hdgmh", "slug=zz").returncode == 1
    run_cli("clients.py", "link", "chnvt-hdgmh", a)
    assert run_cli("clients.py", "unlink", "chnvt-hdgmh", a).returncode == 0
    assert profile(home, "chnvt-hdgmh")["styles"] == []

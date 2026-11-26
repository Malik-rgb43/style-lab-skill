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


def test_readd_does_not_overwrite_kept_folder(home):
    add_client()
    run_cli("clients.py", "set", "chnvt-hdgmh", "notes=keep me")
    run_cli("clients.py", "remove", "chnvt-hdgmh")
    f = home / "clients" / "chnvt-hdgmh" / "client.json"
    before = f.read_text(encoding="utf-8")
    r = run_cli("clients.py", "add", "--slug", "chnvt-hdgmh", "--name", "חנות הדגמה")
    assert r.returncode == 1 and "restore" in r.stderr and "Traceback" not in r.stderr
    assert f.read_text(encoding="utf-8") == before


def test_restore_reregisters(home):
    add_client()
    run_cli("clients.py", "remove", "chnvt-hdgmh")
    assert run_cli("clients.py", "restore", "nope").returncode == 1
    r = run_cli("clients.py", "restore", "chnvt-hdgmh")
    assert r.returncode == 0, r.stderr
    assert run_cli("clients.py", "show", "chnvt-hdgmh").returncode == 0
    assert run_cli("clients.py", "restore", "chnvt-hdgmh").returncode == 1


def test_colour_inputs(home):
    for i, (raw, want) in enumerate([("008754", "#008754"), ("#0a5", "#00AA55"), (" #008754 ", "#008754")]):
        r = run_cli("clients.py", "add", "--slug", f"c{i}", "--name", f"C{i}", "--color", raw)
        assert r.returncode == 0, r.stderr
        assert profile(home, f"c{i}")["brand_colors"] == [want]
    for bad in ("##008754", "rgb(1,2,3)"):
        r = run_cli("clients.py", "add", "--slug", "bad", "--name", "Bad", "--color", bad)
        assert r.returncode == 1 and "Traceback" not in r.stderr


def test_auto_slug_skips_kept_folder(home):
    add_client()
    run_cli("clients.py", "remove", "chnvt-hdgmh")
    r = run_cli("clients.py", "add", "--name", "חנות הדגמה")
    assert r.returncode == 0, r.stderr
    assert "chnvt-hdgmh-2" in r.stdout and "restore" in r.stdout
    assert profile(home, "chnvt-hdgmh-2")["name"] == "חנות הדגמה"


def test_restore_rejects_bad_slug(home):
    outside = home / "x"  # a profile outside clients/ that ../x would reach
    outside.mkdir(parents=True)
    (outside / "client.json").write_text(json.dumps({"slug": "x", "name": "X"}), encoding="utf-8")
    for bad in ("../x", "..", "A", "x/y"):
        r = run_cli("clients.py", "restore", bad)
        assert r.returncode == 1 and "Traceback" not in r.stderr, bad
        assert "lowercase latin letters" in r.stderr, bad
    assert not (home / "clients.json").exists()


def test_clients_uses_palette_parse_hex():
    import clients
    import palette
    assert clients.parse_hex is palette.parse_hex
